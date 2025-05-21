#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning with FSDP2 (modern FSDP).

• 4-bit NF4 base weights  +  LoRA adapters (r=16, α=32)
• FSDP2 style sharding.
• Flash-Attention-2 auto-enabled on Hopper (H100)
"""

# ───────────────────────── Std lib ──────────────────────────
import argparse
import math
import os
import sys
from collections import Counter
from contextlib import nullcontext

# ─────────────────────── 3rd-party deps ────────────────────
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from datasets import load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.amp.grad_scaler import GradScaler
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

# ──────────────────────── CLI args ─────────────────────────
cli = argparse.ArgumentParser()
# infrastructure
cli.add_argument("--batch_size_per_device", type=int, default=16)
cli.add_argument("--gradient_accumulation_steps", type=int, default=4)
cli.add_argument("--learning_rate", type=float, default=6e-5)
cli.add_argument("--num_epochs", type=int, default=3)
cli.add_argument("--max_training_steps", type=int, default=-1)
cli.add_argument("--warmup_ratio", type=float, default=0.1)
cli.add_argument("--gradient_checkpointing", action="store_true")
cli.add_argument(
    "--disable_amp",
    action="store_true",
    help="Disable Automatic Mixed Precision (train in FP32)",
)
cli.add_argument(
    "--amp_precision_mode",
    type=str,
    default="bf16",
    choices=["bf16", "fp16"],
    help="Mixed precision mode when AMP is enabled",
)
cli.add_argument(
    "--no_fsdp",
    action="store_true",
    help="Disable FSDP (use DDP instead)",
)
cli.add_argument(
    "--reshard_after_forward",
    action="store_true",
    help="FSDP2: Reshard parameters after forward pass (ZeRO-3 style)",
)

# model / data
cli.add_argument("--model_name_or_path", default="ibm-granite/granite-3.3-2b-instruct")
cli.add_argument(
    "--processed_dataset_path",
    required=True,
    help="Path to processed dataset on shared filesystem",
)
cli.add_argument("--output_dir", default="./checkpoints")

# LoRA / QLoRA
cli.add_argument("--use_qlora", action="store_true")
cli.add_argument("--lora_r", type=int, default=16)
cli.add_argument("--lora_alpha", type=int, default=32)
cli.add_argument("--lora_dropout", type=float, default=0.05)
cli.add_argument("--lora_target_modules", default="q_proj,v_proj")

cli.add_argument(
    "--disable_torch_compile",
    action="store_true",
    help="Disable torch.compile even if available",
)
args = cli.parse_args()


# ────────────── Stateful App Class for DCP ─────────────────
class AppState(Stateful):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer  # Though optimizer state is not saved in this script's current logic

    def state_dict(self):
        # get_state_dict from torch.distributed.checkpoint.state_dict
        # handles FSDP models and returns model and optimizer state dicts.
        # If self.optimizer is None, optim_state_dict will be None.
        model_state_dict, optim_state_dict = get_state_dict(
            self.model, optimizers=self.optimizer
        )
        return {"model": model_state_dict, "optimizer": optim_state_dict}

    def load_state_dict(self, state_dict):
        # This method would be used if loading with dcp.load and AppState
        # Not strictly needed for this script's save-only focus, but good for completeness.
        from torch.distributed.checkpoint.state_dict import set_state_dict

        set_state_dict(
            self.model,
            optimizers=self.optimizer,
            model_state_dict=state_dict.get("model"),
            optim_state_dict=state_dict.get("optimizer"),
        )


# ────────────── Logging & distributed init ─────────────────
def print_rank0_info(msg):
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[INFO] {msg}", flush=True)


def print_rank0_error(msg):
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


dist.init_process_group("nccl")
rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print_rank0_info(f"[rank {rank}] ready on {torch.cuda.get_device_name(device)}")


# ───────────── Dataset wrapper ─────────────────────────────
class Split(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split
        self.ds.set_format("torch", ["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    @staticmethod
    def collate(rows):
        return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


# ──────────────── Tokenizer & model load ───────────────────
tok = AutoTokenizer.from_pretrained(
    args.model_name_or_path, cache_dir=".cache", trust_remote_code=True
)
if tok.pad_token is None:
    if tok.eos_token:
        tok.pad_token = tok.eos_token
        print_rank0_info(f"Set pad_token to eos_token: {tok.eos_token}")
    else:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        print_rank0_info("Added [PAD] as pad_token")

# Configure amp_dtype and GradScaler
scaler = None
if args.disable_amp:
    amp_dtype = torch.float32
    print_rank0_info("AMP disabled. Training in FP32")
elif args.amp_precision_mode == "bf16":
    amp_dtype = torch.bfloat16
    print_rank0_info("AMP enabled with BF16")
elif args.amp_precision_mode == "fp16":
    amp_dtype = torch.float16
    scaler = GradScaler()
    print_rank0_info("AMP enabled with FP16")
else:
    amp_dtype = torch.bfloat16
    print_rank0_error(
        f"Invalid amp_precision_mode '{args.amp_precision_mode}'; defaulting to BF16"
    )

# BitsAndBytes QLoRA config
model_load_torch_dtype = amp_dtype
if args.use_qlora:
    print_rank0_info(f"QLoRA active: using {model_load_torch_dtype}")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=args.use_qlora,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=amp_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=model_load_torch_dtype if args.use_qlora else None,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=".cache",
    quantization_config=bnb_cfg if args.use_qlora else None,
    torch_dtype=model_load_torch_dtype,
    trust_remote_code=True,
)
if args.gradient_checkpointing:
    model.config.use_cache = False
else:
    model.config.use_cache = True
if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )

# Apply LoRA
lconf = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
    bias="none",
)
model = get_peft_model(model, lconf)
if rank == 0:
    model.print_trainable_parameters()

# Cast remaining parameters to amp_dtype
for name, param in model.named_parameters():
    if param.dtype != amp_dtype:
        param.data = param.data.to(amp_dtype)
print_rank0_info(f"Dtype histogram: {Counter(p.dtype for p in model.parameters())}")

# FSDP2 wrapping
device_mesh = init_device_mesh("cuda", (world_size,))
fsdp_mp_policy = None
if amp_dtype == torch.bfloat16:
    fsdp_mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
elif amp_dtype == torch.float16:
    fsdp_mp_policy = MixedPrecision(
        param_dtype=torch.float16, reduce_dtype=torch.float32
    )
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
    if args.reshard_after_forward
    else ShardingStrategy.SHARD_GRAD_OP,
    device_mesh=device_mesh,
    mixed_precision=fsdp_mp_policy,
    use_orig_params=True,
)
print_rank0_info("FSDP2 wrapping complete")

# Data loaders
ds_loaded = load_from_disk(args.processed_dataset_path)
# Handle cases where ds_loaded might be a DatasetDict
if isinstance(ds_loaded, dict) and "train" in ds_loaded:
    ds_for_split = ds_loaded["train"]
elif not isinstance(ds_loaded, dict):  # if it's a plain Dataset
    ds_for_split = ds_loaded
else:
    raise ValueError(
        f"Loaded dataset at {args.processed_dataset_path} is a dict but does not contain a 'train' key. Found keys: {list(ds_loaded.keys())}"
    )

cut = int(0.9 * len(ds_for_split))
train_ds = Split(ds_for_split.select(range(cut)))
val_ds = Split(ds_for_split.select(range(cut, len(ds_for_split))))
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(train_ds, shuffle=True),
    num_workers=8,
    collate_fn=Split.collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(val_ds, shuffle=False),
    num_workers=4,
    collate_fn=Split.collate,
)

# Optimizer & scheduler
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
total_steps = (
    args.num_epochs * steps_per_epoch
    if args.max_training_steps < 0
    else args.max_training_steps
)
warmup_steps = int(args.warmup_ratio * total_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda step: step / warmup_steps
    if step < warmup_steps
    else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
)


@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with (
            torch.autocast("cuda", amp_dtype) if not args.disable_amp else nullcontext()
        ):
            total_loss += model(**batch).loss.item()
        count += 1
    return total_loss / max(1, count)


def save_checkpoint(tag):
    if rank != 0:
        dist.barrier()  # Ensure rank 0 waits for other ranks if they were doing something before save
        return

    checkpoint_dir = os.path.join(args.output_dir, tag)
    # DCP saves to a directory, os.makedirs will be handled by dcp.save if it doesn't exist.
    # However, tok.save_pretrained needs the directory to exist.
    os.makedirs(checkpoint_dir, exist_ok=True)

    app_state = AppState(
        model, optimizer=None
    )  # Pass model; optimizer state not saved here

    # The state to save with dcp.save is a dictionary of Stateful objects
    state_to_save = {"app": app_state}

    dcp.save(state_to_save, checkpoint_id=checkpoint_dir)

    # Tokenizer is not part of the PyTorch module state, save it separately.
    # dcp.save saves into the checkpoint_dir, so tokenizer should also go there.
    tok.save_pretrained(checkpoint_dir)

    print_rank0_info(f"Saved distributed checkpoint and tokenizer to: {checkpoint_dir}")
    dist.barrier()  # Ensure all ranks complete before rank 0 might proceed/exit


# Training loop
model.train()
step_count = 0
best_val = float("inf")
for epoch in range(args.num_epochs):
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with (
            torch.autocast("cuda", amp_dtype) if not args.disable_amp else nullcontext()
        ):
            loss = model(**batch).loss / args.gradient_accumulation_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad()
            sched.step()
            step_count += 1
            if step_count % 200 == 0:
                val_loss = evaluate()
                model.train()
                if rank == 0 and val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint("best")
    if step_count >= args.max_training_steps > 0:
        break

if rank == 0:
    save_checkpoint("final")
dist.destroy_process_group()
print_rank0_info("Training complete")
