#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning with FSDP for H100 (PyTorch 2.4).
• Single-shard FSDP with use_orig_params=True (no dtype clashes)
• 4-bit weight storage in FP16, LoRA adapters train in AMP (BF16 default)
• Optional Hopper FP8 through NVIDIA Transformer-Engine (flag --use_fp8)
• Large micro-batch + gradient-accum to exploit 80 GB VRAM
• Flash-Attention-2 kernels auto-enabled on H100
• DataLoader tuned (workers, prefetch, pinned)
• Clean shutdown (destroy_process_group) and rank-0 checkpoint gather
"""

# ────────────────────────────────────────────────────────────────────────────
# 0   Standard libs
# ────────────────────────────────────────────────────────────────────────────
import argparse
import logging
import math
import os
import subprocess
from collections import Counter

# ────────────────────────────────────────────────────────────────────────────
# 1   PyTorch / HF / BnB / TE imports
# ────────────────────────────────────────────────────────────────────────────
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import transformer_engine.pytorch as te  # FP8 optional

    TE_AVAILABLE = True
except ModuleNotFoundError:
    TE_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────
# 2   CLI
# ────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
# === training infra ===
parser.add_argument("--no_fsdp", action="store_true")
parser.add_argument("--batch_size_per_device", type=int, default=48)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=6e-5)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--max_training_steps", type=int, default=-1)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument(
    "--disable_amp",
    action="store_true",
    help="Disable mixed-precision; when off we default to BF16",
)
# === model / data ===
parser.add_argument(
    "--model_name_or_path", default="ibm-granite/granite-3.3-2b-instruct"
)
parser.add_argument("--processed_dataset_path", required=True)
parser.add_argument("--max_seq_length", type=int, default=4096)
# === LoRA / QLoRA ===
parser.add_argument("--use_qlora", action="store_true")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--use_fp8", type=bool, default=True)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
# === Hopper extras ===
parser.add_argument(
    "--use_fp8",
    action="store_true",
    help="Enable FP8 layers via Transformer-Engine (H100 only)",
)
args = parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────
# 3   Logging + distributed init
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if int(os.environ.get("RANK", 0)) == 0 else logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
logger.info(f"Rank {torch.distributed.get_rank()} ready on GPU {device}")


# ────────────────────────────────────────────────────────────────────────────
# 4   Dataset wrapper
# ────────────────────────────────────────────────────────────────────────────
class PreprocessedDataset(Dataset):
    def __init__(self, split):
        self.ds = split
        self.ds.set_format("torch", ["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]

    @staticmethod
    def collate(ex):
        return {k: torch.stack([e[k] for e in ex]) for k in ex[0]}


# ────────────────────────────────────────────────────────────────────────────
# 5   Tokenizer & model
# ────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, cache_dir=".cache", trust_remote_code=True
)

amp_dtype = torch.bfloat16 if not args.disable_amp else torch.float32
scaler = GradScaler() if not args.disable_amp else None

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=args.use_qlora,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=amp_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=".cache",
    quantization_config=bnb_cfg if args.use_qlora else None,
    torch_dtype=None if args.use_qlora else amp_dtype,
    trust_remote_code=True,
)

logger.info(f"Param dtype histogram: {Counter(p.dtype for p in model.parameters())}")

if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    lora_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

# optional FP8 for LoRA adapters
if args.use_fp8 and TE_AVAILABLE:
    for name, mod in model.named_modules():
        if "lora_" in name and isinstance(mod, torch.nn.Linear):
            fp8_layer = te.Linear(
                mod.in_features, mod.out_features, bias=mod.bias is not None, fp8=True
            )
            fp8_layer.weight.data.copy_(mod.weight.data)
            parent, child_name = name.rsplit(".", 1)
            setattr(eval("model." + parent), child_name, fp8_layer)
    logger.info("Replaced LoRA adapters with Transformer-Engine FP8 layers.")

# ────────────────────────────────────────────────────────────────────────────
# 6   FSDP wrap (single shard, orig params)
# ────────────────────────────────────────────────────────────────────────────
if args.no_fsdp:
    model = torch.nn.parallel.DistributedDataParallel(model.to(device))
else:
    # find every Linear4bit (or any module with an int8 param)
    ignored = [
        m
        for m in model.modules()
        if any(p.dtype is torch.int8 for p in m.parameters(recurse=False))
    ]

    model = FSDP(
        model,
        device_id=device,
        auto_wrap_policy=transformer_auto_wrap_policy,  # keep your layer shards
        ignored_modules=ignored,  # <- new
        sync_module_states=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# 7   Data loading
# ────────────────────────────────────────────────────────────────────────────
full = load_from_disk(args.processed_dataset_path)
train_size = int(0.9 * len(full))
train_set = PreprocessedDataset(full.select(range(train_size)))
eval_set = PreprocessedDataset(full.select(range(train_size, len(full))))

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(train_set, shuffle=True, seed=42),
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    collate_fn=train_set.collate,
)

eval_loader = DataLoader(
    eval_set,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(eval_set, shuffle=False, seed=42),
    num_workers=4,
    pin_memory=True,
    collate_fn=eval_set.collate,
)

# ────────────────────────────────────────────────────────────────────────────
# 8   Optimiser + LR schedule
# ────────────────────────────────────────────────────────────────────────────
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
max_steps = (
    args.num_epochs * steps_per_epoch
    if args.max_training_steps < 0
    else args.max_training_steps
)
warmup_steps = int(args.warmup_ratio * max_steps)

lr_sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: s / warmup_steps
    if s < warmup_steps
    else max(0.0, (max_steps - s) / max(1, max_steps - warmup_steps)),
)


# ────────────────────────────────────────────────────────────────────────────
# 9   Training / eval helpers
# ────────────────────────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    tot, n = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", amp_dtype) if scaler else torch.no_grad():
                tot += model(**batch).loss.float()
            n += 1
    loss = (tot / n).to(device)
    torch.distributed.all_reduce(loss)
    return (loss / torch.distributed.get_world_size()).item()


def save_ckpt(tag):
    if torch.distributed.get_rank() != 0:
        return
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.module.save_pretrained(path)
    logger.info(f"Saved {path}")


# ────────────────────────────────────────────────────────────────────────────
# 10 Train loop
# ────────────────────────────────────────────────────────────────────────────
if torch.distributed.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(subprocess.check_output(["nvidia-smi"]).decode())

model.train()
global_step = best = 0
best_loss = 1e9
for epoch in range(args.num_epochs):
    train_loader.sampler.set_epoch(epoch)
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast("cuda", amp_dtype) if scaler else torch.no_grad()
        with ctx:
            loss = model(**batch).loss / args.gradient_accumulation_steps
        (scaler.scale(loss) if scaler else loss).backward()
        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt) if scaler else opt.step()
            scaler.update() if scaler else None
            opt.zero_grad()
            lr_sched.step()
            global_step += 1

            if global_step % args.eval_steps == 0:
                val = evaluate()
                if torch.distributed.get_rank() == 0:
                    logger.info(f"step {global_step} eval {val:.4f}")
                if val < best_loss:
                    best_loss, best = val, global_step
                    save_ckpt("best")

            if global_step % args.save_steps == 0 and torch.distributed.get_rank() == 0:
                save_ckpt(f"step-{global_step}")
            if 0 < args.max_training_steps == global_step:
                break
    if 0 < args.max_training_steps == global_step:
        break

if torch.distributed.get_rank() == 0:
    save_ckpt("final")
torch.distributed.destroy_process_group()
