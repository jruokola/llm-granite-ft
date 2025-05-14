#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FSDP + QLoRA + LoRA fine-tuning script for multi-GPU clusters (PyTorch 2.2).
Key improvements vs. original draft
• logging defined before first use
• correct GradScaler import and unified dtype handling
• optional AMP off path (uses bfloat16)
• custom FSDP auto-wrap skips Linear4bit sub-modules
• removed per-batch torch.cuda.empty_cache()
• optional nvidia-smi print only on rank 0
• dataset collate & sampler untouched
"""

# -------------------------------------------------------------------------- #
# 0   Standard libs
# -------------------------------------------------------------------------- #
import argparse
import functools
import logging
import math
import os
import subprocess
import time
from collections import Counter

# -------------------------------------------------------------------------- #
# 1   PyTorch / HF imports
# -------------------------------------------------------------------------- #
import torch
from bitsandbytes.nn.modules import Linear4bit
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# -------------------------------------------------------------------------- #
# 2   CLI
# -------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--no_fsdp", action="store_true")
parser.add_argument("--no_layer_wrap_policy", action="store_true")
parser.add_argument("--batch_size_per_device", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--max_training_steps", type=int, default=-1)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--disable_amp", action="store_true")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument(
    "--model_name_or_path", type=str, default="ibm-granite/granite-3.3-2b-instruct"
)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--dataset_subset_size", type=int, default=-1)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
# QLoRA / LoRA
parser.add_argument("--use_qlora", action="store_true")
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj")
parser.add_argument("--processed_dataset_path", type=str, required=True)
args = parser.parse_args()

# -------------------------------------------------------------------------- #
# 3   Logging and dist init
# -------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO if int(os.environ.get("RANK", 0)) == 0 else logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

torch.distributed.init_process_group(backend="nccl")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

logger.info(f"Rank {torch.distributed.get_rank()} initialised on GPU {device}.")


# -------------------------------------------------------------------------- #
# 4   Dataset helper
# -------------------------------------------------------------------------- #
class PreprocessedFunctionCallingDataset(Dataset):
    def __init__(self, split, subset_size=-1):
        self.dataset = (
            split.select(range(subset_size)) if 0 < subset_size < len(split) else split
        )
        self.dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        if len(self.dataset):
            logger.info(f"Sample keys: {list(self.dataset[0].keys())}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @staticmethod
    def collate_fn(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# -------------------------------------------------------------------------- #
# 5   Tokenizer & model (QLoRA or full)
# -------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, cache_dir=".cache", trust_remote_code=True
)

amp_dtype = torch.float16 if not args.disable_amp else torch.bfloat16
scaler = GradScaler() if not args.disable_amp else None

if args.use_qlora:
    logger.info("Using QLoRA (4-bit)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=amp_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.float16,  # keeps 4-bit blocks FP16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=".cache",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    # Print model layer dtypes
    print(Counter(p.dtype for p in model.parameters()))
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    lora_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    peft_cfg = LoraConfig(
        TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=".cache",
        torch_dtype=amp_dtype,
        trust_remote_code=True,
    )

# -------------------------------------------------------------------------- #
# 6   FSDP / DDP wrap
# -------------------------------------------------------------------------- #
if args.no_fsdp:
    model = DDP(model.to(device))
else:

    def custom_wrap(m):
        if isinstance(m, Linear4bit):
            return False
        return "layer" in m.__class__.__name__.lower()

    auto_policy = (
        None
        if args.no_layer_wrap_policy
        else functools.partial(lambda_auto_wrap_policy, lambda_fn=custom_wrap)
    )

    # ----- Cast leftover FP32 tensors ----- 5-10% less memory used
    # target_dtype = torch.float16 if not args.disable_amp else torch.bfloat16
    # if not args.disable_amp:  # we train in FP16
    #    for p in model.parameters():
    #       if p.dtype == torch.float32:
    #            p.data = p.data.to(torch.float16)

    mp_policy = MixedPrecision(
        param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype
    )
    model = FSDP(
        model,
        device_id=device,
        auto_wrap_policy=auto_policy,
        sync_module_states=True,
        use_orig_params=True,
        mixed_precision=mp_policy,
    )

# -------------------------------------------------------------------------- #
# 7   Data loading
# -------------------------------------------------------------------------- #
full_ds = load_from_disk(args.processed_dataset_path)
train_sz = int(0.9 * len(full_ds))
train_set = PreprocessedFunctionCallingDataset(
    full_ds.select(range(train_sz)), args.dataset_subset_size
)
eval_set = PreprocessedFunctionCallingDataset(
    full_ds.select(range(train_sz, len(full_ds))),
    min(1000, len(full_ds)) if args.dataset_subset_size > 0 else -1,
)

train_sampler = DistributedSampler(train_set, shuffle=True, seed=42)
eval_sampler = DistributedSampler(eval_set, shuffle=False, seed=42)
train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size_per_device,
    sampler=train_sampler,
    num_workers=4,
    collate_fn=train_set.collate_fn,
)
eval_loader = DataLoader(
    eval_set,
    batch_size=args.batch_size_per_device,
    sampler=eval_sampler,
    collate_fn=eval_set.collate_fn,
)

# -------------------------------------------------------------------------- #
# 8   Optimiser & schedulers
# -------------------------------------------------------------------------- #
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
max_steps = (
    args.num_epochs * steps_per_epoch
    if args.max_training_steps < 0
    else min(args.max_training_steps, args.num_epochs * steps_per_epoch)
)
warmup_steps = int(args.warmup_ratio * max_steps)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (max_steps - step) / max(1, max_steps - warmup_steps))


scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# -------------------------------------------------------------------------- #
# 9   Train & eval helpers
# -------------------------------------------------------------------------- #
def evaluate():
    model.eval()
    loss_sum, n = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast("cuda", torch.float16) if scaler else torch.no_grad()
            with ctx:
                loss_sum += model(**batch).loss.detach().float()
            n += 1
    loss = (loss_sum / n).to(device)
    torch.distributed.all_reduce(loss)
    return (loss / torch.distributed.get_world_size()).item()


def save_ckpt(name):
    if torch.distributed.get_rank() != 0:
        return
    out = os.path.join(args.output_dir, name)
    os.makedirs(out, exist_ok=True)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
        model.module.save_pretrained(out, state_dict=state)
    else:
        (model.module if hasattr(model, "module") else model).save_pretrained(out)
    logger.info(f"Checkpoint saved: {out}")


# -------------------------------------------------------------------------- #
# 10  Training loop
# -------------------------------------------------------------------------- #
if torch.distributed.get_rank() == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(subprocess.check_output(["nvidia-smi"]).decode())

model.train()
global_step = 0
best_eval = float("inf")

for epoch in range(args.num_epochs):
    train_sampler.set_epoch(epoch)
    epoch_loss = 0
    start = time.time()
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        ctx = torch.autocast("cuda", torch.float16) if scaler else torch.no_grad()
        with ctx:
            loss = model(**batch).loss / args.gradient_accumulation_steps
        (scaler.scale(loss) if scaler else loss).backward()

        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad()
            scheduler.step()
            global_step += 1

            if global_step % args.eval_steps == 0:
                eval_loss = evaluate()
                if torch.distributed.get_rank() == 0:
                    logger.info(f"Eval loss: {eval_loss:.4f} @ step {global_step}")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    save_ckpt("best")
            if global_step % args.save_steps == 0:
                save_ckpt(f"step-{global_step}")
            if 0 < args.max_training_steps <= global_step:
                break
        epoch_loss += loss.detach().float()

    if torch.distributed.get_rank() == 0:
        logger.info(
            f"Epoch {epoch + 1} finished in {time.time() - start:.1f}s "
            f"‒ avg loss {(epoch_loss / len(train_loader)).item():.4f}"
        )
        save_ckpt(f"epoch-{epoch + 1}")
    if 0 < args.max_training_steps <= global_step:
        break

logger.info("Training done.")
save_ckpt("final")
torch.distributed.destroy_process_group()
torch.distributed.barrier()
