#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning with single-shard FSDP on H100 pairs (PyTorch 2.4).

 • 4-bit NF4 base weights (bnb) + LoRA adapters (r=8)
 • Optional FP8 forward pass via NVIDIA-Transformer-Engine
 • Works on 2× H100 (1 GPU / node) – no shared file-system required
"""

# ── Std lib ──────────────────────────────────────────────────────────────
import argparse
import logging
import math
import os
import subprocess
from collections import Counter
from contextlib import nullcontext

# ── PT / HF / bnb ─────────────────────────────────────────────────────────
import torch
import torch.distributed as dist
from datasets import load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── Transformer-Engine (optional FP8) ─────────────────────────────────────
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import fp8_autocast

    TE_AVAILABLE = True
except ModuleNotFoundError:
    TE_AVAILABLE = False

# ── CLI ───────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
# infra
p.add_argument("--batch_size_per_device", type=int, default=128)
p.add_argument("--gradient_accumulation_steps", type=int, default=4)
p.add_argument("--learning_rate", type=float, default=6e-5)
p.add_argument("--num_epochs", type=int, default=3)
p.add_argument("--warmup_ratio", type=float, default=0.1)
p.add_argument("--max_training_steps", type=int, default=-1)
p.add_argument("--gradient_checkpointing", action="store_true")
p.add_argument("--disable_amp", action="store_true")
p.add_argument("--no_fsdp", action="store_true")
# model / data
p.add_argument("--model_name_or_path", default="ibm-granite/granite-3.3-2b-instruct")
p.add_argument("--processed_dataset_path", required=True)
p.add_argument("--output_dir", default="./checkpoints")
# LoRA / QLoRA
p.add_argument("--use_qlora", action="store_true")
p.add_argument("--lora_r", type=int, default=8)
p.add_argument("--lora_alpha", type=int, default=32)
p.add_argument("--lora_dropout", type=float, default=0.05)
p.add_argument("--lora_target_modules", default="q_proj,v_proj")
# Hopper FP8
p.add_argument("--use_fp8", action="store_true", help="H100-only")
args = p.parse_args()

# ── Logging & dist init ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO if int(os.getenv("RANK", 0)) == 0 else logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

dist.init_process_group("nccl")
rank, local_rank = int(os.getenv("RANK", 0)), int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
log.info(f"Rank {rank} initialised on {torch.cuda.get_device_name(device)}")


# ── Dataset helper ────────────────────────────────────────────────────────
class Preprocessed(Dataset):
    def __init__(self, split):
        self.ds = split
        self.ds.set_format("torch", ["input_ids", "attention_mask", "labels"])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]

    @staticmethod
    def col(ex):
        return {k: torch.stack([e[k] for e in ex]) for k in ex[0]}


# ── Tokenizer & model load ────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(
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

log.info(f"Dtype histogram: {Counter(p.dtype for p in model.parameters())}")

# ── LoRA attach ───────────────────────────────────────────────────────────
if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    lora_cfg = LoraConfig(
        TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_target_modules.split(",")],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

# ── FSDP (single shard, ignore int8) ───────────────────────────────────────
if not args.no_fsdp:
    ignored = [
        m
        for m in model.modules()
        if any(p.dtype == torch.int8 for p in m.parameters(recurse=False))
    ]
    model = torch.compile(model, backend="inductor", mode="max-autotune")
    model = FSDP(
        model,
        device_id=device,
        use_orig_params=True,
        auto_wrap_policy=transformer_auto_wrap_policy,
        ignored_modules=ignored,
        sync_module_states=True,
    )
else:
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device), device_ids=[local_rank]
    )
# ── Data ──────────────────────────────────────────────────────────────────
full = load_from_disk(args.processed_dataset_path)
split = int(0.9 * len(full))
train_set = Preprocessed(full.select(range(split)))
eval_set = Preprocessed(full.select(range(split, len(full))))

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(train_set, shuffle=True, seed=42),
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persisten_workers=True,
    collate_fn=train_set.col,
)
eval_loader = DataLoader(
    eval_set,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(eval_set, shuffle=False, seed=42),
    num_workers=4,
    pin_memory=True,
    persisten_workers=True,
    collate_fn=eval_set.col,
)

# ── Optimiser & scheduler ────────────────────────────────────────────────
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
steps_ep = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
max_steps = (
    args.num_epochs * steps_ep
    if args.max_training_steps < 0
    else args.max_training_steps
)
warm = int(args.warmup_ratio * max_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: s / warm
    if s < warm
    else max(0, (max_steps - s) / max(1, max_steps - warm)),
)


# ── Eval helper ───────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    tot = torch.zeros([], device=device)
    with torch.no_grad():
        for b in eval_loader:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast("cuda", amp_dtype) if scaler else torch.no_grad():
                tot += model(**b).loss.float()
    dist.all_reduce(tot)
    return (tot / (len(eval_loader) * dist.get_world_size())).item()


def save(tag):
    if rank:
        return
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.module.save_pretrained(path)
    log.info(f"Checkpoint {tag} saved.")


# ── Training loop ─────────────────────────────────────────────────────────
if rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(subprocess.check_output(["nvidia-smi"]).decode())

model.train()
gstep, best = 0, 1e9
for epoch in range(args.num_epochs):
    train_loader.sampler.set_epoch(epoch)
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}

        ctx1 = torch.autocast("cuda", amp_dtype) if scaler else torch.no_grad()
        fp8_recipe = DelayedScaling(fp8_format="HYBRID", margin=0)
        ctx2 = (
            fp8_autocast(enabled=True, recipe=fp8_recipe)
            if args.use_fp8 and TE_AVAILABLE
            else nullcontext()
        )
        with ctx1, ctx2:
            loss = model(**batch).loss / args.gradient_accumulation_steps

        (scaler.scale(loss) if scaler else loss).backward()

        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            (scaler.step(opt) if scaler else opt.step())
            (scaler.update() if scaler else None)
            opt.zero_grad()
            sched.step()
            gstep += 1

            if gstep % 100 == 0 and rank == 0:
                log.info(f"epoch {epoch + 1} step {gstep} loss {loss.item():.4f}")

            if gstep % 500 == 0:
                val = evaluate()
                if rank == 0:
                    log.info(f"eval {val:.4f} best {best:.4f}")
                    if val < best:
                        best = val
                        save("best")

            if 0 < args.max_training_steps == gstep:
                break
    if 0 < args.max_training_steps == gstep:
        break

if rank == 0:
    save("final")
dist.destroy_process_group()
