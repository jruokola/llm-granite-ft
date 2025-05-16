#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning on 2 × H100 (1 GPU / node) with single-shard FSDP.

• 4-bit NF4 base weights  +  LoRA adapters (r=8, α=32)
• Optional FP8 forward via NVIDIA-Transformer-Engine (--use_fp8)
• One FSDP shard per rank (use_orig_params=True) → no dtype clashes
• Flash-Attention-2 auto-enabled on Hopper
• No shared filesystem needed – only a common dataset cache inside the container
"""

# ───────────────────────── Std lib ──────────────────────────
import argparse
import logging
import math
import os
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext

# ─────────────────────── 3rd-party deps ────────────────────
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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    StateDictType,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# optional FP8
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling
    from transformer_engine.pytorch import fp8_autocast

    TE_OK = True
except ModuleNotFoundError:  # container w/o TE
    TE_OK = False

# ──────────────────────── CLI args ─────────────────────────
cli = argparse.ArgumentParser()
# infra
cli.add_argument("--batch_size_per_device", type=int, default=128)
cli.add_argument("--gradient_accumulation_steps", type=int, default=4)
cli.add_argument("--learning_rate", type=float, default=6e-5)
cli.add_argument("--num_epochs", type=int, default=3)
cli.add_argument("--max_training_steps", type=int, default=-1)
cli.add_argument("--warmup_ratio", type=float, default=0.1)
cli.add_argument("--gradient_checkpointing", action="store_true")
cli.add_argument("--disable_amp", action="store_true")
cli.add_argument("--no_fsdp", action="store_true")
# model / data
cli.add_argument("--model_name_or_path", default="ibm-granite/granite-3.3-2b-instruct")
cli.add_argument("--processed_dataset_path", required=True)
cli.add_argument("--output_dir", default="./checkpoints")
# LoRA / QLoRA
cli.add_argument("--use_qlora", action="store_true")
cli.add_argument("--lora_r", type=int, default=8)
cli.add_argument("--lora_alpha", type=int, default=32)
cli.add_argument("--lora_dropout", type=float, default=0.05)
cli.add_argument("--lora_target_modules", default="q_proj,v_proj")
# Hopper FP8
cli.add_argument("--use_fp8", action="store_true", help="Enable FP8 on H100")
args = cli.parse_args()

# ────────────── Logging & distributed init ─────────────────
logging.basicConfig(
    level=logging.INFO if int(os.getenv("RANK", 0)) == 0 else logging.WARNING,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
log = logging.getLogger("finetune")

# Check for lora_r compatibility with FP8 if QLoRA and FP8 are enabled
if args.use_fp8 and TE_OK and args.use_qlora and (args.lora_r % 16 != 0):
    error_msg = (
        f"FP8 execution with QLoRA requires --lora_r to be a multiple of 16. "
        f"Current --lora_r is {args.lora_r}. Please adjust --lora_r (e.g., to 16, 32, etc.)."
    )
    log.error(error_msg)  # Log the error
    # Rank 0 is typically responsible for user-facing messages.
    if int(os.getenv("RANK", 0)) == 0:
        print(f"CRITICAL ERROR: {error_msg}", file=sys.stderr)
    # Exit all processes. This check is before dist.init_process_group(), so direct exit is fine.
    sys.exit(1)

dist.init_process_group("nccl")
rank, local_rank = int(os.getenv("RANK", 0)), int(os.getenv("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
log.info(f"[rank {rank}] ready on {torch.cuda.get_device_name(device)}")


# ───────────── Dataset wrapper (already tokenised) ─────────
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
    torch_dtype=amp_dtype,  # Ensures non-quantized parts match amp_dtype consistently
    trust_remote_code=True,
)
log.info(f"Dtype histogram: {Counter(p.dtype for p in model.parameters())}")

# attach LoRA
if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    lconf = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
        bias="none",
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()


# replace LoRA linear layers with FP8 (optional)
def fp8_ctx():
    if args.use_fp8 and TE_OK:
        from transformer_engine.common.recipe import Format  # Import the Enum

        recipe = DelayedScaling(fp8_format=Format.HYBRID, margin=0)
        return fp8_autocast(enabled=True, fp8_recipe=recipe)
    return nullcontext()


# Replace LoRA adapters with FP8 Transformer-Engine layers, avoiding eval()
if args.use_fp8 and TE_OK:
    for name, mod in model.named_modules():
        if "lora_" in name and isinstance(mod, torch.nn.Linear):
            # create a new FP8-enabled layer
            repl = te.Linear(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
            )
            # copy over the trained weights
            repl.weight.data.copy_(mod.weight.data)

            # split the full module path into parts
            # e.g. "base_model.model.layers.0.self_attn.q_proj"
            parts = name.split(".")
            child_name = parts[-1]
            parent_parts = parts[:-1]

            # walk down from `model` to the parent module
            parent_mod = model
            for p in parent_parts:
                if p.isdigit():
                    # numeric index into a ModuleList or list
                    parent_mod = parent_mod[int(p)]
                else:
                    parent_mod = getattr(parent_mod, p)

            # set the new FP8 layer in place of the old Linear
            setattr(parent_mod, child_name, repl)

    log.info("LoRA adapters swapped to FP8 TE layers.")

# Cast parameters of non-BitsAndBytes modules to amp_dtype before FSDP
# This ensures uniformity for parameters FSDP will handle.
# PEFT might upcast LayerNorms/lm_head to float32; this brings them to amp_dtype.
fsdp_ignored_modules = []
if args.use_qlora:  # Only ignore BnB modules if QLoRA is active
    for module_name, module in model.named_modules():
        if "bitsandbytes.nn.modules" in str(type(module)).lower():
            # Check if module is already added to avoid duplicates if nested
            is_already_added = False
            for added_module in fsdp_ignored_modules:
                if module is added_module:
                    is_already_added = True
                    break
            if not is_already_added:
                fsdp_ignored_modules.append(module)

for module_name, module in model.named_modules():
    is_ignored_for_casting = False
    for ignored_type_module in fsdp_ignored_modules:
        if module is ignored_type_module:  # Check actual module instance
            is_ignored_for_casting = True
            break

    if is_ignored_for_casting:
        continue  # Skip casting parameters of ignored (BnB) modules

    for param_name, param in module.named_parameters(recurse=False):
        if param.dtype != amp_dtype:
            # Avoid casting LoRA FP8 layers if they are handled by Transformer Engine
            is_te_fp8_lora = False
            if args.use_fp8 and TE_OK and "lora_" in param_name:
                # Check if the module is a TE Linear layer
                if hasattr(te, "pytorch") and isinstance(module, te.pytorch.Linear):
                    is_te_fp8_lora = True

            if not is_te_fp8_lora:
                if rank == 0:
                    # This logging can be very verbose, enable if needed for deep debugging
                    # log.info(f"Casting {module_name}.{param_name} from {param.dtype} to {amp_dtype}")
                    pass
                param.data = param.data.to(amp_dtype)

if rank == 0:
    log.info(
        f"Dtype histogram before FSDP: {Counter(p.dtype for p in model.parameters())}"
    )
    # To see which modules are being ignored by FSDP:
    # log.info(f"FSDP ignored_modules types: {[type(m) for m in fsdp_ignored_modules]}")


# ──────── FSDP (single shard → ignore int8 safely) ─────────
if not args.no_fsdp:
    # torch.compile can be applied before or after FSDP.
    # If applied before, FSDP wraps the compiled module.
    # If after, FSDP itself is compiled (less common for FSDP itself).
    # Let's keep it before FSDP for now as in the original script.
    if hasattr(torch, "compile") and not (
        args.use_fp8 and TE_OK
    ):  # TE and compile can conflict
        model = torch.compile(model, backend="inductor", mode="max-autotune")

    mixed_precision_policy = MixedPrecision(
        param_dtype=amp_dtype,
        reduce_dtype=amp_dtype,
        buffer_dtype=amp_dtype,
    )

    model = FSDP(
        model,
        device_id=device,
        use_orig_params=True,  # Crucial for PEFT/QLoRA
        ignored_modules=fsdp_ignored_modules if args.use_qlora else None,
        sync_module_states=True,  # Ensure all ranks start with the same model states
        mixed_precision=mixed_precision_policy,
    )
else:
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device), device_ids=[local_rank]
    )

# ─────────────────────── Dataloaders ───────────────────────
ds = load_from_disk(args.processed_dataset_path)
cut = int(0.9 * len(ds))
train_ds, val_ds = Split(ds.select(range(cut))), Split(ds.select(range(cut, len(ds))))

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(train_ds, shuffle=True, seed=42),
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
    collate_fn=train_ds.collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(val_ds, shuffle=False, seed=42),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=val_ds.collate,
)

# ───────────── Optimiser & LR schedule ─────────────────────
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
    else max(0.0, (max_steps - s) / max(1, max_steps - warm)),
)


# eval helper
@torch.no_grad()
def val_loss():
    model.eval()
    tot = torch.zeros([], device=device)
    for b in val_loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", amp_dtype) if scaler else nullcontext():
            tot += model(**b).loss.float()
    dist.all_reduce(tot)
    return (tot / (len(val_loader) * dist.get_world_size())).item()


def save(tag):
    if rank:
        return
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.module.save_pretrained(path)
    log.info(f"✓ checkpoint {tag}")


# ───────────────────── Main train loop ─────────────────────
if rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(subprocess.check_output(["nvidia-smi"]).decode().strip())

model.train()
gstep = 0
best = 1e9
for ep in range(args.num_epochs):
    train_loader.sampler.set_epoch(ep)
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", amp_dtype) if scaler else nullcontext(), fp8_ctx():
            loss = model(**batch).loss / args.gradient_accumulation_steps
        (scaler.scale(loss) if scaler else loss).backward()

        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt) if scaler else opt.step()
            scaler.update() if scaler else None
            opt.zero_grad()
            sched.step()
            gstep += 1

            if gstep % 100 == 0 and rank == 0:
                log.info(f"E{ep + 1} S{gstep} loss {loss.item():.4f}")

            if gstep % 500 == 0:
                vl = val_loss()
                if rank == 0:
                    mb = torch.cuda.max_memory_allocated() / 1e9
                    log.info(f"eval {vl:.4f} (best {best:.4f}) - {mb:.1f} GB max_alloc")
                    if vl < best:
                        best = vl
                        save("best")

            if 0 < args.max_training_steps == gstep:
                break
    if 0 < args.max_training_steps == gstep:
        break

if rank == 0:
    save("final")
dist.destroy_process_group()
