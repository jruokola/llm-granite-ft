#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning on a single NVIDIA Consumer GPU.

• 4-bit NF4 base weights  +  LoRA adapters (r=16, α=32)
• Flash-Attention-2 auto-enabled on Ampere (e.g. RTX 30x0/40x0) and newer GPUs.
"""

# ───────────────────────── Std lib ──────────────────────────
import argparse
import logging  # For simplified logging
import math
import os
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext

# ─────────────────────── 3rd-party deps ────────────────────
import torch

# Removed: import torch.distributed as dist (not needed for single GPU)
from datasets import load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.amp import GradScaler  # Standard GradScaler

# Removed: FSDP specific imports
from torch.utils.data import DataLoader, Dataset

# Removed: DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Removed Transformer Engine related imports for FP8

# ──────────────────────── CLI args ─────────────────────────
cli = argparse.ArgumentParser(description="Single GPU QLoRA + LoRA fine-tuning script.")
# infra
cli.add_argument(
    "--batch_size_per_device", type=int, default=16
)  # Name kept for consistency, but it's total batch size now
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
    help="Mixed precision mode when AMP is enabled and FP8 is not used (default: bf16)",
)
# Removed: --no_fsdp argument

# model / data
cli.add_argument("--model_name_or_path", default="ibm-granite/granite-3.3-2b-instruct")
cli.add_argument("--processed_dataset_path", required=True)
cli.add_argument(
    "--output_dir", default="./checkpoints_local_nvidia"
)  # Adjusted default
# LoRA / QLoRA
cli.add_argument("--use_qlora", action="store_true")
cli.add_argument("--lora_r", type=int, default=16)
cli.add_argument("--lora_alpha", type=int, default=32)
cli.add_argument("--lora_dropout", type=float, default=0.05)
cli.add_argument("--lora_target_modules", default="q_proj,v_proj")
# Removed --use_fp8 CLI argument
cli.add_argument(
    "--disable_torch_compile",
    action="store_true",
    help="Disable torch.compile even if available",
)
args = cli.parse_args()

# ────────────── Logging & Device Setup ─────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    logger.error("CUDA is not available. This script requires an NVIDIA GPU.")
    sys.exit(1)

device = torch.device("cuda:0")  # Use the first available CUDA device
torch.cuda.set_device(device)
logger.info(f"Using device: {torch.cuda.get_device_name(device)}")


# Removed FP8 related checks and TE_OK logic


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

if tok.pad_token is None:
    if tok.eos_token:
        tok.pad_token = tok.eos_token
        logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token: {tok.eos_token}")
    else:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        logger.info("Added [PAD] as pad_token, as eos_token was not found.")

SOT = "<|start_of_role|>"
EOTR = "<|end_of_role|>"
EOTXT = "<|end_of_text|>"
TOOL_CALL_MARKER_GRANITE = "<|tool_call|>"
granite_special_tokens = [SOT, EOTR, EOTXT, TOOL_CALL_MARKER_GRANITE]
newly_added_tokens = []
for special_token in granite_special_tokens:
    if special_token not in tok.vocab:
        newly_added_tokens.append(special_token)

if newly_added_tokens:
    tok.add_special_tokens({"additional_special_tokens": newly_added_tokens})
    logger.info(f"Added new special tokens: {newly_added_tokens}")
else:
    logger.info("Granite special tokens already exist in tokenizer vocabulary.")


# Configure amp_dtype and GradScaler
scaler = None
if args.disable_amp:
    amp_dtype = torch.float32
    logger.info("AMP disabled. Training in FP32. GradScaler is disabled.")
# Removed FP8 specific amp_dtype logic: elif args.use_fp8 and TE_OK:
else:  # AMP enabled
    if args.amp_precision_mode == "bf16":
        amp_dtype = torch.bfloat16
        if torch.cuda.is_bf16_supported():
            logger.info(
                f"AMP enabled with BF16 ({amp_dtype}). GradScaler is disabled (not typically used with BF16)."
            )
        else:
            logger.warning(
                "BF16 not supported on this GPU. Falling back to FP16 for AMP."
            )
            amp_dtype = torch.float16
            scaler = GradScaler()
            logger.info(
                f"AMP enabled with FP16 ({amp_dtype}). Standard GradScaler is ENABLED."
            )
    elif args.amp_precision_mode == "fp16":
        amp_dtype = torch.float16
        scaler = GradScaler()
        logger.info(
            f"AMP enabled with FP16 ({amp_dtype}). Standard GradScaler is ENABLED."
        )
    else:  # Should not happen
        amp_dtype = torch.bfloat16
        logger.error(
            f"Invalid amp_precision_mode: {args.amp_precision_mode}. Defaulting to BF16. Check GPU support."
        )
        if not torch.cuda.is_bf16_supported():
            amp_dtype = torch.float16
            scaler = GradScaler()


model_load_torch_dtype = amp_dtype
if args.use_qlora:
    logger.info(
        f"QLoRA active: Model `torch_dtype` and `bnb_4bit_quant_storage` will be {model_load_torch_dtype}. "
        f"QLoRA compute dtype (`bnb_4bit_compute_dtype`) will be {amp_dtype}."
    )
else:
    logger.info(
        f"QLoRA not active: Loading model with `torch_dtype` {model_load_torch_dtype}."
    )


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

if newly_added_tokens:
    model.resize_token_embeddings(len(tok))
    logger.info(
        f"Resized model token embeddings to {len(tok)} to accommodate new special tokens."
    )

if args.gradient_checkpointing:
    model.config.use_cache = False
else:
    model.config.use_cache = True

logger.info(
    f"Dtype histogram after model load: {Counter(p.dtype for p in model.parameters())}"
)
if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if (
        args.gradient_checkpointing
    ):  # PEFT handles this, but explicit call for clarity/safety
        base_model_for_gc = model.base_model if hasattr(model, "base_model") else model
        if hasattr(base_model_for_gc, "gradient_checkpointing_enable"):
            base_model_for_gc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled with use_reentrant=False.")

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


# Removed fp8_ctx function
# Removed TE layer swapping logic: if args.use_fp8 and TE_OK:

# Cast remaining parameters to amp_dtype if not already
# This is important after all model modifications (QLoRA, LoRA)
# QLoRA base layers are handled by BitsAndBytesConfig.
# Other layers (embeddings, norms, final head, LoRA adapters) should be in amp_dtype.
for module_name, module in model.named_modules():
    # Skip BnB modules as their dtype is managed by BnB
    if "bitsandbytes.nn.modules" in str(type(module)).lower():
        continue
    # No TE layers to skip anymore

    for param_name, param in module.named_parameters(recurse=False):
        if param.dtype != amp_dtype:
            try:
                param.data = param.data.to(amp_dtype)
            except Exception as e:
                logger.error(
                    f"Failed to cast {module_name}.{param_name} to {amp_dtype}: {e}"
                )

logger.info(
    f"Dtype histogram before model.to(device): {Counter(p.dtype for p in model.parameters())}"
)

model.to(device)  # Move the entire model to the device
logger.info(
    f"Model moved to {device}. Dtype histogram after model.to(device): {Counter(p.dtype for p in model.parameters())}"
)


# torch.compile (optional)
if hasattr(torch, "compile") and not args.disable_torch_compile:
    # Skip compile if BF16 AMP is used.
    # For single GPU, this might be less problematic, but keep skip for now for consistency.
    skip_compile_for_bf16_amp = (
        not args.disable_amp
        and amp_dtype == torch.bfloat16
        and torch.cuda.is_bf16_supported()
    )
    # Removed (args.use_fp8 and TE_OK) from skip_compile_logic

    if skip_compile_for_bf16_amp:
        logger.info("Skipping torch.compile due to BF16 AMP configuration.")
    else:
        logger.info("Attempting torch.compile on model...")
        try:
            model = torch.compile(model, backend="inductor", mode="max-autotune")
            logger.info("torch.compile successful.")
        except Exception as e:
            logger.warning(
                f"torch.compile failed: {e}. Proceeding with uncompiled model."
            )


# Load dataset from disk
ds = load_from_disk(args.processed_dataset_path)
# Assuming ds is a single HuggingFace Dataset object, not a DatasetDict
train_size = int(0.9 * len(ds))
eval_size = len(ds) - train_size

train_indices = list(range(train_size))
val_indices = list(range(train_size, len(ds)))

if not train_indices:
    logger.error("Training dataset is empty after split. Check dataset path and size.")
    sys.exit(1)

train_ds_raw = ds.select(train_indices)
val_ds_raw = (
    ds.select(val_indices) if val_indices else ds.select([])
)  # Handle empty val set

logger.info(f"Raw train dataset size: {len(train_ds_raw)}")
logger.info(f"Raw validation dataset size: {len(val_ds_raw)}")


train_dataset = Split(train_ds_raw)
val_dataset = Split(val_ds_raw)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=True,  # Shuffle for training
    num_workers=2,  # Adjust as needed
    pin_memory=True,
    collate_fn=Split.collate,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size_per_device,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=Split.collate,
)
opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
steps_ep = (
    math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if len(train_loader) > 0
    else 1
)
max_steps = (
    args.num_epochs * steps_ep
    if args.max_training_steps < 0
    else args.max_training_steps
)
if (
    max_steps == 0 and args.num_epochs > 0 and len(train_loader) > 0
):  # Ensure max_steps is at least 1 if training is intended
    max_steps = steps_ep * args.num_epochs
    if max_steps == 0:
        max_steps = 1  # If dataset is tiny

warm = int(args.warmup_ratio * max_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: s / warm
    if s < warm
    else max(0.0, (max_steps - s) / max(1, max_steps - warm)),
)


@torch.no_grad()
def val_loss_fn():  # Renamed to avoid conflict
    model.eval()
    tot_loss = torch.zeros([], device=device)  # Ensure tensor is on the correct device
    num_batches = 0
    if not val_loader or len(val_loader) == 0:  # Check if val_loader is empty
        logger.warning("Validation loader is empty. Skipping validation.")
        return float("nan")  # Return NaN or some indicator of no validation

    for b in val_loader:
        b = {k: v.to(device) for k, v in b.items()}
        with (
            torch.autocast(
                "cuda", dtype=amp_dtype
            )  # 'cuda' is the device_type for autocast with NVIDIA GPUs
            if not args.disable_amp and amp_dtype != torch.float32
            else nullcontext(),
            # Removed fp8_ctx()
        ):
            outputs = model(**b)
            loss = outputs.loss
            if loss is not None:  # Ensure loss is not None
                tot_loss += loss.float()
        num_batches += 1

    if num_batches == 0:
        logger.warning("No batches processed in validation. Returning NaN.")
        return float("nan")
    return (tot_loss / num_batches).item()


def save_checkpoint(model_to_save, tokenizer_to_save, output_path_tag):
    path = os.path.join(args.output_dir, output_path_tag)
    os.makedirs(path, exist_ok=True)

    # For PEFT models, save_pretrained is preferred.
    # The model object here is already the PEFT model if LoRA/QLoRA is used.
    if hasattr(model_to_save, "save_pretrained"):
        model_to_save.save_pretrained(path)
    else:  # Fallback for non-PEFT models (though this script focuses on PEFT)
        torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))

    if tokenizer_to_save:
        tokenizer_to_save.save_pretrained(path)
    logger.info(f"✓ Checkpoint '{output_path_tag}' saved to {path}")


if args.output_dir:  # Ensure output_dir is created
    os.makedirs(args.output_dir, exist_ok=True)

try:  # nvidia-smi check
    smi_output = subprocess.check_output(["nvidia-smi"]).decode().strip()
    logger.info(f"nvidia-smi output:\n{smi_output}")
except Exception as e:
    logger.warning(f"Could not run nvidia-smi: {e}")

model.train()
gstep = 0
total_loss_agg = 0.0  # For aggregating loss over log_interval
best_val_loss = float("inf")

logger.info(f"Starting training for {args.num_epochs} epochs, {max_steps} total steps.")

for ep in range(args.num_epochs):
    logger.info(f"--- Epoch {ep + 1}/{args.num_epochs} ---")
    epoch_loss_agg = 0.0
    num_epoch_steps = 0

    for step, batch_data in enumerate(
        train_loader, 1
    ):  # Renamed batch to batch_data to avoid conflict
        batch = {
            k: v.to(device) for k, v in batch_data.items()
        }  # Corrected: use batch_data.items()
        with (
            torch.autocast("cuda", dtype=amp_dtype)
            if not args.disable_amp and amp_dtype != torch.float32
            else nullcontext(),
            # Removed fp8_ctx()
        ):
            outputs = model(**batch)
            raw_loss = outputs.loss
            if raw_loss is None:
                logger.warning(f"Step {gstep}: Loss is None. Skipping batch.")
                continue

            loss = raw_loss.float() / args.gradient_accumulation_steps

        if torch.isinf(loss).any() or torch.isnan(loss).any():
            logger.error(
                f"Loss became inf/nan BEFORE backward (value: {loss.item()}). Raw loss: {raw_loss.item()}. Skipping step."
            )
            opt.zero_grad()  # Clear any potentially bad grads from previous partial accumulations
            continue

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss_agg += (
            loss.item() * args.gradient_accumulation_steps
        )  # Log un-normalized loss
        epoch_loss_agg += loss.item() * args.gradient_accumulation_steps
        num_epoch_steps += 1

        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)  # Unscale before clipping

            # Clip gradients - applied to all model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scaler:
                scaler_step_output = scaler.step(opt)
                if scaler_step_output is None:  # Check if grads were inf/nan
                    logger.warning(
                        f"Step {gstep}: Gradients were inf/nan. Optimizer step skipped."
                    )
                scaler.update()
            else:
                opt.step()

            opt.zero_grad()
            sched.step()
            gstep += 1

            if gstep % 10 == 0:  # Log every 10 global steps
                avg_loss_interval = (
                    total_loss_agg / (10 * args.gradient_accumulation_steps)
                    if gstep > 0
                    else total_loss_agg
                )
                current_lr = sched.get_last_lr()[0]
                logger.info(
                    f"Epoch {ep + 1} | Step {gstep}/{max_steps} | Loss {avg_loss_interval:.4f} | LR {current_lr:.2e}"
                )
                total_loss_agg = 0.0  # Reset for next interval

            if gstep > 0 and gstep % args.eval_steps == 0 and len(val_loader) > 0:
                current_val_loss = val_loss_fn()
                logger.info(
                    f"Validation Loss @ GStep {gstep}: {current_val_loss:.4f} (Best: {best_val_loss:.4f})"
                )
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    logger.info(
                        f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint 'best_model'."
                    )
                    save_checkpoint(model, tok, "best_model")
                model.train()  # Set model back to train mode

            if gstep > 0 and gstep % args.save_steps == 0:
                save_checkpoint(model, tok, f"checkpoint_step_{gstep}")

            if 0 < args.max_training_steps <= gstep:
                break

    avg_epoch_loss = (
        epoch_loss_agg / num_epoch_steps if num_epoch_steps > 0 else float("nan")
    )
    logger.info(f"--- Epoch {ep + 1} finished. Average Loss: {avg_epoch_loss:.4f} ---")
    save_checkpoint(model, tok, f"checkpoint_epoch_{ep + 1}")

    if 0 < args.max_training_steps <= gstep:
        logger.info(
            f"Reached max_training_steps ({args.max_training_steps}). Stopping training."
        )
        break

logger.info("Training finished.")
save_checkpoint(model, tok, "final_model")
logger.info("Final model saved.")
