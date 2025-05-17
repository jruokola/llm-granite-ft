#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning on 2 × H100 (1 GPU / node) with single-shard FSDP.

• 4-bit NF4 base weights  +  LoRA adapters (r=16, α=32)
• Optional FP8 forward via NVIDIA-Transformer-Engine (--use_fp8)
• One FSDP shard per rank (use_orig_params=True) → no dtype clashes
• Flash-Attention-2 auto-enabled on Hopper (H100)
"""

# ───────────────────────── Std lib ──────────────────────────
import argparse
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
from torch.distributed.fsdp.sharded_grad_scaler import (
    ShardedGradScaler,  # Import for FSDP+FP16
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
    help="Mixed precision mode when AMP is enabled and FP8 is not used (default: bf16)",
)
cli.add_argument(
    "--no_fsdp",
    action="store_true",
    help="Disable FSDP (use DDP instead if multiple GPUs)",
)
# model / data
cli.add_argument("--model_name_or_path", default="ibm-granite/granite-3.3-2b-instruct")
cli.add_argument("--processed_dataset_path", required=True)
cli.add_argument("--output_dir", default="./checkpoints")
# LoRA / QLoRA
cli.add_argument("--use_qlora", action="store_true")
cli.add_argument("--lora_r", type=int, default=16)
cli.add_argument("--lora_alpha", type=int, default=32)
cli.add_argument("--lora_dropout", type=float, default=0.05)
cli.add_argument("--lora_target_modules", default="q_proj,v_proj")
# Hopper FP8
cli.add_argument(
    "--use_fp8",
    action="store_true",
    help="Enable FP8 on H100 (implies AMP with FP16 for non-FP8 parts)",
)
cli.add_argument(
    "--disable_torch_compile",
    action="store_true",
    help="Disable torch.compile even if available",
)
args = cli.parse_args()


# ────────────── Logging & distributed init ─────────────────
# Define these early as they might be used by test toggles below
def print_rank0_info(msg):
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[INFO] {msg}", flush=True)


def print_rank0_error(msg):
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


if args.use_fp8 and TE_OK and args.use_qlora and (args.lora_r % 16 != 0):
    error_msg = (
        f"FP8 execution with QLoRA requires --lora_r to be a multiple of 16. "
        f"Current --lora_r is {args.lora_r}. Please adjust --lora_r (e.g., to 16, 32, etc.)."
    )
    print_rank0_error(f"CRITICAL ERROR: {error_msg}")
    sys.exit(1)

dist.init_process_group("nccl")
rank, local_rank = int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print_rank0_info(f"[rank {rank}] ready on {torch.cuda.get_device_name(device)}")


# ───────────── Dataset wrapper (already tokenised) ─────────
class Split(Dataset):  # (Content unchanged)
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

# Set pad_token if not already set (mirroring generation script)
if tok.pad_token is None:
    if tok.eos_token:
        tok.pad_token = tok.eos_token
        print_rank0_info(
            f"Set tokenizer.pad_token to tokenizer.eos_token: {tok.eos_token}"
        )
    else:
        # Add a standard pad token if no EOS token is available
        tok.add_special_tokens({"pad_token": "[PAD]"})
        print_rank0_info("Added [PAD] as pad_token, as eos_token was not found.")

# Define and add Granite-specific special tokens
# Based on generate_granite_fc_examples.py and IBM Granite documentation
SOT = "<|start_of_role|>"
EOTR = "<|end_of_role|>"
EOTXT = "<|end_of_text|>"
TOOL_CALL_MARKER_GRANITE = "<|tool_call|>"
# Note: Roles like "system", "user", "assistant", "available_tools", "tool_response"
# are part of the string construction, not necessarily separate special tokens themselves,
# unless the base tokenizer for Granite treats them as such. The key structural tokens are above.
# FIM tokens like <fim_prefix>, <fim_suffix>, <fim_middle> are for FIM tasks, not directly function calling.

granite_special_tokens = [SOT, EOTR, EOTXT, TOOL_CALL_MARKER_GRANITE]
newly_added_tokens = []
for special_token in granite_special_tokens:
    if special_token not in tok.vocab:
        newly_added_tokens.append(special_token)

if newly_added_tokens:
    tok.add_special_tokens({"additional_special_tokens": newly_added_tokens})
    print_rank0_info(f"Added new special tokens: {newly_added_tokens}")
else:
    print_rank0_info("Granite special tokens already exist in tokenizer vocabulary.")


# Configure amp_dtype and GradScaler
scaler = None
if args.disable_amp:
    amp_dtype = torch.float32
    print_rank0_info("AMP disabled. Training in FP32. GradScaler is disabled.")
elif args.use_fp8 and TE_OK:
    amp_dtype = torch.float16
    print_rank0_info(
        f"FP8 enabled (TE_OK=True). Using {amp_dtype} for non-FP8 parts. GradScaler is disabled."
    )
else:  # AMP enabled, FP8 not used
    if args.amp_precision_mode == "bf16":
        amp_dtype = torch.bfloat16
        print_rank0_info(
            f"AMP enabled with BF16 ({amp_dtype}). GradScaler is disabled."
        )
        # scaler remains None
    elif args.amp_precision_mode == "fp16":
        amp_dtype = torch.float16
        if not args.no_fsdp:  # FSDP is active
            scaler = ShardedGradScaler()
            print_rank0_info(
                f"AMP enabled with FP16 ({amp_dtype}) and FSDP. ShardedGradScaler is ENABLED."
            )
        else:  # DDP or single GPU with FP16
            scaler = GradScaler()
            print_rank0_info(
                f"AMP enabled with FP16 ({amp_dtype}) and DDP/SingleGPU. Standard GradScaler is ENABLED."
            )
    else:  # Should not happen due to choices in argparse
        amp_dtype = torch.bfloat16
        print_rank0_error(
            f"Invalid amp_precision_mode: {args.amp_precision_mode}. Defaulting to BF16. GradScaler disabled."
        )

# Determine model_load_torch_dtype for AutoModelForCausalLM.from_pretrained
# This will also be used for bnb_4bit_quant_storage if QLoRA is active.
model_load_torch_dtype = amp_dtype
if args.use_qlora:
    print_rank0_info(
        f"QLoRA active: Model `torch_dtype` and `bnb_4bit_quant_storage` will be {model_load_torch_dtype}. "
        f"QLoRA compute dtype (`bnb_4bit_compute_dtype`) will be {amp_dtype}."
    )
else:
    print_rank0_info(
        f"QLoRA not active: Loading model with `torch_dtype` {model_load_torch_dtype}."
    )

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=args.use_qlora,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=amp_dtype,  # Compute dtype for BnB operations
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=model_load_torch_dtype
    if args.use_qlora
    else None,  # Storage dtype for 4-bit weights, must match model's float type for FSDP
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=".cache",
    quantization_config=bnb_cfg if args.use_qlora else None,
    torch_dtype=model_load_torch_dtype,  # Use the determined loading dtype
    trust_remote_code=True,
)

# Resize token embeddings if new special tokens were added
if newly_added_tokens:
    model.resize_token_embeddings(len(tok))
    print_rank0_info(
        f"Resized model token embeddings to {len(tok)} to accommodate new special tokens."
    )
    # After resizing, it's good practice to check the embedding layer's new size.
    # For many models, this is model.get_input_embeddings().weight.size(0)
    # or model.transformer.wte.weight.size(0) for GPT-like models.
    # This check can be added if further debugging is needed.

# ... (rest of model setup, LoRA, FP8 adapter swap, FSDP wrapping etc. remains the same as the last provided full file) ...
if args.gradient_checkpointing:
    model.config.use_cache = False
else:
    model.config.use_cache = True
print_rank0_info(
    f"Dtype histogram after model load: {Counter(p.dtype for p in model.parameters())}"
)
if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if args.gradient_checkpointing:
        base_model_for_gc = model.base_model if hasattr(model, "base_model") else model
        if hasattr(base_model_for_gc, "gradient_checkpointing_enable"):
            base_model_for_gc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print_rank0_info("Gradient checkpointing enabled with use_reentrant=False.")
        else:
            print_rank0_info(
                "Attempted to enable gradient checkpointing, but base model lacks `gradient_checkpointing_enable`."
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
    if rank == 0:
        model.print_trainable_parameters()


def fp8_ctx():
    if args.use_fp8 and TE_OK:
        from transformer_engine.common.recipe import Format

        recipe = DelayedScaling(fp8_format=Format.HYBRID, margin=0)
        return fp8_autocast(enabled=True, fp8_recipe=recipe)
    return nullcontext()


if args.use_fp8 and TE_OK:
    for name, mod in model.named_modules():
        if "lora_" in name and isinstance(mod, torch.nn.Linear):
            repl = te.Linear(
                mod.in_features, mod.out_features, bias=(mod.bias is not None)
            )
            repl.weight.data.copy_(mod.weight.data)
            parts = name.split(".")
            child_name = parts[-1]
            parent_mod = model
            for p in parts[:-1]:
                parent_mod = (
                    parent_mod[int(p)] if p.isdigit() else getattr(parent_mod, p)
                )
            setattr(parent_mod, child_name, repl)
    print_rank0_info("LoRA adapters swapped to FP8 TE layers.")

fsdp_ignored_modules = []
if args.use_qlora:
    for module_name, module in model.named_modules():
        if "bitsandbytes.nn.modules" in str(type(module)).lower():
            if not any(module is added_module for added_module in fsdp_ignored_modules):
                fsdp_ignored_modules.append(module)
for module_name, module in model.named_modules():
    if any(module is ignored_module for ignored_module in fsdp_ignored_modules):
        continue
    for param_name, param in module.named_parameters(recurse=False):
        if param.dtype != amp_dtype:
            is_te_fp8_lora = (
                args.use_fp8
                and TE_OK
                and "lora_" in param_name
                and hasattr(te, "pytorch")
                and isinstance(module, te.pytorch.Linear)
            )
            if not is_te_fp8_lora:
                param.data = param.data.to(amp_dtype)
print_rank0_info(
    f"Dtype histogram before FSDP: {Counter(p.dtype for p in model.parameters())}"
)

if not args.no_fsdp:
    compile_model = hasattr(torch, "compile") and not args.disable_torch_compile
    skip_compile_for_amp_fp8 = (
        not args.disable_amp and args.amp_precision_mode == "bf16"
    ) or (args.use_fp8 and TE_OK)  # Adjusted for bf16
    if compile_model:
        if skip_compile_for_amp_fp8:
            print_rank0_info(
                "Skipping torch.compile due to BF16 AMP or FP8+TE configuration (potential FSDP conflicts)."
            )
        else:
            print_rank0_info(
                "Attempting torch.compile on model BEFORE FSDP wrapping..."
            )
            try:
                model_to_compile = model
                model = torch.compile(model, backend="inductor", mode="max-autotune")
                print_rank0_info("torch.compile BEFORE FSDP successful.")
            except Exception as e:
                model = model_to_compile
                print_rank0_error(
                    f"torch.compile BEFORE FSDP failed: {e}. Proceeding with uncompiled model."
                )
    if amp_dtype == torch.float16:
        # More conservative mixed precision for FP16 to enhance stability
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,  # Model parameters compute in FP16
            reduce_dtype=torch.float32,  # Gradients reduced in FP32
            buffer_dtype=torch.float32,  # Buffers (e.g., LayerNorm params) in FP32
        )
        print_rank0_info(
            "Using FSDP MixedPrecision: param_dtype=fp16, reduce_dtype=fp32, buffer_dtype=fp32"
        )
    else:
        # Default policy for bf16 or fp32
        mixed_precision_policy = MixedPrecision(
            param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype
        )
    model = FSDP(
        model,
        device_id=device,
        use_orig_params=True,
        ignored_modules=fsdp_ignored_modules if args.use_qlora else None,
        sync_module_states=True,
        mixed_precision=mixed_precision_policy,
    )
    print_rank0_info("FSDP wrapping complete.")
else:
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device), device_ids=[local_rank]
    )
    print_rank0_info("Using DDP instead of FSDP.")

ds = load_from_disk(args.processed_dataset_path)
cut_idx = int(0.9 * len(ds))
train_indices = list(range(cut_idx))
val_indices = list(range(cut_idx, len(ds)))
if dist.get_world_size() > len(train_indices):
    print_rank0_error(
        f"Training dataset size ({len(train_indices)}) is smaller than world size ({dist.get_world_size()}). This can lead to issues."
    )
if dist.get_world_size() > len(val_indices) and len(val_indices) > 0:
    print_rank0_error(
        f"Validation dataset size ({len(val_indices)}) is smaller than world size ({dist.get_world_size()})."
    )
train_ds, val_ds = Split(ds.select(train_indices)), Split(ds.select(val_indices))
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(train_ds, shuffle=True, seed=42, drop_last=False),
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
    collate_fn=Split.collate,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size_per_device,
    sampler=DistributedSampler(val_ds, shuffle=False, seed=42, drop_last=False),
    num_workers=4,
    persistent_workers=True,
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
warm = int(args.warmup_ratio * max_steps)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: s / warm
    if s < warm
    else max(0.0, (max_steps - s) / max(1, max_steps - warm)),
)


@torch.no_grad()
def val_loss():  # (Content mostly unchanged, uses amp_dtype and fp8_ctx)
    model.eval()
    tot_loss = torch.zeros([], device=device)
    num_batches = 0
    if len(val_loader) == 0:
        pass
    for b in val_loader:
        b = {k: v.to(device) for k, v in b.items()}
        with (
            torch.autocast("cuda", amp_dtype)
            if not args.disable_amp
            else nullcontext(),
            fp8_ctx(),
        ):
            tot_loss += model(**b).loss.float()
        num_batches += 1
    loss_sum_tensor = tot_loss.clone().detach()
    num_batches_tensor = torch.tensor(num_batches, device=device, dtype=torch.int64)
    dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
    if num_batches_tensor.item() == 0:
        return 0.0
    return (loss_sum_tensor / num_batches_tensor).item()


def save(tag):  # (Content mostly unchanged)
    if rank != 0:
        return
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)
    if not args.no_fsdp:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            cpu_state_dict = model.state_dict()
        if hasattr(model, "module") and hasattr(model.module, "save_pretrained"):
            model.module.save_pretrained(path, state_dict=cpu_state_dict)
        else:
            torch.save(cpu_state_dict, os.path.join(path, "pytorch_model.bin"))
            print_rank0_info(
                f"Saved FSDP full state_dict to {os.path.join(path, 'pytorch_model.bin')}"
            )
        tok.save_pretrained(path)
    else:
        if hasattr(model, "module"):
            model.module.save_pretrained(path)
        else:
            model.save_pretrained(path)
        tok.save_pretrained(path)
    print_rank0_info(f"✓ Checkpoint '{tag}' saved to {path}")


if rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        smi_output = subprocess.check_output(["nvidia-smi"]).decode().strip()
        print_rank0_info(f"nvidia-smi output:\n{smi_output}")
    except Exception as e:
        print_rank0_error(f"Could not run nvidia-smi: {e}")

model.train()
gstep = 0
best_val_loss = float("inf")
for ep in range(args.num_epochs):
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(ep)
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with (
            torch.autocast("cuda", amp_dtype)
            if not args.disable_amp
            else nullcontext(),
            fp8_ctx(),
        ):
            raw_loss = model(**batch).loss
            # Cast to float32 for accumulation and division to improve stability
            loss = raw_loss.float()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

        # Check for NaN/inf in the loss *before* backward pass
        if torch.isinf(loss).any() or torch.isnan(loss).any():
            print_rank0_error(
                f"Rank {rank} - Loss became inf/nan BEFORE backward (value: {loss.item()}). Raw loss: {raw_loss.item()}. Skipping step."
            )
            opt.zero_grad()
            continue
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % args.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(opt)  # Unscale before clipping

            # Check for NaN/inf in gradients *after* unscaling and *before* clipping/optimizer step
            # This requires iterating through model parameters if FSDP is used.
            # For simplicity, we'll check the loss again, assuming if loss was fine,
            # and backward didn't explode, grads *might* be okay.
            # A more thorough check would involve FSDP's API to inspect grad norms if available,
            # or summing squared grads.
            # However, the FSDP clip_grad_norm_ itself might raise errors if grads are NaN.

            if not args.no_fsdp:
                # Potentially, FSDP's clip_grad_norm_ might handle/error on NaN grads internally.
                # The warning "Called FSDP.clip_grad_norm_() on rank 0 with no gradients"
                # from the initial problem description suggests that sometimes gradients might be zero
                # or not computed, which is different from NaN.
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Final check before optimizer step
            # This is tricky because gradients are distributed.
            # We rely on the optimizer step or scaler.step() to potentially fail or log if issues persist.

            if scaler:
                # scaler.step() can have issues if grads are NaN/inf
                scaler_step_output = scaler.step(opt)
                # scaler_step_output is None if gradients were not finite.
                if scaler_step_output is None and rank == 0:
                    print_rank0_error(
                        f"Rank {rank} - scaler.step() reported non-finite gradients. Skipping optimizer update."
                    )
                    # Grads were not finite, optimizer not stepped.
                    # We might need to zero_grad again if opt.step wasn't called.
                scaler.update()
            else:
                opt.step()
            opt.zero_grad()
            sched.step()
            gstep += 1
            if rank == 0:
                current_lr = sched.get_last_lr()[0]
                if gstep % 10 == 0:
                    print_rank0_info(
                        f"Epoch {ep + 1}/{args.num_epochs} | Step {gstep}/{max_steps} | Batch {step // args.gradient_accumulation_steps}/{len(train_loader) // args.gradient_accumulation_steps if args.gradient_accumulation_steps > 0 else len(train_loader)} | Loss {loss.item() * args.gradient_accumulation_steps:.4f} | LR {current_lr:.2e}"
                    )
            if gstep > 0 and gstep % 200 == 0:
                current_val_loss = val_loss()
                model.train()
                if rank == 0:
                    max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    print_rank0_info(
                        f"Validation Loss after GStep {gstep}: {current_val_loss:.4f} (Best: {best_val_loss:.4f}) | Max GPU Mem: {max_mem_gb:.2f} GB"
                    )
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        print_rank0_info(
                            f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint 'best'."
                        )
                        save("best")
            if 0 < args.max_training_steps <= gstep:
                break
    if 0 < args.max_training_steps <= gstep:
        print_rank0_info(
            f"Reached max_training_steps ({args.max_training_steps}). Stopping training."
        )
        break
if rank == 0:
    print_rank0_info("Training finished. Saving final model.")
    save("final")
dist.destroy_process_group()
print_rank0_info("Distributed process group destroyed. Exiting.")
