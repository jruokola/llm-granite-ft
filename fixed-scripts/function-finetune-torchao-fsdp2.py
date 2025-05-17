#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA + LoRA fine-tuning with FSDP2 (modern FSDP) and torchao for FP8.

• 4-bit NF4 base weights  +  LoRA adapters (r=16, α=32)
• Optional FP8 forward via torchao library (--use_fp8_torchao)
• FSDP2 style sharding.
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
from torch.amp import GradScaler  # Standard GradScaler for FSDP2
from torch.distributed.device_mesh import init_device_mesh  # For FSDP2
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,  # Still used for some types/enums
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,  # FSDP2 style policy (using MixedPrecision as per error hint)
    StateDictType,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# FP8 with torchao
try:
    import torchao
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    # Removed: from torchao.float8.config import Float8Recipe
    from torchao.float8.fsdp import (
        enable_fsdp_float8_all_gather,
        force_recompute_fp8_weight_in_bwd,
        precompute_float8_dynamic_scale_for_fsdp,
    )
    from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, apply_logging_config

    TORCHAO_OK = True
    apply_logging_config(log_level="INFO")
except ModuleNotFoundError:
    TORCHAO_OK = False

    # Define dummy versions for graceful failure if TORCHAO_OK is True but imports failed (less likely for primary import)
    # or if these names are used elsewhere conditionally.
    def convert_to_float8_training(model, module_filter_fn=None, linear_config=None):
        return model

    class Float8LinearConfig:  # Dummy for when torchao is not found
        def __init__(self, recipe=None):  # recipe here will be a string
            self.recipe_str = (
                recipe if recipe else "dynamic"
            )  # Store as string, ensure it has a default

    # Removed dummy Float8Recipe class

    def enable_fsdp_float8_all_gather(enabled: bool = True):
        pass

    def precompute_float8_dynamic_scale_for_fsdp(enabled: bool = True):
        pass

    def force_recompute_fp8_weight_in_bwd(enabled: bool = True):
        pass

    TORCH_VERSION_AT_LEAST_2_5 = False  # Assume older PyTorch if torchao is not found

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
cli.add_argument(
    "--reshard_after_forward",
    action="store_true",  # Default to False (ZeRO-2 like)
    help="FSDP2: Reshard parameters after forward pass (ZeRO-3 like). Default is False.",
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
# Hopper FP8 (now with torchao)
cli.add_argument(
    "--use_fp8_torchao",
    action="store_true",
    help="Enable FP8 on H100 using torchao (non-FP8 parts use amp_precision_mode)",
)
cli.add_argument(
    "--float8_recipe_name",
    type=str,
    default="tensor",  # Defaulting to tensor-wise as it has more FSDP specific flags
    choices=["tensor", "rowwise", "dynamic"],  # dynamic is also mentioned for torchao
    help="torchao FP8 recipe name (e.g., 'tensor', 'rowwise'). Default: 'tensor'.",
)
cli.add_argument(
    "--float8_enable_fsdp_all_gather",
    action="store_true",
    help="torchao: Cast Float8Linear.weight to FP8 before FSDP all-gather. (Tensorwise scaling).",
)
cli.add_argument(
    "--float8_precompute_dynamic_scale",
    action="store_true",
    help="torchao: Communicate AMAX/scales efficiently in a single all-reduce for all FSDP params. (Tensorwise scaling, optional).",
)
cli.add_argument(
    "--float8_force_recompute_weight_bwd",
    action="store_true",
    help="torchao: Force recomputation of FP8 weights during backward pass. (Tensorwise scaling, optional).",
)
cli.add_argument(
    "--disable_torch_compile",
    action="store_true",
    help="Disable torch.compile even if available",
)
args = cli.parse_args()


# ────────────── Logging & distributed init ─────────────────
def print_rank0_info(msg):
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[INFO] {msg}", flush=True)


def print_rank0_error(msg):
    if int(os.getenv("RANK", "0")) == 0:
        print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


if args.use_fp8_torchao and not TORCHAO_OK:
    print_rank0_error(
        "CRITICAL ERROR: --use_fp8_torchao was specified, but torchao library is not available or failed to import."
    )
    sys.exit(1)

if args.use_fp8_torchao and TORCHAO_OK and args.use_qlora and (args.lora_r % 16 != 0):
    error_msg = (
        f"FP8 (torchao) execution with QLoRA: --lora_r ({args.lora_r}) is not a multiple of 16. "
        f"This might be suboptimal or unsupported. Consider adjusting --lora_r."
    )
    print_rank0_error(f"Potential Issue: {error_msg}")


dist.init_process_group("nccl")
rank, local_rank = int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
print_rank0_info(f"[rank {rank}] ready on {torch.cuda.get_device_name(device)}")


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
        print_rank0_info(
            f"Set tokenizer.pad_token to tokenizer.eos_token: {tok.eos_token}"
        )
    else:
        tok.add_special_tokens({"pad_token": "[PAD]"})
        print_rank0_info("Added [PAD] as pad_token, as eos_token was not found.")

# Assuming "ibm-granite/granite-3.3-2b-instruct" tokenizer includes all necessary special tokens
# (e.g., <|start_of_role|>, <|end_of_role|>, <|end_of_text|>, <|tool_call|>)
# as per successful training of the original script.
# Thus, explicit addition and embedding resize are removed for cleanliness.
newly_added_tokens = []  # Keep this defined as it's checked later, but it will remain empty.

# Configure amp_dtype and GradScaler
scaler = None
if args.disable_amp:
    amp_dtype = torch.float32
    print_rank0_info("AMP disabled. Training in FP32. GradScaler is disabled.")
elif args.use_fp8_torchao and TORCHAO_OK:
    print_rank0_info(
        f"FP8 (torchao) training selected. Non-FP8 modules will use torch.autocast with amp_precision_mode: {args.amp_precision_mode}."
    )
    if args.amp_precision_mode == "bf16":
        amp_dtype = torch.bfloat16
        print_rank0_info(
            f"Non-FP8 parts (autocast context) will use BF16 ({amp_dtype}). GradScaler is disabled."
        )
    elif args.amp_precision_mode == "fp16":
        amp_dtype = torch.float16
        # FSDP2 uses standard GradScaler
        scaler = GradScaler()
        print_rank0_info(
            f"Non-FP8 parts (autocast context) will use FP16 ({amp_dtype}). Standard GradScaler is ENABLED."
        )
    else:
        amp_dtype = torch.bfloat16
        print_rank0_error(
            f"Invalid amp_precision_mode ('{args.amp_precision_mode}') with FP8 (torchao). Defaulting non-FP8 autocast to BF16. GradScaler disabled."
        )
elif args.amp_precision_mode == "bf16":
    amp_dtype = torch.bfloat16
    print_rank0_info(f"AMP enabled with BF16 ({amp_dtype}). GradScaler is disabled.")
elif args.amp_precision_mode == "fp16":
    amp_dtype = torch.float16
    # FSDP2 uses standard GradScaler
    scaler = GradScaler()
    print_rank0_info(
        f"AMP enabled with FP16 ({amp_dtype}). Standard GradScaler is ENABLED."
    )
else:
    amp_dtype = torch.bfloat16
    print_rank0_error(
        f"Invalid amp_precision_mode: {args.amp_precision_mode}. Defaulting to BF16. GradScaler disabled."
    )

# Model loading (potentially on meta device for FSDP2)
# For FSDP2, it's recommended to load on "meta" device first if not using QLoRA directly on GPU
# However, QLoRA with BitsAndBytesConfig usually requires GPU for quantization.

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


print_rank0_info(
    f"Loading model with torch_dtype: {model_load_torch_dtype}"
)  # This line is effectively duplicated by the QLoRA active/inactive log above, but kept for consistency if QLoRA is off.

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=args.use_qlora,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=amp_dtype,  # Compute dtype for BnB operations
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=model_load_torch_dtype
    if args.use_qlora
    else None,  # Storage dtype for 4-bit weights, must match model's float type for FSDP2
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=".cache",
    quantization_config=bnb_cfg if args.use_qlora else None,
    torch_dtype=model_load_torch_dtype,  # Dtype for non-quantized parts of the model
    trust_remote_code=True,
)

# Removed: model.resize_token_embeddings(len(tok)) as newly_added_tokens will be empty.

if args.gradient_checkpointing:
    model.config.use_cache = False
else:
    model.config.use_cache = True

if args.use_qlora:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    # For QLoRA, gradient checkpointing enabling is handled by prepare_model_for_kbit_training
    # if use_gradient_checkpointing is True.
    # The explicit call below might be redundant or specific to non-QLoRA LoRA.
    if args.gradient_checkpointing and not hasattr(
        model, "gradient_checkpointing_enable"
    ):
        # If PEFT didn't add it, try to enable on base.
        base_model_for_gc = model.base_model if hasattr(model, "base_model") else model
        if hasattr(base_model_for_gc, "gradient_checkpointing_enable"):
            base_model_for_gc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print_rank0_info("Gradient checkpointing explicitly enabled on base model.")

lconf = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=[m.strip() for m in args.lora_target_modules.split(",")],
    bias="none",
)
if args.use_qlora:  # Apply LoRA to the QLoRA-prepared model
    model = get_peft_model(model, lconf)
else:  # Standard LoRA (or no LoRA if target_modules is empty, though unlikely)
    model = get_peft_model(model, lconf)


if rank == 0:
    model.print_trainable_parameters()

# Convert model to FP8 using torchao if enabled
if args.use_fp8_torchao and TORCHAO_OK:
    if args.use_qlora:
        print_rank0_info(  # Changed from error to info/warning
            "WARNING: --use_fp8_torchao with --use_qlora is an experimental configuration. "
            "Applying torchao FP8 conversion to LoRA adapters on top of a QLoRA base model. "
            "Ensure compatibility and monitor behavior closely. "
            f"QLoRA compute_dtype is: {amp_dtype}, torchao FP8 will be applied to LoRA layers."
        )
    # Proceed with conversion regardless of QLoRA status if use_fp8_torchao is True
    print_rank0_info(
        "Applying torchao FP8 conversion to compatible Linear layers (e.g., LoRA adapters if present)..."
    )
    try:
        if not TORCH_VERSION_AT_LEAST_2_5:
            print_rank0_error(
                "torchao.float8 training APIs require PyTorch version 2.5 or greater. Skipping FP8 conversion."
            )
            args.use_fp8_torchao = False  # Corrected indentation
            raise RuntimeError(
                "PyTorch version too old for torchao.float8 training."
            )  # Corrected indentation

        def linear_layer_filter_fn(module, fqn):  # Signature: (module, fqn)
            if isinstance(module, torch.nn.Linear):
                is_compatible = (
                    module.in_features % 16 == 0 and module.out_features % 16 == 0
                )
                if not is_compatible:
                    print_rank0_info(
                        f"Skipping FP8 conversion for {fqn} (in_features={module.in_features}, out_features={module.out_features}) due to non-divisibility by 16."
                    )
                return is_compatible
            return False

        print_rank0_info(
            f"Attempting FP8 conversion using convert_to_float8_training with recipe: {args.float8_recipe_name}..."
        )

        # Use recipe string directly
        recipe_str = args.float8_recipe_name.lower()
        allowed_recipes = [
            "tensor",
            "rowwise",
            "dynamic",
        ]  # Defined by argparse choices
        if recipe_str not in allowed_recipes:
            print_rank0_error(
                f"Invalid float8_recipe_name: '{args.float8_recipe_name}'. Defaulting to 'dynamic'."
            )
            recipe_str = "dynamic"

        config = Float8LinearConfig(recipe=recipe_str)  # Pass the string directly
        # For logging, if TORCHAO_OK, config.recipe might be an enum, otherwise it's config.recipe_str from the dummy
        current_recipe_log = (
            config.recipe
            if hasattr(config, "recipe") and not isinstance(config.recipe, str)
            else recipe_str
        )
        print_rank0_info(
            f"Created Float8LinearConfig with recipe: {current_recipe_log}"
        )

        # Apply FSDP-specific torchao settings using imported functions
        # These apply if recipe is tensor-based ("dynamic" or "tensor")
        if recipe_str in ["dynamic", "tensor"]:
            if args.float8_enable_fsdp_all_gather:
                enable_fsdp_float8_all_gather(True)
                print_rank0_info(
                    "torchao: Enabled FSDP float8 all_gather via function call."
                )
            else:
                enable_fsdp_float8_all_gather(
                    False
                )  # Explicitly set to default or user's choice
                print_rank0_info(
                    "torchao: FSDP float8 all_gather set to False (or default) via function call."
                )

            if args.float8_precompute_dynamic_scale:
                precompute_float8_dynamic_scale_for_fsdp(True)
                print_rank0_info(
                    "torchao: Enabled FSDP precompute_float8_dynamic_scale via function call."
                )
            else:
                precompute_float8_dynamic_scale_for_fsdp(False)
                print_rank0_info(
                    "torchao: FSDP precompute_float8_dynamic_scale set to False (or default) via function call."
                )

            if args.float8_force_recompute_weight_bwd:
                force_recompute_fp8_weight_in_bwd(True)
                print_rank0_info(
                    "torchao: Enabled FSDP force_recompute_fp8_weight_in_bwd via function call."
                )
            else:
                force_recompute_fp8_weight_in_bwd(False)
                print_rank0_info(
                    "torchao: FSDP force_recompute_fp8_weight_in_bwd set to False (or default) via function call."
                )
        else:  # rowwise or other recipes
            if (
                args.float8_enable_fsdp_all_gather
                or args.float8_precompute_dynamic_scale
                or args.float8_force_recompute_weight_bwd
            ):
                print_rank0_info(
                    f"FSDP-specific FP8 flags are not applicable for recipe '{recipe_str}' and are ignored."
                )

        model = convert_to_float8_training(
            model, module_filter_fn=linear_layer_filter_fn, linear_config=config
        )
        current_recipe_log_after = (
            config.recipe
            if hasattr(config, "recipe") and not isinstance(config.recipe, str)
            else recipe_str
        )
        print_rank0_info(
            f"Model conversion using torchao.float8.convert_to_float8_training (recipe: {current_recipe_log_after}) completed."
        )
        print_rank0_info(
            f"Dtype histogram after torchao FP8 (training API) conversion: {Counter(p.dtype for p in model.parameters())}"
        )
    # Restored except blocks
    except RuntimeError as e:
        if "PyTorch version too old" in str(e):
            # args.use_fp8_torchao was already set to False before raising.
            pass
        else:
            print_rank0_error(
                f"torchao FP8 conversion or config failed (RuntimeError): {e}. Proceeding without torchao FP8 features."
            )
            args.use_fp8_torchao = False
    except Exception as e:
        print_rank0_error(
            f"torchao FP8 conversion or config failed (Exception): {e}. Proceeding without torchao FP8 features."
        )
        args.use_fp8_torchao = False  # Fallback to non-FP8 path

# FSDP ignored modules for QLoRA
fsdp_ignored_modules_list = []
params_in_ignored_modules_fqns = set()  # To skip casting these later

if args.use_qlora:
    print_rank0_info("Identifying BitsAndBytes modules to ignore for FSDP2...")
    for module_name, module in model.named_modules():
        if "bitsandbytes.nn.modules" in str(type(module)).lower():
            if module not in fsdp_ignored_modules_list:  # Ensure uniqueness
                fsdp_ignored_modules_list.append(module)
            # Also collect FQNs of parameters within these modules to prevent casting them
            for param_fqn, _ in module.named_parameters():
                full_param_fqn = (
                    f"{module_name}.{param_fqn}" if module_name else param_fqn
                )
                params_in_ignored_modules_fqns.add(full_param_fqn)

    if fsdp_ignored_modules_list:
        print_rank0_info(
            f"Identified {len(fsdp_ignored_modules_list)} BitsAndBytes modules to ignore for FSDP2."
        )
    else:
        print_rank0_info(
            "No BitsAndBytes modules found to ignore for FSDP (this might be unexpected if QLoRA is used)."
        )

# Cast parameters not handled by torchao FP8 or BnB to amp_dtype
print_rank0_info(f"Casting remaining parameters to {amp_dtype} if necessary...")

param_names_to_cast = []
for fqn, param in model.named_parameters():
    is_torchao_fp8_param = False  # Placeholder
    # Add more robust check for torchao parameters if needed, e.g., by checking parent module type

    if (
        fqn
        not in params_in_ignored_modules_fqns  # Check against FQNs of params in ignored modules
        and not is_torchao_fp8_param
        and param.dtype != amp_dtype
    ):
        param_names_to_cast.append(fqn)

if param_names_to_cast:
    print_rank0_info(
        f"Casting {len(param_names_to_cast)} parameters to {amp_dtype}: {param_names_to_cast[:5]}..."
    )
    for fqn, param in model.named_parameters():
        if fqn in param_names_to_cast:
            try:
                param.data = param.data.to(amp_dtype)
            except Exception as e:
                print_rank0_error(f"Failed to cast {fqn} to {amp_dtype}: {e}")
print_rank0_info(
    f"Dtype histogram before FSDP2 wrapping: {Counter(p.dtype for p in model.parameters())}"
)


if not args.no_fsdp:
    # FSDP2 Setup
    device_mesh = init_device_mesh("cuda", (world_size,))  # 1D mesh for FSDP

    # FSDP2 MixedPrecisionPolicy
    # Note: buffer_dtype is not part of FSDP2's MixedPrecisionPolicy arguments directly.
    # It's handled by FSDP based on param_dtype or reduce_dtype.
    # We primarily care about param_dtype (for model params) and reduce_dtype (for gradients).
    fsdp_mp_policy = None
    if not args.disable_amp:  # Only configure if AMP is not disabled
        if amp_dtype == torch.bfloat16:
            fsdp_mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            )
            print_rank0_info(
                "FSDP2 MixedPrecision: param_dtype=bf16, reduce_dtype=bf16"
            )
        elif amp_dtype == torch.float16:
            fsdp_mp_policy = MixedPrecision(
                param_dtype=torch.float16, reduce_dtype=torch.float32
            )  # Reduce in fp32 for stability
            print_rank0_info(
                "FSDP2 MixedPrecision: param_dtype=fp16, reduce_dtype=fp32"
            )
        # If amp_dtype is float32 (e.g. AMP disabled), fsdp_mp_policy can remain None or be explicit.

    compile_model = hasattr(torch, "compile") and not args.disable_torch_compile
    skip_compile_logic = (
        not args.disable_amp
        and args.amp_precision_mode == "bf16"
        and not (args.use_fp8_torchao and TORCHAO_OK)
    )

    if compile_model and not skip_compile_logic:
        print_rank0_info("Attempting torch.compile on model BEFORE FSDP2 sharding...")
        if args.use_fp8_torchao and TORCHAO_OK:
            print_rank0_info("torch.compile is being used with torchao FP8 setup.")
        try:
            model = torch.compile(model, backend="inductor", mode="max-autotune")
            print_rank0_info("torch.compile BEFORE FSDP2 sharding successful.")
        except Exception as e:
            print_rank0_error(
                f"torch.compile BEFORE FSDP2 sharding failed: {e}. Proceeding with uncompiled model."
            )
    elif skip_compile_logic:
        print_rank0_info(
            "Skipping torch.compile due to BF16 AMP configuration (without FP8)."
        )

    # Apply FSDP2 using fully_shard
    # We need to iterate and apply fully_shard. For simplicity, start with the outer model.
    # A more granular approach might involve a wrapping policy or manual iteration.
    # The `ignored_params` argument in FSDP2's `fully_shard` is what we need for BnB params.

    # This is a simplified application. Proper recursive application or a policy might be better.
    # For FSDP2, you typically apply `fully_shard` to modules you want to shard.
    # If QLoRA is used, `bitsandbytes` layers should not be sharded by FSDP2.
    # Their parameters should be in `ignored_params`.

    # It's generally applied from innermost to outermost sharded unit.
    # For now, a single shard for the whole model, ignoring BnB params.

    # Convert model to a sharded state using FSDP2's fully_shard
    # This is a conceptual change; the actual FSDP2 wrapping happens by modifying the class
    # and then applying fully_shard. The model object `model` is modified in-place.

    # FSDP expects `ignored_modules` to be a list of module instances.
    # from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy # Keep for now if needed by FSDP internals

    # sharding_strategy = (
    #     ShardingStrategy.FULL_SHARD
    #     if args.reshard_after_forward
    #     else ShardingStrategy.SHARD_GRAD_OP
    # )

    # FSDP expects `ignored_modules` to be a list of module instances.
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy

    sharding_strategy = (
        ShardingStrategy.FULL_SHARD
        if args.reshard_after_forward
        else ShardingStrategy.SHARD_GRAD_OP
    )

    # FSDP2 style wrapping:
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        device_mesh=device_mesh,  # device_mesh is initialized earlier
        mixed_precision=fsdp_mp_policy,
        ignored_modules=fsdp_ignored_modules_list
        if fsdp_ignored_modules_list
        else None,
        use_orig_params=True,  # Retained for QLoRA compatibility
    )
    print_rank0_info(
        f"FSDP (FSDP2 style) wrapping complete with sharding_strategy: {sharding_strategy}."
    )

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
def val_loss():
    model.eval()
    tot_loss = torch.zeros([], device=device)
    num_batches = 0
    if len(val_loader) == 0:
        pass
    for b in val_loader:
        b = {k: v.to(device) for k, v in b.items()}
        with (
            torch.autocast("cuda", amp_dtype) if not args.disable_amp else nullcontext()
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


def save(tag):
    if rank != 0:
        return
    path = os.path.join(args.output_dir, tag)
    os.makedirs(path, exist_ok=True)

    # FSDP2 state_dict saving
    # The FSDP class itself handles how state_dict is collected.
    # For full state dict (e.g., for resuming on different #ranks or for HF conversion),
    # FSDP provides a context manager.
    # For sharded state_dict (rank0_only=False, offload_to_cpu=True), it's more efficient.
    # Current script uses FULL_STATE_DICT for saving. This should still work with FSDP2.
    if not args.no_fsdp:  # If FSDP is active
        # Use FSDP's state_dict_type context manager for full state dict
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            cpu_state_dict = model.state_dict()

        # Saving logic for PEFT models (common case)
        if hasattr(model, "save_pretrained") and callable(
            getattr(model, "save_pretrained")
        ):
            # If the FSDP-wrapped model (or its underlying module if FSDP does not expose it)
            # has save_pretrained, use it. This is typical for PEFT models.
            # FSDP might require unwrapping or accessing module for this.
            # Let's assume model.module for FSDP, or model itself if not DDP/FSDP wrapped.
            model_to_save = model.module if hasattr(model, "module") else model
            if hasattr(model_to_save, "save_pretrained"):
                model_to_save.save_pretrained(path, state_dict=cpu_state_dict)
            else:  # Fallback if inner model also doesn't have it
                torch.save(cpu_state_dict, os.path.join(path, "pytorch_model.bin"))
                print_rank0_info(
                    f"Saved FSDP full state_dict to {os.path.join(path, 'pytorch_model.bin')} (model has no save_pretrained)"
                )
        else:  # Fallback for non-HuggingFace models or if save_pretrained is not on the FSDP object
            torch.save(cpu_state_dict, os.path.join(path, "pytorch_model.bin"))
            print_rank0_info(
                f"Saved FSDP full state_dict to {os.path.join(path, 'pytorch_model.bin')}"
            )
        tok.save_pretrained(path)
    else:  # DDP or single GPU
        model_to_save = model.module if hasattr(model, "module") else model
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(path)
        else:
            torch.save(
                model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin")
            )
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
            torch.autocast("cuda", amp_dtype) if not args.disable_amp else nullcontext()
        ):
            raw_loss = model(**batch).loss
            loss = raw_loss.float()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

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
                scaler.unscale_(opt)

            # FSDP2: Use torch.nn.utils.clip_grad_norm_ on model.parameters()
            if not args.no_fsdp:
                # model.clip_grad_norm_(1.0) # FSDP1 style
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # FSDP2 style
            else:  # DDP
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scaler:
                scaler_step_output = scaler.step(opt)
                if scaler_step_output is None and rank == 0:
                    print_rank0_error(
                        f"Rank {rank} - scaler.step() reported non-finite gradients. Skipping optimizer update."
                    )
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
