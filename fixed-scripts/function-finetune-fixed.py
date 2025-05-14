import argparse
import functools
import logging
import math
import os
import subprocess
import time

import torch
from datasets import load_from_disk  # Added load_from_disk
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,  # prepare_model_for_kbit_training might be needed for older peft
)
from torch.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--no_fsdp", action="store_true", help="Train with DDP instead of FSPD."
)
parser.add_argument(
    "--no_layer_wrap_policy",
    action="store_true",
    help="Don't use custom FSDP layer wrap policy.",
)
parser.add_argument(
    "--batch_size_per_device",
    type=int,
    default=24,
    help="Per-device training batch size",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=4,
    help="Gradient accumulation steps",
)
parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning Rate")
parser.add_argument(
    "--max_training_steps", type=int, default=-1, help="Interrupt training early."
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    default=False,
    help="Use gradient checkpointing (default: False, to disable it)",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=2048,
    help="Maximum sequence length for truncation.",
)
parser.add_argument(
    "--disable_amp",
    action="store_true",
    default=False,
    help="Disable automatic mixed precision (AMP is enabled by default)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./function_calling_output",
    help="Directory to save checkpoints and logs.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="ibm-granite/granite-3.3-2b-instruct",
    help="Path to pretrained model or model identifier from huggingface.co/models",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=5,
    help="Number of epochs to train for.",
)
parser.add_argument(
    "--dataset_subset_size",
    type=int,
    default=-1,
    help="Limit dataset to a smaller subset for testing. -1 uses full dataset.",
)
parser.add_argument(
    "--save_steps",
    type=int,
    default=500,
    help="Number of steps between checkpoint saves",
)
parser.add_argument(
    "--eval_steps",
    type=int,
    default=100,
    help="Number of steps between evaluations",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.1,
    help="Ratio of steps for warmup out of total training steps",
)
# QLoRA / LoRA arguments
parser.add_argument(
    "--use_qlora", action="store_true", help="Enable QLoRA for training."
)
parser.add_argument(
    "--lora_r", type=int, default=8, help="LoRA attention dimension (rank)."
)
parser.add_argument(
    "--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor."
)
parser.add_argument(
    "--lora_dropout", type=float, default=0.05, help="LoRA dropout probability."
)
parser.add_argument(
    "--lora_target_modules",
    type=str,
    default="q_proj,v_proj",  # Common for many models, adjust as needed
    help="Comma-separated list of module names to apply LoRA to (e.g., 'q_proj,v_proj').",
)
parser.add_argument(
    "--processed_dataset_path",
    type=str,
    default=None,  # Required if not using on-the-fly processing
    help="Path to the preprocessed dataset saved by test_dataproc.py.",
)
args = parser.parse_args()


# Simplified Dataset class for preprocessed data
class PreprocessedFunctionCallingDataset(Dataset):
    def __init__(self, processed_dataset_split, subset_size=-1):
        if subset_size > 0 and subset_size < len(processed_dataset_split):
            self.dataset = processed_dataset_split.select(range(subset_size))
        else:
            self.dataset = processed_dataset_split

        # Set format to PyTorch tensors for direct tensor output from __getitem__
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        if len(self.dataset) > 0:
            logger.info(
                f"Preprocessed dataset split loaded. Sample keys: {list(self.dataset[0].keys())}"
            )
            logger.info(f"Sample input_ids shape: {self.dataset[0]['input_ids'].shape}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]  # Returns a dict of tensors

    # Collate function remains the same as it expects a list of dicts of tensors
    def collate_fn(self, examples):
        batch = {
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
            "attention_mask": torch.stack(
                [example["attention_mask"] for example in examples]
            ),
            "labels": torch.stack([example["labels"] for example in examples]),
        }
        return batch


def train(model, train_dataset, eval_dataset, args):
    # Create samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        seed=42,
        drop_last=True,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
    )
    eval_sampler = DistributedSampler(
        eval_dataset,
        shuffle=False,
        seed=42,
        drop_last=False,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=8,  # Adjusted num_workers
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        pin_memory=True,  # Added pin_memory
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=8,  # Adjusted num_workers
        collate_fn=eval_dataset.collate_fn,
        sampler=eval_sampler,
        pin_memory=True,  # Added pin_memory
    )

    model.train()
    if args.gradient_checkpointing:
        logger.info(
            f"Rank {torch.distributed.get_rank()}: Enabling gradient checkpointing."
        )
        model.gradient_checkpointing_enable()
    else:
        logger.info(
            f"Rank {torch.distributed.get_rank()}: Gradient checkpointing is DISABLED."
        )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Training steps calculation
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # Set up max_training_steps if specified
    if args.max_training_steps > 0:
        max_train_steps = min(args.max_training_steps, max_train_steps)

    # Warmup steps calculation
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    # Linear learning rate schedule with warmup
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(max_train_steps - current_step)
            / float(max(1, max_train_steps - num_warmup_steps)),
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Gradient scaler for mixed precision training
    scaler = GradScaler() if not args.disable_amp else None

    # For demonstration purposes only: Print nvidia-smi GPU interface
    logger.info(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))

    # Training metrics
    global_step = 0
    total_loss = 0
    epoch_loss = 0
    best_eval_loss = float("inf")
    start_time = time.time()
    log_interval = 100

    logger.info(
        f"Starting training for {args.num_epochs} epochs ({max_train_steps} steps)"
    )
    logger.info(f"Warmup for {num_warmup_steps} steps")
    logger.info(
        f"Rank {torch.distributed.get_rank()}: Preparing to start training epochs."
    )

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        logger.info(
            f"Rank {torch.distributed.get_rank()}: Starting epoch {epoch + 1}/{args.num_epochs}"
        )

        epoch_start_time = time.time()
        epoch_loss = 0
        logger.info(
            f"Rank {torch.distributed.get_rank()}: Epoch {epoch + 1}: Initializing DataLoader iteration..."
        )
        for step, batch in enumerate(train_dataloader):
            if step == 0 and epoch == 0:  # Log only for the very first batch attempt
                logger.info(
                    f"Rank {torch.distributed.get_rank()}: Epoch {epoch + 1}, Step {step}: Successfully fetched first batch from DataLoader."
                )
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if not args.disable_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):  # Changed to float16
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            # Accumulate loss statistics
            total_loss += loss.detach().float()
            epoch_loss += loss.detach().float()

            # Backward pass with gradient accumulation
            if not args.disable_amp:  # AMP is ON, scaler is GradScaler()
                scaler.scale(loss).backward()
            else:  # AMP is OFF, scaler is None
                loss.backward()

            # Update weights after accumulating gradients
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (
                step == len(train_dataloader) - 1
            ):
                if not args.disable_amp:  # AMP is ON, scaler is GradScaler()
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:  # AMP is OFF, scaler is None
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

                # Log training progress
                if global_step % log_interval == 0:
                    avg_loss = (total_loss / log_interval).item()
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Epoch: {epoch + 1}/{args.num_epochs} | "
                        f"Step: {global_step}/{max_train_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.6f} | "
                        f"Time: {elapsed:.2f}s"
                    )
                    total_loss = 0
                    start_time = time.time()

                # Run evaluation
                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader, args)
                    logger.info(f"Eval Loss: {eval_loss:.4f}")

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        if torch.distributed.get_rank() == 0:
                            save_checkpoint(model, args.output_dir, "best_model")
                            logger.info(
                                f"New best model saved with eval loss: {best_eval_loss:.4f}"
                            )

                    # Return to training mode
                    model.train()

                # Save checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if torch.distributed.get_rank() == 0:
                        save_checkpoint(
                            model, args.output_dir, f"checkpoint-{global_step}"
                        )
                        logger.info(f"Saved checkpoint at step {global_step}")

                # Check if we've reached max training steps
                if (
                    args.max_training_steps > 0
                    and global_step >= args.max_training_steps
                ):
                    logger.info(
                        f"Reached maximum training steps ({args.max_training_steps}). Stopping training."
                    )
                    return

            # Clear CUDA cache to save memory
            torch.cuda.empty_cache()

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {(epoch_loss / len(train_dataloader)).item():.4f}"
        )

        # Save checkpoint at the end of each epoch
        if torch.distributed.get_rank() == 0:
            save_checkpoint(model, args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            logger.info(f"Saved checkpoint for epoch {epoch + 1}")

    logger.info("Training complete!")


def evaluate(model, eval_dataloader, args):
    """Evaluate the model on the evaluation dataset"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Fix autocast usage here
            if not args.disable_amp:
                with torch.amp.autocast(
                    device_type="cuda", dtype=torch.float16
                ):  # Changed to float16
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            total_loss += outputs.loss.detach().float()
            num_batches += 1

    # Compute average loss across all processes
    avg_loss = total_loss / max(num_batches, 1)

    # Gather losses from all processes
    all_losses = torch.tensor([avg_loss], device=device)
    torch.distributed.all_reduce(all_losses, op=torch.distributed.ReduceOp.SUM)
    avg_loss = all_losses.item() / torch.distributed.get_world_size()

    return avg_loss


def save_checkpoint(model, output_dir, checkpoint_name):
    """Save model checkpoint"""
    save_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)

    # If using FSDP, we need to consolidate the model before saving
    if isinstance(model, FSDP):
        logger.info("Gathering FSDP model full state_dict for saving...")
        # Use the FSDP.state_dict_type context manager to get the full state dict.
        # This will be on CPU and only on rank 0 by default.
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()

        # The save_checkpoint function is called within a rank 0 check in the main script,
        # so the following operations will correctly occur on rank 0.
        model_to_save = model.module  # Get the original, unwrapped module.
        model_to_save.load_state_dict(state_dict)  # Load the full state_dict into it.
    else:
        model_to_save = model.module if hasattr(model, "module") else model

    logger.info(f"Saving model to {save_dir}")
    model_to_save.save_pretrained(save_dir)


# Make sure to initialize distributed training
torch.distributed.init_process_group(backend="nccl")

# Setup logging
logging.basicConfig(
    level=(logging.INFO if torch.distributed.get_rank() == 0 else logging.WARNING),
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

logger.info(f"Initialized process group: {torch.distributed.get_world_size()}")
logger.info(f"Args: {args}")

# Fix seed
torch.manual_seed(42)

# Set device - use LOCAL_RANK for device assignment to fix previous bug
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()
logger.info(
    f"Using GPU: {torch.cuda.get_device_name(device)} (local_rank: {local_rank}, global_rank: {torch.distributed.get_rank()})"
)

# Load the tokenizer and model
logger.info(f"Loading tokenizer and model from: {args.model_name_or_path}...")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path, cache_dir=".cache", trust_remote_code=True
)

# Determine model dtype based on AMP setting
if not args.disable_amp:
    model_torch_dtype = (
        torch.float16
    )  # Use float16 when AMP (and GradScaler) is enabled
    logger.info("AMP is enabled. Loading model with torch_dtype=torch.float16.")
else:
    model_torch_dtype = torch.bfloat16  # Use bfloat16 when AMP is disabled
    logger.info("AMP is disabled. Loading model with torch_dtype=torch.bfloat16.")

if args.use_qlora:  # This is the line that needs to change
    logger.info(
        "QLoRA is enabled. Preparing BitsAndBytesConfig and loading model in 4-bit."
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Align compute_dtype with AMP: if AMP is float16, use float16.
        # If AMP is disabled, bnb_config used float32, which is fine.
        # The original script's AMP uses float16.
        bnb_4bit_compute_dtype=torch.float16
        if not args.disable_amp
        else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # When using FSDP, device_map should ideally be handled by FSDP.
    # Load model on CPU first or let FSDP handle it. For simplicity here, we might load it
    # without an explicit device_map and let FSDP place it.
    # Or, for initial QLoRA setup before FSDP, one might load to current device if single-GPU testing.
    # For multi-GPU FSDP, it's safer to load on meta device or CPU if possible, then FSDP handles sharding.
    # However, bitsandbytes 4-bit loading often requires a device.
    # Let's try loading directly to the current device for the PEFT wrapping step,
    # FSDP should then re-distribute. This can be tricky with FSDP.
    # A common pattern is to load on rank 0's GPU then FSDP.
    # If LOCAL_RANK is 0, load with quantization config. Other ranks load meta device then sync.
    # This part needs careful handling with FSDP.
    # For now, let's assume a simpler path and see if FSDP handles it.
    # If issues arise, this is a key area to revisit for FSDP+QLoRA.

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=".cache",
        quantization_config=bnb_config,
        # device_map={"":torch.cuda.current_device()}, # This might conflict with FSDP if not handled carefully
        trust_remote_code=True,
    )
    logger.info("Base model loaded in 4-bit for QLoRA.")

    # PEFT recommends this for QLoRA, though some parts might be handled by get_peft_model
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    # logger.info("Model prepared for k-bit training.")

    # Define LoRA config
    lora_target_modules_list = [m.strip() for m in args.lora_target_modules.split(",")]
    logger.info(f"Applying LoRA to modules: {lora_target_modules_list}")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules_list,
        bias="none",  # Common for QLoRA
    )
    model = get_peft_model(model, peft_config)
    logger.info("LoRA applied to the model for QLoRA.")
    model.print_trainable_parameters()

else:  # Original full fine-tuning path
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=".cache",
        torch_dtype=model_torch_dtype,  # Use the determined dtype
        trust_remote_code=True,
    )
logger.info("Model initialized.")
if not args.use_qlora:  # This is the line that needs to change
    logger.info("Full model fine-tuning (not QLoRA).")
else:  # This is new
    logger.info("QLoRA fine-tuning is active.")


# Load the dataset
if args.processed_dataset_path:
    logger.info(
        f"Loading pre-processed dataset from disk: {args.processed_dataset_path}"
    )
    try:
        processed_full_dataset = load_from_disk(args.processed_dataset_path)
        logger.info(
            f"Pre-processed dataset loaded. Total examples: {len(processed_full_dataset)}"
        )
    except Exception as e:
        logger.error(
            f"Failed to load preprocessed dataset from {args.processed_dataset_path}: {e}"
        )
        exit(1)

    # Split into train and eval
    # Assuming the loaded dataset from disk is the 'train' part that needs splitting
    train_size = int(0.9 * len(processed_full_dataset))
    eval_size = len(processed_full_dataset) - train_size

    if train_size <= 0 or eval_size <= 0:
        logger.error(
            f"Dataset too small to split. Train size: {train_size}, Eval size: {eval_size}. Full dataset has {len(processed_full_dataset)} samples."
        )
        exit(1)

    train_dataset_processed = processed_full_dataset.select(range(train_size))
    eval_dataset_processed = processed_full_dataset.select(
        range(
            train_size, train_size + eval_size
        )  # Corrected to use train_size + eval_size
    )
    logger.info(f"Train dataset size (from processed): {len(train_dataset_processed)}")
    logger.info(f"Eval dataset size (from processed): {len(eval_dataset_processed)}")

    train_dataset = PreprocessedFunctionCallingDataset(
        train_dataset_processed,
        subset_size=args.dataset_subset_size,  # Apply subset size if specified for training
    )
    eval_dataset = PreprocessedFunctionCallingDataset(
        eval_dataset_processed,
        subset_size=min(
            1000, len(eval_dataset_processed)
        )  # Limit eval subset for speed if main subset_size is active
        if args.dataset_subset_size > 0
        else -1,  # Or use all of eval_dataset_processed if no main subset_size
    )

else:
    # Fallback to original on-the-fly processing if no processed_dataset_path is given
    # This part would require the original FunctionCallingDataset class to be defined or imported.
    # For this refactor, we'll assume processed_dataset_path is mandatory.
    logger.error(
        "Error: --processed_dataset_path is required. On-the-fly processing is disabled in this version."
    )
    logger.error(
        "Please preprocess the data first using test_dataproc.py and provide the path."
    )
    exit(1)
    # --- Code for on-the-fly processing would go here if we were to keep it ---
    # logger.info("Loading Function Calling dataset for on-the-fly processing...")
    # raw_dataset = load_dataset(
    #     "glaiveai/glaive-function-calling-v2",
    #     data_files={"train": "glaive-function-calling-v2.json"},
    # )
    # logger.info(f"Dataset loaded. Size: {len(raw_dataset['train'])} examples")
    # ... (rest of original data loading and splitting) ...
    # train_dataset = OriginalFunctionCallingDataset(...)
    # eval_dataset = OriginalFunctionCallingDataset(...)


if args.no_fsdp:
    # Move the model to the GPU and wrap model with DDP (no model sharding)
    model = DDP(model.to(device))
else:
    auto_wrap_policy = None
    if not args.no_layer_wrap_policy:
        # Identify which modules have "layer" in their class name and use these
        # as the basic FSDP blocks that are sharded and exchanged between GPUs
        def layer_policy_fn(module):
            return "layer" in module.__class__.__name__.lower()

        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=layer_policy_fn
        )

    # Wrap model as FSDP model
    logger.info(f"Rank {torch.distributed.get_rank()}: Starting FSDP model wrapping...")
    # Add sync_module_states=True for pre-modified models (like QLoRA)
    model = FSDP(
        model,
        device_id=device,
        auto_wrap_policy=auto_wrap_policy,
        sync_module_states=True,
    )
    logger.info(f"Rank {torch.distributed.get_rank()}: FSDP model wrapping complete.")

# Create output directory (only on rank 0)
if torch.distributed.get_rank() == 0 and args.output_dir:
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

# Start training
train(model, train_dataset, eval_dataset, args)
logger.info("Fine-tuning complete!")

# Save final model (only on rank 0)
if torch.distributed.get_rank() == 0:
    save_checkpoint(model, args.output_dir, "final_model")
    logger.info(f"Final model saved to {os.path.join(args.output_dir, 'final_model')}")
