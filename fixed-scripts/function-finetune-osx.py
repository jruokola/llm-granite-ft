import argparse
import logging
import math
import os
import sys  # Import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

# Removed: FSDP and DDP imports, will be handled differently for single device
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from datasets import load_from_disk  # Added load_from_disk
from torch.amp import GradScaler  # Will be made conditional or adapted
from torch.utils.data import DataLoader, Dataset

# Removed: from torch.utils.data.distributed import DistributedSampler # Not needed for single process
from transformers import AutoModelForCausalLM, AutoTokenizer

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--processed_dataset_path",
    type=str,
    required=True,
    help="Path to the processed dataset on disk.",
)
parser.add_argument(
    "--batch_size_per_device",
    type=int,
    default=1,
    help="Per-device training batch size",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=8,
    help="Gradient accumulation steps",
)
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning Rate")
parser.add_argument(
    "--max_training_steps", type=int, default=-1, help="Interrupt training early."
)
parser.add_argument(
    "--gradient_checkpointing",
    action="store_true",
    default=True,  # Keep True, can be beneficial for memory
    help="Use gradient checkpointing (default: True)",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=512,
    help="Maximum sequence length for truncation.",
)
parser.add_argument(
    "--disable_amp", action="store_true", help="Disable automatic mixed precision"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./function_calling_output_osx",  # Changed default output dir
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
    default=10,
    help="Number of steps between evaluations",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.1,
    help="Ratio of steps for warmup out of total training steps",
)
args = parser.parse_args()

# --- macOS/Single Device Setup ---
# Determine device
# Logger needs to be defined before this block if it's used within.
# Moved logger definition up.

# Distributed settings for single process (rank 0, world size 1)
IS_DISTRIBUTED = False
RANK = 0
WORLD_SIZE = 1

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)  # Define logger earlier

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # torch.mps.empty_cache() # Good practice for MPS
else:
    device = torch.device("cpu")
    logger.info("MPS not available, defaulting to CPU. Training might be very slow.")


_mps_bf16_warning_logged = False  # Flag to ensure warning is logged only once


# Helper function to safely check for MPS bfloat16 support
def mps_check_bf16_support():
    if device.type == "mps":
        try:
            return torch.backends.mps.is_bf16_supported()
        except AttributeError:
            global _mps_bf16_warning_logged
            if not _mps_bf16_warning_logged:
                logger.warning(
                    "torch.backends.mps.is_bf16_supported() not found. "
                    "Assuming bfloat16 not supported on MPS for this PyTorch version. "
                    "Falling back to float16 for AMP if applicable."
                )
                _mps_bf16_warning_logged = True
            return False
    return False


# Function calling dataset processor
class FunctionCallingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, subset_size=-1):
        self.tokenizer = tokenizer
        if subset_size > 0:
            self.dataset = dataset.select(range(min(subset_size, len(dataset))))
        else:
            self.dataset = dataset
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if len(self.dataset) > 0 and RANK == 0:
            logger.debug(f"Dataset sample keys: {list(self.dataset[0].keys())}")
            logger.debug(
                f"Dataset sample first example chat content (first 200 chars): {self.dataset[0]['chat'][:200] if 'chat' in self.dataset[0] else None}"
            )

    def __len__(self):
        return len(self.dataset)

    def safe_process_message(self, message, idx=None):
        try:
            if isinstance(message, dict):
                role = str(message.get("role", "")).lower()
                content = str(message.get("content", ""))
                if role == "system":
                    return f"<|system|>\n{content}\n"
                elif role == "user":
                    return f"<|user|>\n{content}\n"
                elif role == "assistant":
                    return f"<|assistant|>\n{content}\n"
                else:
                    return f"<|user|>\n{content}\n"
            elif isinstance(message, str):
                return f"<|user|>\n{message}\n"
            elif isinstance(message, list):
                return "".join(
                    self.safe_process_message(submessage, idx) for submessage in message
                )
            else:
                if idx is not None and idx < 3 and RANK == 0:
                    logger.debug(
                        f"Unknown message type: {type(message)} for example {idx}. Content: {message}"
                    )
                return f"<|user|>\n{str(message)}\n"
        except Exception as e:
            if idx is not None and idx < 3 and RANK == 0:
                logger.error(f"Error processing message: {str(e)}. Content: {message}")
            return ""

    def __getitem__(self, idx):
        try:
            example = self.dataset[idx]
            if idx < 3 and RANK == 0:
                logger.debug(f"Processing example {idx}")
            formatted_text = ""
            if "system" in example and example["system"]:
                formatted_text += f"<|system|>\n{str(example['system']).strip()}\n"

            chat_content = example.get("chat") or example.get("text") or str(example)

            if isinstance(chat_content, str):
                lines = chat_content.split("\n")
                current_role, current_content = None, []
                role_map = {"SYSTEM:": "system", "USER:": "user", "A:": "assistant"}

                def append_content():
                    nonlocal formatted_text
                    if current_role and current_content:
                        role_content = "\n".join(current_content).strip()
                        if current_role == "system" and not formatted_text.startswith(
                            "<|system|>"
                        ):
                            formatted_text += f"<|system|>\n{role_content}\n"
                        elif current_role in ["user", "assistant"]:
                            formatted_text += f"<|{current_role}|>\n{role_content}\n"

                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    new_role_found = False
                    for marker, role_name in role_map.items():
                        if line_stripped.startswith(marker):
                            append_content()
                            current_role = role_name
                            current_content = [
                                line_stripped.replace(marker, "").strip()
                            ]
                            new_role_found = True
                            break

                    if new_role_found:
                        continue

                    if line_stripped.startswith("FUNCTION RESPONSE:"):
                        if current_role == "assistant":
                            current_content.append(line_stripped)
                        else:
                            append_content()
                            current_role = "assistant"
                            current_content = [line_stripped]
                    elif current_role:
                        current_content.append(line_stripped)
                    else:
                        current_role = "system"
                        current_content = [line_stripped]
                append_content()

            elif isinstance(chat_content, list):
                for message in chat_content:
                    formatted_text += self.safe_process_message(message, idx)
            else:
                formatted_text += f"<|user|>\n{str(chat_content)}\n"

            formatted_text = formatted_text.replace("<|endoftext|>", "")
            if not formatted_text:
                formatted_text = (
                    "<|user|>\nHello\n<|assistant|>\nHello! How can I help you today?\n"
                )

            if idx < 3 and RANK == 0:
                logger.debug(
                    f"Formatted text for example {idx} (sample): {formatted_text[:200]}..."
                )

            encodings = self.tokenizer(
                formatted_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings.input_ids[0]
            labels = input_ids.clone()

            assistant_marker_ids = self.tokenizer.encode(
                "<|assistant|>", add_special_tokens=False
            )
            user_marker_ids = self.tokenizer.encode(
                "<|user|>", add_special_tokens=False
            )
            system_marker_ids = self.tokenizer.encode(
                "<|system|>", add_special_tokens=False
            )

            assistant_positions = []
            for i in range(len(input_ids) - len(assistant_marker_ids) + 1):
                if torch.equal(
                    input_ids[i : i + len(assistant_marker_ids)],
                    torch.tensor(assistant_marker_ids, device=input_ids.device),
                ):
                    assistant_positions.append(i)

            if assistant_positions:
                in_assistant_turn = False
                for i in range(len(labels)):
                    is_start_of_assistant = any(i == pos for pos in assistant_positions)

                    is_start_of_user = False
                    if i <= len(input_ids) - len(user_marker_ids):
                        if torch.equal(
                            input_ids[i : i + len(user_marker_ids)],
                            torch.tensor(user_marker_ids, device=input_ids.device),
                        ):
                            is_start_of_user = True

                    is_start_of_system = False
                    if i <= len(input_ids) - len(system_marker_ids):
                        if torch.equal(
                            input_ids[i : i + len(system_marker_ids)],
                            torch.tensor(system_marker_ids, device=input_ids.device),
                        ):
                            is_start_of_system = True

                    if is_start_of_assistant:
                        in_assistant_turn = True
                        labels[i : i + len(assistant_marker_ids)] = -100

                    if is_start_of_user or is_start_of_system:
                        in_assistant_turn = False

                    if not in_assistant_turn:
                        labels[i] = -100
            else:
                labels[:] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": encodings.attention_mask[0],
                "labels": labels,
            }
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}", exc_info=True)
            dummy_ids = torch.zeros(self.max_length, dtype=torch.long)
            dummy_mask = torch.zeros(self.max_length, dtype=torch.long)
            dummy_labels = -100 * torch.ones(self.max_length, dtype=torch.long)
            return {
                "input_ids": dummy_ids,
                "attention_mask": dummy_mask,
                "labels": dummy_labels,
            }

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
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=2,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=2,
        collate_fn=eval_dataset.collate_fn,
        shuffle=False,
    )

    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    if args.max_training_steps > 0:
        max_train_steps = min(args.max_training_steps, max_train_steps)
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(max_train_steps - current_step)
            / float(max(1, max_train_steps - num_warmup_steps)),
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = None
    if not args.disable_amp:
        if device.type == "cuda":
            scaler = GradScaler()
        elif device.type == "cpu" and model.dtype == torch.bfloat16:
            logger.info(
                "CPU AMP with bfloat16. GradScaler might be needed if not automatic."
            )

    global_step, total_loss, epoch_loss = 0, 0, 0
    best_eval_loss = float("inf")
    start_time = time.time()
    log_interval = 1  # Changed for more frequent logging

    logger.info(
        f"Starting training for {args.num_epochs} epochs ({max_train_steps} steps)"
    )
    logger.info(f"Warmup for {num_warmup_steps} steps")

    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        epoch_start_time = time.time()
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            amp_dtype = (
                torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16
            )
            if device.type == "mps" and not mps_check_bf16_support():
                amp_dtype = torch.float16

            if not args.disable_amp:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=True
                ):
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            total_loss += loss.detach().float()
            epoch_loss += loss.detach().float()

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((step + 1) % args.gradient_accumulation_steps == 0) or (
                step == len(train_dataloader) - 1
            ):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

                if global_step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Epoch: {epoch + 1}/{args.num_epochs} | Step: {global_step}/{max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.2f}s"
                    )
                    total_loss = 0
                    start_time = time.time()

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader, args, device)
                    logger.info(f"Eval Loss: {eval_loss:.4f}")
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        save_checkpoint(model, args.output_dir, "best_model")
                        logger.info(
                            f"New best model saved with eval loss: {best_eval_loss:.4f}"
                        )
                    model.train()

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(model, args.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"Saved checkpoint at step {global_step}")

                if (
                    args.max_training_steps > 0
                    and global_step >= args.max_training_steps
                ):
                    logger.info(
                        f"Reached maximum training steps ({args.max_training_steps}). Stopping."
                    )
                    return

            if device.type == "mps":
                torch.mps.empty_cache()
            # Removed CUDA specific empty_cache

        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Avg loss: {epoch_loss / len(train_dataloader):.4f}"
        )
        save_checkpoint(model, args.output_dir, f"checkpoint-epoch-{epoch + 1}")
        logger.info(f"Saved checkpoint for epoch {epoch + 1}")
    logger.info("Training complete!")


def evaluate(model, eval_dataloader, args, current_device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(current_device) for k, v in batch.items()}

            amp_dtype = (
                torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16
            )
            if current_device.type == "mps" and not mps_check_bf16_support():
                amp_dtype = torch.float16

            if not args.disable_amp:
                with torch.autocast(
                    device_type=current_device.type, dtype=amp_dtype, enabled=True
                ):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            total_loss += outputs.loss.detach().float()
            num_batches += 1
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss.item()


def save_checkpoint(model, output_dir, checkpoint_name):
    save_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model
    logger.info(f"Saving model to {save_dir}")
    model_to_save.save_pretrained(save_dir)


if __name__ == "__main__":
    logger.info(f"Running on device: {device}")  # Moved here from global scope
    logger.info(f"Args: {args}")  # Moved here from global scope

    # Fix seed
    torch.manual_seed(42)
    # Removed CUDA specific seed

    # --- Main Execution ---
    logger.info(f"Loading tokenizer and model from: {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=".cache", trust_remote_code=True
    )

    model_dtype = torch.bfloat16
    if args.disable_amp:
        model_dtype = torch.float32
    elif device.type == "mps" and not mps_check_bf16_support():
        logger.info(
            "MPS device detected, bfloat16 not supported by this device/PyTorch version. Using float16 for model loading if AMP enabled."
        )
        model_dtype = torch.float16
    elif (
        device.type == "cpu" and not args.disable_amp
    ):  # Ensure bfloat16 is attempted on CPU if AMP not disabled
        logger.info(
            "CPU device detected. Using bfloat16 for model loading if AMP enabled (requires PyTorch 1.10+ for CPU AMP)."
        )
        # model_dtype remains torch.bfloat16 as per initial assignment

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=".cache",
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    logger.info(
        f"Tokenizer and model loaded. Model is on {model.device} with dtype {model.dtype}"
    )

    logger.info(f"Loading dataset from disk: {args.processed_dataset_path}...")
    try:
        processed_dataset = load_from_disk(args.processed_dataset_path)
    except Exception as e:
        logger.error(
            f"Failed to load dataset from {args.processed_dataset_path}: {e}",
            exc_info=True,
        )
        sys.exit(1)

    logger.info(f"Dataset loaded. Size: {len(processed_dataset)} examples")

    # Assuming the loaded dataset is a single split (e.g., 'train') or needs to be treated as such.
    # If it's a DatasetDict, you might need to select a split, e.g., processed_dataset['train']
    # For simplicity, assuming it's directly usable or the user provides a pre-split dataset.

    # Splitting the loaded dataset
    try:
        # Attempt to split if it's a single dataset, common for preprocessed data
        total_size = len(processed_dataset)
        train_size = int(0.9 * total_size)
        # Ensure eval_size is at least 0, even if 0.9 * total_size is total_size (e.g. for small datasets)
        eval_size = max(0, total_size - train_size)

        train_dataset_raw = processed_dataset.select(range(train_size))
        if eval_size > 0:
            eval_dataset_raw = processed_dataset.select(range(train_size, total_size))
        else:  # Handle case where dataset is too small for a 90/10 split to yield a non-empty eval set
            eval_dataset_raw = processed_dataset.select([])  # Empty dataset
            logger.warning(
                "Evaluation dataset is empty due to small total dataset size or 90/10 split resulting in zero eval samples."
            )

    except AttributeError as e:
        logger.error(
            f"Failed to split dataset. Ensure '{args.processed_dataset_path}' points to a single Hugging Face Dataset object that can be .select()ed. Error: {e}"
        )
        sys.exit(1)
    except Exception as e:  # Catch other potential errors during split
        logger.error(
            f"An unexpected error occurred during dataset splitting: {e}", exc_info=True
        )
        sys.exit(1)

    logger.info(f"Train dataset size: {len(train_dataset_raw)}")
    logger.info(f"Eval dataset size: {len(eval_dataset_raw)}")

    train_dataset = FunctionCallingDataset(
        train_dataset_raw,
        tokenizer,
        max_length=args.max_seq_length,
        subset_size=args.dataset_subset_size,  # subset_size here applies *after* the 90/10 split
    )

    # Adjust eval_subset_size logic based on the already split eval_dataset_raw
    effective_eval_subset_size = -1  # Default to full (split) eval set
    if args.dataset_subset_size > 0:  # If a global subset size is requested for testing
        # Apply this as a cap to the eval set, or a smaller fixed number if eval set is large
        if len(eval_dataset_raw) > 0:
            effective_eval_subset_size = min(
                args.dataset_subset_size, len(eval_dataset_raw), 1000
            )  # Cap at 1000 for eval subset
        else:
            effective_eval_subset_size = 0  # If eval_dataset_raw is empty

    eval_dataset = FunctionCallingDataset(
        eval_dataset_raw,
        tokenizer,
        max_length=args.max_seq_length,
        subset_size=effective_eval_subset_size,
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")

    train(model, train_dataset, eval_dataset, args)
    logger.info("Fine-tuning complete!")
    save_checkpoint(model, args.output_dir, "final_model")
    logger.info(f"Final model saved to {os.path.join(args.output_dir, 'final_model')}")
