import torch
import argparse
import subprocess
import logging
import math
import time
import functools
import os

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from datasets import load_dataset
from torch.amp import GradScaler

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
    default=True,
    help="Use gradient checkpointing (default: True)",
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=2048,
    help="Maximum sequence length for truncation.",
)
parser.add_argument(
    "--disable_amp", action="store_true", help="Disable automatic mixed precision"
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
    default=1,
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
args = parser.parse_args()


# Function calling dataset processor
class FunctionCallingDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048, subset_size=-1):
        self.tokenizer = tokenizer
        # If a subset size is specified, take only that many examples
        if subset_size > 0:
            self.dataset = dataset.select(range(min(subset_size, len(dataset))))
        else:
            self.dataset = dataset

        self.max_length = max_length
        # Set special tokens for the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Print sample data to understand structure
        if len(self.dataset) > 0:
            print(f"Dataset sample keys: {list(self.dataset[0].keys())}")
            print(
                f"Dataset sample first example chat content (first 200 chars): {self.dataset[0]['chat'][:200] if 'chat' in self.dataset[0] else None}"
            )

    def __len__(self):
        return len(self.dataset)

    def safe_process_message(self, message, idx=None):
        """Process a single message safely without causing attribute errors."""
        try:
            if isinstance(message, dict):
                # Handle dictionary message
                role = ""
                content = ""

                # Safely get role and content
                if "role" in message and isinstance(message["role"], str):
                    role = message["role"].lower()
                elif "role" in message:
                    # Non-string role, convert to string
                    role = str(message["role"]).lower()

                if "content" in message and isinstance(message["content"], str):
                    content = message["content"]
                elif "content" in message:
                    # Non-string content, convert to string
                    content = str(message["content"])

                # Format based on role
                if role == "system":
                    return f"<|system|>\n{content}\n"
                elif role == "user":
                    return f"<|user|>\n{content}\n"
                elif role == "assistant":
                    return f"<|assistant|>\n{content}\n"
                else:
                    # Unknown role, default to user
                    return f"<|user|>\n{content}\n"
            elif isinstance(message, str):
                # Handle string message (default to user)
                return f"<|user|>\n{message}\n"
            elif isinstance(message, list):
                # Handle list of messages recursively
                result = ""
                for submessage in message:
                    result += self.safe_process_message(submessage, idx)
                return result
            else:
                # Unknown type, convert to string and treat as user message
                if idx is not None and idx < 3:
                    print(f"Unknown message type: {type(message)} for example {idx}")
                    print(f"Message content: {message}")
                return f"<|user|>\n{str(message)}\n"
        except Exception as e:
            # Log any errors and return empty string
            if idx is not None and idx < 3:
                print(f"Error processing message: {str(e)}")
                print(f"Message content: {message}")
            return ""

    def __getitem__(self, idx):
        try:
            # Get the example from the dataset
            example = self.dataset[idx]

            # Debug first few examples
            if idx < 3:
                print(f"Processing example {idx}")

            # Initialize the formatted text
            formatted_text = ""

            # Handle system content if available
            if "system" in example:
                system_content = (
                    str(example["system"]).strip()
                    if example["system"] is not None
                    else ""
                )
                if system_content:
                    formatted_text += f"<|system|>\n{system_content}\n"

            # Get chat content from various possible fields
            chat_content = None
            if "chat" in example and example["chat"] is not None:
                chat_content = example["chat"]
            elif "text" in example and example["text"] is not None:
                chat_content = example["text"]
            else:
                # Fallback: use entire example as JSON string
                chat_content = str(example)

            # Process the chat content based on its type
            if isinstance(chat_content, str):
                # String content: process as conversation with markers
                lines = chat_content.split("\n")
                current_role = None
                current_content = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check for role markers
                    if line.startswith("SYSTEM:"):
                        # Process previous role if any
                        if current_role and current_content:
                            role_content = "\n".join(current_content).strip()
                            if current_role == "system":
                                # Only add if not already added
                                if not formatted_text.startswith("<|system|>"):
                                    formatted_text += f"<|system|>\n{role_content}\n"
                            elif current_role == "user":
                                formatted_text += f"<|user|>\n{role_content}\n"
                            elif current_role == "assistant":
                                formatted_text += f"<|assistant|>\n{role_content}\n"

                        # Start new system content
                        current_role = "system"
                        current_content = [line.replace("SYSTEM:", "").strip()]

                    elif line.startswith("USER:"):
                        # Process previous role if any
                        if current_role and current_content:
                            role_content = "\n".join(current_content).strip()
                            if current_role == "system":
                                # Only add if not already added
                                if not formatted_text.startswith("<|system|>"):
                                    formatted_text += f"<|system|>\n{role_content}\n"
                            elif current_role == "user":
                                formatted_text += f"<|user|>\n{role_content}\n"
                            elif current_role == "assistant":
                                formatted_text += f"<|assistant|>\n{role_content}\n"

                        # Start new user content
                        current_role = "user"
                        current_content = [line.replace("USER:", "").strip()]

                    elif line.startswith("A:"):
                        # Process previous role if any
                        if current_role and current_content:
                            role_content = "\n".join(current_content).strip()
                            if current_role == "system":
                                # Only add if not already added
                                if not formatted_text.startswith("<|system|>"):
                                    formatted_text += f"<|system|>\n{role_content}\n"
                            elif current_role == "user":
                                formatted_text += f"<|user|>\n{role_content}\n"
                            elif current_role == "assistant":
                                formatted_text += f"<|assistant|>\n{role_content}\n"

                        # Start new assistant content
                        current_role = "assistant"
                        current_content = [line.replace("A:", "").strip()]

                    elif line.startswith("FUNCTION RESPONSE:"):
                        # Special case for function responses
                        if current_role == "assistant":
                            # Function response is part of the assistant conversation
                            current_content.append(line)
                        else:
                            # If not in an assistant message, start a new one
                            if current_role and current_content:
                                role_content = "\n".join(current_content).strip()
                                if current_role == "system":
                                    formatted_text += f"<|system|>\n{role_content}\n"
                                elif current_role == "user":
                                    formatted_text += f"<|user|>\n{role_content}\n"
                                elif current_role == "assistant":
                                    formatted_text += f"<|assistant|>\n{role_content}\n"

                            current_role = "assistant"
                            current_content = [line]
                    else:
                        # Continue with current role
                        if current_role:
                            current_content.append(line)
                        else:
                            # If no role defined yet, assume it's the system message
                            current_role = "system"
                            current_content = [line]

                # Process the last role if any
                if current_role and current_content:
                    role_content = "\n".join(current_content).strip()
                    if current_role == "system":
                        # Only add if not already added
                        if not formatted_text.startswith("<|system|>"):
                            formatted_text += f"<|system|>\n{role_content}\n"
                    elif current_role == "user":
                        formatted_text += f"<|user|>\n{role_content}\n"
                    elif current_role == "assistant":
                        formatted_text += f"<|assistant|>\n{role_content}\n"

            elif isinstance(chat_content, list):
                # Debug content format for the first few examples
                if idx < 3:
                    print(
                        f"Example {idx} has list chat_content with {len(chat_content)} items"
                    )
                    if len(chat_content) > 0:
                        print(f"First item type: {type(chat_content[0])}")

                # Handle list of messages using our safe processor
                for message in chat_content:
                    formatted_text += self.safe_process_message(message, idx)
            else:
                # Unknown content type, convert to string and add as user message
                if idx < 3:
                    print(
                        f"Unknown chat_content type: {type(chat_content)} for example {idx}"
                    )
                formatted_text += f"<|user|>\n{str(chat_content)}\n"

            # Remove special tokens if present
            formatted_text = formatted_text.replace("<|endoftext|>", "")

            # If we couldn't parse anything useful, provide a fallback
            if not formatted_text:
                formatted_text = (
                    "<|user|>\nHello\n<|assistant|>\nHello! How can I help you today?\n"
                )

            # Debug the formatted text for the first few examples
            if idx < 3:
                print(
                    f"Formatted text for example {idx} (sample): {formatted_text[:200]}..."
                )

            # Encode the formatted text
            encodings = self.tokenizer(
                formatted_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # Create the labels (same as input_ids but with -100 for non-assistant tokens)
            input_ids = encodings.input_ids[0]
            attention_mask = encodings.attention_mask[0]

            # Create labels: -100 for non-assistant tokens, actual token ids for assistant tokens
            labels = input_ids.clone()

            # Find positions of <|assistant|> tokens to mask everything before them
            assistant_positions = []
            for i in range(len(input_ids) - 1):
                token_pair = self.tokenizer.decode(input_ids[i : i + 2])
                if "<|assistant|>" in token_pair:
                    assistant_positions.append(i)

            # Set labels to -100 for non-assistant parts
            if assistant_positions:
                in_assistant = False
                for i in range(len(labels)):
                    # Check if this is the start of an assistant section
                    if i in assistant_positions:
                        in_assistant = True
                        # Skip the assistant token itself in the loss
                        labels[i : i + 2] = -100
                        continue

                    # If not in assistant section, mask the token
                    if not in_assistant:
                        labels[i] = -100

                    # Check if this is the end of an assistant section
                    if in_assistant and i < len(labels) - 1:
                        token_pair = self.tokenizer.decode(input_ids[i : i + 2])
                        if "<|user|>" in token_pair or "<|system|>" in token_pair:
                            in_assistant = False
            else:
                # If no assistant token found, don't compute loss on this example
                labels[:] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        except Exception as e:
            print(f"Error processing example {idx}: {str(e)}")
            import traceback

            traceback.print_exc()
            # Return a dummy example that won't affect training
            dummy_ids = torch.zeros(self.max_length, dtype=torch.long)
            dummy_mask = torch.zeros(self.max_length, dtype=torch.long)
            dummy_labels = -100 * torch.ones(self.max_length, dtype=torch.long)
            return {
                "input_ids": dummy_ids,
                "attention_mask": dummy_mask,
                "labels": dummy_labels,
            }

    def collate_fn(self, examples):
        # This function is called by the DataLoader to collate multiple examples into a batch
        # All examples are already padded to max_length, so we can just stack them
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
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size_per_device,
        num_workers=4,
        collate_fn=eval_dataset.collate_fn,
        sampler=eval_sampler,
    )

    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
    log_interval = 10

    logger.info(
        f"Starting training for {args.num_epochs} epochs ({max_train_steps} steps)"
    )
    logger.info(f"Warmup for {num_warmup_steps} steps")

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")

        epoch_start_time = time.time()
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if not args.disable_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            # Accumulate loss statistics
            total_loss += loss.detach().float()
            epoch_loss += loss.detach().float()

            # Backward pass with gradient accumulation
            if args.disable_amp:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            # Update weights after accumulating gradients
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (
                step == len(train_dataloader) - 1
            ):
                if args.disable_amp:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

                # Log training progress
                if global_step % log_interval == 0:
                    avg_loss = total_loss / log_interval
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
            f"Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {epoch_loss / len(train_dataloader):.4f}"
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
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        logger.info("Consolidating FSDP model for saving...")
        model_to_save = FSDP.consolidate_model_parallel_state(model)
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
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    cache_dir=".cache",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
logger.info("Tokenizer and model loaded.")

# Load the dataset
logger.info("Loading Function Calling dataset...")
raw_dataset = load_dataset("glaiveai/glaive-function-calling-v2")
logger.info(f"Dataset loaded. Size: {len(raw_dataset['train'])} examples")

# Create train and eval datasets
train_size = int(0.9 * len(raw_dataset["train"]))
eval_size = len(raw_dataset["train"]) - train_size

# Split into train and eval
train_dataset_raw = raw_dataset["train"].select(range(train_size))
eval_dataset_raw = raw_dataset["train"].select(
    range(train_size, train_size + eval_size)
)

logger.info(f"Train dataset size: {len(train_dataset_raw)}")
logger.info(f"Eval dataset size: {len(eval_dataset_raw)}")

# Prepare datasets
train_dataset = FunctionCallingDataset(
    train_dataset_raw,
    tokenizer,
    max_length=args.max_seq_length,
    subset_size=args.dataset_subset_size,
)
eval_dataset = FunctionCallingDataset(
    eval_dataset_raw,
    tokenizer,
    max_length=args.max_seq_length,
    subset_size=min(1000, len(eval_dataset_raw))
    if args.dataset_subset_size > 0
    else -1,
)

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
    model = FSDP(model, device_id=device, auto_wrap_policy=auto_wrap_policy)

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
