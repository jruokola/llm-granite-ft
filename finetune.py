import mlflow
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import (
    SFTTrainer,  # simpler SFT loop  [oai_citation:9‡discuss.huggingface.co](https://discuss.huggingface.co/t/when-to-use-sfttrainer/40998)
)
import argparse  # Added for command-line arguments
import os  # For environment variables
import torch  # For distributed training
import torch.distributed as dist  # For distributed training
# from torch.utils.data.distributed import DistributedSampler # May be needed if Trainer doesn't handle it

# ---- Distributed Training Setup ----
local_rank = -1
rank = -1
world_size = -1

if "WORLD_SIZE" in os.environ:  # Check if launched in a distributed environment
    # These should be set by torchrun or your Slurm/Soperator setup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    print(
        f"Initializing distributed training: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}"
    )
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)  # Crucial for DDP
    device = torch.device("cuda", local_rank)
else:
    print(
        "Not a distributed run or environment variables not set. Running on single device."
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Argument Parser ----
parser = argparse.ArgumentParser(description="Finetune a Granite model.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/checkpts/granite3.3-lora-default",
    help="Directory to save checkpoints and final model.",
)
# Add other arguments if needed, e.g., for data paths, model_id, etc.
args = parser.parse_args()

# ---- MLflow Setup ----
# Only log from rank 0 process in distributed training
if rank == 0 or rank == -1:  # rank == -1 for non-distributed runs
    print(f"MLflow Autologging enabled for rank {rank}.")
    mlflow.autolog(
        log_models=False,
        log_datasets=False,
        # Below are important for distributed training if Trainer supports them directly
        # log_only_on_rank0=True # Ideal if SFTTrainer/TrainingArguments supports this
    )
elif rank > 0:
    # For non-rank 0 processes, disable mlflow to avoid conflicts if autolog is not rank-aware.
    # A more robust way is if the Trainer itself handles rank-based logging.
    os.environ["MLFLOW_DISABLE"] = "TRUE"
    print(f"MLflow Autologging disabled for rank {rank}.")

MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"  # base model 8 B  [oai_citation:10‡huggingface.co](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
TRAIN_CSV = "data/train_extended.csv"
EVAL_CSV = "data/test_extended.csv"
SEQ_LEN = 1024

# ---- 1.  load in 4-bit (QLoRA)  ----
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16"
)

# For DDP, each process loads the full model initially. device_map="auto" is problematic.
# We will load to CPU first if memory is an issue, then move to specific GPU,
# or load directly to the target GPU if feasible.
# If OOM on direct GPU load, use device_map='cpu' or device_map={'':'cpu'} then model.to(device)
print(f"[Rank {rank if rank != -1 else 'N/A'}] Loading base model {MODEL_ID}...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # device_map="auto", # Remove for DDP; manage device placement manually
    quantization_config=bnb_cfg,
    # low_cpu_mem_usage=True # Can help if loading large models on CPU first
)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

if local_rank != -1:  # If distributed, ensure model is on the correct device before DDP
    print(f"[Rank {rank}] Moving base model to device: {device}")
    base.to(device)

# ---- 2.  attach LoRA adapters  ----
lora_cfg = LoraConfig(
    r=64, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(base, lora_cfg)

if local_rank != -1:  # If distributed, wrap model with DDP
    print(f"[Rank {rank}] Wrapping model with DDP for device: {device}")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=False
    )
    # Note: find_unused_parameters might be needed depending on model structure and LoRA
else:  # Non-distributed: move model to device if not already via device_map
    if (
        hasattr(base, "hf_device_map") and not base.hf_device_map
    ):  # if no device_map was used
        print(f"Moving model to device: {device}")
        model.to(device)

if rank == 0 or rank == -1:
    model.module.print_trainable_parameters() if hasattr(
        model, "module"
    ) else model.print_trainable_parameters()  # Access original model for PEFT methods

# ---- 3.  load tiny dataset  ----
print(f"[Rank {rank if rank != -1 else 'N/A'}] Loading dataset...")
ds = load_dataset(
    "csv",
    data_files={"train": TRAIN_CSV, "eval": EVAL_CSV},
    column_names=["system", "user", "assistant"],
)


# Helper to turn row → single prompt+completion
def fmt(ex):
    return {"prompt": f"{ex['system']}\n{ex['user']}", "completion": ex["assistant"]}


ds = ds.map(fmt, remove_columns=ds["train"].column_names)

# ---- 4.  trainer  ----

# Define Training Arguments separately
print(f"[Rank {rank if rank != -1 else 'N/A'}] Initializing TrainingArguments...")
training_args_dict = {
    "per_device_train_batch_size": 4,
    "num_train_epochs": 5,
    "learning_rate": 2e-4,
    "logging_steps": 5,
    "output_dir": args.output_dir,  # Use output_dir from command-line arguments
    "report_to": "mlflow"
    if (rank == 0 or rank == -1)
    else "none",  # Report to mlflow only on rank 0
    # Args for distributed training (Trainer should pick these up from env if torch.distributed is initialized):
    # "local_rank": local_rank if local_rank != -1 else -1, # Usually not needed to pass explicitly
    # "ddp_find_unused_parameters": False, # If using DDP via Trainer args
}

# For FSDP with PEFT, specific configurations might be needed
# if using HuggingFace Trainer's FSDP integration instead of DDP wrapper.
# For now, we are using explicit DDP wrapping.

if world_size > 1:  # If distributed training
    training_args_dict["gradient_accumulation_steps"] = 1  # Example, adjust as needed
    # Potentially adjust other DDP related args like ddp_find_unused_parameters if not wrapping manually
    # training_args_dict["ddp_find_unused_parameters"] = False

training_args = TrainingArguments(**training_args_dict)

# MLflow run context is managed by autolog triggered by trainer.train()
# Removed: with mlflow.start_run():
# Removed: manual mlflow.log_param calls
# Removed: manual mlflow.log_artifact("ds_config_zero3.json") # Autolog might not capture this

print(f"[Rank {rank if rank != -1 else 'N/A'}] Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,  # This is the DDP-wrapped model if distributed
    tokenizer=tok,
    train_dataset=ds["train"],
    eval_dataset=ds["eval"],
    max_seq_length=SEQ_LEN,
    args=training_args,
)

print(f"[Rank {rank if rank != -1 else 'N/A'}] Starting training...")
train_result = trainer.train()

# Removed: manual mlflow.log_metric calls for train_result
# Verify in MLflow UI what metrics are captured by autolog

# Save the final model (adapter)
if rank == 0 or rank == -1:
    print(f"[Rank {rank}] Training finished. Saving model to {args.output_dir}")
    # When using DDP, the model to save is model.module to get the original underlying model
    model_to_save = model.module if hasattr(model, "module") else model
    trainer.model = model_to_save  # Temporarily set trainer.model to the unwrapped model for saving PEFT adapters
    trainer.save_model()  # This saves the LoRA adapters
    print(f"[Rank {rank}] Model adapters saved.")

    # Optional: merge and save full model (if desired, and if you have enough memory on rank 0)
    # print(f"[Rank {rank}] Merging adapters and saving full model...")
    # merged_model = model_to_save.merge_and_unload()
    # merged_path = os.path.join(args.output_dir, "merged_model_final")
    # merged_model.save_pretrained(merged_path)
    # tok.save_pretrained(merged_path)
    # print(f"[Rank {rank}] Full merged model saved to {merged_path}")
    # if mlflow.active_run():
    #     mlflow.log_artifacts(merged_path, artifact_path="merged_model_final")

if world_size > 1:
    dist.destroy_process_group()

print(f"[Rank {rank if rank != -1 else 'N/A'}] Script finished.")
