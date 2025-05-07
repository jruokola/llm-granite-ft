import mlflow
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import (
    SFTTrainer,  # simpler SFT loop  [oai_citation:9‡discuss.huggingface.co](https://discuss.huggingface.co/t/when-to-use-sfttrainer/40998)
)
import argparse  # Added for command-line arguments
import os  # For environment variables
import torch  # For distributed training
import torch.distributed as dist  # For distributed training

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    print(
        "WARN: pynvml library not found. GPU utilization and memory metrics will not be logged by this script."
    )
    NVML_AVAILABLE = False

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
parser = argparse.ArgumentParser(
    description="Finetune a Granite model with FSDP-QLoRA."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/job_outputs/checkpoints/granite_fsdp_qlora_default",
    help="Directory to save checkpoints and final model.",
)
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
parser.add_argument(
    "--batches_per_epoch",
    type=int,
    default=0,
    help="Limit batches per epoch for quick test (0 for all).",
)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="Learning rate."
)  # Common for QLoRA
parser.add_argument(
    "--data_path_train",
    type=str,
    default="data/train_extended.csv",
    help="Path to training CSV.",
)
parser.add_argument(
    "--data_path_eval",
    type=str,
    default="data/test_extended.csv",
    help="Path to evaluation CSV.",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="ibm-granite/granite-3.3-8b-instruct",
    help="Base model ID from Hugging Face Hub.",
)
parser.add_argument(
    "--seq_len", type=int, default=1024, help="Maximum sequence length."
)
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=4,
    help="Batch size per device for training.",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=4,
    help="Batch size per device for evaluation.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Gradient accumulation steps.",
)
parser.add_argument("--logging_steps", type=int, default=5, help="Log every X steps.")
parser.add_argument(
    "--save_steps", type=int, default=500, help="Save checkpoint every X steps."
)  # Or based on epochs
parser.add_argument(
    "--dry_run_cpu",
    action="store_true",
    help="Run a minimal test on CPU for 1 batch (for CI).",
)
parser.add_argument(
    "--no_mlflow", action="store_true", help="Disable MLflow logging for this script."
)
# Add FSDP specific args if needed, or use an accelerate config file
# parser.add_argument("--fsdp_config_path", type=str, default=None, help="Path to FSDP config file for Accelerate/Trainer.")

cli_args = parser.parse_args()

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

MODEL_ID = cli_args.model_id  # base model 8 B  [oai_citation:10‡huggingface.co](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)
TRAIN_CSV = cli_args.data_path_train
EVAL_CSV = cli_args.data_path_eval
SEQ_LEN = cli_args.seq_len

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
    "per_device_train_batch_size": cli_args.per_device_train_batch_size,
    "num_train_epochs": cli_args.epochs,
    "learning_rate": cli_args.lr,
    "logging_steps": cli_args.logging_steps,
    "output_dir": cli_args.output_dir,  # Use output_dir from command-line arguments
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
    training_args_dict["gradient_accumulation_steps"] = (
        cli_args.gradient_accumulation_steps
    )
    # Potentially adjust other DDP related args like ddp_find_unused_parameters if not wrapping manually
    # training_args_dict["ddp_find_unused_parameters"] = False

training_args = TrainingArguments(**training_args_dict)

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
    print(f"[Rank {rank}] Training finished. Saving model to {cli_args.output_dir}")
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


class ShowcaseMetricsCallback(TrainerCallback):
    def __init__(self, no_mlflow_arg=False):
        super().__init__()
        self.nvml_initialized = False
        self.no_mlflow = no_mlflow_arg
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                print("[ShowcaseMetricsCallback] NVML Initialized.")
            except Exception as e:
                print(f"[ShowcaseMetricsCallback] WARN: Could not initialize NVML: {e}")

    def _get_gpu_metrics(self, local_rank):
        metrics = {}
        if self.nvml_initialized and torch.cuda.is_available() and local_rank != -1:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics["gpu_util_percent_snapshot"] = float(util.gpu)
                metrics["vram_used_mib_snapshot"] = float(mem_info.used / (1024**2))
                metrics["vram_total_mib_snapshot"] = float(mem_info.total / (1024**2))
                if mem_info.total > 0:
                    metrics["vram_percent_used_snapshot"] = float(
                        (mem_info.used / mem_info.total) * 100
                    )
            except Exception as e:
                print(
                    f"[ShowcaseMetricsCallback] WARN: Could not log NVML metrics for device {local_rank}: {e}"
                )
        return metrics

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not state.is_world_process_zero or self.no_mlflow or not mlflow.active_run():
            return

        current_step = state.global_step
        if logs is not None:
            # Log GPU metrics snapshot
            if torch.cuda.is_available():
                # local_rank should be available via state.local_process_index or args.local_rank
                local_rank_for_nvml = (
                    state.local_process_index if state.local_process_index != -1 else 0
                )
                gpu_metrics = self._get_gpu_metrics(local_rank_for_nvml)
                if gpu_metrics:
                    mlflow.log_metrics(gpu_metrics, step=current_step)
                    # print(f"[MetricsCallback Rank {state.process_index}] Logged GPU Snapshot at step {current_step}")

            # Calculate and log Tokens/s if throughput info is available
            # Trainer often logs 'train_samples_per_second' or 'throughput_samples_per_second'
            # For SFTTrainer, packing might make exact token count per sample variable.
            # We will rely on samples_per_second and multiply by sequence length as an approximation.
            samples_per_second = logs.get(
                "train_samples_per_second", None
            )  # From Trainer
            if samples_per_second is None:
                samples_per_second = logs.get(
                    "samples_per_second", None
                )  # Sometimes just this

            if (
                samples_per_second is not None
                and hasattr(args, "max_seq_length")
                and args.max_seq_length is not None
            ):
                tokens_per_second_approx = samples_per_second * args.max_seq_length
                mlflow.log_metric(
                    "tokens_per_second_interval_approx",
                    tokens_per_second_approx,
                    step=current_step,
                )
                # print(f"[MetricsCallback Rank {state.process_index}] Logged Tokens/s (approx): {tokens_per_second_approx:.2f} at step {current_step}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Ensure NVML shutdown happens for the master process that initialized it.
        # Other ranks might not have initialized it if self.no_mlflow was true for them somehow or NVML_AVAILABLE was false.
        if self.nvml_initialized and state.is_world_process_zero:
            try:
                pynvml.nvmlShutdown()
                print("[ShowcaseMetricsCallback] NVML Shutdown on train end.")
            except Exception as e:
                print(f"[ShowcaseMetricsCallback] WARN: Error shutting down NVML: {e}")

        if not state.is_world_process_zero or self.no_mlflow or not mlflow.active_run():
            return

        if torch.cuda.is_available():
            local_rank_for_nvml = (
                state.local_process_index if state.local_process_index != -1 else 0
            )
            gpu_metrics = self._get_gpu_metrics(local_rank_for_nvml)
            if gpu_metrics:
                final_gpu_metrics = {f"final_{k}": v for k, v in gpu_metrics.items()}
                mlflow.log_metrics(final_gpu_metrics, step=state.global_step)

        final_metrics = next(
            (
                l
                for l in reversed(state.log_history)
                if l.get("train_runtime") is not None
                and l.get("train_samples_per_second") is not None
            ),
            None,
        )
        if (
            final_metrics
            and hasattr(args, "max_seq_length")
            and args.max_seq_length is not None
        ):
            overall_samples_per_second = final_metrics["train_samples_per_second"]
            overall_tokens_per_second_approx = (
                overall_samples_per_second * args.max_seq_length
            )
            mlflow.log_metric(
                "overall_avg_tokens_per_second_approx",
                overall_tokens_per_second_approx,
                step=state.global_step,
            )
            print(
                f"[ShowcaseMetricsCallback Rank {state.process_index}] Logged Overall Avg Tokens/s (approx): {overall_tokens_per_second_approx:.2f}"
            )
