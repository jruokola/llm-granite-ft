# LLM Fine-tuning for Chess and Function Calling (Distributed Training & Slurm on Nebius)

This project supports fine-tuning Large Language Models for two distinct tasks:

1. **CodeLlama for Chess**: Fine-tunes a CodeLlama LLM (e.g., `codellama/CodeLlama-7b-hf`) on a chess dataset (`strategic_game_chess.jsonl`) using PyTorch Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).

2. **Granite-3.3-2B-Instruct for Function Calling**: Fine-tunes an IBM Granite model (e.g., `ibm-granite/granite-3.3-2b-instruct`) using a synthetically generated dataset for function calling, orchestrated with PyTorch DDP/FSDP. For demonstration and testing purposes, external datasets like `glaiveai/glaive-function-calling-v2` or `NousResearch/hermes-function-calling-v1` are not currently used due to previous processing challenges.

Both fine-tuning processes are orchestrated using Slurm for distributed training on a Nebius Kubernetes cluster.

## Prerequisites

1. **Nebius Kubernetes Cluster with Slurm/Soperator:** Functional cluster with Slurm/Soperator installed.
2. **Nebius Filestore (or equivalent shared storage):** Mounted and accessible by Slurm worker nodes for storing model checkpoints and outputs.
3. **Docker Registry:** Accessible registry (e.g., Nebius Container Registry) for the training image.
4. **Local Environment:** Python and Docker installed for building the training image.
5. **Nebius CLI & `kubectl`:** Configured access (optional for debugging).
6. **Slurm Client Tools:** `sbatch`, `squeue`, etc., on a login node.

## Project Structure

```text
llm-granite-ft/
├── Dockerfile.chess                     # Defines the container image for chess training
├── Dockerfile.function                  # Defines the container image for function calling training (example name)
├── chess-finetune.py                    # Python script for chess fine-tuning
├── fixed-scripts/
│   └── function-finetune-fixed.py       # Python script for function calling fine-tuning
├── strategic_game_chess.jsonl            # Chess dataset file (for chess-finetune.py)
├── submit_finetune_chess.sbatch          # Slurm batch submission script for chess
├── submit_finetune_function_calling.sbatch # Slurm batch submission script for function calling
├── pyproject.toml                       # Project dependencies and metadata (used by uv)
└── README.md                            # This file
```

(Note: `function-finetune.py` and `Dockerfile.osx` are also present for local/OSX debugging but not detailed here for Slurm deployment.)

## Setup and Deployment Workflow

### 1. Prepare Data

* **For Chess Fine-tuning:** The `strategic_game_chess.jsonl` file should be present in the `llm-granite-ft` directory when building the Docker image, as `Dockerfile.chess` copies it into the image.

* **For Function Calling Fine-tuning:**
  * The primary method for obtaining data for this task is now by using the synthetic data generator script: `llm-granite-ft/fixed-scripts/generate_granite_fc_examples.py`.
  * This script creates a small, correctly formatted dataset with examples of function calls in the Granite format.
  * Run this script first to generate the dataset. Example usage:

```bash
python llm-granite-ft/fixed-scripts/generate_granite_fc_examples.py \
    --output_path ./my_synthetic_fc_dataset \
    --num_examples 25 \
    --tokenizer_name_or_path "ibm-granite/granite-3.3-2b-instruct" \
    --max_seq_length 512 
```

* The `--output_path` (e.g., `./my_synthetic_fc_dataset`) will then be used as the `--processed_dataset_path` argument for the `function-finetune-fixed.py` script via the Slurm sbatch file.
* This synthetic dataset is primarily for demonstration and testing the fine-tuning pipeline.

### 2. Build and Push Docker Images

You will need to build and push separate Docker images for each fine-tuning task.

**a. Chess Fine-tuning Image:**

1. Navigate to the `llm-granite-ft` directory.
2. Build the Docker image using `Dockerfile.chess`:

    ```bash
    # Replace <tag> with your desired tag, e.g., latest
    docker build -f Dockerfile.chess -t <your_registry>/llm-chess-ft:<tag> .
    ```

    (Example: `docker build -f Dockerfile.chess -t cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-chess-ft:latest .`)
3. Push the image:

    ```bash
    docker push <your_registry>/llm-chess-ft:<tag>
    ```

**b. Function Calling Fine-tuning Image:**

1. Navigate to the `llm-granite-ft` directory.
2. Build the Docker image using `Dockerfile.function` (or your equivalent Dockerfile for this task):

    ```bash
    # Replace <tag> with your desired tag, e.g., latest
    # The image name should match what's in submit_finetune_function_calling.sbatch
    # Example:
    docker build -f Dockerfile.function -t cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-transformers:latest .
    ```

    (Ensure `Dockerfile.function` is correctly set up to include `fixed-scripts/function-finetune-fixed.py` and its dependencies like `peft`, `bitsandbytes`, `transformer_engine` if using FP8, etc.)
3. Push the image:

    ```bash
    docker push cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-transformers:latest
    ```

    (Use your actual image name and tag).

### 3. Configure Slurm Submission Scripts

You will use different sbatch scripts for the different fine-tuning tasks.

#### a. Chess Fine-tuning (`submit_finetune_chess.sbatch`)

Place this script on the Slurm login node or accessible shared storage.

**Review and update these sections before submitting:**

* **Resource Requests (`#SBATCH` directives):** Adjust `--nodes`, `--ntasks-per-node`, `--gres=gpu:...`, `--cpus-per-task`, `--mem`, `--time` as needed.
* **`srun` options for Pyxis/Enroot:**
  * `--container-image`: Ensure this points to the **chess image URI** you pushed (e.g., `docker://cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-chess-ft:latest`).
  * `--container-workdir`: Should be `/workspace`.
  * `--container-mounts`:
    * The script constructs `CONTAINER_MOUNTS_ARG` to map a host path to `/job_outputs` inside the container.
    * The default host path is `/root/slurm_outputs/${SLURM_JOB_ID}`. **Ensure `/root/slurm_outputs` exists on the host file system accessible by workers and is writable by the job user.** You might need to adjust this path based on your shared storage setup (e.g., map to your Nebius Filestore mount point).
* **`chess-finetune.py` Arguments (`FINETUNE_CLI_ARGS`):**
  * `--output_dir` is set automatically based on the mounted path.
  * `--data_path` points to the dataset copied into the image.
  * Uncomment and set other arguments like `--batch_size_per_device`, `--learning_rate`, `--gradient_accumulation_steps` if you need to override the defaults in `chess-finetune.py`.
* **`torchrun` parameters:**
  * `--nproc_per_node`: The script attempts to calculate this based on allocated GPUs (`$CUDA_VISIBLE_DEVICES`, `$SLURM_GPUS_PER_TASK`, etc.). Ensure your `#SBATCH --gres=gpu:N` request aligns with how many processes you expect per node.

#### b. Function Calling Fine-tuning (`submit_finetune_function_calling.sbatch`)

This script is used to launch `fixed-scripts/function-finetune-fixed.py`. Place it on the Slurm login node or accessible shared storage.

**Key configurations in `submit_finetune_function_calling.sbatch`:**

* **Job Name:** Set to `fc-qlora-h100` (as per `#SBATCH --job-name`).
* **Output Log:** Main Slurm log is directed to `/root/slurm_logs/fc_%j.log` (as per `#SBATCH --output`). Node-specific logs may be created by the application within the job's shared directory.
* **Container Image:** The sbatch script uses `IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-13:latest"`. Ensure this matches the image you built and pushed.
* **Shared Job Directory:** Base directory for outputs, logs, and coordination files is `/slurm_jobs/${SLURM_JOB_ID}` on the host, mounted into the container at `/job_data`. (Note: The sbatch script uses `/mnt/jail/` prefix for `HOST_JOBDIR` internally, but the conceptual host path for user understanding is `/slurm_jobs/...`).
* **Python Script:** Executes `function-finetune-fixed.py` (located in the container's `/workspace`).
* **NCCL Environment Variables:** The sbatch script sets `NCCL_DEBUG=INFO`. Other NCCL variables like `NCCL_COLLNET_ENABLE`, `NCCL_IB_HCA` are not explicitly set in the current version of the sbatch script.
* **Distributed Setup (using `torchrun`):**
  * The script now uses `torchrun` to launch `function-finetune-fixed.py`.
  * `torchrun` parameters include `--nnodes`, `--nproc_per_node`, `--rdzv_backend=c10d`, `--rdzv_id`, and `--rdzv_endpoint` (constructed from `MASTER_IP` and `MASTER_PORT` derived from Slurm).
  * While the sbatch script sets `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`, `torchrun` typically manages these for the application. The Python script `function-finetune-fixed.py` is designed for `env://` initialization, which `torchrun` provides.
* **Python Script Arguments (`FINETUNE_CLI_ARGS`):**
  * `--output_dir` is set to `${CONT_JOBDIR}/checkpoints` (where `CONT_JOBDIR` is `/job_data` inside the container).
  * `--processed_dataset_path` should be set to the path where the synthetic dataset was saved by `generate_granite_fc_examples.py` (e.g., `/path/on/shared/storage/my_synthetic_fc_dataset` if generated outside the container, or a path within the container if copied during image build or mounted). The sbatch script example might use a placeholder like `/job_data/synthetic_dataset` which you would need to ensure is correctly populated or mounted.
  * `--use_qlora` is passed by default, enabling QLoRA.
  * `--use_fp8` is also passed by default in the sbatch script. If enabled:
    * LoRA layers are converted to Transformer Engine FP8 layers.
    * The script internally sets the precision for non-TE components to `torch.float16`. Standard `torch.autocast` for AMP will use `torch.float16` for these parts.
  * The `--disable_amp` argument is **not** passed by default.
    * If `--use_fp8` is active, non-TE parts run in `torch.float16` AMP. To run non-TE parts in FP32 while TE layers use FP8, you would add `--disable_amp` to `FINETUNE_ARGS`.
    * If `--use_fp8` is *not* active, the script uses the `--amp_precision_mode` (defaulting to `bf16`) for AMP.
  * Other arguments for `function-finetune-fixed.py` (e.g., `--batch_size_per_device`, `--learning_rate`, other LoRA parameters) can be added to `FINETUNE_ARGS` in the sbatch script.
  * **Important Considerations for `FINETUNE_ARGS` (previously `FINETUNE_CLI_ARGS`):**
    * **`--batch_size_per_device`**: The script `function-finetune-fixed.py` defaults this to 16. For large models like Granite 3.3B, especially on GPUs with ~80GB memory, even this might be too high. It's recommended to start with a smaller value (e.g., 4 or 8) and enable `--gradient_checkpointing` to prevent Out-Of-Memory errors. Adjust based on your specific GPU memory and model size.
    * **`--lora_r`**: If using QLoRA (`--use_qlora`) in conjunction with FP8 (`--use_fp8`), the `--lora_r` value **must be a multiple of 16**. The script `function-finetune-fixed.py` defaults `--lora_r` to 16 and includes a check to enforce this.
    * **`--gradient_checkpointing`**: Strongly recommended to reduce memory usage, especially with large batch sizes or models. Ensure this flag is passed in `FINETUNE_CLI_ARGS` if needed.

### 4. Configure Enroot Authentication (If using private registry)

If pulling `llm-chess-ft:<tag>` requires authentication:

1. Create/edit `/root/.config/enroot/.credentials` (or user equivalent) on the environment where `srun` executes (login node, potentially propagated to workers).
2. Add line: `machine <your_registry_host> login <KEY_ID> password <SECRET_KEY>`
    * Use appropriate credentials (e.g., Nebius SA Static Access Key).
3. Set permissions: `chmod 600 .../.credentials`.

### 5. Submit the Fine-tuning Job to Slurm

From the Slurm login node, submit the appropriate sbatch script for your desired task:

**a. For Chess Fine-tuning:**

```bash
sbatch /path/to/your/llm-granite-ft/submit_finetune_chess.sbatch
```

**b. For Function Calling Fine-tuning:**

```bash
sbatch /path/to/your/llm-granite-ft/submit_finetune_function_calling.sbatch
```

Ensure the paths to the sbatch scripts are correct.

### 6. Monitor the Job

* **Slurm:** `squeue -u $USER`, `scontrol show job <jobid>`, check output log (e.g., `/root/chess_finetune_<jobid>.log` as defined in the sbatch script).
* **Kubernetes (for Soperator debugging):** `kubectl get pods -A -l slurm.nebius.ai/job-id=<jobid>`, `kubectl logs <pod_name> -n <namespace>`.

## Fine-tuning Scripts

### a. Chess Fine-tuning (`chess-finetune.py`)

* Uses standard PyTorch DDP (`torch.distributed.init_process_group`, `torch.nn.parallel.DistributedDataParallel`).
* Parses command-line arguments for hyperparameters (batch size, LR, etc.).
* Includes a basic `JsonlDataset` class for the chess data (`strategic_game_chess.jsonl`).
* Implements a standard training loop using `torch.optim.AdamW`, `GradScaler` for AMP (bf16), and a simple linear warmup LR schedule.
* Typically launched via `torchrun` (often implicitly handled by `srun` with correct arguments if the sbatch script is set up for it, though the example `submit_finetune_chess.sbatch` might need review for this specific launch method).
* May contain optional FSDP logic.

### b. Function Calling Fine-tuning (`fixed-scripts/function-finetune-fixed.py`)

* **Current Version (as of last update):** This script is significantly refactored for fine-tuning models like `ibm-granite/granite-3.3-2b-instruct` using PyTorch FSDP with QLoRA.
* **Key Features:**
  * **FSDP:** Uses `torch.distributed.fsdp.FullyShardedDataParallel` by default.
    * Utilizes `use_orig_params=True` and `ignored_modules` (especially for QLoRA compatibility) when initializing FSDP. It does not use a specific Hugging Face `transformer_auto_wrap_policy`.
    * Includes logic to set `ignored_modules` for FSDP, particularly for handling potential `int8` parameters from quantization.
    * Uses `sync_module_states=True` and `StateDictType.FULL_STATE_DICT` for robust checkpointing.
  * **QLoRA:** Enabled via the `--use_qlora` flag.
    * Configures `BitsAndBytesConfig` for 4-bit quantization (e.g., `nf4`). The `bnb_4bit_quant_storage` is aligned with the model's `amp_dtype` for FSDP compatibility.
    * Uses `peft.prepare_model_for_kbit_training` and `peft.get_peft_model` with `LoraConfig`.
  * **FP8 Support (Optional via Transformer Engine):** Includes experimental support for NVIDIA Transformer Engine FP8 for LoRA adapters (`--use_fp8`), if `transformer_engine` is available.
    * **Important Note on FP8 and AMP:** When using `--use_fp8` with Transformer Engine, standard PyTorch Automatic Mixed Precision (`torch.autocast`) should generally be disabled (`--disable_amp`). Transformer Engine's `fp8_autocast` context manages its own precision for FP8 layers, and the surrounding non-TE operations will run in the precision determined by `--disable_amp` (FP32) or implicitly by the TE FP8 setup (often FP16 for non-TE parts if AMP is not explicitly disabled). The script `function-finetune-fixed.py` sets `amp_dtype` to `torch.float16` if `--use_fp8` is active and AMP is not disabled, which means non-TE parts run under `torch.autocast("cuda", dtype=torch.float16)`.
  * **Data Handling:**
    * **Requires preprocessed data:** The script loads data using `datasets.load_from_disk` via the mandatory `--processed_dataset_path` argument. On-the-fly processing of raw datasets is no longer supported in this version.
    * Uses a custom `Split` class (a wrapper around a Hugging Face `Dataset` split).
  * **Mixed Precision (AMP):**
    * If `--use_fp8` is **not** enabled, AMP is active by default, typically using `torch.bfloat16` (if supported) or `torch.float16`. A `GradScaler` (standard or sharded for FSDP) is used with `torch.float16`.
    * If `--use_fp8` **is** enabled, non-TE parts of the model operate under `torch.autocast` with `torch.float16` (unless `--disable_amp` is also passed, then they use FP32).
    * AMP can be fully disabled with `--disable_amp`.
  * **Optimizer & Scheduler:** Uses `torch.optim.AdamW` and a linear warmup LR schedule.
  * **Gradient Checkpointing:** Supported via `--gradient_checkpointing`.
  * **Logging & Checkpointing:** Standard logging and checkpoint saving logic, compatible with FSDP.
  * **Distributed Launch:** The `submit_finetune_function_calling.sbatch` script now uses `torchrun` to launch this Python script. The Python script itself is compatible with `torchrun`'s `env://` initialization method for distributed training.

## Dockerization

### a. `Dockerfile.chess` (for Chess Fine-tuning)

* Uses a base image like `nvcr.io/nvidia/pytorch:24.07-py3`.
* Installs minimal extra dependencies (e.g., `transformers`, `datasets`).
* Copies the `chess-finetune.py` script and `strategic_game_chess.jsonl` dataset into `/workspace`.

### b. `Dockerfile.function` (for Function Calling Fine-tuning)

* This Dockerfile (e.g., named `Dockerfile.function` or similar, corresponding to the image `llm-granite-function-ft-fix`) should be set up to:
  * Use a suitable PyTorch base image (e.g., `nvcr.io/nvidia/pytorch:24.07-py3` or newer, as used in recent examples).
  * Install dependencies like `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, and optionally `transformer_engine` (for FP8).
  * Copy the `fixed-scripts/function-finetune-fixed.py` script and `fixed-scripts/generate_granite_fc_examples.py` (if you intend to generate data within a job step, though typically it's a pre-step).
  * If using the synthetic data generator as a pre-step, the `Dockerfile.function` does not necessarily need to copy the dataset itself, as the path will be provided at runtime.

## Troubleshooting

* **Slurmctld Down / PartitionConfig Errors:** Check `sinfo`, `slurmctld` logs, and `slurmd` logs on worker pods via cluster admin or `kubectl`.
* **Image Pull Errors (401 Unauthorized):** Verify Enroot authentication credentials for your registry.
* **Pyxis Errors (`couldn't start container`):** Check image pull success, `--container-mounts` validity (host path exists and has permissions), and basic container functionality.
* **`torchrun` Errors / DDP Init Errors (for `chess-finetune.py` or similar `torchrun`-based scripts):** Ensure `srun` is passing necessary Slurm environment variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.) correctly into the container for `torchrun` auto-detection.
* **`torchrun` / DDP/FSDP Init Errors (for `function-finetune-fixed.py`):** If `torch.distributed.init_process_group` (called internally by the script for `env://` init) fails or hangs:
  * Verify `torchrun` parameters in the sbatch script (`--nnodes`, `--nproc_per_node`, rendezvous endpoint).
  * Check that `MASTER_IP` and `MASTER_PORT` are correctly determined and accessible between nodes.
  * Examine `NCCL_DEBUG=INFO` output and any node-specific logs within the job directory for clues.
  * Ensure the network configuration (e.g., InfiniBand, Ethernet) is correctly utilized by NCCL.
* **`ModuleNotFoundError`:** Ensure the respective Dockerfile (`Dockerfile.chess` or `Dockerfile.function`) installs all required Python packages. Check `PYTHONPATH` if necessary, although direct script execution in `/workspace` (or `/workspace/fixed-scripts/`) should generally work if scripts and dependencies are correctly placed.
* **CUDA Errors / `nvidia-smi` fails inside container:** Likely an issue with Pyxis/Enroot setup, host drivers, or Slurm GPU resource allocation (`gres.conf`, cgroups). Escalate to admin if basic checks fail.
* **Node Failure:** As seen previously, check `slurmctld.log` via admin for hardware/daemon issues on the specific worker node.
