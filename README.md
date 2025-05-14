# LLM Fine-tuning for Chess and Function Calling (Distributed Training & Slurm on Nebius)

This project supports fine-tuning Large Language Models for two distinct tasks:

1. **CodeLlama for Chess**: Fine-tunes a CodeLlama LLM (e.g., `codellama/CodeLlama-7b-hf`) on a chess dataset (`strategic_game_chess.jsonl`) using PyTorch Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).

2. **Granite-3.3B-Instruct for Function Calling**: Fine-tunes an IBM Granite model (e.g., `ibm-granite/granite-3.3-2b-instruct`) on the `glaiveai/glaive-function-calling-v2` dataset, also using PyTorch DDP/FSDP.

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

* **For Function Calling Fine-tuning:** The `fixed-scripts/function-finetune-fixed.py` script now **requires** a preprocessed dataset. You must first process your raw data (e.g., from `glaiveai/glaive-function-calling-v2`) into the format expected by the script and provide the path to this processed dataset via the `--processed_dataset_path` argument in the sbatch script. The script uses `datasets.load_from_disk` to load this.

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

* **Job Name:** Set to `function-finetune`.
* **Output Log:** Main Slurm log is directed to `/root/slurm_logs/function_finetune_%j.log`. Node-specific logs are created within the job's shared directory.
* **Container Image:** The sbatch script uses `CONTAINER_IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-transformers:latest"`. Ensure this matches the image you built and pushed.
* **Shared Job Directory:** Base directory for outputs, logs, and coordination files is `/root/slurm_jobs/function_finetune/${SLURM_JOB_ID}`. This is mounted into the container at `/job_data`.
* **Python Script:** Executes `/workspace/function-finetune-fixed.py`.
* **NCCL Environment Variables:** The sbatch script now sets `NCCL_COLLNET_ENABLE=0`, `NCCL_IB_HCA`, and `NCCL_NET_GDR_LEVEL`.
* **Distributed Setup (Manual):**
  * The script manually sets up `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` environment variables for the Python script. This setup does **not** use `torchrun`.
* **Python Script Arguments (`FINETUNE_CLI_ARGS`):**
  * `--output_dir` is set to `${CONTAINER_JOB_DIR}/checkpoints`.
  * `--processed_dataset_path` is set to `/processed_datasets/glaive_fc_v2`. This path is inside the container and assumes your `Dockerfile.function` copies your preprocessed data to this location. **This is a critical change: the script now requires a preprocessed dataset.**
  * `--use_qlora` is passed by default, enabling QLoRA.
  * The `--disable_amp` argument is **not** passed by default in the current sbatch script (the line is commented out). This means the Python script's default AMP behavior (BF16 with GradScaler) will be active.
  * Other arguments for `function-finetune-fixed.py` (e.g., `--batch_size_per_device`, `--learning_rate`, LoRA parameters) can be added to `FINETUNE_CLI_ARGS`.

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
    * Employs `transformer_auto_wrap_policy` for layer sharding.
    * Includes logic to set `ignored_modules` for FSDP, particularly for handling potential `int8` parameters from quantization.
    * Uses `sync_module_states=True` and `StateDictType.FULL_STATE_DICT` for robust checkpointing.
  * **QLoRA:** Enabled via the `--use_qlora` flag.
    * Configures `BitsAndBytesConfig` for 4-bit quantization (`nf4`, `bnb_4bit_quant_storage=torch.float16`).
    * Uses `peft.prepare_model_for_kbit_training` and `peft.get_peft_model` with `LoraConfig`.
  * **FP8 Support (Optional):** Includes experimental support for NVIDIA Transformer Engine FP8 for LoRA adapters (`--use_fp8`), if `transformer_engine` is available.
  * **Data Handling:**
    * **Requires preprocessed data:** The script loads data using `datasets.load_from_disk` via the mandatory `--processed_dataset_path` argument. On-the-fly processing of raw datasets is no longer supported in this version.
    * Uses a custom `PreprocessedDataset` class.
  * **Mixed Precision:** AMP is enabled by default using `torch.bfloat16` and `GradScaler`. Can be disabled with `--disable_amp` (falls back to `torch.float32` for `bnb_4bit_compute_dtype` if QLoRA is also off, otherwise `amp_dtype` becomes `float32`).
  * **Optimizer & Scheduler:** Uses `torch.optim.AdamW` and a linear warmup LR schedule.
  * **Gradient Checkpointing:** Supported via `--gradient_checkpointing`.
  * **Logging & Checkpointing:** Standard logging and checkpoint saving logic, compatible with FSDP.
  * **Distributed Launch:** Designed to be launched directly by `python` with environment variables for distributed setup (as done by the sbatch script), not `torchrun`.

## Dockerization

### a. `Dockerfile.chess` (for Chess Fine-tuning)

* Uses a base image like `nvcr.io/nvidia/pytorch:24.07-py3`.
* Installs minimal extra dependencies (e.g., `transformers`, `datasets`).
* Copies the `chess-finetune.py` script and `strategic_game_chess.jsonl` dataset into `/workspace`.

### b. `Dockerfile.function` (for Function Calling Fine-tuning)

* This Dockerfile (e.g., named `Dockerfile.function` or similar, corresponding to the image `llm-granite-function-ft-fix`) should be set up to:
  * Use a suitable PyTorch base image (e.g., `nvcr.io/nvidia/pytorch:24.07-py3` or newer, as used in recent examples).
  * Install dependencies like `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, and optionally `transformer_engine` (for FP8).
  * Copy the `fixed-scripts/function-finetune-fixed.py` script and any other necessary local files (like preprocessed datasets if built into the image, though the sbatch script assumes `/processed_datasets/glaive_fc_v2` is already in the image from a `COPY` command).

## Troubleshooting

* **Slurmctld Down / PartitionConfig Errors:** Check `sinfo`, `slurmctld` logs, and `slurmd` logs on worker pods via cluster admin or `kubectl`.
* **Image Pull Errors (401 Unauthorized):** Verify Enroot authentication credentials for your registry.
* **Pyxis Errors (`couldn't start container`):** Check image pull success, `--container-mounts` validity (host path exists and has permissions), and basic container functionality.
* **`torchrun` Errors / DDP Init Errors (for `chess-finetune.py` or similar `torchrun`-based scripts):** Ensure `srun` is passing necessary Slurm environment variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.) correctly into the container for `torchrun` auto-detection.
* **Direct Python Launch DDP/FSDP Init Errors (for `function-finetune-fixed.py`):** If `torch.distributed.init_process_group` fails or hangs, double-check that `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` are correctly set by the `submit_finetune_function_calling.sbatch` script (e.g., by examining the node-specific logs in `/root/slurm_jobs/function_finetune/${SLURM_JOB_ID}/logs/`). Also check the new NCCL environment variables in the sbatch script.
* **`ModuleNotFoundError`:** Ensure the respective Dockerfile (`Dockerfile.chess` or `Dockerfile.function`) installs all required Python packages. Check `PYTHONPATH` if necessary, although direct script execution in `/workspace` (or `/workspace/fixed-scripts/`) should generally work if scripts and dependencies are correctly placed.
* **CUDA Errors / `nvidia-smi` fails inside container:** Likely an issue with Pyxis/Enroot setup, host drivers, or Slurm GPU resource allocation (`gres.conf`, cgroups). Escalate to admin if basic checks fail.
* **Node Failure:** As seen previously, check `slurmctld.log` via admin for hardware/daemon issues on the specific worker node.
