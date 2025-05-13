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

* **For Function Calling Fine-tuning:** The `function-finetune-fixed.py` script loads the `glaiveai/glaive-function-calling-v2` dataset directly using the Hugging Face `datasets` library. Ensure the compute nodes have internet access to download the dataset, or that it's pre-cached in a location accessible by the script (e.g., via `HF_DATASETS_CACHE` environment variable pointing to a shared drive location, or by building it into the container if small enough).

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
    docker build -f Dockerfile.function -t cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-function-ft-fix:<tag> .
    ```

    (Ensure `Dockerfile.function` is correctly set up to include `fixed-scripts/function-finetune-fixed.py` and its dependencies.)
3. Push the image:

    ```bash
    docker push cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-function-ft-fix:<tag>
    ```

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
* **Output Log:** Main Slurm log is directed to `/new_folder_name/slurm_logs/function_finetune_%j.log`. Node-specific logs are created within the job's shared directory.
* **Container Image:** Uses `cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-function-ft-fix:latest`.
* **Shared Job Directory:** Base directory for outputs, logs, and coordination files is `/new_folder_name/slurm_jobs/function_finetune/${SLURM_JOB_ID}`. This is mounted into the container at `/job_data`.
* **Python Script:** Executes `/workspace/fixed-scripts/function-finetune-fixed.py` (ensure this path is correct within your container).
* **Distributed Setup:**
  * The script manually sets up `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` environment variables.
  * `MASTER_ADDR` and `MASTER_PORT` are determined from the first node in `SLURM_JOB_NODELIST` and a fixed port (29510).
  * `WORLD_SIZE` is `SLURM_NNODES`.
  * `RANK` is determined by node hostname.
  * `LOCAL_RANK` is hardcoded to `0` (suitable for 1 task/GPU per node).
  * This setup does **not** use `torchrun`; the Python script is launched directly with `python ...`.
* **Python Script Arguments (`FINETUNE_CLI_ARGS`):**
  * `--output_dir` is set to `${CONTAINER_JOB_DIR}/checkpoints` (maps to the shared drive).
  * `--disable_amp` is passed by default, meaning Automatic Mixed Precision is turned off. You can remove this from the sbatch script if you wish to use the Python script's default AMP behavior (which is typically AMP enabled).
  * Other arguments for `function-finetune-fixed.py` can be added to `FINETUNE_CLI_ARGS` as needed.
* **No On-the-fly Patching:** Unlike the example chess script, this sbatch script directly runs the Python script without `sed` modifications, assuming `function-finetune-fixed.py` is already correctly configured.

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

* Designed for distributed training using PyTorch FSDP (default) or DDP (`--no_fsdp` flag).
* Fine-tunes models like `ibm-granite/granite-3.3-2b-instruct` on the `glaiveai/glaive-function-calling-v2` dataset (loaded via Hugging Face `datasets`).
* Parses a comprehensive set of command-line arguments for controlling training parameters, including AMP, FSDP/DDP, learning rate, batch sizes, etc.
* Implements a detailed `FunctionCallingDataset` class for processing the specific data format.
* Uses `torch.optim.AdamW`, `GradScaler` (conditional on AMP), and a linear warmup LR schedule.
* Launched directly via `python fixed-scripts/function-finetune-fixed.py ...` within the `srun` command in `submit_finetune_function_calling.sbatch`. The distributed environment (`MASTER_ADDR`, `RANK`, etc.) is manually configured by the sbatch script.

## Dockerization

### a. `Dockerfile.chess` (for Chess Fine-tuning)

* Uses a base image like `nvcr.io/nvidia/pytorch:24.07-py3`.
* Installs minimal extra dependencies (e.g., `transformers`, `datasets`).
* Copies the `chess-finetune.py` script and `strategic_game_chess.jsonl` dataset into `/workspace`.

### b. `Dockerfile.function` (for Function Calling Fine-tuning)

* This Dockerfile (e.g., named `Dockerfile.function` or similar, corresponding to the image `llm-granite-function-ft-fix`) should be set up to:
  * Use a suitable PyTorch base image compatible with NVIDIA H100s (e.g., a recent NVIDIA PyTorch container).
  * Install necessary dependencies specified in `pyproject.toml` (e.g., using `uv pip install ...`).
  * Copy the `fixed-scripts/function-finetune-fixed.py` script into the container (e.g., to `/workspace/fixed-scripts/function-finetune-fixed.py`).
  * Set up any other necessary environment configurations.

## Troubleshooting

* **Slurmctld Down / PartitionConfig Errors:** Check `sinfo`, `slurmctld` logs, and `slurmd` logs on worker pods via cluster admin or `kubectl`.
* **Image Pull Errors (401 Unauthorized):** Verify Enroot authentication credentials for your registry.
* **Pyxis Errors (`couldn't start container`):** Check image pull success, `--container-mounts` validity (host path exists and has permissions), and basic container functionality.
* **`torchrun` Errors / DDP Init Errors (for `chess-finetune.py` or similar `torchrun`-based scripts):** Ensure `srun` is passing necessary Slurm environment variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.) correctly into the container for `torchrun` auto-detection.
* **Direct Python Launch DDP Init Errors (for `function-finetune-fixed.py`):** If `torch.distributed.init_process_group` fails or hangs, double-check that `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` are correctly set by the `submit_finetune_function_calling.sbatch` script (e.g., by examining the node-specific logs in `/new_folder_name/slurm_jobs/function_finetune/${SLURM_JOB_ID}/logs/`).
* **`ModuleNotFoundError`:** Ensure the respective Dockerfile (`Dockerfile.chess` or `Dockerfile.function`) installs all required Python packages. Check `PYTHONPATH` if necessary, although direct script execution in `/workspace` (or `/workspace/fixed-scripts/`) should generally work if scripts and dependencies are correctly placed.
* **CUDA Errors / `nvidia-smi` fails inside container:** Likely an issue with Pyxis/Enroot setup, host drivers, or Slurm GPU resource allocation (`gres.conf`, cgroups). Escalate to admin if basic checks fail.
* **Node Failure:** As seen previously, check `slurmctld.log` via admin for hardware/daemon issues on the specific worker node.
