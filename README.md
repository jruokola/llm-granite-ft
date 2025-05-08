# LLM Fine-tuning for Chess (DDP & Slurm on Nebius)

This project fine-tunes a CodeLlama LLM (e.g., `codellama/CodeLlama-7b-hf`) on a chess dataset (`strategic_game_chess.jsonl`) using PyTorch Distributed Data Parallel (DDP). The fine-tuning process is orchestrated using Slurm (managed by Soperator on a Nebius Kubernetes cluster).

## Prerequisites

1. **Nebius Kubernetes Cluster with Slurm/Soperator:** Functional cluster with Slurm/Soperator installed.
2. **Nebius Filestore (or equivalent shared storage):** Mounted and accessible by Slurm worker nodes for storing model checkpoints and outputs.
3. **Docker Registry:** Accessible registry (e.g., Nebius Container Registry) for the training image.
4. **Local Environment:** Python and Docker installed for building the training image.
5. **Nebius CLI & `kubectl`:** Configured access (optional for debugging).
6. **Slurm Client Tools:** `sbatch`, `squeue`, etc., on a login node.

## Project Structure

```
llm-granite-ft/ # (Directory name might be legacy)
├── Dockerfile.chess          # Defines the container image for chess training
├── chess-finetune.py         # Main Python script for DDP fine-tuning
├── strategic_game_chess.jsonl # Chess dataset file
├── submit_finetune_chess.sbatch # Slurm batch submission script for chess
├── requirements.txt          # (Optional) Might contain additional deps beyond Dockerfile
└── README.md                 # This file
```

## Setup and Deployment Workflow

### 1. Prepare Data

The `strategic_game_chess.jsonl` file should be present in the `llm-granite-ft` directory when building the Docker image, as `Dockerfile.chess` copies it into the image.

### 2. Build and Push Docker Image

1. Navigate to the `llm-granite-ft` directory.
2. Build the Docker image using `Dockerfile.chess`:

    ```bash
    # Replace <tag> with your desired tag, e.g., latest
    # Use a distinct image name like llm-chess-ft
    docker build -f Dockerfile.chess -t <your_registry>/llm-chess-ft:<tag> .
    ```

    (e.g., `docker build -f Dockerfile.chess -t cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-chess-ft:latest .`)
3. Push the image:

    ```bash
    docker push <your_registry>/llm-chess-ft:<tag>
    ```

### 3. Configure Slurm Submission Script (`submit_finetune_chess.sbatch`)

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

### 4. Configure Enroot Authentication (If using private registry)

If pulling `llm-chess-ft:<tag>` requires authentication:

1. Create/edit `/root/.config/enroot/.credentials` (or user equivalent) on the environment where `srun` executes (login node, potentially propagated to workers).
2. Add line: `machine <your_registry_host> login <KEY_ID> password <SECRET_KEY>`
    * Use appropriate credentials (e.g., Nebius SA Static Access Key).
3. Set permissions: `chmod 600 .../.credentials`.

### 5. Submit the Fine-tuning Job to Slurm

From the Slurm login node:

```bash
# Make sure the path is correct
sbatch /path/to/your/llm-granite-ft/submit_finetune_chess.sbatch
```

### 6. Monitor the Job

* **Slurm:** `squeue -u $USER`, `scontrol show job <jobid>`, check output log (e.g., `/root/chess_finetune_<jobid>.log` as defined in the sbatch script).
* **Kubernetes (for Soperator debugging):** `kubectl get pods -A -l slurm.nebius.ai/job-id=<jobid>`, `kubectl logs <pod_name> -n <namespace>`.

## Fine-tuning Script (`chess-finetune.py`)

* Uses standard PyTorch DDP (`torch.distributed.init_process_group`, `torch.nn.parallel.DistributedDataParallel`).
* Parses command-line arguments for hyperparameters (batch size, LR, etc.).
* Includes a basic `JsonlDataset` class for the chess data.
* Implements a standard training loop using `torch.optim.AdamW`, `GradScaler` for AMP (bf16), and a simple linear warmup LR schedule.
* Launched via `torchrun` called by `srun` in the sbatch script.
* Still contains optional FSDP logic via `--no_fsdp` flag, but the default launch script does not enable FSDP.

## Dockerization (`Dockerfile.chess`)

* Uses `nvcr.io/nvidia/pytorch:24.07-py3` base image.
* Installs minimal extra dependencies (`transformers`, `datasets`).
* Copies the script and dataset into `/workspace`.

## Troubleshooting

* **Slurmctld Down / PartitionConfig Errors:** Check `sinfo`, `slurmctld` logs, and `slurmd` logs on worker pods via cluster admin or `kubectl`.
* **Image Pull Errors (401 Unauthorized):** Verify Enroot authentication credentials for your registry.
* **Pyxis Errors (`couldn't start container`):** Check image pull success, `--container-mounts` validity (host path exists and has permissions), and basic container functionality.
* **`torchrun` Errors / DDP Init Errors (`RANK not set`, etc.):** Ensure `srun` is passing the necessary Slurm environment variables (`SLURM_PROCID`, `SLURM_NTASKS`, etc.) correctly into the container environment where `torchrun` executes. The current script relies on `torchrun` auto-detection, which usually works with `srun`.
* **`ModuleNotFoundError`:** Ensure `Dockerfile.chess` installs all required Python packages and check `PYTHONPATH` if necessary, although direct script execution in `/workspace` should usually work.
* **CUDA Errors / `nvidia-smi` fails inside container:** Likely an issue with Pyxis/Enroot setup, host drivers, or Slurm GPU resource allocation (`gres.conf`, cgroups). Escalate to admin if basic checks fail.
* **Node Failure:** As seen previously, check `slurmctld.log` via admin for hardware/daemon issues on the specific worker node.
