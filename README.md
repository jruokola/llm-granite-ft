# LLM Granite Fine-tuning with FSDP-QLoRA (Slurm & Soperator on Nebius)

This project fine-tunes an IBM Granite LLM (e.g., `ibm-granite/granite-3.3-8b-instruct`) using FSDP-QLoRA for custom instruction-following tasks. The fine-tuning process is orchestrated using Slurm (managed by Soperator on a Nebius Kubernetes cluster) and Hugging Face Accelerate.

## Prerequisites

1. **Nebius Kubernetes Cluster with Slurm/Soperator:** A functional Kubernetes cluster by Nebius with Slurm and Soperator installed and configured. This is typically set up by the `infra-nebius-slurm-poc` Terraform configuration.
2. **Nebius Filestore:** A Nebius Filestore instance mounted and accessible by the Slurm worker nodes. This will be used for:
    * (Optional) Storing training and evaluation datasets if not baked into the image.
    * Storing model checkpoints and final LoRA adapters.
3. **Docker Registry:** A Docker container registry (e.g., Nebius Container Registry, Docker Hub, GHCR) accessible from the Kubernetes cluster. This project's Docker image will be pushed here.
4. **MLflow Tracking Server:** An MLflow Tracking Server instance for logging. The MLflow CA certificate (`mlflow-cert/ca.pem`) is required if using TLS.
5. **Local Environment:** Python and Docker installed for building the training image.
6. **Nebius CLI & `kubectl`:** Configured for access to your Nebius resources and Kubernetes cluster (for debugging, `kubectl` is not strictly needed for job submission if Slurm is configured).
7. **Slurm Client Tools:** `sbatch`, `squeue`, etc., configured on a Slurm login node to submit jobs to your Soperator-managed cluster.

## Project Structure

```
llm-granite-ft/
├── Dockerfile                # Defines the container image for training
├── pyproject.toml            # Python project metadata (can be used by uv or for reference)
├── requirements.txt          # Python dependencies for pip
├── finetune.py               # Main Python script for FSDP-QLoRA fine-tuning
├── evaluate.py               # (Optional) Script for evaluating the fine-tuned model
├── submit_finetune.sbatch    # Slurm batch submission script
├── fsdp_config.yaml          # Hugging Face Accelerate config for FSDP
├── data/                       # Directory for training/evaluation data (if baked into image)
│   └── train_extended.csv
│   └── test_extended.csv
├── mlflow-cert/              # Directory containing MLflow CA certificate (if needed)
│   └── ca.pem
└── README.md                 # This file
```

## Setup and Deployment Workflow

### 1. Configure Environment Variables (for sbatch script)

The `submit_finetune.sbatch` script sets necessary environment variables like `MLFLOW_TRACKING_URI` and `MLFLOW_TRACKING_SERVER_CERT_PATH` for the job environment. Ensure these are correct within the script if not using external environment configuration for the job.

### 2. Prepare Data

* **Option A (Data in Image - Default):** Place your `train_extended.csv` and `test_extended.csv` files into the `llm-granite-ft/data/` directory. The `Dockerfile` copies these into `/workspace/data/` inside the image, and `finetune.py` uses these paths by default.
* **Option B (Data on Shared Filestore - Recommended for large datasets):
    1. Upload your data to a specific location on your Nebius Filestore (e.g., `/path_on_filestore/my_data/`).
    2. In `submit_finetune.sbatch`:
        * Set `HOST_SHARED_FS_ROOT_PATH` to the mount point of your filestore on the worker nodes (e.g., `/` or `/mnt/shared_fs`).
        * Uncomment and define `HOST_ACTUAL_DATA_PATH` to point to your data on the filestore (e.g., `${HOST_SHARED_FS_ROOT_PATH}/path_on_filestore/my_data`).
        * Uncomment and define `CONTAINER_MOUNTED_DATA_DIR` (e.g., `/mounted_data_in_container`).
        * Add the data mount to `CONTAINER_MOUNTS_ARG`: `CONTAINER_MOUNTS_ARG+=",${HOST_ACTUAL_DATA_PATH}:${CONTAINER_MOUNTED_DATA_DIR}"`.
        * Update `DATA_TRAIN_ARG` and `DATA_EVAL_ARG` to use `CONTAINER_MOUNTED_DATA_DIR`.

### 3. Build and Push Docker Image

1. Navigate to the `llm-granite-ft` directory on your local machine.
2. Build the Docker image:

    ```bash
    docker build -t <your_registry>/llm-granite-ft:<tag> .
    ```

    (e.g., `docker build -t cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-ft:latest .`)
3. Push the image to your registry:

    ```bash
    docker push <your_registry>/llm-granite-ft:<tag>
    ```

### 4. Configure Slurm Submission Script (`submit_finetune.sbatch`)

This script must be placed on the Slurm login node or accessible shared storage.

**Critically review and update these sections ON THE LOGIN NODE before submitting:**

* **Resource Requests (`#SBATCH` directives):** Adjust `--nodes`, `--ntasks-per-node`, `--gres=gpu:...`, `--cpus-per-task`, `--mem`, `--time` to match your desired scale and cluster resources.
* **`srun` options for Pyxis/Enroot:**
  * `--container-image`: Ensure this points to the exact image URI you pushed (e.g., `docker://cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-ft:latest`).
  * `--container-workdir`: Should be `/workspace` (aligns with Dockerfile).
  * `--container-mounts`:
    * The script constructs `CONTAINER_MOUNTS_ARG`. You **MUST** set the `HOST_SHARED_FS_ROOT_PATH` variable within the script to the correct absolute path where your Nebius Filestore is mounted on the Slurm worker nodes (e.g., `/` or `/mnt/filestore_on_host`).
    * `HOST_JOB_OUTPUTS_BASE_PATH` (e.g., `/root/slurm_jobs_output` on the filestore) will be mapped to `CONTAINER_CHECKPOINT_DIR_BASE` (e.g., `/job_outputs`) inside the container.
    * Ensure `HOST_JOB_OUTPUTS_BASE_PATH` exists on the shared filestore and is writable by the job user (`mkdir -p /root/slurm_jobs_output` on the login node if path is `/root/...` on filestore).
* **MLflow Environment Variables:** Verify `MLFLOW_TRACKING_URI` and credentials if needed.
* **`finetune.py` Arguments (`FINETUNE_CLI_ARGS`):** Adjust epochs, batch sizes, learning rate, etc., as needed.

### 5. Configure Enroot Authentication (on Login/Worker Nodes)

For pulling from a private Nebius Container Registry, Enroot (used by Pyxis) needs credentials. On the **login node** (and this configuration needs to be effective on worker nodes, possibly via shared home or Soperator-managed config):

1. Create/edit `/root/.config/enroot/.credentials` (if jobs run as root).
2. Add a line like: `machine cr.eu-north1.nebius.cloud login <KEY_ID> password <SECRET_KEY>`
    * Replace `<KEY_ID>` and `<SECRET_KEY>` with your Nebius Service Account **Static Access Key ID and Secret** that has permission to pull from the registry.
3. Set permissions: `chmod 600 /root/.config/enroot/.credentials`.

### 6. Submit the Fine-tuning Job to Slurm

From the Slurm login node:

```bash
sbatch /path/to/your/submit_finetune.sbatch
```

### 7. Monitor the Job

* **Slurm:** `squeue -u $USER`, `scontrol show job <jobid>`, check output log specified in `sbatch` (e.g., `/root/granite_fsdp_qlora_<jobid>.log`).
* **MLflow UI:** For parameters, metrics, and artifacts.
* **Kubernetes (for Soperator debugging):** `kubectl get pods -A -l slurm.nebius.ai/job-id=<jobid>`, `kubectl logs <pod_name> -n <namespace>`. Pods are typically in the `soperator` or `default` namespace.

## Fine-tuning Script (`finetune.py`)

Now adapted for FSDP-QLoRA using Hugging Face Accelerate and Trainer:

* Uses `BitsAndBytesConfig` with `bnb_4bit_quant_storage` for FSDP compatibility.
* Model loading uses `torch_dtype` matching the quant storage.
* Manual DDP setup is removed.
* `TrainingArguments` includes `fsdp` parameters (e.g., `fsdp="full_shard auto_wrap"`).
* Launched via `accelerate launch` (called by `srun` in the sbatch script).
* Handles various command-line arguments for configurability.

## Accelerate Configuration (`fsdp_config.yaml`)

A basic configuration file for Hugging Face Accelerate to enable FSDP with common strategies like `TRANSFORMER_BASED_WRAP` and `FULL_SHARD`.

## Dockerization (`Dockerfile`)

* Uses `nvcr.io/nvidia/pytorch:24.07-py3` base image.
* Installs dependencies from `requirements.txt` using `pip`.
* Sets `WORKDIR /workspace` and `ENV PYTHONPATH="/app/src:${PYTHONPATH}"` (Corrected: should be `/workspace/src` if `src` is inside `llm-granite-ft` copied to `/workspace`, or just `/workspace` if `gpu_probe` package is directly in `/workspace`).
  * **Note:** If your project structure is `llm-granite-ft/src/gpu_probe_package`, and `Dockerfile` is in `llm-granite-ft`, then `COPY . .` makes `src` appear at `/workspace/src`. So `PYTHONPATH` should be `/workspace` for `import src.gpu_probe_package` or if `finetune.py` is in `/workspace` and uses relative imports to `src`. If `finetune.py` is `/workspace/finetune.py` and tries `import data.utils`, then `PYTHONPATH` should include `/workspace`. For `-m module_name`, the parent of `module_name` dir must be on path. **Let's assume `finetune.py` is top-level in `/workspace` for now.** The `PYTHONPATH` in the current `Dockerfile` is `/app/src` from the `gpu-probe` edits; this will need to be `/workspace` if `finetune.py` and `evaluate.py` are directly in `/workspace`.

## Evaluation (`evaluate.py`)

(To be detailed: how to load the FSDP-QLoRA trained adapters and run evaluation.)

## Troubleshooting

* **Slurmctld Down (Login Node MOTD):** The `controller-0` pod in the `soperator` namespace (or equivalent) is not running or healthy. Check `kubectl describe pod controller-0 -n soperator` and its logs. Often due to insufficient resources on the controller node or PVC issues.
* **Job PENDING with `Reason=PartitionConfig`:** Nodes are not correctly configured in the Slurm partition or `slurmd` is not registering. Check `sinfo`, `slurmctld` logs, and `slurmd` logs on worker pods.
* **Image Pull Errors (e.g., 401 Unauthorized from `cr.eu-north1.nebius.cloud/v2/token/`):** Enroot authentication issue. Verify `/root/.config/enroot/.credentials` on the execution environment (login or worker nodes) uses the correct format and valid credentials (preferably SA Static Key) for Nebius Container Registry.
* **Pyxis Errors (`couldn't start container`, `child failed`):** Can be due to failed image pull, incorrect `--container-mounts` (host path doesn't exist or permissions), or issues with the container image itself (entrypoint/cmd error, missing dependencies).
* **`torchrun: command not found` or `ModuleNotFoundError: No module named 'torch'` in job log:** Indicates Python environment or `PATH` issues within the container when run by `srun`. Ensure your `Dockerfile` installs PyTorch correctly and that `accelerate launch` (or `python -m torch.distributed.run`) uses the correct Python environment. Explicitly setting `ENV PATH` in Dockerfile to include Python script directories (like `/usr/local/bin` or custom venv paths) can help.
