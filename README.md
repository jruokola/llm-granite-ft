# LLM Granite Fine-tuning with Slurm and Soperator on Nebius

This project fine-tunes an IBM Granite LLM (e.g., `ibm-granite/granite-3.3-8b-instruct`) using LoRA (Low-Rank Adaptation) for a custom instruction-following task. The fine-tuning process is orchestrated using Slurm managed by Soperator on a Nebius Kubernetes cluster.

## Prerequisites

1. **Nebius Kubernetes Cluster with Slurm/Soperator:** A functional Kubernetes cluster managed by Nebius with Slurm and Soperator installed and configured. This is typically set up by the `infra-nebius-slurm-poc` Terraform configuration.
2. **Nebius Filestore:** A Nebius Filestore instance mounted and accessible by the Slurm worker nodes. This will be used for:
    * Storing training and evaluation datasets.
    * Storing model checkpoints and final LoRA adapters.
3. **Docker Registry:** A Docker container registry (e.g., Nebius Container Registry, Docker Hub) accessible from the Kubernetes cluster. This project's Docker image will be pushed here.
4. **MLflow Tracking Server:** An MLflow Tracking Server instance for logging metrics, parameters, and artifacts. The MLflow CA certificate (`mlflow-cert/ca.pem`) must be present.
5. **Python Environment & `uv`:** A local Python environment with `uv` installed for managing dependencies (`pip install uv`).
6. **Nebius CLI & `kubectl`:** Configured for access to your Nebius resources and Kubernetes cluster.
7. **Slurm Client Tools:** `sbatch`, `squeue`, `scancel`, `scontrol` configured to submit jobs to your Slurm/Soperator cluster.

## Project Structure

```
llm-granite-ft/
├── Dockerfile            # Defines the container image for training
├── pyproject.toml        # Python project metadata and dependencies (used by uv)
├── finetune.py           # Main Python script for fine-tuning the model
├── evaluate.py           # (Optional) Script for evaluating the fine-tuned model
├── submit_finetune.sbatch # Slurm batch submission script
├── ds_config_zero3.json  # DeepSpeed Zero3 config (if using DeepSpeed)
├── data/                   # Directory for training/evaluation data (if baked into image)
│   └── train_extended.csv
│   └── test_extended.csv
├── mlflow-cert/          # Directory containing MLflow CA certificate
│   └── ca.pem
└── README.md             # This file
```

## Setup and Deployment Workflow

### 1. Configure Environment Variables

Ensure the following environment variables are set in your shell or sourced from a script before interacting with MLflow or submitting Slurm jobs. The `submit_finetune.sbatch` script also sets some of these for the job environment.

* `MLFLOW_TRACKING_URI`: URI of your MLflow tracking server.
* `MLFLOW_TRACKING_SERVER_CERT_PATH`: Path to the MLflow CA certificate (used by the Python script if the path in the container differs from the one set in Dockerfile, but generally ENV in Dockerfile is sufficient).
* `MLFLOW_TRACKING_USERNAME` (if MLflow is authenticated)
* `MLFLOW_TRACKING_PASSWORD` (if MLflow is authenticated)
* `MLFLOW_EXPERIMENT_NAME` (optional, defaults can be used)

### 2. Prepare Data

* **Option A (Data in Image):** Place your `train_extended.csv` and `test_extended.csv` files into the `data/` directory. They will be copied into the Docker image.
* **Option B (Data on Shared Filestore - Recommended for large datasets):**
    1. Upload your `train_extended.csv` and `test_extended.csv` to a specific location on your Nebius Filestore.
    2. Modify `finetune.py` to accept data paths as command-line arguments.
    3. Update the `submit_finetune.sbatch` script to:
        * Ensure the Nebius Filestore is mounted into the container (via Soperator's `--container-mounts` or equivalent).
        * Pass the correct paths to your data on the mounted filestore to `finetune.py`.

### 3. Build and Push Docker Image

1. Navigate to the `llm-granite-ft` directory.
2. Build the Docker image:

    ```bash
    docker build -t <your_registry>/llm-granite-ft:latest .
    ```

3. Push the image to your registry:

    ```bash
    docker push <your_registry>/llm-granite-ft:latest
    ```

    Replace `<your_registry>` with your actual container registry path (e.g., `cr.eu-north1.nebius.cloud/your-registry-id`).

### 4. Configure Slurm Submission Script (`submit_finetune.sbatch`)

Open `submit_finetune.sbatch` and critically review/update the following sections:

* **Resource Requests:** Adjust `#SBATCH` directives for `--nodes`, `--ntasks-per-node`, `--gres=gpu:...`, `--cpus-per-task`, `--mem`, and `--time` to match your desired scale and available resources.
* **Soperator Directives (Very Important!):**
  * `#SBATCH --container-image=...`: Ensure this points to the image you pushed.
  * `#SBATCH --container-workdir=...`: Should likely be `/workspace` as per the Dockerfile.
  * `#SBATCH --container-mounts=<HOST_PATH_TO_NEBIUS_FILESTORE_MOUNT>:/mnt/shared_storage`: **This is crucial.** Replace `<HOST_PATH_TO_NEBIUS_FILESTORE_MOUNT>` with the actual path on the Slurm worker nodes where your Nebius Filestore (used for `filestore_jail` in Terraform) is mounted. `/mnt/shared_storage` will be its path inside the job container. This mount is used for checkpoints and potentially for data.
* **MLflow Environment Variables:** Verify `MLFLOW_TRACKING_URI` and other MLflow variables if they are hardcoded in the script.
* **Checkpoint Directory:** The script creates `JOB_CHECKPOINT_DIR` based on the container mount. Ensure this logic aligns with your filestore structure.
* **Data Paths (if using Option B for data):** Update `FINETUNE_ARGS` to pass the correct data paths to `finetune.py`.
* **Launch Command:** The script currently uses `srun torchrun ...` for distributed training. This assumes `finetune.py` has been adapted for `torch.distributed` as discussed. If using DeepSpeed or another method, adjust the launch command accordingly.

### 5. Submit the Fine-tuning Job to Slurm

Once the `sbatch` script is configured and your data is in place:

```bash
sbatch submit_finetune.sbatch
```

### 6. Monitor the Job

* Check job status: `squeue -u $USER`
* View job output log: `cat granite_ft_<job_id>.log` (or path specified in `--output`)
* Inspect Slurm control daemon logs or Soperator logs if issues arise with job scheduling or pod creation.
* Check MLflow UI for experiment tracking.
* If Kubernetes pods are created by Soperator for the job, you can use `kubectl logs <pod_name> -n <namespace_soperator_uses>` for detailed container logs.

## Fine-tuning Script (`finetune.py`)

The script `finetune.py` is responsible for:

1. Loading the base IBM Granite model with 4-bit quantization (QLoRA).
2. Attaching LoRA adapters.
3. Loading the training and evaluation datasets.
4. Formatting the dataset into prompt-completion pairs.
5. Using Hugging Face `SFTTrainer` for fine-tuning.
6. Logging metrics and parameters to MLflow via `mlflow.autolog()`.
7. Saving the trained LoRA adapters to the specified output directory.

**For Distributed Training (Multi-Node/Multi-GPU):**
The script has been adapted to support PyTorch DistributedDataParallel (DDP):

* Initializes `torch.distributed.init_process_group(backend='nccl', init_method='env://')`.
* Determines `local_rank` from `os.environ["LOCAL_RANK"]` (set by `torchrun`).
* Moves the model to the correct GPU and wraps it with `torch.nn.parallel.DistributedDataParallel`.
* MLflow logging and model saving are performed only on `rank 0`.

## Evaluation (`evaluate.py`)

(Details to be added on how to use `evaluate.py` with the saved LoRA adapters, likely loading them onto the base model and running on a test set. This might also be run as a Slurm job.)

## Dockerization (`Dockerfile`)

The `Dockerfile`:

* Starts from an NVIDIA PyTorch base image.
* Installs `uv` for Python package management.
* Copies the MLflow CA certificate into the image and sets `MLFLOW_TRACKING_SERVER_CERT_PATH`.
* Uses `uv sync` to install dependencies from `pyproject.toml`.
* Sets the `WORKDIR` to `/workspace` and copies the project code.
* Sets `CMD ["python", "finetune.py"]` (though this `CMD` is typically overridden by the `srun` command in the `sbatch` script).

## Contributing

(Add guidelines if this is a shared project.)

## Troubleshooting

* Ensure Slurm, Soperator, and Kubernetes are healthy.
* Verify that the Nebius Filestore is correctly mounted on all Slurm worker nodes at the expected `<HOST_PATH_TO_NEBIUS_FILESTORE_MOUNT>`.
* Check Soperator logs for issues translating Slurm jobs to Kubernetes pods.
* Check container logs via `kubectl logs` if pods are created but failing.
* Ensure the Docker image is accessible and contains all necessary dependencies and scripts.
* Verify MLflow server is reachable and credentials/certificates are correct.
