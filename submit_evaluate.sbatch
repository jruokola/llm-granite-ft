#!/bin/bash
#SBATCH --job-name=granite-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1 # Evaluation typically needs one GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G # Adjust based on model size and batch size for pipeline
#SBATCH --time=01:00:00
#SBATCH --output=/root/granite_evaluate_%j.log
#SBATCH --export=ALL

# --- Host Path for Shared Storage (MUST BE SET CORRECTLY) ---
HOST_SHARED_FS_ROOT_PATH="/" # Example: if filestore is mounted at / on worker nodes

# --- Paths to Resources on Shared Storage (relative to HOST_SHARED_FS_ROOT_PATH) ---
# This is where the training job (submit_finetune.sbatch) saved its checkpoints/adapters.
# You need to know the SLURM_JOB_ID of the training job to point to the correct adapters.
# Example: /root/slurm_jobs_output/checkpoints/granite_fsdp_qlora_slurm_<TRAIN_JOB_ID>
HOST_PATH_TO_TRAINED_ADAPTERS="/root/slurm_jobs_output/checkpoints/granite_fsdp_qlora_slurm_<REPLACE_WITH_TRAIN_JOB_ID>"

# Path on shared storage where evaluation results will be written by this job.
HOST_PATH_FOR_EVAL_RESULTS="/root/slurm_jobs_output/eval_results/granite_eval_${SLURM_JOB_ID}"

# Path to the test data CSV on shared storage (if not using data from image)
# HOST_PATH_TO_TEST_CSV="/root/data/test_extended.csv"

# --- Container Image (must be the same as used for training or compatible) ---
CONTAINER_IMAGE="cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-ft:latest"
CONTAINER_WORKDIR="/workspace"

# --- Define Paths for INSIDE the Container ---
# Where the host shared storage root will be mounted inside the container
CONTAINER_SHARED_STORAGE_MOUNT_POINT="/mnt/shared_storage"

# Paths inside the container that evaluate.py will use
CONTAINER_ADAPTER_PATH="${CONTAINER_SHARED_STORAGE_MOUNT_POINT}/root/slurm_jobs_output/checkpoints/granite_fsdp_qlora_slurm_<REPLACE_WITH_TRAIN_JOB_ID>"
CONTAINER_RESULTS_OUTPUT_DIR="${CONTAINER_SHARED_STORAGE_MOUNT_POINT}/root/slurm_jobs_output/eval_results/granite_eval_${SLURM_JOB_ID}"
CONTAINER_TEST_CSV_PATH="${CONTAINER_WORKDIR}/data/test_extended.csv" # Default: uses data from image
# If using data from shared storage, uncomment and adjust:
# CONTAINER_TEST_CSV_PATH="${CONTAINER_SHARED_STORAGE_MOUNT_POINT}/root/data/test_extended.csv"

# --- MLflow Setup ---
export MLFLOW_TRACKING_URI="public-tracking.mlflow-e00spsxj9fckt2k6pz.backbone-e00qwnewt2vzn1dh4s.msp.eu-north1.nebius.cloud"
export MLFLOW_TRACKING_SERVER_CERT_PATH="/etc/mlflow/certs/ca.pem"
export MLFLOW_EXPERIMENT_NAME="LLM_Granite_Evaluation"
export MLFLOW_TRACKING_USERNAME="jruokola"
export MLFLOW_TRACKING_PASSWORD="K3m1k44l1!!666"

echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID (evaluate) on node: $SLURMD_NODENAME"

# Create the results directory on the host shared storage (if it doesn't exist)
# This sbatch script runs on the login node, so this mkdir uses the HOST_PATH
# The container will then write into this via its mounted path.
# Best if the python script itself handles creating its output dir if it can.
mkdir -p "${HOST_PATH_FOR_EVAL_RESULTS}"

# Arguments for evaluate.py
EVAL_ARGS=""
EVAL_ARGS+=" --base_model_id='ibm-granite/granite-3.3-8b-instruct'"
EVAL_ARGS+=" --adapter_load_path='$CONTAINER_ADAPTER_PATH'"
EVAL_ARGS+=" --test_csv_path='$CONTAINER_TEST_CSV_PATH'"
EVAL_ARGS+=" --output_results_file='$CONTAINER_RESULTS_OUTPUT_DIR/evaluation_metrics_and_predictions.json'"
# EVAL_ARGS+=" --device='cuda:0'" # Optional, script will default to cuda:0 if available
# EVAL_ARGS+=" --no_mlflow" # Uncomment if you don't want this eval job to log to MLflow

echo "Evaluate.py arguments: $EVAL_ARGS"

# Construct --container-mounts argument
# Mount the entire shared FS root to the container's shared storage mount point.
# This makes adapter paths and output paths simpler to manage.
CONTAINER_MOUNTS_ARG="${HOST_SHARED_FS_ROOT_PATH}:${CONTAINER_SHARED_STORAGE_MOUNT_POINT}"
echo "Using --container-mounts=$CONTAINER_MOUNTS_ARG"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="${CONTAINER_WORKDIR}" \
    --container-mounts="${CONTAINER_MOUNTS_ARG}" \
    python evaluate.py ${EVAL_ARGS}

JOB_EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID (evaluate) finished with exit code $JOB_EXIT_CODE."
exit $JOB_EXIT_CODE 