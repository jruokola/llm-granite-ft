#!/bin/bash
export PYXIS_VERBOSE=3
#SBATCH --job-name=granite-fsdp-qlora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # Number of processes Accelerate will launch per node (usually 1, then Accelerate handles GPUs)
#SBATCH --gres=gpu:1        # GPUs per task. If ntasks-per-node=1, this is GPUs per node.
#SBATCH --cpus-per-task=16 
#SBATCH --mem=180G 
#SBATCH --time=04:00:00
#SBATCH --output=/root/granite_fsdp_qlora_%j.log
#SBATCH --export=ALL

echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES, Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "SLURM_PROCID=$SLURM_PROCID, SLURM_NPROCS=$SLURM_NPROCS, SLURMD_NODENAME=$SLURMD_NODENAME"

# Debug: Print relevant Slurm environment variables
echo "SLURM_PROCID=$SLURM_PROCID, SLURM_NPROCS=$SLURM_NPROCS"
echo "SLURM_LOCALID=$SLURM_LOCALID, SLURM_NODEID=$SLURM_NODEID"
echo "SLURMD_NODENAME=$SLURMD_NODENAME"

# MLflow Configuration (exported to srun via --export=ALL)
export MLFLOW_TRACKING_URI="public-tracking.mlflow-e00spsxj9fckt2k6pz.backbone-e00qwnewt2vzn1dh4s.msp.eu-north1.nebius.cloud"
export MLFLOW_TRACKING_SERVER_CERT_PATH="/etc/mlflow/certs/ca.pem" # Path inside container from Dockerfile
# export MLFLOW_TRACKING_USERNAME="your_mlflow_username" # Uncomment and set if MLflow is authenticated
# export MLFLOW_TRACKING_PASSWORD="your_mlflow_password" # Uncomment and set if MLflow is authenticated

# --- Define Paths for INSIDE the Container ---
# Checkpoints will be written to this path inside the container.
# This path will be mapped from a shared host path via --container-mounts.
CONTAINER_CHECKPOINT_DIR_BASE="/job_outputs" # Base mount point in container for all job outputs
JOB_CHECKPOINT_DIR="${CONTAINER_CHECKPOINT_DIR_BASE}/checkpoints/granite_fsdp_qlora_slurm_${SLURM_JOB_ID}"

# Data directory inside the container.
# Assumes data is IN THE IMAGE at /workspace/data/ as per current finetune.py and Dockerfile.
# If data is mounted from shared storage, CONTAINER_DATA_DIR would be the container-side mount point.
CONTAINER_DATA_DIR_IN_IMAGE="/workspace/data" 

# --- Host Path Definitions (CRITICAL - REPLACE PLACEHOLDERS) ---
# This is the path on the HOST (Slurm worker node) where your Nebius Filestore is mounted.
# Example: /mnt/nebius_shared_fs  OR /shared_fs OR simply /
HOST_SHARED_FS_ROOT_PATH="/"

# Define where job outputs (checkpoints, logs from sbatch --output) on shared FS will go.
HOST_JOB_OUTPUTS_BASE_PATH="${HOST_SHARED_FS_ROOT_PATH}/root/slurm_jobs_output" # Example: /root/slurm_jobs_output on the filestore

# Ensure the parent directory for job outputs exists on the shared filesystem (run this MANUALLY once on login node or via other means)
# mkdir -p "${HOST_JOB_OUTPUTS_BASE_PATH}/checkpoints"

echo "Host path for job outputs base: $HOST_JOB_OUTPUTS_BASE_PATH"
echo "Container path for job outputs base: $CONTAINER_CHECKPOINT_DIR_BASE"
echo "Container path for this job's checkpoints: $JOB_CHECKPOINT_DIR"

# Create the specific job checkpoint directory using the CONTAINER path.
# This will be created inside the mounted volume by the script running in the container.
# The mkdir command in the sbatch script itself might not work as expected for container paths
# before srun initiates the container environment and mounts.
# It's safer if finetune.py (rank 0) creates args.output_dir.
# However, if the parent mount exists, torchrun/python might be able to create subdirs.

# Data arguments (assuming data in image at /workspace/data for now)
DATA_TRAIN_ARG="--data_path_train=/workspace/data/train_extended.csv"
DATA_EVAL_ARG="--data_path_eval=/workspace/data/test_extended.csv"

# Arguments for finetune.py
# Paths for output_dir, data_path_train, data_path_eval are inside the container.
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir='$JOB_CHECKPOINT_DIR'"
FINETUNE_CLI_ARGS+=" $DATA_TRAIN_ARG"
FINETUNE_CLI_ARGS+=" $DATA_EVAL_ARG"
FINETUNE_CLI_ARGS+=" --epochs=3" # Example
FINETUNE_CLI_ARGS+=" --per_device_train_batch_size=2" # Adjust based on GPU memory
FINETUNE_CLI_ARGS+=" --gradient_accumulation_steps=4" # Adjust for effective batch size
# Add other finetune.py arguments as needed

echo "------------------------------------------------------------------------------"
echo "DEBUG: Current directory: $(pwd)"
echo "DEBUG: SLURM Environment Variables:"
printenv | grep SLURM
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (should be set by Slurm/Pyxis)"
echo "------------------------------------------------------------------------------"
echo "Launching FSDP training with Accelerate on $SLURM_NNODES nodes..."

# MASTER_ADDR and MASTER_PORT for Accelerate if not using Slurm process group init
# Accelerate with Slurm often auto-detects. If not, these are needed.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 # Ensure this port is free
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT (for Accelerate/torch.dist)"

# Construct --container-mounts argument
CONTAINER_MOUNTS_ARG="${HOST_JOB_OUTPUTS_BASE_PATH}:${CONTAINER_CHECKPOINT_DIR_BASE}"
# If data is on shared storage, add data mounts here too.

echo "Using --container-mounts=$CONTAINER_MOUNTS_ARG"

# Accelerate launch command
# --num_processes will be SLURM_NNODES * num_gpus_per_node (from accelerate config or auto-detected)
# --num_machines will be SLURM_NNODES
# --machine_rank will be SLURM_NODEID (or derived if multi-job-step)
# Accelerate should pick these up from Slurm + PyTorch env vars

srun \
    --container-image="docker://cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-ft:latest" \
    --container-workdir="/workspace" \
    --container-mounts="$CONTAINER_MOUNTS_ARG" \
    accelerate launch --config_file /workspace/fsdp_config.yaml finetune.py ${FINETUNE_CLI_ARGS}

JOB_EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE."
exit $JOB_EXIT_CODE 