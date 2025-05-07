#!/bin/bash
export PYXIS_VERBOSE=3
#SBATCH --job-name=granite-fsdp-qlora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # Number of processes Accelerate will launch per node (usually 1, then Accelerate handles GPUs)
#SBATCH --gres=gpu:1        # GPUs per task. If ntasks-per-node=1, this is GPUs per node.
#SBATCH --cpus-per-task=16 
#SBATCH --mem=180G 
#SBATCH --time=04:00:00
#SBATCH --output=/slurm_job_data/job_logs/granite_fsdp_qlora_%j.log
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
export MLFLOW_TRACKING_USERNAME="jruokola" # Uncomment and set if MLflow is authenticated
export MLFLOW_TRACKING_PASSWORD="K3m1k44l1!!666" # Uncomment and set if MLflow is authenticated

# --- Define Paths for INSIDE the Container ---
CONTAINER_OUTPUTS_BASE="/job_outputs" # Base in container for all outputs
JOB_CHECKPOINT_DIR="${CONTAINER_OUTPUTS_BASE}/checkpoints/granite_fsdp_qlora_slurm_${SLURM_JOB_ID}"
CONTAINER_DATA_DIR_IN_IMAGE="/workspace/data"

# --- Host Path Definitions (CRITICAL - REPLACE PLACEHOLDERS) ---
# This is the path on the HOST (Slurm worker node) where your Nebius Filestore is mounted.
# Example: /mnt/nebius_shared_fs  OR /shared_fs OR simply /
HOST_SHARED_FS_ROOT_PATH="/" # Assuming shared filestore is mounted at / on worker nodes

# New base path for job outputs on the shared filestore
HOST_JOB_OUTPUTS_BASE_PATH="${HOST_SHARED_FS_ROOT_PATH}slurm_job_data/llm_granite_outputs" # No leading // if root is /

# Ensure the HOST_JOB_OUTPUTS_BASE_PATH directory exists (run by sbatch script itself on submission host, hopefully seen by workers)
# This should ideally be created with appropriate permissions beforehand by an admin or setup script.
# For this test, the sbatch script will attempt to create it.
# Note: If this script is run as root on login node, it can create /slurm_job_data.
# If worker nodes have a different view or stricter perms on root of shared FS, this might need adjustment.
echo "Ensuring host output base directory exists: ${HOST_JOB_OUTPUTS_BASE_PATH}"
mkdir -p "${HOST_JOB_OUTPUTS_BASE_PATH}" # sbatch script creates this on shared FS

echo "Host path for job outputs base: $HOST_JOB_OUTPUTS_BASE_PATH"
echo "Container path for job outputs base (mount target): $CONTAINER_OUTPUTS_BASE"
echo "Container path for this job's checkpoints: $JOB_CHECKPOINT_DIR"

# Create the specific job checkpoint directory using the CONTAINER path.
# This will be created inside the mounted volume by the script running in the container.
# The mkdir command in the sbatch script itself might not work as expected for container paths
# before srun initiates the container environment and mounts.
# It's safer if finetune.py (rank 0) creates args.output_dir.
# However, if the parent mount exists, torchrun/python might be able to create subdirs.

# Data arguments (assuming data in image at /workspace/data for now)
DATA_TRAIN_ARG="--data_path_train=${CONTAINER_DATA_DIR_IN_IMAGE}/train_extended.csv"
DATA_EVAL_ARG="--data_path_eval=${CONTAINER_DATA_DIR_IN_IMAGE}/test_extended.csv"

# Arguments for finetune.py
# Paths for output_dir, data_path_train, data_path_eval are inside the container.
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir='$JOB_CHECKPOINT_DIR'"
FINETUNE_CLI_ARGS+=" $DATA_TRAIN_ARG"
FINETUNE_CLI_ARGS+=" $DATA_EVAL_ARG"
FINETUNE_CLI_ARGS+=" --epochs=3" # Example
FINETUNE_CLI_ARGS+=" --per_device_train_batch_size=2" # Adjust based on GPU memory
FINETUNE_CLI_ARGS+=" --gradient_accumulation_steps=4" # Adjust for effective batch size
FINETUNE_CLI_ARGS+=" --logging_steps=10"
# Add other finetune.py arguments as needed

# Define the container image URI consistently
NEBIUS_CONTAINER_IMAGE="docker://cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-ft:u5rmsco7gkx0leqy9uk7i06oy"

echo "------------------------------------------------------------------------------"
echo "DEBUG: Current directory: $(pwd)"
echo "DEBUG: SLURM Environment Variables:"
printenv | grep SLURM
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (should be set by Slurm/Pyxis)"
echo "------------------------------------------------------------------------------"

# MASTER_ADDR and MASTER_PORT for Accelerate if not using Slurm process group init
# Accelerate with Slurm often auto-detects. If not, these are needed.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 # Ensure this port is free
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT (for Accelerate/torch.dist)"

# Construct --container-mounts argument
CONTAINER_MOUNTS_ARG="${HOST_JOB_OUTPUTS_BASE_PATH}:${CONTAINER_OUTPUTS_BASE}"
# If data is on shared storage, add data mounts here too.

echo "Using --container-mounts=$CONTAINER_MOUNTS_ARG"

# --- DEBUGGING FILE PRESENCE --- 
echo "DEBUG: Listing /workspace contents inside container before accelerate launch..."
srun \
    --container-image="${NEBIUS_CONTAINER_IMAGE}" \
    --container-workdir="/workspace" \
    --container-mounts="$CONTAINER_MOUNTS_ARG" \
    ls -la /workspace
echo "DEBUG: Attempting to cat /workspace/fsdp_config.yaml inside container..."
srun \
    --container-image="${NEBIUS_CONTAINER_IMAGE}" \
    --container-workdir="/workspace" \
    --container-mounts="$CONTAINER_MOUNTS_ARG" \
    cat /workspace/fsdp_config.yaml || echo "DEBUG: cat /workspace/fsdp_config.yaml FAILED or file not found"
echo "DEBUG: End of pre-launch file checks."
# --- END DEBUGGING --- 

echo "Launching FSDP training with Accelerate on $SLURM_NNODES nodes..."
srun \
    --container-image="${NEBIUS_CONTAINER_IMAGE}" \
    --container-workdir="/workspace" \
    --container-mounts="$CONTAINER_MOUNTS_ARG" \
    accelerate launch --config_file /workspace/fsdp_config.yaml finetune.py ${FINETUNE_CLI_ARGS}

JOB_EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE."
exit $JOB_EXIT_CODE 