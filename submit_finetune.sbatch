#!/bin/bash
#SBATCH --job-name=granite-dist-ft
#SBATCH --nodes=2                   # Utilize 2 H100 nodes
#SBATCH --ntasks-per-node=1         # 1 task (PyTorch process) per node, each managing one H100
#SBATCH --gres=gpu:1                # Each task gets 1 H100 GPU
#SBATCH --cpus-per-task=16          # Allocate more CPUs, e.g., 16 per H100 for data loading, etc.
#SBATCH --mem=200G                  # Allocate more memory, e.g., 128GB per node
#SBATCH --time=04:00:00             # Example: 4 hours wall time for a distributed job
#SBATCH --output=granite_dist_ft_%j.log

# --- Soperator Specifics - THESE ARE CRITICAL --- 
# You MUST confirm these with Soperator/Nebius documentation.

# 1. Container Image
#SBATCH --container-image=registry-e00xn9gpx27cp05wsr/granite-ft:latest 

# 2. Container Working Directory
#SBATCH --container-workdir=/workspace 

# 3. Container Mounts for Shared Storage (Nebius Filestore)
#    Replace <HOST_PATH_TO_NEBIUS_FILESTORE_MOUNT> with the actual path.
#SBATCH --container-mounts=<HOST_PATH_TO_NEBIUS_FILESTORE_MOUNT>:/mnt/shared_storage


echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES, Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_ON_NODE (from scontrol, may need parsing or specific Soperator env var)"

# Debug: Print relevant Slurm environment variables for distributed setup
echo "SLURM_PROCID=$SLURM_PROCID, SLURM_NPROCS=$SLURM_NPROCS"
echo "SLURM_LOCALID=$SLURM_LOCALID, SLURM_NODEID=$SLURM_NODEID"
echo "SLURMD_NODENAME=$SLURMD_NODENAME"

# MLflow Configuration
export MLFLOW_TRACKING_URI="public-tracking.mlflow-e00spsxj9fckt2k6pz.backbone-e00qwnewt2vzn1dh4s.msp.eu-north1.nebius.cloud"
export MLFLOW_TRACKING_SERVER_CERT_PATH="/etc/mlflow/certs/ca.pem"

# Checkpoint directory on shared storage
CHECKPOINT_PARENT_DIR="/mnt/shared_storage/checkpoints"
JOB_CHECKPOINT_DIR="${CHECKPOINT_PARENT_DIR}/granite_dist_ft_slurm_${SLURM_JOB_ID}"

echo "Creating checkpoint directory: $JOB_CHECKPOINT_DIR"
# Create checkpoint directory only on the master process (rank 0) to avoid race conditions
if [ "$SLURM_PROCID" -eq 0 ]; then
  mkdir -p "$JOB_CHECKPOINT_DIR"
fi
# Ensure all processes wait for directory creation if needed, though usually not an issue if parent exists
scontrol Barrier

# Arguments for finetune.py
FINETUNE_ARGS=""
FINETUNE_ARGS+=" --output_dir='$JOB_CHECKPOINT_DIR'"
# Add other arguments like data paths if made configurable in finetune.py

echo "------------------------------------------------------------------------------"
echo "Launching distributed training on $SLURM_NPROCS processes..."

# Launching distributed training with torchrun (or deepspeed)
# `srun` will execute this command on each allocated task (SLURM_NPROCS times in total).
# `torchrun` uses environment variables set by Slurm/Soperator (or its own auto-discovery) 
# for MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK.

# Ensure your Docker image has torchrun or deepspeed CLIs.
# The `uv sync` in your Dockerfile should install dependencies from pyproject.toml, 
# make sure `torch`, `deepspeed` (if using) are listed there.

# Option 1: Using torchrun (recommended for PyTorch DDP/FSDP if not using DeepSpeed launcher)
# --nproc_per_node should match the number of GPUs each task/node is meant to handle (here, 1)
# The master_addr and master_port might be automatically set by Slurm or need specific srun options.
# Often, one can rely on init_method='env://' in the Python script.
# This example assumes finetune.py is adapted for torch.distributed.

# First, get the master address from the first node in the job
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 # Default PyTorch port, ensure it's free

echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

srun torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    finetune.py $FINETUNE_ARGS

# Option 2: Using DeepSpeed launcher (if finetune.py is adapted for DeepSpeed)
# srun deepspeed --num_nodes=$SLURM_NNODES --num_gpus=$SLURM_GPUS_ON_NODE \
#     finetune.py $FINETUNE_ARGS --deepspeed ds_config_zero3.json

JOB_EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE."
exit $JOB_EXIT_CODE 