#!/bin/bash
#SBATCH --job-name=chess-finetune # Simple name
#SBATCH --nodes=2                  # Use 2 nodes for distributed training
#SBATCH --ntasks-per-node=1        # One task per node
#SBATCH --gres=gpu:1               # One GPU per task
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=180G                 # Memory per node
#SBATCH --time=23:59:00            # Time limit
#SBATCH --output=/root/chess_finetune_%j.log # Output to /root
#SBATCH --export=ALL               # Export environment

# Basic Info
echo "============================================================"
echo "DISTRIBUTED PYTORCH TRAINING JOB"
echo "============================================================"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "NNODES: $SLURM_NNODES, NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"

# Environment variables for distributed training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_TIMEOUT=600

# Set container image path - using correct syntax without docker:// prefix
CONTAINER_IMAGE="cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-chess-ft:latest"
echo "Using container image: $CONTAINER_IMAGE"

# Define paths for job data and outputs
JOB_DIR="/slurm_jobs/${SLURM_JOB_ID}"
COORD_DIR="${JOB_DIR}/coord"
LOGS_DIR="${JOB_DIR}/logs"
CHECKPOINTS_DIR="${JOB_DIR}/checkpoints"

# Create directories
echo "Creating job directories in ${JOB_DIR}"
mkdir -p "${COORD_DIR}" "${LOGS_DIR}" "${CHECKPOINTS_DIR}"
chmod -R 777 "${JOB_DIR}"  # Ensure permissions

# Container paths
CONTAINER_JOB_DIR="/job_data"
CONTAINER_DATA_PATH="/workspace/strategic_game_chess.jsonl"

# Container mounts
CONTAINER_MOUNTS="${JOB_DIR}:${CONTAINER_JOB_DIR}"

# Training arguments - add precision overrides
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir=${CONTAINER_JOB_DIR}/checkpoints"
FINETUNE_CLI_ARGS+=" --data_path=${CONTAINER_DATA_PATH}"
FINETUNE_CLI_ARGS+=" --disable_amp=True"  # Add parameter to disable automatic mixed precision

# Node and rank configuration
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(getent hosts ${HEAD_NODE} | awk '{print $1}' | head -n 1)
MASTER_PORT=29510

echo "Head node: ${HEAD_NODE} (${HEAD_NODE_IP}:${MASTER_PORT})"
echo "Container mounts: ${CONTAINER_MOUNTS}"
echo "Finetune CLI args: ${FINETUNE_CLI_ARGS}"

# Write configuration files for nodes to read
echo "${HEAD_NODE}" > "${COORD_DIR}/master_addr.txt"
echo "${HEAD_NODE_IP}" > "${COORD_DIR}/master_ip.txt"
echo "${MASTER_PORT}" > "${COORD_DIR}/master_port.txt"
echo "${SLURM_NNODES}" > "${COORD_DIR}/world_size.txt"

# Create node rank mapping
ALL_NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
for i in "${!ALL_NODES[@]}"; do
    echo "${ALL_NODES[$i]}: $i" >> "${COORD_DIR}/node_ranks.txt"
    echo "Node ${i}: ${ALL_NODES[$i]}"
done

# Make all files readable
chmod -R 755 "${COORD_DIR}"

# Launch the distributed training job
echo "============================================================"
echo "Launching distributed training on ${SLURM_NNODES} nodes"
echo "============================================================"

srun \
    --export=ALL,PYTORCH_CUDA_ALLOC_CONF,TORCH_DISTRIBUTED_TIMEOUT \
    --container-image=${CONTAINER_IMAGE} \
    --container-workdir=/workspace \
    --container-mounts=${CONTAINER_MOUNTS} \
    sh -c '
# Get node information
NODE_HOSTNAME=$(hostname)
NODE_LOG_FILE="/job_data/logs/${NODE_HOSTNAME}.log"

# Use simple redirection compatible with dash shell
echo "==================== Node: ${NODE_HOSTNAME} ====================" | tee "$NODE_LOG_FILE"
date | tee -a "$NODE_LOG_FILE"

# Define a simple logging function
log() {
  echo "$@" | tee -a "$NODE_LOG_FILE"
}

# Read configuration
MASTER_ADDR=$(cat /job_data/coord/master_ip.txt)
MASTER_PORT=$(cat /job_data/coord/master_port.txt)
WORLD_SIZE=$(cat /job_data/coord/world_size.txt)

# Find node rank
NODE_RANK=""
while IFS=": " read -r host rank; do
    if [ "$host" = "$NODE_HOSTNAME" ]; then
        NODE_RANK=$rank
        break
    fi
done < "/job_data/coord/node_ranks.txt"

if [ -z "$NODE_RANK" ]; then
    log "ERROR: Could not determine rank for node ${NODE_HOSTNAME}"
    cat "/job_data/coord/node_ranks.txt" | tee -a "$NODE_LOG_FILE"
    exit 1
fi

# Always ensure LOCAL_RANK is 0 as we have 1 GPU per node
LOCAL_RANK=0

log "This node: ${NODE_HOSTNAME}, Global Rank: ${NODE_RANK}, Local Rank: ${LOCAL_RANK}"
log "Master: ${MASTER_ADDR}:${MASTER_PORT}, World Size: ${WORLD_SIZE}"

# System diagnostics
log "--- System Info ---"
nvidia-smi | tee -a "$NODE_LOG_FILE"
python --version | tee -a "$NODE_LOG_FILE"
pip list | grep torch | tee -a "$NODE_LOG_FILE"

# Staggered start - workers wait for head node
if [ "$NODE_RANK" -ne "0" ]; then
    log "Worker node waiting 30 seconds for head node to initialize..."
    sleep 30
    log "Worker node resuming"
fi

# Set distributed training environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export RANK=$NODE_RANK
export LOCAL_RANK=$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=0  # Important: ensure only GPU 0 is visible
export PYTHONUNBUFFERED=1

log "Starting PyTorch distributed training with direct Python execution"
log "ENV: RANK=${RANK}, LOCAL_RANK=${LOCAL_RANK}, WORLD_SIZE=${WORLD_SIZE}"

# Create a Python script to do the patching
python -c "
import os
import sys
import contextlib

# Get the original script path
original_script = \"/workspace/chess-finetune.py\"

# Read the original script
with open(original_script, \"r\") as f:
    content = f.read()

# Make necessary patches
patches = []

# 1. Fix CUDA device selection
if \"torch.cuda.set_device(torch.distributed.get_rank())\" in content:
    content = content.replace(
        \"torch.cuda.set_device(torch.distributed.get_rank())\",
        \"torch.cuda.set_device(int(os.environ.get(\\\"LOCAL_RANK\\\", 0)))\"
    )
    patches.append(\"CUDA device selection\")

# 2. Fix BFloat16 issue with gradient scaler
# Add disable_amp argument if it doesn't exist
if \"parser.add_argument(\\\"--disable_amp\\\"\" not in content:
    content = content.replace(
        \"parser.add_argument(\\\"--output_dir\\\"\",
        \"parser.add_argument(\\\"--disable_amp\\\", action=\\\"store_true\\\", help=\\\"Disable automatic mixed precision\\\")\\n    parser.add_argument(\\\"--output_dir\\\"\"
    )

# Modify the scaler creation code
if \"scaler = GradScaler()\" in content:
    content = content.replace(
        \"scaler = GradScaler()\",
        \"scaler = GradScaler() if not args.disable_amp else None\"
    )
    patches.append(\"Added disable_amp option\")

# Modify the scaler usage
if \"with autocast(device_type=\\\"cuda\\\", dtype=torch.bfloat16):\" in content:
    content = content.replace(
        \"with autocast(device_type=\\\"cuda\\\", dtype=torch.bfloat16):\",
        \"with autocast(device_type=\\\"cuda\\\", dtype=torch.bfloat16) if not args.disable_amp else contextlib.nullcontext():\"
    )
    # Add import if not present
    if \"import contextlib\" not in content:
        content = content.replace(
            \"import torch\",
            \"import torch\\nimport contextlib\"
        )
    patches.append(\"Modified autocast usage\")

# Fix scaler.scale calls
if \"scaler.scale(loss).backward()\" in content:
    content = content.replace(
        \"scaler.scale(loss).backward()\",
        \"loss.backward() if args.disable_amp else scaler.scale(loss).backward()\"
    )
    patches.append(\"Modified backward pass\")

if \"scaler.step(optimizer)\" in content:
    content = content.replace(
        \"scaler.step(optimizer)\",
        \"optimizer.step() if args.disable_amp else scaler.step(optimizer)\"
    )
    patches.append(\"Modified optimizer step\")

if \"scaler.update()\" in content:
    content = content.replace(
        \"scaler.update()\",
        \"scaler.update() if not args.disable_amp else None\"
    )
    patches.append(\"Modified scaler update\")

# Write the patched script
patched_script = \"/tmp/patched_chess-finetune.py\"
with open(patched_script, \"w\") as f:
    f.write(content)

print(f\"Patched script created at {patched_script} with patches: {patches}\")
" || log "Error in patching script creation"

# Execute the patched script
if [ -f "/tmp/patched_chess-finetune.py" ]; then
    log "Running patched script"
    python /tmp/patched_chess-finetune.py ${FINETUNE_CLI_ARGS} 2>&1 | tee -a "$NODE_LOG_FILE"
    PYTHON_EXIT_CODE=$?
else
    log "ERROR: Patched script was not created successfully"
    PYTHON_EXIT_CODE=1
fi

log "Python process exited with code: ${PYTHON_EXIT_CODE}"
exit ${PYTHON_EXIT_CODE}
'

JOB_EXIT_CODE=$?
echo "============================================================"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE"
echo "============================================================"
exit ${JOB_EXIT_CODE} 