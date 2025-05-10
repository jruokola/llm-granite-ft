#!/bin/bash
#SBATCH --job-name=function-finetune # Simple name
#SBATCH --nodes=2                  # Use 2 nodes for distributed training
#SBATCH --ntasks-per-node=1        # One task per node
#SBATCH --gres=gpu:1               # One GPU per task
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=180G                 # Memory per node
#SBATCH --time=23:59:00            # Time limit
#SBATCH --output=/root/function_finetune_%j.log # Output to /root
#SBATCH --export=ALL               # Export environment

# Basic Info
echo "============================================================"
echo "DISTRIBUTED PYTORCH TRAINING JOB"
echo "============================================================"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "NNODES: $SLURM_NNODES, NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"

# Debug steps - show directory contents
echo "============================================================"
echo "DEBUG: Current directory and contents"
pwd
echo "Contents of current directory:"
ls -la
echo "Contents of fixed-scripts directory (if exists):"
ls -la fixed-scripts/ || echo "fixed-scripts directory not found"
echo "Contents of parent directory:"
ls -la ..
echo "============================================================"

# Environment variables for distributed training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_TIMEOUT=600

# Set container image path - using correct syntax without docker:// prefix
CONTAINER_IMAGE="cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-function-ft-fix:latest"
echo "Using container image: $CONTAINER_IMAGE"

# Define paths for job data and outputs
JOB_DIR="/slurm_jobs/${SLURM_JOB_ID}"
COORD_DIR="${JOB_DIR}/coord"
LOGS_DIR="${JOB_DIR}/logs"
CHECKPOINTS_DIR="${JOB_DIR}/checkpoints"
SCRIPTS_DIR="${JOB_DIR}/scripts"

# Create directories
echo "Creating job directories in ${JOB_DIR}"
mkdir -p "${COORD_DIR}" "${LOGS_DIR}" "${CHECKPOINTS_DIR}" "${SCRIPTS_DIR}"
chmod -R 777 "${JOB_DIR}"  # Ensure permissions

# Note: We do not need to copy the script locally as it should be in the container at /workspace
echo "Will use function-finetune-fixed.py from container image at /workspace"

# Container paths
CONTAINER_JOB_DIR="/job_data"

# Container mounts - just mount the job directory
CONTAINER_MOUNTS="${JOB_DIR}:${CONTAINER_JOB_DIR}"

# Training arguments
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir=${CONTAINER_JOB_DIR}/checkpoints"
FINETUNE_CLI_ARGS+=" --disable_amp"  # Pass as a flag
FINETUNE_CLI_ARGS+=" --num_epochs=1" # Set to 1 epoch
FINETUNE_CLI_ARGS+=" --dataset_subset_size=10000" # Limit dataset size for testing
FINETUNE_CLI_ARGS+=" --model_name_or_path=ibm-granite/granite-3.3-2b-instruct" # Correct model name format

# Node and rank configuration
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_NODE_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}' | head -n 1)
MASTER_PORT=29510

echo "Head node: ${HEAD_NODE} (${HEAD_NODE_IP}:${MASTER_PORT})"
echo "Container mounts: ${CONTAINER_MOUNTS}"
echo "Finetune CLI args: $FINETUNE_CLI_ARGS"

# Write configuration files for nodes to read
echo "${HEAD_NODE}" > "${COORD_DIR}/master_addr.txt"
echo "${HEAD_NODE_IP}" > "${COORD_DIR}/master_ip.txt"
echo "${MASTER_PORT}" > "${COORD_DIR}/master_port.txt"
echo "${SLURM_NNODES}" > "${COORD_DIR}/world_size.txt"

# Create node rank mapping
# Use read to properly handle command output
mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
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
    --container-image="${CONTAINER_IMAGE}" \
    --container-workdir="/workspace" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    sh -c "
# Get node information
NODE_HOSTNAME=\$(hostname)
NODE_LOG_FILE=\"/job_data/logs/\${NODE_HOSTNAME}.log\"

# Use simple redirection compatible with dash shell
echo \"==================== Node: \${NODE_HOSTNAME} ====================\" | tee \"\$NODE_LOG_FILE\"
date | tee -a \"\$NODE_LOG_FILE\"

# Define a simple logging function
log() {
  echo \"\$@\" | tee -a \"\$NODE_LOG_FILE\"
}

# Read configuration
MASTER_ADDR=\$(cat /job_data/coord/master_ip.txt)
MASTER_PORT=\$(cat /job_data/coord/master_port.txt)
WORLD_SIZE=\$(cat /job_data/coord/world_size.txt)

# Find node rank
NODE_RANK=\"\"
while IFS=\": \" read -r host rank; do
    if [ \"\$host\" = \"\$NODE_HOSTNAME\" ]; then
        NODE_RANK=\$rank
        break
    fi
done < \"/job_data/coord/node_ranks.txt\"

if [ -z \"\$NODE_RANK\" ]; then
    log \"ERROR: Could not determine rank for node \${NODE_HOSTNAME}\"
    cat \"/job_data/coord/node_ranks.txt\" | tee -a \"\$NODE_LOG_FILE\"
    exit 1
fi

# Always ensure LOCAL_RANK is 0 as we have 1 GPU per node
LOCAL_RANK=0

log \"This node: \${NODE_HOSTNAME}, Global Rank: \${NODE_RANK}, Local Rank: \${LOCAL_RANK}\"
log \"Master: \${MASTER_ADDR}:\${MASTER_PORT}, World Size: \${WORLD_SIZE}\"

# System diagnostics
log \"--- System Info ---\"
nvidia-smi | tee -a \"\$NODE_LOG_FILE\"
python --version | tee -a \"\$NODE_LOG_FILE\"
pip list | grep torch | tee -a \"\$NODE_LOG_FILE\"

# Staggered start - workers wait for head node
if [ \"\$NODE_RANK\" -ne \"0\" ]; then
    log \"Worker node waiting 30 seconds for head node to initialize...\"
    sleep 30
    log \"Worker node resuming\"
fi

# Set distributed training environment variables
export MASTER_ADDR=\$MASTER_ADDR
export MASTER_PORT=\$MASTER_PORT
export WORLD_SIZE=\$WORLD_SIZE
export RANK=\$NODE_RANK
export LOCAL_RANK=\$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=0  # Important: ensure only GPU 0 is visible
export PYTHONUNBUFFERED=1

log \"Starting PyTorch distributed training using fixed script\"
log \"ENV: RANK=\${RANK}, LOCAL_RANK=\${LOCAL_RANK}, WORLD_SIZE=\${WORLD_SIZE}\"
log \"DEBUG: FINETUNE_CLI_ARGS before python call: $FINETUNE_CLI_ARGS\"

# List workspace directory content for debugging
log \"DEBUG: Listing workspace directory content:\"
ls -la /workspace || log \"Cannot list /workspace\"
log \"DEBUG: Listing mounted job directory content:\"
ls -la ${CONTAINER_JOB_DIR} || log \"Cannot list ${CONTAINER_JOB_DIR}\"

# Run the script directly from the workspace directory in the container
python /workspace/function-finetune-fixed.py $FINETUNE_CLI_ARGS 2>&1 | tee -a \"\$NODE_LOG_FILE\"
PYTHON_EXIT_CODE=\$?

log \"Python process exited with code: \${PYTHON_EXIT_CODE}\"
exit \${PYTHON_EXIT_CODE}
"

JOB_EXIT_CODE=$?
echo "============================================================"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE"
echo "============================================================"
exit ${JOB_EXIT_CODE} 