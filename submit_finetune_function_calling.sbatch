#!/bin/bash
#SBATCH --job-name=function-finetune # Updated job name
#SBATCH --nodes=2                  # Use 2 nodes for distributed training
#SBATCH --ntasks-per-node=1        # One task per node
#SBATCH --gres=gpu:1               # One GPU per task
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=180G                 # Memory per node
#SBATCH --time=23:59:00            # Time limit
#SBATCH --output=/root/slurm_logs/function_finetune_%j.log # Updated output path
#SBATCH --export=ALL               # Export environment

# Basic Info
echo "============================================================"
echo "DISTRIBUTED PYTORCH TRAINING JOB (FUNCTION CALLING)"
echo "============================================================"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "NNODES: $SLURM_NNODES, NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"

# Environment variables for distributed training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DISTRIBUTED_TIMEOUT=600

# Set container image path
CONTAINER_IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-transformers:latest" # Updated for function calling
echo "Using container image: $CONTAINER_IMAGE"

# Define paths for job data and outputs on shared drive
JOB_DIR="/root/slurm_jobs/function_finetune/${SLURM_JOB_ID}" # Updated JOB_DIR
COORD_DIR="${JOB_DIR}/coord"
LOGS_DIR="${JOB_DIR}/logs"
CHECKPOINTS_DIR="${JOB_DIR}/checkpoints"

# Create directories
echo "Creating job directories in ${JOB_DIR}"
mkdir -p "${COORD_DIR}" "${LOGS_DIR}" "${CHECKPOINTS_DIR}"
chmod -R 777 "${JOB_DIR}"  # Ensure permissions

# Container paths
CONTAINER_JOB_DIR="/job_data" # This path is inside the container, referring to the mounted JOB_DIR

# Container mounts
CONTAINER_MOUNTS="${JOB_DIR}:${CONTAINER_JOB_DIR}"

# Training arguments
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir=${CONTAINER_JOB_DIR}/checkpoints"
FINETUNE_CLI_ARGS+=" --processed_dataset_path=/processed_datasets/glaive_fc_v2" # Path inside the Docker container
FINETUNE_CLI_ARGS+=" --use_qlora" # Enable QLoRA
# FINETUNE_CLI_ARGS+=" --data_path=${CONTAINER_DATA_PATH}" # Removed for function calling script
# FINETUNE_CLI_ARGS+=" --disable_amp"  # Keeping this for now
export FINETUNE_CLI_ARGS

echo "Finetune CLI args (before srun): $FINETUNE_CLI_ARGS" # For sbatch log

# Node and rank configuration (same as chess_sbatch)
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_NODE_IP=$(getent hosts "${HEAD_NODE}" | awk '{print $1}' | head -n 1)
MASTER_PORT=29510 # Using the same default port

echo "Head node: ${HEAD_NODE} (${HEAD_NODE_IP}:${MASTER_PORT})"
echo "Container mounts: ${CONTAINER_MOUNTS}"
echo "Finetune CLI args (in main sbatch script): $FINETUNE_CLI_ARGS"

# Write configuration files for nodes to read (same as chess_sbatch)
echo "${HEAD_NODE}" > "${COORD_DIR}/master_addr.txt"
echo "${HEAD_NODE_IP}" > "${COORD_DIR}/master_ip.txt"
echo "${MASTER_PORT}" > "${COORD_DIR}/master_port.txt"
echo "${SLURM_NNODES}" > "${COORD_DIR}/world_size.txt"

mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
for i in "${!ALL_NODES[@]}"; do
    echo "${ALL_NODES[$i]}: $i" >> "${COORD_DIR}/node_ranks.txt"
    echo "Node ${i}: ${ALL_NODES[$i]}"
done
chmod -R 755 "${COORD_DIR}"

# Attempt to ensure filesystem changes are synced before srun
sync
sleep 5 

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
# Construct NODE_LOG_FILE using CONTAINER_JOB_DIR which should be exported by srun --export=ALL
# and was set to /job_data in the main sbatch script.
# Ensure CONTAINER_JOB_DIR is correctly inherited, default to /job_data if empty for safety.
EFFECTIVE_CONTAINER_JOB_DIR=\${CONTAINER_JOB_DIR:-/job_data}
NODE_LOG_FILE=\"\${EFFECTIVE_CONTAINER_JOB_DIR}/logs/\${NODE_HOSTNAME}.log\"

# Ensure log directory exists
mkdir -p \"\$(dirname \"\$NODE_LOG_FILE\")\"

# Initial log entries directly using constructed NODE_LOG_FILE
echo \"==================== Node: \${NODE_HOSTNAME} ====================\" | tee \"\$NODE_LOG_FILE\"
date | tee -a \"\$NODE_LOG_FILE\"

# Define log function as in the original working script
log() {
  echo \"\$@\" | tee -a \"\$NODE_LOG_FILE\"
}

# Debug: Check if CONTAINER_JOB_DIR and FINETUNE_CLI_ARGS are available
log \"DEBUG: Inherited CONTAINER_JOB_DIR is: '\${CONTAINER_JOB_DIR}'\"
log \"DEBUG: EFFECTIVE_CONTAINER_JOB_DIR is: '\${EFFECTIVE_CONTAINER_JOB_DIR}'\"
log \"DEBUG: Inherited FINETUNE_CLI_ARGS is: '\${FINETUNE_CLI_ARGS}'\"
log \"DEBUG: NODE_LOG_FILE is: '\${NODE_LOG_FILE}'\"

MASTER_ADDR=\$(cat \${EFFECTIVE_CONTAINER_JOB_DIR}/coord/master_ip.txt)
MASTER_PORT=\$(cat \${EFFECTIVE_CONTAINER_JOB_DIR}/coord/master_port.txt)
WORLD_SIZE=\$(cat \${EFFECTIVE_CONTAINER_JOB_DIR}/coord/world_size.txt)

NODE_RANK=\"\"
# Add check for node_ranks.txt existence
if [ ! -f \"\${EFFECTIVE_CONTAINER_JOB_DIR}/coord/node_ranks.txt\" ]; then
    log \"ERROR: \${EFFECTIVE_CONTAINER_JOB_DIR}/coord/node_ranks.txt not found!\"
    exit 1
fi
while IFS=\": \" read -r host rank; do
    if [ \"\$host\" = \"\$NODE_HOSTNAME\" ]; then
        NODE_RANK=\$rank
        break
    fi
done < \"\${EFFECTIVE_CONTAINER_JOB_DIR}/coord/node_ranks.txt\"

if [ -z \"\$NODE_RANK\" ]; then
    log \"ERROR: Could not determine rank for node \${NODE_HOSTNAME}\"
    cat \"\${EFFECTIVE_CONTAINER_JOB_DIR}/coord/node_ranks.txt\" | tee -a \"\$NODE_LOG_FILE\"
    exit 1
fi

LOCAL_RANK=0 # Correct for --ntasks-per-node=1 and --gres=gpu:1

log \"This node: \${NODE_HOSTNAME}, Global Rank: \${NODE_RANK}, Local Rank: \${LOCAL_RANK}\"
log \"Master: \${MASTER_ADDR}:\${MASTER_PORT}, World Size: \${WORLD_SIZE}\"

log \"--- System Info ---\"
nvidia-smi | tee -a \"\$NODE_LOG_FILE\"
python --version | tee -a \"\$NODE_LOG_FILE\"
pip list | grep torch | tee -a \"\$NODE_LOG_FILE\"
pip list | grep peft | tee -a \"\$NODE_LOG_FILE\"
pip list | grep bitsandbytes | tee -a \"\$NODE_LOG_FILE\"


if [ \"\$NODE_RANK\" -ne \"0\" ]; then
    log \"Worker node waiting 30 seconds for head node to initialize...\"
    sleep 30
    log \"Worker node resuming\"
fi

# Export variables needed by the Python script within this sh -c context
export MASTER_ADDR=\$MASTER_ADDR
export MASTER_PORT=\$MASTER_PORT
export WORLD_SIZE=\$WORLD_SIZE
export RANK=\$NODE_RANK
export LOCAL_RANK=\$LOCAL_RANK
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false


log \"Starting PyTorch distributed training for function calling\"
log \"ENV: RANK=\${RANK}, LOCAL_RANK=\${LOCAL_RANK}, WORLD_SIZE=\${WORLD_SIZE}\"

# Define the target script path within the container
TARGET_SCRIPT_PATH=\"/workspace/function-finetune-fixed.py\"

log \"Running training script \${TARGET_SCRIPT_PATH}\"
# FINETUNE_CLI_ARGS should be available here due to 'export FINETUNE_CLI_ARGS' in main script and srun --export=ALL
log \"DEBUG: FINETUNE_CLI_ARGS (inside srun sh -c): \$FINETUNE_CLI_ARGS\"

python \"\${TARGET_SCRIPT_PATH}\" \$FINETUNE_CLI_ARGS 2>&1 | tee -a \"\$NODE_LOG_FILE\"
PYTHON_EXIT_CODE=\$?

log \"Python process exited with code: \${PYTHON_EXIT_CODE}\"
exit \${PYTHON_EXIT_CODE}
"

JOB_EXIT_CODE=$?
echo "============================================================"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE"
echo "============================================================"
exit ${JOB_EXIT_CODE}
