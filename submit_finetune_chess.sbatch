#!/bin/bash
export PYXIS_VERBOSE=3
export ENROOT_VERBOSE=3
#SBATCH --job-name=chess-qlora-ddp # Updated job name
#SBATCH --nodes=2                  # Keep node count from previous script, adjust as needed
#SBATCH --ntasks-per-node=1      # Keep tasks per node from previous script
#SBATCH --gres=gpu:1               # Keep GPUs per task from previous script (adjust NPROC_PER_NODE if changing)
#SBATCH --cpus-per-task=16         # Keep CPU count
#SBATCH --mem=180G                 # Keep memory
#SBATCH --time=23:59:00          # Use longer time limit from example
#SBATCH --output=/root/chess_qlora_ddp_%j.log # Updated log file name
#SBATCH --export=ALL

# NCCL performance tuning (from example)
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES, SLURM_NTASKS: $SLURM_NTASKS, SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE, SLURM_JOB_GPUS: $SLURM_JOB_GPUS, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Paths
CONTAINER_OUTPUTS_BASE="/job_outputs"
# Updated checkpoint dir name
JOB_CHECKPOINT_DIR="${CONTAINER_OUTPUTS_BASE}/checkpoints/chess_qlora_slurm_${SLURM_JOB_ID}"
# Dataset path within the container (assuming copied to /workspace)
CONTAINER_DATA_PATH="/workspace/strategic_game_chess.jsonl" 
HOST_SHARED_FS_ROOT_PATH="/" # Assuming outputs go to a shared FS mounted at root
HOST_JOB_OUTPUTS_BASE_PATH="${HOST_SHARED_FS_ROOT_PATH}slurm_job_data/llm_chess_outputs" # Separate output dir

echo "Ensuring host output base directory exists: ${HOST_JOB_OUTPUTS_BASE_PATH}"
mkdir -p "${HOST_JOB_OUTPUTS_BASE_PATH}" 
CONTAINER_MOUNTS_ARG="${HOST_JOB_OUTPUTS_BASE_PATH}:${CONTAINER_OUTPUTS_BASE}"
echo "Using --container-mounts=$CONTAINER_MOUNTS_ARG"

# Arguments for chess-finetune.py
# Use defaults from chess-finetune.py argparse where suitable
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir=${JOB_CHECKPOINT_DIR}" 
FINETUNE_CLI_ARGS+=" --data_path=${CONTAINER_DATA_PATH}" # Pass the data path
#FINETUNE_CLI_ARGS+=" --learning_rate=1e-5" # Example override
#FINETUNE_CLI_ARGS+=" --batch_size_per_device=1" # Example override
#FINETUNE_CLI_ARGS+=" --gradient_accumulation_steps=8" # Example override

# Torchrun DDP Setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Use MASTER_PORT calculation from example
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Determine NPROC_PER_NODE based on allocated GPUs
if [[ -n "$SLURM_GPUS_PER_TASK" ]]; then
    NPROC_PER_NODE=$SLURM_GPUS_PER_TASK
elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    # Count GPUs assigned by Slurm/Pyxis via CUDA_VISIBLE_DEVICES
    NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F ',' '{print NF}') 
elif [[ -n "$SLURM_JOB_GPUS" ]]; then 
    # Fallback: Count total GPUs in job if visible devices isn't set (less reliable for per-node count)
    NPROC_PER_NODE=$(echo "$SLURM_JOB_GPUS" | awk -F ',' '{print NF}')
else
    # Default if no GPU info found (e.g., CPU job or config issue)
    NPROC_PER_NODE=1 
fi
# Ensure it's at least 1
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

NNODES=${SLURM_NNODES:-1}

echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"
echo "NPROC_PER_NODE: $NPROC_PER_NODE, NNODES: $NNODES, SLURM_JOB_ID: $SLURM_JOB_ID"

echo "Launching training with torchrun..."

# Define the command to run inside the container
# Use chess-finetune.py and llm-chess-ft image
COMMAND_TO_RUN=" \
    echo \\\"--- Running inside container on RANK: \${SLURM_PROCID}, LOCAL_RANK: \${SLURM_LOCALID}, NODE: \$(hostname) ---\\\" && \
    printenv && \
    echo \\\"--- Checking files in /workspace ---\\\" && ls -l /workspace && \
    echo \\\"--- nvidia-smi ---\\\" && \
    nvidia-smi && \
    echo \\\"--- Starting torchrun ---\\\" && \
    torchrun \\\n        --nproc_per_node=\\${NPROC_PER_NODE} \\\n        --nnodes=\${NNODES} \\\n        --rdzv_id=\\${SLURM_JOB_ID} \\\n        --rdzv_backend=c10d \\\n        --rdzv_endpoint=\\${MASTER_ADDR}:\\${MASTER_PORT} \\\n        chess-finetune.py ${FINETUNE_CLI_ARGS} \\\n"

echo "Command to be run by srun: ${COMMAND_TO_RUN}"

# srun launches the container and executes the bash command inside
srun \
    --container-image=docker://cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-chess-ft:latest \
    --container-workdir=/workspace \
    --container-mounts="${CONTAINER_MOUNTS_ARG}" \
    --container-image-save="" \
    bash -c "${COMMAND_TO_RUN}"

JOB_EXIT_CODE=$?

echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE."
exit ${JOB_EXIT_CODE} 