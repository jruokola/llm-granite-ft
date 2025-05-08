#!/bin/bash
#SBATCH --job-name=chess-finetune # Simple name
#SBATCH --nodes=2                  # Keep node count, adjust as needed
#SBATCH --ntasks-per-node=1      # Keep tasks per node 
#SBATCH --gres=gpu:1               # Keep GPUs per task (adjust NPROC_PER_NODE if changing)
#SBATCH --cpus-per-task=16         # Keep CPU count
#SBATCH --mem=180G                 # Keep memory
#SBATCH --time=23:59:00          # Keep longer time limit
#SBATCH --output=/root/chess_finetune_%j.log # Output to /root
#SBATCH --export=ALL # Export environment

# Removed NCCL tuning vars for simplicity
# Removed MLflow vars

# Basic Info
echo "------------------------------------------------------------------------------"
echo "Starting Slurm job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"
echo "NNODES: $SLURM_NNODES, NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"

# Define paths inside the container
CONTAINER_OUTPUTS_BASE="/job_outputs"
JOB_CHECKPOINT_DIR="${CONTAINER_OUTPUTS_BASE}/checkpoints/chess_qlora_slurm_${SLURM_JOB_ID}"
CONTAINER_DATA_PATH="/workspace/strategic_game_chess.jsonl" 

# Define host path for output mount (ensure /root/slurm_outputs exists or has write perms)
HOST_JOB_OUTPUTS_PATH="/root/slurm_outputs/${SLURM_JOB_ID}"
echo "Ensuring host output directory exists: ${HOST_JOB_OUTPUTS_PATH}"
mkdir -p "${HOST_JOB_OUTPUTS_PATH}" 
CONTAINER_MOUNTS_ARG="${HOST_JOB_OUTPUTS_PATH}:${CONTAINER_OUTPUTS_BASE}"
echo "Using --container-mounts=$CONTAINER_MOUNTS_ARG"

# Arguments for chess-finetune.py
FINETUNE_CLI_ARGS=""
FINETUNE_CLI_ARGS+=" --output_dir=${JOB_CHECKPOINT_DIR}" 
FINETUNE_CLI_ARGS+=" --data_path=${CONTAINER_DATA_PATH}" 
# Add other args like batch size, lr, gradient accumulation if needed, e.g.:
# FINETUNE_CLI_ARGS+=" --batch_size_per_device=1"
# FINETUNE_CLI_ARGS+=" --gradient_accumulation_steps=8"

# Determine nproc_per_node based on allocated GPUs
# Use CUDA_VISIBLE_DEVICES count if available, fallback to gres count or 1
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F ',' '{print NF}') 
elif [[ "$SLURM_GPUS_PER_TASK" -gt 0 ]]; then 
    NPROC_PER_NODE=$SLURM_GPUS_PER_TASK
elif [[ "$SLURM_STEP_GPUS" -gt 0 ]]; then # Check step GPUs as another fallback
    NPROC_PER_NODE=$SLURM_STEP_GPUS
else
    NPROC_PER_NODE=1 
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-1} # Final fallback to 1

echo "Determined NPROC_PER_NODE: $NPROC_PER_NODE based on GPU allocation"
echo "Launching training with torchrun..."

# srun directly executes torchrun within the container context
# torchrun should auto-configure DDP from Slurm + srun environment
srun \
    --container-image=docker://cr.eu-north1.nebius.cloud/e00xn9gpx27cp05wsr/llm-granite-chess-ft:latest \
    --container-workdir=/workspace \
    --container-mounts="${CONTAINER_MOUNTS_ARG}" \
    --container-image-save="" \
    torchrun \
        --nproc_per_node=${NPROC_PER_NODE} \
        chess-finetune.py ${FINETUNE_CLI_ARGS}

JOB_EXIT_CODE=$?

echo "------------------------------------------------------------------------------"
echo "Slurm job $SLURM_JOB_ID finished with exit code $JOB_EXIT_CODE."
exit ${JOB_EXIT_CODE} 