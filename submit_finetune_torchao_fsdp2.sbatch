#!/bin/bash
#SBATCH --job-name=fc-torchao-fsdp2
#SBATCH --nodes=2                # Assuming 2 nodes, one H100 per node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12       # Adjust as needed
#SBATCH --mem=180G               # Adjust as needed
#SBATCH --time=23:00:00
#SBATCH --output=/root/slurm_logs/fc_torchao_fsdp2_%j.log # Updated log name
#SBATCH --export=ALL             # propagate env to Pyxis

############################ 0. constants & job paths #########################
# finetune-fsdp2-*:latest NVIDIA PyTorch25.04 + CUDA12.6 Base Image
# finetune-fsdp2v2-*:latest PyTorch27.04 + CUDA12.8 Base Image
IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-fsdp2-1:latest"
JOB_DIR="/slurm_jobs/${SLURM_JOB_ID}"
COORD_DIR="${JOB_DIR}/coord" # For c10d backend if needed, though endpoint is hardcoded
HOST_JOBDIR="/mnt/jail/${JOB_DIR}" # Example host path, adjust if different
CONT_JOBDIR="/job_data"
CONT_DATADIR="/processed_datasets/glaive_fc_v2/"

mkdir -p "${COORD_DIR}"
chmod -R 777 "${JOB_DIR}" # Ensure permissions for container access
############################ 1. rendez-vous params ############################
CONTAINER_MOUNTS="${JOB_DIR}:${CONT_JOBDIR}:rw" # Mount job-specific directory
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_IP=$(getent hosts "${MASTER_NODE}" | awk '{print $1}')
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))   

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "IMAGE: ${IMAGE}"
echo "CONTAINER_MOUNTS: ${CONTAINER_MOUNTS}"
# echo "HEAD NODE for rdzv (if dynamic): ${MASTER_NODE} ${MASTER_IP}:${MASTER_PORT}"

############################ 2. environment ###################################
export WORLD_SIZE=${SLURM_NNODES}
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export NCCL_DEBUG=INFO # Changed to INFO for more verbose NCCL logs during testing
export PYXIS_VERBOSE=3
export TORCH_DISTRIBUTED_TIMEOUT=1800
export HF_DATASETS_CACHE="${CONT_JOBDIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
# export FLASH_ATTENTION_FORCE_CUDA=1 # H100 has flash attention 2 by default
export PYTHONUNBUFFERED=1

############################ 3. python CLI ####################################
# --- Training Script Arguments ---
# Common arguments
BASE_TRAIN_ARGS="--output_dir=${CONT_JOBDIR}/checkpoints \
  --processed_dataset_path=${CONT_DATADIR} \
  --batch_size_per_device=4 \
  --learning_rate=6e-5 \
  --num_epochs=3 \
  --amp_precision_mode=bf16"
# --- FP8 torchao specific arguments (enable as needed for testing) ---
# To test FSDP2 without FP8: Do not include --use_fp8_torchao
# To test FSDP2 with FP8: Include --use_fp8_torchao and other --float8_* flags
# Example for enabling FP8 with tensorwise recipe and FSDP all-gather:
FP8_TORCHAO_ARGS="--use_fp8_torchao \
   --float8_recipe_name tensor \
   --float8_enable_fsdp_all_gather \
   --float8_precompute_dynamic_scale \
   --float8_force_recompute_weight_bwd"

# --- QLoRA arguments (enable if testing QLoRA, but be cautious with FP8 torchao) ---
QLORA_ARGS="--use_qlora" # Note: --use_fp8_torchao with --use_qlora is experimental/discouraged

# --- FSDP2 specific arguments ---
FSDP2_ARGS="--reshard_after_forward" # Example: Enable ZeRO-3 like behavior

# Combine arguments
FINETUNE_ARGS="${BASE_TRAIN_ARGS} ${FP8_TORCHAO_ARGS} ${QLORA_ARGS} ${FSDP2_ARGS}"

# Note: The rdzv_endpoint below is hardcoded. 
TORCHRUN_CMD="\
torchrun \
    --nnodes=${SLURM_NNODES} \
    --max-restarts=0 \
    --nproc-per-node=1 \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint="worker-0.soperator-worker-svc.soperator.svc.cluster.local:${MASTER_PORT}" \
    function-finetune-torchao-fsdp2.py ${FINETUNE_ARGS}"

echo "--- TORCHRUN COMMAND ---"
echo "${TORCHRUN_CMD}"
echo "------------------------"

############################ 4. launch containers #############################
srun \
     --export=ALL,MASTER_IP,MASTER_PORT,PYXIS_VERBOSE,TORCH_DISTRIBUTED_TIMEOUT,HF_DATASETS_CACHE,TOKENIZERS_PARALLELISM,NCCL_DEBUG,PYTHONUNBUFFERED \
     --container-image="${IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir="/workspace" \
     bash -c "${TORCHRUN_CMD}"

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}
