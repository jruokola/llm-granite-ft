#!/bin/bash
#SBATCH --job-name=fc-qlora-h100
#SBATCH --nodes=2                # one H100 per node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=23:00:00
#SBATCH --output=/root/slurm_logs/fc_%j.log
#SBATCH --export=ALL             # propagate env to Pyxis

############################ 0. constants & job paths #########################
# Tested finetune-39:latest 
# function-finetune-fixed.py
# QLoRa+FSDP+AMP works - lr:6e-5, batch_size_per_device=16
# QLoRa+DDP+AMP works - lr:6e-5, batch_size_per_device=16
# QLoRa+FSDP+AMP+FP8 doesn't work - unstability and nan loss with lr:6e-5, batch_size_per_device=16
# QLoRa+DDP+AMP+FP8 doesn't work - unstability and nan loss with lr:6e-5, batch_size_per_device=16
# QLoRa+FSDP+FP8 works - lr: 6e-5, batch_size_per_device=2
# QLoRa+DDP+FP8 works - lr: 6e-5, batch_size_per_device=2
# LoRa+FSDP+AMP+FP8 works -lr: 6e-5, batch_size_per_device=2
# LoRa+FSDP+FP8 works -  lr: 6e-5, batch_size_per_device=2
# LoRa+DDP+AMP+FP8 doesn't work - unstability and nan loss with lr:6e-5, batch_size_per_device=2
IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-39:latest"
JOB_DIR="/slurm_jobs/${SLURM_JOB_ID}"
COORD_DIR="${JOB_DIR}/coord"
HOST_JOBDIR="/mnt/jail/${JOB_DIR}"
CONT_JOBDIR="/job_data"
CONT_DATADIR="/processed_datasets/glaive_fc_v2/"

mkdir -p "${COORD_DIR}"
chmod -R 777 "${JOB_DIR}"
############################ 1. rendez-vous params ############################
CONTAINER_MOUNTS="${JOB_DIR}:${CONT_JOBDIR}:rw"
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_IP=$(getent hosts "${MASTER_NODE}" | awk '{print $1}')
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))               # free high port

echo "HEAD NODE: ${MASTER_NODE} ${MASTER_IP}:${MASTER_PORT}"
echo "IMAGE  ${IMAGE}"
echo "CONTAINER_MOUNTS ${CONTAINER_MOUNTS}"

############################ 2. environment ###################################
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export NCCL_DEBUG=INFO
export PYXIS_VERBOSE=3
export TORCH_DISTRIBUTED_TIMEOUT=1800
export HF_DATASETS_CACHE=${CONT_JOB_DIR}/hf_cache
export TOKENIZERS_PARALLELISM=false
export FLASH_ATTENTION_FORCE_CUDA=1
export PYTHONBUFFERED=1

############################ 3. python CLI ####################################
FINETUNE_ARGS="--output_dir=${CONT_JOBDIR}/checkpoints \
  --processed_dataset_path=${CONT_DATADIR} \
  --batch_size_per_device=16 \
  --learning_rate=6e-5 \
  --use_qlora"

TORCHRUN_CMD="\
torchrun \
    --nnodes=${SLURM_NNODES} \
    --max-restarts=0 \
    --nproc-per-node=1 \
    --rdzv-id=${SLURM_JOB_ID} \
    --rdzv-backend=c10d \
    --rdzv-endpoint="worker-0.soperator-worker-svc.soperator.svc.cluster.local:${MASTER_PORT}" \
    function-finetune-fixed.py ${FINETUNE_ARGS}"

############################ 4. launch containers #############################
## ─── Launch with srun + env:// init ──────────────────────────────────────────
srun \
     --export=ALL,PYTHONBUFFERED,MASTER_IP,MASTER_PORT,PYXIS_VERBOSE,TORCH_DISTRIBUTED_TIMEOUT,HF_DATASETS_CACHE,TOKENIZERS_PARALLELISM,FLASH_ATTENTION_FORCE_CUDA,WORLD_SIZE,LOCAL_RANK,RANK \
     --container-image=${IMAGE} \
     --container-mounts=${CONTAINER_MOUNTS} \
     --container-workdir="/workspace" \
     bash -c "${TORCHRUN_CMD}"

EXIT_CODE=$?
echo "Job finished with exit code ${EXIT_CODE}"
exit ${EXIT_CODE}

