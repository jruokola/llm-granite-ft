#!/bin/bash
###############################################################################
#  Function-calling fine-tune on 2 × H100 nodes (single-GPU per node)
###############################################################################
#SBATCH --job-name=function-finetune              # job name
#SBATCH --nodes=2                                 # 2 nodes
#SBATCH --ntasks-per-node=1                       # 1 task  (=1 container) per node
#SBATCH --gres=gpu:1                              # 1 H100 per node
#SBATCH --cpus-per-task=8                         # host threads for dataloader / NCCL
#SBATCH --mem=180G                                # RAM per node
#SBATCH --time=23:59:00                           # wall-clock limit
#SBATCH --output=/root/slurm_logs/function_finetune_%j.log
#SBATCH --export=ALL
#SBATCH --kill-on-bad-exit=1                      # cancel job if any task dies   [oai_citation:0‡slurm.schedmd.com](https://slurm.schedmd.com/srun.html?utm_source=chatgpt.com)

###############################################################################
# 0.  Cluster / container constants
###############################################################################
CONTAINER_IMAGE="cr.eu-north1.nebius.cloud/e00hdcpaq6azg81mmp/finetune-5:latest"
JOB_DIR="/root/slurm_jobs/function_finetune/${SLURM_JOB_ID}"
CHECKPOINTS_DIR="${JOB_DIR}/checkpoints"
LOGS_DIR="${JOB_DIR}/logs"
mkdir -p "${CHECKPOINTS_DIR}" "${LOGS_DIR}"
chmod -R 777 "${JOB_DIR}"

# expose JOB_DIR inside the container at /job_data
CONTAINER_MOUNTS="${JOB_DIR}:/job_data"
CONTAINER_WORKDIR="/workspace"

###############################################################################
# 1.  Performance-tuning env vars  (all tasks inherit via --export=ALL)
###############################################################################
export NCCL_DEBUG=INFO                              # helpful diagnostics  [oai_citation:1‡Dongda's homepage](https://dongdongbh.tech/blog/torchrun/?utm_source=chatgpt.com)
export NCCL_COLLNET_ENABLE=0                        # disable CollNet on ≤ 2 GPUs  [oai_citation:2‡Dongda's homepage](https://dongdongbh.tech/blog/torchrun/?utm_source=chatgpt.com)
export NCCL_NET_GDR_LEVEL=2                         # prefer NVLink/IB RDMA  [oai_citation:3‡Dongda's homepage](https://dongdongbh.tech/blog/torchrun/?utm_source=chatgpt.com)
export NCCL_IB_HCA=$(ibv_devinfo -l | head -n1)     # pick the first HCA   [oai_citation:4‡swaglu.com](https://swaglu.com/llama-405b-vllm-slurm-multinode/?utm_source=chatgpt.com)
export NCCL_SOCKET_IFNAME=^docker0,lo               # skip docker / loopback  [oai_citation:5‡GitHub](https://github.com/NVIDIA/pyxis?utm_source=chatgpt.com)
export CUDA_DEVICE_MAX_CONNECTIONS=1                # fewer GPU peer queues improves latency  [oai_citation:6‡Markaicode](https://markaicode.com/pytorch-3-nvidia-h100-optimization-guide/?utm_source=chatgpt.com)
export TORCH_DISTRIBUTED_TIMEOUT=600                # 10 min rendez-vous timeout  [oai_citation:7‡PyTorch Forums](https://discuss.pytorch.org/t/distributeddataparallel-socket-timeout/174192?utm_source=chatgpt.com)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}       # use all allocated CPU cores
export HF_DATASETS_CACHE="/job_data/hf_cache"       # keep cache on fast SSD  [oai_citation:8‡Hugging Face](https://huggingface.co/docs/datasets/cache?utm_source=chatgpt.com)
export FLASH_ATTENTION_FORCE=1                      # ensure FA2 kernels on H100  [oai_citation:9‡manpages.ubuntu.com](https://manpages.ubuntu.com/manpages/trusty/man1/srun.1.html?utm_source=chatgpt.com)

###############################################################################
# 2.  Build CLI for the fine-tune script (passed unchanged to torchrun)
###############################################################################
FINETUNE_CLI_ARGS="\
 --output_dir=/job_data/checkpoints \
 --processed_dataset_path=/processed_datasets/glaive_fc_v2 \
 --use_qlora \
 --use_fp8"

echo "FINETUNE_CLI_ARGS: ${FINETUNE_CLI_ARGS}"

###############################################################################
# 3.  Decide rendez-vous endpoint (rank-0 node’s IP + random high port)
###############################################################################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

###############################################################################
# 4.  Launch: one container per node, torchrun inside each container
###############################################################################
srun --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${CONTAINER_MOUNTS}" \
     --container-workdir="${CONTAINER_WORKDIR}" \
     --label --verbose \
     torchrun \
       --nnodes=${SLURM_NNODES} \
       --nproc_per_node=${SLURM_GPUS_ON_NODE} \
       --rdzv_backend=c10d \
       --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
       /workspace/function-finetune-fixed.py \
       ${FINETUNE_CLI_ARGS}

###############################################################################
# 5.  Final status
###############################################################################
EXIT_CODE=$?
echo "============================================================"
echo "Job ${SLURM_JOB_ID} finished with exit code ${EXIT_CODE}"
echo "============================================================"
exit ${EXIT_CODE}