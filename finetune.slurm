#!/bin/bash
#SBATCH --job-name=granite_ft
#SBATCH --nodes=2
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

module load cuda/12.3     # from GPU-Operator on the node

torchrun --nnodes=${SLURM_JOB_NUM_NODES} \
         --nproc_per_node=1 \
         finetune.py