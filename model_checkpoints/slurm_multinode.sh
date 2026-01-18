#!/bin/bash
#SBATCH --job-name=neutts-1.5b-v2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Multi-Node Training: 4 nodes x 8 H100 GPUs = 32 GPUs
# =============================================================================
#
# Submit: sbatch slurm_multinode.sh
# Monitor: squeue -u $USER
# Cancel: scancel <job_id>
#
# =============================================================================

set -e

# Create logs directory
mkdir -p logs

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 8))"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Master: $SLURM_NODELIST"
echo "=============================================="

# Network settings for NCCL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_GID_INDEX=3

# Get master address (first node)
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Paths
WORK_DIR="/scratch/vikram.solanki/workspace/vs/neutts"
CONFIG="model_checkpoints/config_1.5b_multinode.yaml"

cd $WORK_DIR

# Launch with torchrun on each node
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    model_checkpoints/finetune_1.5b_hf.py $CONFIG

echo "Training complete!"
