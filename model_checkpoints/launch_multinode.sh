#!/bin/bash
# =============================================================================
# Manual Multi-Node Launch Script (without SLURM)
# 4 nodes x 8 H100 GPUs = 32 GPUs
# =============================================================================
#
# Prerequisites:
# 1. SSH passwordless access between all nodes
# 2. Same paths/environment on all nodes
# 3. NCCL/InfiniBand configured
#
# Usage:
#   # On master node (node0):
#   bash launch_multinode.sh
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================
MASTER_ADDR="dgx-41"                       # First node hostname/IP
MASTER_PORT=29500
NNODES=4
GPUS_PER_NODE=8

# Node hostnames (edit to match your cluster)
NODES=(
    "dgx-41"
    "dgx-42"
    "dgx-43"
    "dgx-44"
)

# Paths
WORK_DIR="/scratch/vikram.solanki/workspace/vs/neutts"
CONFIG="model_checkpoints/config_1.5b_multinode.yaml"
SCRIPT="model_checkpoints/finetune_1.5b_hf.py"

# =============================================================================
# ENVIRONMENT
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# =============================================================================
# LAUNCH
# =============================================================================
echo "=============================================="
echo "Multi-Node Training: ${NNODES} nodes x ${GPUS_PER_NODE} GPUs"
echo "Total GPUs: $((NNODES * GPUS_PER_NODE))"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Nodes: ${NODES[*]}"
echo "=============================================="

# Launch on each node
for ((i=0; i<${NNODES}; i++)); do
    NODE=${NODES[$i]}
    
    if [ "$NODE" == "$MASTER_ADDR" ] && [ "$i" -eq 0 ]; then
        # Local launch on master
        echo "Launching on master node: $NODE (local)"
        cd $WORK_DIR
        torchrun \
            --nnodes=$NNODES \
            --nproc_per_node=$GPUS_PER_NODE \
            --node_rank=$i \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            $SCRIPT $CONFIG &
    else
        # Remote launch via SSH
        echo "Launching on worker node: $NODE (ssh)"
        ssh $NODE "cd $WORK_DIR && \
            NCCL_DEBUG=INFO \
            NCCL_IB_DISABLE=0 \
            torchrun \
                --nnodes=$NNODES \
                --nproc_per_node=$GPUS_PER_NODE \
                --node_rank=$i \
                --master_addr=$MASTER_ADDR \
                --master_port=$MASTER_PORT \
                $SCRIPT $CONFIG" &
    fi
done

echo "Waiting for all nodes..."
wait

echo "Training complete!"
