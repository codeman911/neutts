#!/bin/bash
# =============================================================================
# DeepSpeed Multi-Node Launch Script
# 4 nodes x 8 H100 GPUs = 32 GPUs with ZeRO-3
# =============================================================================
#
# Usage:
#   bash launch_deepspeed.sh
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
WORK_DIR="/scratch/vikram.solanki/workspace/vs/neutts"
CONFIG="model_checkpoints/config_1.5b_multinode.yaml"
DS_CONFIG="model_checkpoints/ds_zero3.json"
HOSTFILE="model_checkpoints/hostfile"

cd $WORK_DIR

# =============================================================================
# CREATE HOSTFILE
# =============================================================================
# Format: hostname slots=<num_gpus>
cat > $HOSTFILE << EOF
dgx-41 slots=8
dgx-42 slots=8
dgx-43 slots=8
dgx-44 slots=8
EOF

echo "Hostfile created: $HOSTFILE"
cat $HOSTFILE

# =============================================================================
# ENVIRONMENT
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# =============================================================================
# LAUNCH WITH DEEPSPEED
# =============================================================================
echo "=============================================="
echo "DeepSpeed Multi-Node Training"
echo "Nodes: 4 x 8 GPUs = 32 total"
echo "ZeRO Stage: 3"
echo "=============================================="

deepspeed --hostfile=$HOSTFILE \
    --master_addr=dgx-41 \
    --master_port=29500 \
    model_checkpoints/finetune_1.5b_hf.py \
    $CONFIG \
    --deepspeed $DS_CONFIG

echo "Training complete!"
