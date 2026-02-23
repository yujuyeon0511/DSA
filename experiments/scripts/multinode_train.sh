#!/bin/bash
# Multi-node DDP Training: 2 nodes × 2 GPUs = 4 GPUs total
# Node 17 (master): 192.168.0.229
# Node 18 (worker): 192.168.0.28
#
# Usage: bash experiments/scripts/multinode_train.sh
# This script launches torchrun on BOTH nodes via SSH.

set -e

MASTER_ADDR="192.168.0.229"
MASTER_PORT=29500
NNODES=2
NPROC_PER_NODE=2
PYTHON="/home/juyeon/miniconda3/envs/docmllm/bin/python"
WORK_DIR="/NetDisk/juyeon/DSA"
SCRIPT="${WORK_DIR}/experiments/scripts/03_sfa_finetune.py"
OUTPUT_DIR="${WORK_DIR}/experiments/results/03_sfa_train"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}"

mkdir -p "${LOG_DIR}"

TRAIN_ARGS="--mode train \
    --model_path /NetDisk/j_son/internvl_35 \
    --train_data '/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train' \
    --output_dir ${OUTPUT_DIR} \
    --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4"

TORCHRUN_ARGS="--nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT}"

echo "=== Multi-Node DDP Training (${NNODES} nodes × ${NPROC_PER_NODE} GPUs) ==="
echo "  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Timestamp: ${TIMESTAMP}"
echo "  Log: ${LOG_DIR}/multinode_train_${TIMESTAMP}.log"
echo ""

# --- Launch Node 18 (worker, node_rank=1) first ---
echo "[$(date)] Launching worker on Node 18 (192.168.0.28)..."
ssh 192.168.0.28 "NCCL_SOCKET_IFNAME=ens3 \
    NCCL_DEBUG=INFO \
    ${PYTHON} -m torch.distributed.run \
        ${TORCHRUN_ARGS} \
        --node_rank=1 \
        ${SCRIPT} ${TRAIN_ARGS} \
    " > "${LOG_DIR}/node18_${TIMESTAMP}.log" 2>&1 &
NODE18_PID=$!
echo "  Node 18 SSH PID: ${NODE18_PID}"

sleep 3

# --- Launch Node 17 (master, node_rank=0) ---
echo "[$(date)] Launching master on Node 17 (local)..."
cd "${WORK_DIR}"
NCCL_SOCKET_IFNAME=ens3 \
NCCL_DEBUG=INFO \
${PYTHON} -m torch.distributed.run \
    ${TORCHRUN_ARGS} \
    --node_rank=0 \
    ${SCRIPT} ${TRAIN_ARGS} \
    2>&1 | tee "${LOG_DIR}/multinode_train_${TIMESTAMP}.log"

MASTER_EXIT=$?

echo ""
echo "[$(date)] Master node finished with exit code: ${MASTER_EXIT}"

# Wait for worker
wait ${NODE18_PID} 2>/dev/null
WORKER_EXIT=$?
echo "[$(date)] Worker node finished with exit code: ${WORKER_EXIT}"

if [ ${MASTER_EXIT} -eq 0 ] && [ ${WORKER_EXIT} -eq 0 ]; then
    echo "[$(date)] Training completed successfully!"
else
    echo "[$(date)] Training failed. Check logs:"
    echo "  Master: ${LOG_DIR}/multinode_train_${TIMESTAMP}.log"
    echo "  Worker: ${LOG_DIR}/node18_${TIMESTAMP}.log"
fi
