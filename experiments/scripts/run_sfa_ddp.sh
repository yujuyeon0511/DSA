#!/bin/bash
# ============================================================
# SFA Fine-tuning — Multi-Node DDP Launch Script
# ============================================================
# Node 17 (master, 192.168.0.229): GPU 0,1
# Node 18 (192.168.0.28):          GPU 0 only (GPU 1 메모리 부족)
#
# Usage:
#   # 이 스크립트를 Node 17 (master)에서 실행
#   bash experiments/scripts/run_sfa_ddp.sh
#
# 또는 단일 노드 2 GPU:
#   bash experiments/scripts/run_sfa_ddp.sh --local-only
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# ─── Configuration ──────────────────────────────────────
MASTER_ADDR="192.168.0.229"
MASTER_PORT=29500
NODE18_ADDR="192.168.0.28"
NODE18_USER="juyeon"

MODEL_PATH="/NetDisk/j_son/internvl_35"
TRAIN_DATA="/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train"
OUTPUT_DIR="experiments/results/03_sfa_train"
CONDA_ENV="docmllm"
CONDA_PYTHON="/home/juyeon/miniconda3/envs/${CONDA_ENV}/bin/python"

EPOCHS=3
LR=2e-5
BATCH_SIZE=2          # per GPU (줄여서 다른 프로세스와 공존)
GRAD_ACCUM=8          # effective = 2 * 8 * N_GPUs

# ─── Parse args ─────────────────────────────────────────
LOCAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --local-only) LOCAL_ONLY=true ;;
    esac
done

# ─── Launch ─────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

TRAIN_CMD="$CONDA_PYTHON experiments/scripts/03_sfa_finetune.py \
    --mode train \
    --model_path $MODEL_PATH \
    --train_data \"$TRAIN_DATA\" \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM"

if [ "$LOCAL_ONLY" = true ]; then
    # ─── Single Node, 2 GPUs ────────────────────────────
    echo "=== SFA DDP Training (Single Node, 2 GPUs) ==="
    echo "  Log: $LOG_FILE"

    $CONDA_PYTHON -m torch.distributed.run \
        --nproc_per_node=2 \
        --master_addr=localhost \
        --master_port=$MASTER_PORT \
        experiments/scripts/03_sfa_finetune.py \
        --mode train \
        --model_path "$MODEL_PATH" \
        --train_data "$TRAIN_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
        2>&1 | tee "$LOG_FILE"
else
    # ─── Multi-Node (Node 17: 2 GPUs + Node 18: 1 GPU) ─
    NNODES=2
    # Node 17: 2 GPUs, Node 18: 1 GPU → total 3 GPUs
    # torchrun handles heterogeneous GPU counts per node
    echo "=== SFA DDP Training (Multi-Node: 3 GPUs) ==="
    echo "  Master: $MASTER_ADDR:$MASTER_PORT"
    echo "  Node 17: 2 GPUs"
    echo "  Node 18: 1 GPU (CUDA_VISIBLE_DEVICES=0)"
    echo "  Log: $LOG_FILE"
    echo ""

    # Start Node 18 worker in background (1 GPU)
    echo "[Node 18] Starting worker..."
    ssh -o StrictHostKeyChecking=no ${NODE18_USER}@${NODE18_ADDR} \
        "cd $PROJECT_DIR && \
         CUDA_VISIBLE_DEVICES=0 \
         $CONDA_PYTHON -m torch.distributed.run \
             --nproc_per_node=1 \
             --nnodes=2 \
             --node_rank=1 \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             experiments/scripts/03_sfa_finetune.py \
             --mode train \
             --model_path '$MODEL_PATH' \
             --train_data '$TRAIN_DATA' \
             --output_dir '$OUTPUT_DIR' \
             --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM" \
        > "$OUTPUT_DIR/node18.log" 2>&1 &
    NODE18_PID=$!
    echo "  Node 18 PID: $NODE18_PID"

    # Start Node 17 master (2 GPUs)
    echo "[Node 17] Starting master..."
    $CONDA_PYTHON -m torch.distributed.run \
        --nproc_per_node=2 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        experiments/scripts/03_sfa_finetune.py \
        --mode train \
        --model_path "$MODEL_PATH" \
        --train_data "$TRAIN_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --epochs $EPOCHS --lr $LR --batch_size $BATCH_SIZE --grad_accum $GRAD_ACCUM \
        2>&1 | tee "$LOG_FILE"

    # Cleanup
    echo "[Cleanup] Waiting for Node 18..."
    wait $NODE18_PID 2>/dev/null || true
    echo "=== Training Complete ==="
fi
