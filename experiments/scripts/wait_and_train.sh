#!/bin/bash
# Qwen eval 종료 대기 후 SFA training 자동 시작
# Usage: nohup bash experiments/scripts/wait_and_train.sh &

echo "[$(date)] Waiting for Qwen eval processes (973383, 978981) to finish..."

while kill -0 973383 2>/dev/null || kill -0 978981 2>/dev/null; do
    sleep 60
    echo "[$(date)] Still waiting... (GPU 0: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0))"
done

echo "[$(date)] Qwen eval finished! Starting SFA training..."
sleep 10

# GPU memory check
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

cd /NetDisk/juyeon/DSA

# Use both GPUs with DDP (single node)
/home/juyeon/miniconda3/envs/docmllm/bin/python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=29500 \
    experiments/scripts/03_sfa_finetune.py \
    --mode train \
    --model_path /NetDisk/j_son/internvl_35 \
    --train_data "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train" \
    --output_dir experiments/results/03_sfa_train \
    --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4 \
    2>&1 | tee experiments/results/03_sfa_train/train.log

echo "[$(date)] Training complete!"
