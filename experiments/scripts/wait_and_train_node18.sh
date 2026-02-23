#!/bin/bash
# Node 18: eval 종료 대기 후 SFA training 자동 시작
echo "[$(date)] [Node18] Waiting for eval processes (819825, 801959) to finish..."

while kill -0 819825 2>/dev/null || kill -0 801959 2>/dev/null; do
    sleep 60
    echo "[$(date)] Still waiting... (GPU 0: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0), GPU 1: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1))"
done

echo "[$(date)] Evals finished! Starting SFA training on 2 GPUs..."
sleep 10

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

cd /NetDisk/juyeon/DSA

/home/juyeon/miniconda3/envs/docmllm/bin/python -m torch.distributed.run     --nproc_per_node=2     --master_addr=localhost     --master_port=29501     experiments/scripts/03_sfa_finetune.py     --mode train     --model_path /NetDisk/j_son/internvl_35     --train_data "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train"     --output_dir experiments/results/03_sfa_train_node18     --epochs 3 --lr 2e-5 --batch_size 4 --grad_accum 4     2>&1 | tee experiments/results/03_sfa_train_node18/train.log

echo "[$(date)] [Node18] Training complete!"
