#!/bin/bash
# =============================================================================
# DSAVE / SFA 전체 실험 파이프라인
# =============================================================================
# Usage:
#   conda activate docmllm
#   cd /NetDisk/juyeon/DSA
#   bash experiments/scripts/run_all.sh [step]
#
# Steps:
#   0 - Baseline evaluation
#   1 - Density estimator training + visualization
#   2 - SFA module test
#   3 - Attention entropy analysis (baseline)
#   4 - Hallucination analysis (baseline)
#   5 - Token efficiency plot (placeholder)
#   all - Run everything sequentially
# =============================================================================

set -e
export PYTHONPATH=/NetDisk/juyeon/DSA:$PYTHONPATH
PYTHON=/home/juyeon/miniconda3/envs/docmllm/bin/python
MODEL=/NetDisk/j_son/internvl_35
RESULTS=/NetDisk/juyeon/DSA/experiments/results
SCRIPTS=/NetDisk/juyeon/DSA/experiments/scripts

STEP=${1:-all}

echo "============================================"
echo "DSAVE Experiment Pipeline"
echo "Step: $STEP"
echo "Model: $MODEL"
echo "Results: $RESULTS"
echo "============================================"

# ─── Step 0: Baseline Evaluation ───
if [[ "$STEP" == "0" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 0] Baseline Evaluation — ChartQA"
    echo "============================================"
    $PYTHON $SCRIPTS/00_baseline_eval.py \
        --model_path $MODEL \
        --output_dir $RESULTS/00_baseline \
        --benchmarks chartqa \
        --max_samples 200
    echo "[Step 0] Done."
fi

# ─── Step 1: Density Estimator ───
if [[ "$STEP" == "1" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 1] Density Estimator Training"
    echo "============================================"
    $PYTHON $SCRIPTS/01_density_estimator.py \
        --mode train \
        --data_dirs \
            "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/train/png" \
            /NetDisk/juyeon/train/dvqa/images \
        --output_dir $RESULTS/01_density \
        --epochs 10 \
        --batch_size 64 \
        --max_images 20000

    echo ""
    echo "[Step 1b] Density Map Visualization"
    $PYTHON $SCRIPTS/01_density_estimator.py \
        --mode visualize \
        --checkpoint $RESULTS/01_density/best.pth \
        --data_dirs \
            "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png" \
        --output_dir $RESULTS/01_density/visualizations
    echo "[Step 1] Done."
fi

# ─── Step 2: SFA Module Test ───
if [[ "$STEP" == "2" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 2] SFA Module Unit Test"
    echo "============================================"
    $PYTHON $SCRIPTS/02_sfa_module.py
    echo "[Step 2] Done."
fi

# ─── Step 3: SFA Integration Test ───
if [[ "$STEP" == "3" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 3] SFA Integration Test"
    echo "============================================"
    $PYTHON $SCRIPTS/03_sfa_integration.py \
        --mode test \
        --model_path $MODEL
    echo "[Step 3] Done."
fi

# ─── Step 4: Attention Entropy Analysis ───
if [[ "$STEP" == "4" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 4] Attention Entropy Analysis"
    echo "============================================"
    $PYTHON $SCRIPTS/04_attention_analysis.py \
        --mode entropy \
        --model_path $MODEL \
        --data_dir "/NetDisk/juyeon/train/chartQA/ChartQA Dataset/test/png" \
        --output_dir $RESULTS/04_analysis \
        --max_samples 100
    echo "[Step 4] Done."
fi

# ─── Step 5: Hallucination Analysis ───
if [[ "$STEP" == "5" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 5] Hallucination Analysis"
    echo "============================================"
    $PYTHON $SCRIPTS/04_attention_analysis.py \
        --mode hallucination \
        --model_path $MODEL \
        --output_dir $RESULTS/04_analysis \
        --max_samples 200
    echo "[Step 5] Done."
fi

# ─── Step 6: Token Efficiency Plot ───
if [[ "$STEP" == "6" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 6] Token Efficiency Plot (placeholder)"
    echo "============================================"
    $PYTHON $SCRIPTS/04_attention_analysis.py \
        --mode token_efficiency \
        --output_dir $RESULTS/04_analysis
    echo "[Step 6] Done."
fi

echo ""
echo "============================================"
echo "All requested steps complete."
echo "Results: $RESULTS/"
echo "============================================"
