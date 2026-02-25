"""
SCR Evaluation
===============
SCR 학습 결과를 평가합니다.
14_adat_eval.py와 동일한 구조 (ChartQA eval + Hallucination subset).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u experiments/scripts/17_scr_eval.py \
        --sfa_checkpoint experiments/results/08_scr/best.pth \
        --output_dir experiments/results/08_scr/eval \
        --label "SFA+ADAT+SCR(entropy)" \
        --device cuda:0
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

# Reuse ADAT eval entirely — same model loading, same evaluation logic
_adat_eval = import_module("14_adat_eval")

if __name__ == "__main__":
    _adat_eval.main()
