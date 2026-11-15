#!/usr/bin/env bash
# Reproduce all analyses, tables and figures from the de-identified data.
# Total run time is roughly 1-3 h on a laptop; ml.py and cv.py are the slowest steps.
set -euo pipefail
cd "$(dirname "$0")/src"
run() { echo "== $1"; python "$1"; }
run build_signals.py      # 1  signals and evaluation masks          -> cache/signals.pkl
run evaluate.py           # 2  metrics for all candidate settings     -> cache/metrics.pkl
run cv.py                 # 3  participant-wise cross-validation      -> cache/cv.pkl
run secondary.py          # 4  Tables 2, 3, 6; Figs 3, 5, 6, 7 inputs  -> results/out_*.csv
run latency.py            # 5  first-warning times (Fig. 4)
run ml.py                 # 6  logistic regression / gradient boosting (Table 3)
run ablation.py           # 7a SCA component variants
run ablation_cv.py        # 7b Table 4
run simplified_sca.py     # 8  simplified SCA (Table 4, last row)
run microsleep.py         # 9  first-microsleep reference (Table 5)
run robustness.py         # 10 frame rate / spurious closures, settings unchanged
run robustness_recal.py   # 11 spurious closures, settings re-selected (Table 7)
run descriptives.py       #    descriptive numbers in Section 2
run make_results.py       # 12 results/results.json
run figures.py            # 13 Figs 2-8 and graphical abstract       -> figures/
run compare_manuscript.py #    results/comparison_with_manuscript.csv
