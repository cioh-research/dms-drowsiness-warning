# Short-closure accumulation (SCA) for earlier drowsiness warnings

Code and de-identified data for

> Oh, C., Kim, W., Lee, M., Pitts, B.J. *Warning drowsy drivers before microsleeps: short eyelid closures
> complement PERCLOS and eye-closure duration criteria.* 

Camera-based driver monitoring systems usually warn on PERCLOS and on the duration of a single eye closure (ECD).
In a simulator study with 46 drivers, these criteria warned only about one-third of drivers within 2 min before
visible drowsiness onset, because what increased before onset were repeated sub-second closures. SCA accumulates
those closures with exponential decay over elapsed time and is OR-ed with the standard alarms, so it never cancels
or delays them.

## Repository layout

```
data/
  processed/Sxx.csv.gz      eye-opening / eye-state time series, one file per participant (random codes)
  annotations.csv           active / ambiguous / drowsy annotations on the same time axis
  participants_summary.csv  aggregate participant characteristics (Table 1)
  README.md                 data dictionary and terms of use
src/
  config.py                 paths (relative to the repository root)
  core.py                   PERCLOS, closure events, SCA and comparison signals; evaluation metrics
  sca_online.py             minimal streaming implementation of SCA (O(1) state)
  build_signals.py ... figures.py   analysis pipeline (see below)
results/                    tables and JSON produced by the pipeline
figures/                    figures produced by the pipeline
run_all.sh                  runs the whole pipeline
```

## Quick start

```bash
git clone https://github.com/cioh-research/dms-drowsiness-warning.git
cd dms-drowsiness-warning
python -m pip install -r requirements.txt
./run_all.sh            # about 1-3 h; ml.py and cv.py are the slowest steps
```

Individual steps can be run from `src/` (for example `python secondary.py`) once their inputs exist.
Intermediate files are written to `cache/` (not versioned).

## Pipeline and where each result appears in the paper

| Step | Script | Output | Paper |
|---|---|---|---|
| 1 | `build_signals.py` | `cache/signals.pkl` | Section 2.2, 2.5 |
| 2 | `evaluate.py` | `cache/metrics.pkl` (all candidate settings × participants) | Section 2.6 |
| 3 | `cv.py` | `cache/cv.pkl` (participant-wise 5-fold CV × 50, LOSO) | Section 2.7 |
| 4 | `secondary.py` | `out_table_standards.csv`, `out_pareto.csv`, `out_stats.csv`, `out_sensitivity.csv`, `out_lambda_profile.csv`, `out_mechanism.csv`, `out_summary.json` | Tables 2, 3, 6; Figs 3, 5, 6, 7 |
| 5 | `latency.py` | `out_latency.csv` | Fig. 4 |
| 6 | `ml.py` | `cache/ml.pkl` | Table 3 (learned models) |
| 7 | `ablation.py`, `ablation_cv.py` | `out_ablation.csv` | Table 4 |
| 8 | `simplified_sca.py` | `out_lite.csv` | Table 4 (simplified SCA) |
| 9 | `microsleep.py` | `out_microsleep.json` | Table 5 |
| 10–11 | `robustness.py`, `robustness_recal.py` | `out_robust.csv`, `out_robust_recal.csv` | Table 7 |
| – | `descriptives.py` | `descriptives.json` | Section 2 numbers |
| 12 | `make_results.py` | `results.json` | – |
| 13 | `figures.py` | `figures/*.png` | Figs 2–8, graphical abstract |

In the code the proposed rule is called `stea`; `base` is the combined standard criteria
(PERCLOS 1/3/5 min ≥ 8/9/10 % or ECD ≥ 1 s), and SCA is always added to it (`base OR SCA`).

## Using SCA

```python
from sca_online import SCA
sca = SCA(d0=0.2, lam=120.0, h=2.5, equal_weight=True)   # simplified SCA, about 5 false alarms/h in this study
for t, closure_duration in stream:                       # duration of a closure that ended at t, else None
    alarm = standard_alarm(t) or sca.update(t, closure_duration)
```

Closure events are defined on the normalised eye opening (start < 0.35, end ≥ 0.3675, ≥ 0.03 s); closures of
1 s or longer are left to the ECD criterion. Thresholds should be re-tuned on data from the target camera and
operating environment (Section 3.8 of the paper).

## Reproducibility

* Identical to the paper: all per-participant signals and the metrics of every candidate setting
  (`cache/signals.pkl`, `cache/metrics.pkl`), Table 2, Figs 3, 5 and 8, the descriptive statistics in Section 2,
  and the standard-criteria results in Tables 5–7.
* Cross-validation folds are drawn from the random participant codes, which are unrelated to the original
  participant order, so cross-validated values differ slightly from the paper
  (`results/comparison_with_manuscript.csv` lists both):
  * means over 50 repeats (Tables 3–5, Figs 6 and 7): within about 1 percentage point;
  * learned models (10 and 3 repeats): within about 4 percentage points;
  * values from a single representative split (Table 6, frame-rate rows of Table 7, Fig. 4): up to 4 participants;
  * spurious-closure rows of Table 7 also depend on the random insertion of closures.
* The fold assignment used for the paper is tied to internal participant numbers and is therefore not published.
  It can be provided on request; placed at `data/folds.csv` (columns `seed, fold, code`), it makes `cv.py` and the
  downstream results reproduce the paper exactly, except for the learned models and the spurious-closure rows,
  whose inner validation split and random insertion follow the participant order.

## Data protection

The study was approved by the Purdue University Institutional Review Board (IRB-2025-646). In accordance with the
IRB requirements, participant numbers, individual demographic information, dates and clock times, and facial video
are not shared. Raw eye-tracker records and uncorrected camera outputs are available from the corresponding author
(cioh@etri.re.kr) on reasonable request within the scope approved by the IRB.

## Citation

```bibtex
@article{Oh_SCA,
  author  = {Oh, Cheonin and Kim, Woojin and Lee, Myeongkyu and Pitts, Brandon J.},
  title   = {Warning drowsy drivers before microsleeps: short eyelid closures complement {PERCLOS} and eye-closure duration criteria},
  journal = {Accident Analysis and Prevention},
  note    = {submitted}
}
```

## License

Code: MIT (see `LICENSE`). Data: see `data/README.md`.
