# Data

All files are de-identified. Participants are labelled with random codes (`S01`–`S46`) that were assigned in a
random order and are unrelated to the internal participant numbers. Calendar dates, clock times, individual
demographic information and facial video are not included. Aggregate participant characteristics are in
`participants_summary.csv` (Table 1 of the paper).

## `processed/Sxx.csv.gz` — one file per participant (46 files)

Camera-based eye-state time series (about 20.8 Hz) after the eye-tracker-assisted correction described in
Section 2.2 of the paper.

| Column | Description |
|---|---|
| `t` | Time since the first sample of the recording (s) |
| `eye_opening` | Normalised eye opening, mean of the two eyes (FaceMesh-V2 / FaceAnalyzer); one eye if only one was valid |
| `valid` | 1 if the sample has a usable eye state, 0 otherwise (treated as missing) |
| `closed` | 1 if the eyelids covered at least 80% of the pupil (mean opening < 0.20; PERCLOS closure state) |
| `source` | Origin of the eye values: 0 camera; 1 replaced with an open-eye value because the time-matched eye tracker tracked the pupil of the same eye (camera value missing, or < 0.35 for >= 0.3 s); 2 initial interval judged alert by two researchers and set to open; -1 unknown |

PERCLOS (time-weighted, 80% valid-time rule) and closure events (start < 0.35, end >= 0.3675, >= 0.03 s, not
bridged across gaps) are computed from these columns by `src/core.py`.

## `annotations.csv` — drowsiness annotations

| Column | Description |
|---|---|
| `code` | Participant code |
| `t` | Start of the annotated state, on the same time axis as `processed/` (s) |
| `state` | `active` (alert), `ambiguous` or `drowsy`; a state lasts until the next row of the same participant |

Drowsiness onset is the first `drowsy` row of a participant (38 participants).

## Terms of use

The data are shared to allow verification and reuse for research. Users must not attempt to re-identify
participants. Please cite the paper when using the data.
