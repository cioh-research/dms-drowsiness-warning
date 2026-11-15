"""Paths used by all scripts. Everything is relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PROCESSED = DATA / 'processed'          # one CSV per participant (S01.csv.gz ...)
ANNOTATIONS = DATA / 'annotations.csv'  # drowsiness annotations on the same relative time axis
CACHE = ROOT / 'cache'                  # intermediate pickles (created on first run, not versioned)
RESULTS = ROOT / 'results'              # tables / json produced by the scripts
FIGURES = ROOT / 'figures'              # figures produced by the scripts

for _d in (CACHE, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)
