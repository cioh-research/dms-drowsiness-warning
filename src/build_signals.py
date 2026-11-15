"""Step 1. Compute all per-participant signals and evaluation masks -> cache/signals.pkl."""
import pickle, time
import numpy as np
from config import CACHE
from core import (load_all, load_labels, label_segments, perclos_time, closure_runs, stea_signal,
                  count_window, cusum_signal, ewma_signal, eval_masks)

WINDOWS = [5, 10, 15, 20, 30, 60, 120, 180, 300]          # PERCLOS windows (s)
LAMS = [15, 30, 60, 120, 180, 300, 600, 900]              # SCA decay time constants (s)
D0S = [0.15, 0.20, 0.30]                                  # SCA lower bound d0 (s)
CUSUM_K = [np.log(0.10), np.log(0.15), np.log(0.20), np.log(0.30)]
EWMA_A = [0.1, 0.2, 0.3]

if __name__ == '__main__':
    subs = load_all(); lab = load_labels()
    out = {}; t0 = time.time()
    for p, df in subs.items():
        t = df.t.values; dt = df.dt.values; valid = df.valid.values
        c80 = df.closed80.values; op = df.open.values
        dur = t[-1]
        segs = label_segments(lab, p, dur)
        S = dict(t=t, dt=dt, segs=segs, dur=dur)
        S['perclos'] = {W: perclos_time(t, dt, valid, c80, W) for W in WINDOWS}
        run, ev = closure_runs(t, dt, valid, op)
        S['run'] = run; S['events'] = ev
        S['stea'] = {(lam, d0): stea_signal(t, ev, lam, d0) for lam in LAMS for d0 in D0S}
        S['count'] = {(30, dmin): count_window(t, ev, 30, dmin) for dmin in [0.03, 0.15, 0.30]}
        S['cusum'] = {k: cusum_signal(t, ev, k) for k in CUSUM_K}
        S['ewma'] = {a: ewma_signal(t, ev, a) for a in EWMA_A}
        S['masks'] = {}
        for H in [60, 120, 180]:
            for ep in ['drowsy', 'nonactive']:
                S['masks'][(H, ep)] = eval_masks(t, segs, H, ep)
        out[p] = S
        print(p, len(ev), 'events', round(time.time() - t0, 1), 's', flush=True)
    pickle.dump(out, open(CACHE / 'signals.pkl', 'wb'), protocol=4)
    print('signals done')
