"""Step 7a. Component variants of SCA (Table 4): per-participant metrics for every variant configuration."""
import pickle, time
from config import CACHE
import numpy as np, pandas as pd
from core import *

LAMS = [15, 30, 60, 120, 180, 300, 600, 900]
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
D0S = [0.15, 0.20, 0.30]
H_STD = np.round(np.exp(np.linspace(np.log(0.05), np.log(8.0), 36)), 4)
H_CNT = np.round(np.exp(np.linspace(np.log(0.2), np.log(40.0), 36)), 4)
H_EVT = np.round(np.exp(np.linspace(np.log(0.05), np.log(40.0), 36)), 4)


def marks(ev, d0, mode):
    d = ev[:, 2]
    if mode == 'noexcl':
        return np.clip(np.minimum(d, 1.0) - d0, 0, None)
    if mode == 'nomark':
        return ((d >= d0) & (d < 1.0)).astype(float)
    return np.clip(np.minimum(d, 1.0) - d0, 0, None) * (d < 1.0)


def exp_decay(t, e, w, lam):
    E = np.zeros(len(t))
    for wj, ej in zip(w, e):
        if wj <= 0: continue
        i = np.searchsorted(t, ej, side='left')
        if i < len(t): E[i:] += wj * np.exp(-(t[i:] - ej) / lam)
    return E


def boxcar(t, e, w, W):
    k = w > 0; e = e[k]; w = w[k]
    o = np.argsort(e); e = e[o]; cw = np.concatenate([[0], np.cumsum(w[o])])
    hi = np.searchsorted(e, t, side='right'); lo = np.searchsorted(e, t - W, side='right')
    return cw[hi] - cw[lo]


def event_decay(t, e, w, g):
    k = w > 0; e = e[k]; w = w[k]
    if len(e) == 0: return np.zeros(len(t))
    s = 0.0; v = []
    for wj in w: s = g * s + wj; v.append(s)
    v = np.array(v); idx = np.searchsorted(e, t, side='right') - 1
    return np.where(idx >= 0, v[np.clip(idx, 0, None)], 0.0)


def configs():
    C = []
    for fam, hs, d0s, keys in [('full', H_STD, D0S, LAMS), ('noexcl', H_STD, D0S, LAMS), ('nomark', H_CNT, D0S, LAMS),
                                ('boxcar', H_STD, D0S, LAMS), ('evdecay', H_EVT, D0S, GAMMAS), ('nod0', H_STD, [0.0], LAMS)]:
        for d0 in d0s:
            for k in keys:
                for p in [0, 3]:
                    for h in hs: C.append((fam, d0, k, p, float(h)))
    return C


if __name__ == '__main__':
    C = configs(); print(len(C), 'configs', flush=True)
    lab = load_labels(); rows = []; t0 = time.time()
    for p_ in participant_codes():
        df = load_participant(p_)
        t = df.t.values; dt = df.dt.values; valid = df.valid.values
        P = {W: perclos_time(t, dt, valid, df.closed80.values, W) for W in [60, 180, 300]}
        run, ev = closure_runs(t, dt, valid, df.open.values)
        ge = lambda x, th: np.where(np.isnan(x), False, x >= th)
        base = ge(P[60], 8) | ge(P[180], 9) | ge(P[300], 10) | (run >= 1.0)
        segs = label_segments(lab, p_, t[-1]); pre, neg, fD = eval_masks(t, segs, 120, 'drowsy')
        cache = {}
        for ci, (fam, d0, k, p, h) in enumerate(C):
            key = (fam, d0, k)
            if key not in cache:
                if len(ev) == 0: cache[key] = np.zeros(len(t))
                else:
                    mode = fam if fam in ('noexcl', 'nomark') else 'full'
                    w = marks(ev, d0, mode); e = ev[:, 1]
                    if fam == 'boxcar': cache[key] = boxcar(t, e, w, k)
                    elif fam == 'evdecay': cache[key] = event_decay(t, e, w, k)
                    else: cache[key] = exp_decay(t, e, w, k)
            al = base | persist(cache[key] >= h, t, p)
            m = subject_metrics(t, dt, al, pre, neg, fD); m.update(cfg=ci, participant=p_); rows.append(m)
        print(p_, round(time.time() - t0), flush=True)
    pickle.dump(dict(configs=C, metrics=pd.DataFrame(rows)), open(CACHE / 'ablation_metrics.pkl', 'wb'))
    print('done')
