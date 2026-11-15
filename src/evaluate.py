"""Step 2. Alarm rules and per-participant metrics for every candidate configuration -> cache/metrics.pkl.

Family names used in the code:
  base          combined standard criteria (PERCLOS 1/3/5 min >= 8/9/10 % or ECD >= 1 s)
  stea          SCA added to the combined standard criteria (proposed rule)
  stea_only     SCA alone (not reported in the paper)
  relax_p1      lower 1-min PERCLOS threshold          relax_multi  lower 1-, 3- and 5-min thresholds
  relax_p30     lower 30-s PERCLOS threshold           relax_ecd    ECD threshold below 1 s
  count30       closure count in the last 30 s         cusum / ewma CUSUM / EWMA of closure duration
"""
import pickle, time
import numpy as np, pandas as pd
from config import CACHE
from core import *

SIG = pickle.load(open(CACHE / 'signals.pkl', 'rb'))
SUBS = sorted(SIG.keys(), key=lambda x: int(x[1:]))
PTH = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
ECD_TH = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 3.0]
H_GRID = np.round(np.exp(np.linspace(np.log(0.05), np.log(8.0), 36)), 4)
CNT_K = list(range(1, 13))
CUS_H = np.round(np.exp(np.linspace(np.log(0.5), np.log(40), 20)), 3)
EW_H = [0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8]


def ge(x, th):
    return np.where(np.isnan(x), False, x >= th)


def base_alarm(S):
    P = S['perclos']
    return ge(P[60], 8) | ge(P[180], 9) | ge(P[300], 10) | (S['run'] >= 1.0)


def alarm_for(S, cfg):
    """cfg: tuple (family, params...)"""
    fam = cfg[0]; P = S['perclos']; t = S['t']
    if fam == 'perclos':            # single window, no base
        _, W, th = cfg; return ge(P[W], th)
    if fam == 'ecd':
        return S['run'] >= cfg[1]
    if fam == 'base':
        return base_alarm(S)
    if fam == 'perclos_union':      # NHTSA early-warning union
        return ge(P[60], 8) | ge(P[180], 9) | ge(P[300], 10)
    if fam == 'perclos_full':
        return ge(P[60], 12) | ge(P[180], 12) | ge(P[300], 12)
    b = base_alarm(S)
    if fam == 'relax_p1':
        return b | ge(P[60], cfg[1])
    if fam == 'relax_multi':
        a = cfg[1]; return b | ge(P[60], a) | ge(P[180], a + 1) | ge(P[300], a + 2)
    if fam == 'relax_p30':
        return b | ge(P[30], cfg[1])
    if fam == 'relax_pw':           # arbitrary short window
        _, W, th = cfg; return b | ge(P[W], th)
    if fam == 'relax_ecd':
        return b | (S['run'] >= cfg[1])
    if fam == 'count30':
        _, dmin, k = cfg; return b | (S['count'][(30, dmin)] >= k)
    if fam == 'stea':
        _, lam, d0, p, h = cfg
        return b | persist(S['stea'][(lam, d0)] >= h, t, p)
    if fam == 'stea_only':
        _, lam, d0, p, h = cfg
        return persist(S['stea'][(lam, d0)] >= h, t, p)
    if fam == 'cusum':
        _, k, h = cfg; return b | (S['cusum'][k] >= h)
    if fam == 'ewma':
        _, a, h = cfg; return b | (S['ewma'][a] >= h)
    raise ValueError(cfg)


def configs():
    C = [('base',), ('perclos_union',), ('perclos_full',)]
    for W in [5, 10, 15, 20, 30, 60, 120, 180, 300]:
        for th in PTH: C.append(('perclos', W, th))
    for th in ECD_TH: C.append(('ecd', th))
    for th in PTH: C.append(('relax_p1', th))
    for a in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7]: C.append(('relax_multi', a))
    for th in PTH: C.append(('relax_p30', th))
    for W in [5, 10, 15, 20]:
        for th in PTH: C.append(('relax_pw', W, th))
    for th in ECD_TH:
        if th < 1: C.append(('relax_ecd', th))
    for dmin in [0.03, 0.15, 0.30]:
        for k in CNT_K: C.append(('count30', dmin, k))
    for lam in [15, 30, 60, 120]:
        for d0 in [0.15, 0.20, 0.30]:
            for p in [0, 3]:
                for h in H_GRID: C.append(('stea', lam, d0, p, float(h)))
    for lam in [15, 30, 60, 120]:
        for d0 in [0.15, 0.20, 0.30]:
            for h in H_GRID: C.append(('stea_only', lam, d0, 0, float(h)))
    for k in [np.log(0.10), np.log(0.15), np.log(0.20), np.log(0.30)]:
        for h in CUS_H: C.append(('cusum', k, float(h)))
    for a in [0.1, 0.2, 0.3]:
        for h in EW_H: C.append(('ewma', a, h))
    # longer SCA memories (appended in this order; the order only matters for exact tie-breaking)
    for lam in [180, 300, 600, 900]:
        for d0 in [0.15, 0.20, 0.30]:
            for p in [0, 3]:
                for h in H_GRID: C.append(('stea', lam, d0, p, float(h)))
            for h in H_GRID: C.append(('stea_only', lam, d0, 0, float(h)))
    return C


if __name__ == '__main__':
    C = configs(); print(len(C), 'configs')
    rows = []
    t0 = time.time()
    for ci, cfg in enumerate(C):
        for p in SUBS:
            S = SIG[p]; al = alarm_for(S, cfg)
            for (H, ep), (pre, neg, fD) in S['masks'].items():
                m = subject_metrics(S['t'], S['dt'], al, pre, neg, fD)
                m.update(cfg=ci, participant=p, H=H, ep=ep); rows.append(m)
        if ci % 100 == 0: print(ci, round(time.time() - t0), flush=True)
    M = pd.DataFrame(rows)
    pickle.dump(dict(configs=C, metrics=M), open(CACHE / 'metrics.pkl', 'wb'))
    print('done', time.time() - t0)
