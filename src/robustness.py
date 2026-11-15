"""Step 10. Robustness of the held-out settings to (A) lower frame rates (every 2nd / 3rd sample) and
(B) added spurious short closures (0.2-0.6 s). Settings are those chosen on the original data
(representative split, seed 0, 10% alarm-on cap) and are NOT re-tuned here (see robustness_recal.py)."""
import pickle, json
import numpy as np, pandas as pd
from config import CACHE, RESULTS
from core import *
import evaluate as E
from cv import folds

D = pickle.load(open(CACHE / 'metrics.pkl', 'rb')); C = D['configs']
res = pickle.load(open(CACHE / 'cv.pkl', 'rb'))[(120, 'drowsy')]
cv0 = res[(res['mode'] == 'cv5') & (res.acap == 0.1) & (res.seed == 0)]
SUBS = E.SUBS; fs = folds(0)
FAMS = ['stea', 'relax_p1', 'cusum', 'ewma', 'count30']; BUD = [2.5, 5.0, 8.0]
HELD = {}
for fam in FAMS:
    for b in BUD:
        r = cv0[(cv0.fam == fam) & (cv0.b == b)].iloc[0]; m = {}
        for te, c in zip(fs, r.chosen):
            for i in te: m[SUBS[i]] = C[c]
        HELD[(fam, b)] = m
need_stea = sorted({(c[1], c[2]) for (f, b), m in HELD.items() if f == 'stea' for c in m.values()})
need_cus = sorted({c[1] for (f, b), m in HELD.items() if f == 'cusum' for c in m.values()})
need_ew = sorted({c[1] for (f, b), m in HELD.items() if f == 'ewma' for c in m.values()})
need_cnt = sorted({c[1] for (f, b), m in HELD.items() if f == 'count30' for c in m.values()})
RAW = {p: load_participant(p) for p in SUBS}
lab = load_labels()


def build(p, t, dt, valid, closed, op):
    S = dict(t=t, dt=dt)
    S['perclos'] = {W: perclos_time(t, dt, valid, closed, W) for W in [30, 60, 180, 300]}
    run, ev = closure_runs(t, dt, valid, op); S['run'] = run; S['events'] = ev
    S['stea'] = {k: stea_signal(t, ev, k[0], k[1]) for k in need_stea}
    S['cusum'] = {k: cusum_signal(t, ev, k) for k in need_cus}
    S['ewma'] = {a: ewma_signal(t, ev, a) for a in need_ew}
    S['count'] = {(30, d): count_window(t, ev, 30, d) for d in need_cnt}
    segs = label_segments(lab, p, t[-1]); S['mask'] = eval_masks(t, segs, 120, 'drowsy')
    return S


def perturb(p, mode, level, rng):
    df = RAW[p]; t = df.t.values.copy(); valid = df.valid.values.copy()
    closed = df.closed80.values.copy(); op = df.open.values.astype(float).copy()
    if mode == 'spur' and level > 0:
        dur = t[-1]; n = rng.poisson(level * dur / 60)
        st = rng.uniform(0, dur, n); d = rng.uniform(0.2, 0.6, n)
        for s0, dd in zip(st, d):
            i0, i1 = np.searchsorted(t, s0), np.searchsorted(t, s0 + dd)
            sl = slice(i0, i1); ok = valid[sl]
            op[sl] = np.where(ok, np.minimum(op[sl], 0.10), op[sl]); closed[sl] = closed[sl] | ok
    if mode == 'fps' and level > 1:
        k = int(level); idx = np.arange(0, len(t), k)
        t, valid, closed, op = t[idx], valid[idx], closed[idx], op[idx]
    dt = np.diff(t, prepend=t[0])
    return build(p, t, dt, valid, closed, op)


def evaluate(SIG):
    out = {}
    cfgs = {('standard', 0): {p: ('base',) for p in SUBS}}; cfgs.update(HELD)
    for key, m in cfgs.items():
        rows = []
        for p in SUBS:
            S = SIG[p]; al = E.alarm_for(S, m[p]); pre, neg, fD = S['mask']
            rows.append(subject_metrics(S['t'], S['dt'], al, pre, neg, fD))
        out[key] = pool(rows)
    return out


if __name__ == '__main__':
  rows = []
  conds = [('clean', 0)] + [('spur', r) for r in [0.5, 1.0, 2.0]] + [('fps', k) for k in [2, 3]]
  for mode, level in conds:
      reps = 3 if mode == 'spur' else 1
      for rep in range(reps):
          rng = np.random.RandomState(100 + rep)
          SIG = {p: perturb(p, mode, level, rng) for p in SUBS}
          ev = evaluate(SIG)
          fps = np.median([1 / np.median(np.diff(S['t'])) for S in SIG.values()])
          for (fam, b), r in ev.items():
              rows.append(dict(mode=mode, level=level, rep=rep, fam=fam, b=b, fps=fps, **{k: r[k] for k in ['cov_n', 'cov_d', 'cov', 'far', 'act']}))
          print(mode, level, rep, round(fps, 1), flush=True)
  R = pd.DataFrame(rows); R.to_csv(RESULTS / 'out_robust.csv', index=False)
  S = R.groupby(['mode', 'level', 'fam', 'b'])[['cov', 'far', 'act', 'fps']].mean().reset_index()
  pd.set_option('display.width', 200)
  print(S.to_string())
