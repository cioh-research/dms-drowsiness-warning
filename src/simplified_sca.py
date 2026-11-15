"""Step 8. Simplified SCA: equal-weight, time-decayed count of closures with d0 <= d < 1 s. d0 and lambda fixed, p = 0;
only the threshold h is selected by subject-wise CV (50 repeats) under the same dual budget."""
import pickle, json
from config import RESULTS
import numpy as np, pandas as pd
from core import *
from evaluate import SIG, SUBS, base_alarm
from cv import folds

H_GRID = np.round(np.exp(np.linspace(np.log(0.2), np.log(20), 45)), 3)


def lite_signal(t, ev, lam, d0):
    E = np.zeros(len(t))
    if len(ev) == 0: return E
    m = (ev[:, 2] >= d0) & (ev[:, 2] < 1.0)
    for ej in ev[m, 1]:
        i = np.searchsorted(t, ej)
        if i < len(t): E[i:] += np.exp(-(t[i:] - ej) / lam)
    return E


rows = []; summ = {}
for d0 in [0.2]:
    for lam in [30, 60, 120, 300]:
        hit = np.full((len(H_GRID), len(SUBS)), np.nan); fa = np.zeros(hit.shape); neg = np.zeros(hit.shape); act = np.zeros(hit.shape)
        for j, p in enumerate(SUBS):
            S = SIG[p]; t = S['t']; sig = lite_signal(t, S['events'], lam, d0); b = base_alarm(S); pre, ng, fD = S['masks'][(120, 'drowsy')]
            for i, h in enumerate(H_GRID):
                m = subject_metrics(t, S['dt'], b | (sig >= h), pre, ng, fD)
                hit[i, j] = m['hit']; fa[i, j] = m['fa']; neg[i, j] = m['neg_time']; act[i, j] = m['act_time']
        for bud in [2.5, 5.0, 8.0]:
            covs, fars, acts, hs = [], [], [], []
            for seed in range(50):
                hh = np.full(len(SUBS), np.nan); f = np.zeros(len(SUBS)); n_ = np.zeros(len(SUBS)); a_ = np.zeros(len(SUBS))
                for te in folds(seed):
                    tr = np.setdiff1d(np.arange(len(SUBS)), te)
                    cov = np.nanmean(hit[:, tr], 1); far = fa[:, tr].sum(1) / (neg[:, tr].sum(1) / 3600); ac = act[:, tr].sum(1) / neg[:, tr].sum(1)
                    feas = (far <= bud) & (ac <= 0.10)
                    i = np.where(feas)[0][np.argmax(cov[feas])] if feas.any() else np.argmin(far)
                    # among equal coverage pick the highest threshold (fewest alarms)
                    if feas.any():
                        best = cov[i]; cands = np.where(feas & (cov == best))[0]; i = cands.max()
                    hs.append(H_GRID[i])
                    hh[te] = hit[i, te]; f[te] = fa[i, te]; n_[te] = neg[i, te]; a_[te] = act[i, te]
                covs.append(np.nanmean(hh)); fars.append(f.sum() / (n_.sum() / 3600)); acts.append(a_.sum() / n_.sum())
            rows.append(dict(d0=d0, lam=lam, b=bud, cov=np.mean(covs), cov_sd=np.std(covs), far=np.mean(fars), act=np.mean(acts),
                             h_med=float(np.median(hs)), h_iqr=list(np.percentile(hs, [25, 75]))))
        # fully fixed: median h per budget applied to everyone (in-sample, for the recommended table)
        print(d0, lam, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(RESULTS / 'out_lite.csv', index=False); print(R.round(3).to_string())
