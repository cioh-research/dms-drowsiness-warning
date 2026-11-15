"""Step 5. Time of the first warning relative to onset (input to the Kaplan-Meier curves, Fig. 4)."""
import pickle, json
from config import CACHE, RESULTS
import numpy as np, pandas as pd
from core import *
from evaluate import alarm_for, SIG, SUBS
from cv import folds

D = pickle.load(open(CACHE / 'metrics.pkl', 'rb')); C = D['configs']
res = pickle.load(open(CACHE / 'cv.pkl', 'rb'))[(120, 'drowsy')]
cv0 = res[(res['mode'] == 'cv5') & (res.acap == 0.1) & (res.seed == 0)]
fs = folds(0)


def heldout_cfg(fam, b):
    r = cv0[(cv0.fam == fam) & (cv0.b == b)].iloc[0]; m = {}
    for te, c in zip(fs, r.chosen):
        for i in te: m[SUBS[i]] = C[c]
    return m


methods = {'PERCLOS 1 min >=8%': ('perclos', 60, 8), 'PERCLOS 3 min >=9%': ('perclos', 180, 9), 'PERCLOS 5 min >=10%': ('perclos', 300, 10),
           'ECD >=1 s': ('ecd', 1.0), 'Standards OR': ('base',)}
held = {f'SCA {b}/h': heldout_cfg('stea', b) for b in [2.5, 5.0, 8.0]}
held['PERCLOS 1-min relax 5/h'] = heldout_cfg('relax_p1', 5.0)
rows = []
for p in SUBS:
    S = SIG[p]; pre, neg, fD = S['masks'][(120, 'drowsy')]
    if fD is None: continue
    t = S['t']; win = (t >= fD - 120)
    allm = {**{k: v for k, v in methods.items()}, **{k: v[p] for k, v in held.items()}}
    for name, cfg in allm.items():
        al = alarm_for(S, cfg) & win
        lat = (t[al][0] - fD) if al.any() else np.nan
        post = t[-1] - fD
        rows.append(dict(participant=p, method=name, latency=lat, followup=post))
L = pd.DataFrame(rows); L.to_csv(RESULTS / 'out_latency.csv', index=False)
summ = {}
for m, g in L.groupby('method'):
    l = g.latency
    summ[m] = dict(before=int((l <= 0).sum()), within30=int(((l > 0) & (l <= 30)).sum()), within120=int(((l > 0) & (l <= 120)).sum()),
                   within300=int(((l > 0) & (l <= 300)).sum()), later=int((l > 300).sum()), never=int(l.isna().sum()),
                   median_delay_after=float(l[l > 0].median()) if (l > 0).any() else None, min_followup=float(g.followup.min()))
print(json.dumps(summ, indent=1))
json.dump(summ, open(RESULTS / 'out_latency_summary.json', 'w'), indent=1)
