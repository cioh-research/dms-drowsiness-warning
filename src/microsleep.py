"""Step 9. Secondary reference event: first microsleep (closure >= 1 s or >= 2 s).
Operating points selected for the onset criterion (main CV, 10% alarm-on cap) are applied unchanged;
we only change the event that defines a timely warning (Table 5)."""
import pickle, json
from config import CACHE, RESULTS
import numpy as np, pandas as pd
from core import wilson
from evaluate import alarm_for, SIG, SUBS
from cv import folds

D = pickle.load(open(CACHE / 'metrics.pkl', 'rb')); C = D['configs']
res = pickle.load(open(CACHE / 'cv.pkl', 'rb'))[(120, 'drowsy')]
cv = res[(res['mode'] == 'cv5') & (res.acap == 0.1)]
H = 120.0; MIN_T = 120.0
FAMS = ['stea', 'relax_p1', 'relax_multi', 'relax_p30', 'relax_ecd', 'count30', 'cusum', 'ewma']
BUD = [2.5, 5.0, 8.0]


def first_ms(S, th):
    ev = S['events']
    if len(ev) == 0: return None
    m = ev[ev[:, 2] >= th]
    return float(m[0, 0]) if len(m) else None


cache = {}
def hit_lead(p, ci, s):
    key = (p, ci, s)
    if key not in cache:
        S = SIG[p]; t = S['t']; al = alarm_for(S, C[ci]); w = (t >= s - H) & (t < s)
        a = al & w
        cache[key] = (bool(a.any()), float(s - t[a][0]) if a.any() else np.nan, bool(al[(t >= s - H) & (t < s - H + 1)].any()))
    return cache[key]


out = {}
for th in [1.0, 2.0]:
    ms = {p: first_ms(SIG[p], th) for p in SUBS}
    elig = [p for p in SUBS if ms[p] is not None and ms[p] >= MIN_T]
    base_ci = C.index(('base',)); pu_ci = C.index(('perclos_union',))
    row = dict(n=len(elig), n_any=sum(v is not None for v in ms.values()))
    bh = [hit_lead(p, base_ci, ms[p]) for p in elig]
    row['standard'] = dict(k=sum(h for h, _, _ in bh), lead_med=float(np.nanmedian([l for _, l, _ in bh])) if any(h for h, _, _ in bh) else None)
    row['standard']['ci'] = wilson(row['standard']['k'], len(elig))
    for fam in FAMS:
        for b in BUD:
            rr = cv[(cv.fam == fam) & (cv.b == b)]
            rates = []; leads = []; on_at_start = []; hits0 = None
            for _, r in rr.iterrows():
                fs = folds(int(r.seed)); hs = []
                for te, c in zip(fs, r.chosen):
                    for i in te:
                        p = SUBS[i]
                        if p in elig:
                            h, l, st = hit_lead(p, c, ms[p]); hs.append(h)
                            if r.seed == 0: leads.append(l); on_at_start.append(st and h)
                rates.append(np.mean(hs))
                if r.seed == 0: hits0 = int(np.sum(hs))
            row[f'{fam}_{b}'] = dict(mean=float(np.mean(rates)), sd=float(np.std(rates)), k0=hits0,
                                     lead_med0=float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else None,
                                     on_at_start0=int(np.sum(on_at_start)))
    # paired comparison seed 0: SCA vs relax_p1 and vs standard
    for b in BUD:
        for other in ['relax_p1', 'base']:
            r_s = cv[(cv.fam == 'stea') & (cv.b == b) & (cv.seed == 0)].iloc[0]
            hs_s = {}; fs = folds(0)
            for te, c in zip(fs, r_s.chosen):
                for i in te:
                    p = SUBS[i]
                    if p in elig: hs_s[p] = hit_lead(p, c, ms[p])[0]
            if other == 'base':
                hs_o = {p: hit_lead(p, base_ci, ms[p])[0] for p in elig}
            else:
                r_o = cv[(cv.fam == other) & (cv.b == b) & (cv.seed == 0)].iloc[0]; hs_o = {}
                for te, c in zip(fs, r_o.chosen):
                    for i in te:
                        p = SUBS[i]
                        if p in elig: hs_o[p] = hit_lead(p, c, ms[p])[0]
            a = sum(hs_s[p] and not hs_o[p] for p in elig); bb = sum(hs_o[p] and not hs_s[p] for p in elig)
            from scipy.stats import binomtest
            pval = binomtest(min(a, bb), a + bb, 0.5).pvalue * 1.0 if a + bb else 1.0
            row[f'mcnemar_{other}_{b}'] = dict(only_sca=a, only_other=bb, p=float(min(1.0, pval)))
    out[th] = row
    print('threshold', th, json.dumps(row, indent=0)[:3000], flush=True)

# relation between references
rel = []
for p in SUBS:
    fD = SIG[p]['masks'][(120, 'drowsy')][2]
    rel.append(dict(participant=p, ms1=first_ms(SIG[p], 1.0), ms2=first_ms(SIG[p], 2.0), onset=fD))
R = pd.DataFrame(rel)
json.dump({str(k): v for k, v in out.items()}, open(RESULTS / 'out_microsleep.json', 'w'), indent=1, default=float)
R.drop(columns='participant').to_csv(RESULTS / 'out_microsleep_relation.csv', index=False)
