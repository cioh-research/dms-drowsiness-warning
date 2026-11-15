"""Step 11. Spurious-closure condition with re-calibration: settings re-selected by participant-wise CV
(10 repeats) on the perturbed data; three random realisations per rate (seeds 100-102). Table 7, spurious rows."""
import numpy as np, pandas as pd
from config import RESULTS
from core import subject_metrics
import evaluate as E
from cv import folds, select
import robustness as RB
from robustness import C, SUBS, perturb
SC = [i for i, c in enumerate(C) if c[0] in ('stea', 'relax_p1', 'base')]
RB.need_stea = sorted({(C[i][1], C[i][2]) for i in SC if C[i][0] == 'stea'})
out = []
for level, rep in [(l, r) for l in [0.5, 1.0] for r in range(3)]:
    rng = np.random.RandomState(100 + rep)
    SIG = {p: perturb(p, 'spur', level, rng) for p in SUBS}
    hit = np.full((len(C), len(SUBS)), np.nan); fa = np.zeros((len(C), len(SUBS))); neg = np.zeros_like(fa); act = np.zeros_like(fa)
    for j, p in enumerate(SUBS):
        S = SIG[p]; pre, ng, fD = S['mask']
        for i in SC:
            m = subject_metrics(S['t'], S['dt'], E.alarm_for(S, C[i]), pre, ng, fD)
            hit[i, j] = m['hit']; fa[i, j] = m['fa']; neg[i, j] = m['neg_time']; act[i, j] = m['act_time']
    T = dict(hit=hit, fa=fa, neg=neg, act=act)
    FAMv = pd.Series([c[0] for c in C])
    bi = C.index(('base',))
    std = dict(cov=np.nanmean(hit[bi]), far=fa[bi].sum() / (neg[bi].sum() / 3600), act=act[bi].sum() / neg[bi].sum())
    out.append(dict(level=level, rep=rep, fam='standard', b=0, **std))
    for fam in ['stea', 'relax_p1']:
        fam_idx = np.where(FAMv == fam)[0]
        for b in [2.5, 5.0, 8.0]:
            covs = []; fars = []; acts = []
            for seed in range(10):
                h = np.full(len(SUBS), np.nan); f = np.zeros(len(SUBS)); n_ = np.zeros(len(SUBS)); a_ = np.zeros(len(SUBS))
                for te in folds(seed):
                    tr = np.setdiff1d(np.arange(len(SUBS)), te)
                    c = select(T, fam_idx, tr, b, 0.10)
                    h[te] = hit[c, te]; f[te] = fa[c, te]; n_[te] = neg[c, te]; a_[te] = act[c, te]
                covs.append(np.nanmean(h)); fars.append(f.sum() / (n_.sum() / 3600)); acts.append(a_.sum() / n_.sum())
            out.append(dict(level=level, rep=rep, fam=fam, b=b, cov=np.mean(covs), cov_sd=np.std(covs), far=np.mean(fars), act=np.mean(acts)))
    print(level, rep, 'done', flush=True)
R = pd.DataFrame(out); R.to_csv(RESULTS / 'out_robust_recal.csv', index=False)
print(R.groupby(['level', 'fam', 'b'])[['cov', 'far', 'act']].mean().round(3).to_string())
