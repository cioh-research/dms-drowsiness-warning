"""Step 3. Participant-wise cross-validation under a false-alarm budget b and an alarm-on cap -> cache/cv.pkl.

Settings are selected on the training participants (max pre-onset warning rate subject to FA/h <= b and
alarm-on time <= cap; ties broken by lower alarm-on time, then lower FA/h) and applied to held-out participants.
Folds are drawn with numpy RandomState(seed) over the participant codes sorted S01..S46.
"""
import pickle
import numpy as np, pandas as pd
from config import CACHE, DATA
from core import wilson

FOLDS_FILE = DATA / 'folds.csv'   # optional, see README
D = pickle.load(open(CACHE / 'metrics.pkl', 'rb'))
C = D['configs']; M = D['metrics']
FAM = pd.Series([c[0] for c in C])
SUBS = sorted(M.participant.unique(), key=lambda x: int(x[1:]))
BUDGETS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0]
FAMILIES = ['relax_p1', 'relax_multi', 'relax_p30', 'relax_ecd', 'count30', 'stea', 'cusum', 'ewma', 'stea_only']


def table(H=120, ep='drowsy'):
    """Pivot per-subject metrics into arrays indexed [cfg, subject]."""
    m = M[(M.H == H) & (M.ep == ep)]
    piv = lambda col: m.pivot(index='cfg', columns='participant', values=col)[SUBS].values
    return dict(hit=piv('hit').astype(float), fa=piv('fa'), neg=piv('neg_time'), act=piv('act_time'), lead=piv('lead'))


def pooled(T, cfg_idx, subj_idx):
    hit = T['hit'][np.ix_(cfg_idx, subj_idx)]; fa = T['fa'][np.ix_(cfg_idx, subj_idx)]
    neg = T['neg'][np.ix_(cfg_idx, subj_idx)]; act = T['act'][np.ix_(cfg_idx, subj_idx)]
    cov = np.nansum(hit, 1) / np.sum(~np.isnan(hit), 1)
    far = fa.sum(1) / (neg.sum(1) / 3600)
    actf = act.sum(1) / neg.sum(1)
    return cov, far, actf


def select(T, fam_idx, train_idx, b, acap=1.0):
    cov, far, actf = pooled(T, fam_idx, train_idx)
    feas = (far <= b) & (actf <= acap)
    if not feas.any():
        return fam_idx[np.argmin(far + 100 * np.maximum(actf - acap, 0))]
    # lexicographic: max coverage, then min active fraction, then min FAR
    key = np.lexsort((far, actf, -cov))
    for i in key:
        if feas[i]:
            return fam_idx[i]


def folds(seed, k=5):
    """Participant-wise folds (indices into SUBS). If data/folds.csv exists (columns seed, fold, code), the fold
    assignment is read from it; otherwise folds are drawn with numpy RandomState(seed) over SUBS."""
    if FOLDS_FILE.exists():
        F = _fold_table()
        g = F[F.seed == seed]
        if len(g):
            pos = {p: i for i, p in enumerate(SUBS)}
            return [np.array([pos[c] for c in g[g.fold == f].code]) for f in range(k)]
    rng = np.random.RandomState(seed); idx = np.array(SUBS)
    order = rng.permutation(len(idx))
    return [order[i::k] for i in range(k)]


_FT = None
def _fold_table():
    global _FT
    if _FT is None:
        _FT = pd.read_csv(FOLDS_FILE)
    return _FT


def run_cv(T, fam, b, seed=0, k=5, loso=False, acap=1.0):
    fam_idx = np.where(FAM == fam)[0]
    n = len(SUBS)
    fs = [[i] for i in range(n)] if loso else folds(seed, k)
    hit = np.full(n, np.nan); fa = np.zeros(n); neg = np.zeros(n); act = np.zeros(n); lead = np.full(n, np.nan)
    chosen = []
    for te in fs:
        tr = np.setdiff1d(np.arange(n), te)
        c = select(T, fam_idx, tr, b, acap); chosen.append(c)
        hit[te] = T['hit'][c, te]; fa[te] = T['fa'][c, te]; neg[te] = T['neg'][c, te]
        act[te] = T['act'][c, te]; lead[te] = T['lead'][c, te]
    cov_n = int(np.nansum(hit)); cov_d = int(np.sum(~np.isnan(hit)))
    return dict(fam=fam, b=b, acap=acap, seed=seed, cov_n=cov_n, cov_d=cov_d, cov=cov_n / cov_d,
                far=fa.sum() / (neg.sum() / 3600), act=act.sum() / neg.sum(),
                lead_med=np.nanmedian(lead), hit=hit, lead=lead, chosen=chosen)


if __name__ == '__main__':
    res = {}
    for (H, ep) in [(120, 'drowsy'), (60, 'drowsy'), (180, 'drowsy'), (120, 'nonactive')]:
        T = table(H, ep)
        rows = []
        acaps = [0.05, 0.10, 0.20, 1.0] if (H, ep) == (120, 'drowsy') else [0.10]
        for acap in acaps:
            for fam in FAMILIES:
                for b in BUDGETS:
                    for seed in range(50):
                        r = run_cv(T, fam, b, seed, acap=acap); r.update(H=H, ep=ep, mode='cv5'); rows.append(r)
                    r = run_cv(T, fam, b, loso=True, acap=acap); r.update(H=H, ep=ep, mode='loso'); rows.append(r)
        res[(H, ep)] = pd.DataFrame(rows)
        print(H, ep, 'done', flush=True)
    pickle.dump(res, open(CACHE / 'cv.pkl', 'wb'))
    print('cv done')
