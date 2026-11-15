"""Step 6. Logistic regression and gradient boosting (multi-feature) under nested participant-wise CV with
the same FA/h budget and 10% alarm-on cap. 10 repeats (LR) and 3 repeats (GB). Slow."""
import pickle
from config import CACHE
import numpy as np, pandas as pd
from core import *
from evaluate import base_alarm, SIG, SUBS
from cv import folds
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

H, EP = 120, 'drowsy'
BUDGETS = [2.5, 5.0, 8.0]


def features(S):
    t = S['t']; sec = np.arange(0, t[-1], 1.0)
    idx = np.searchsorted(t, sec, side='right') - 1; idx = np.clip(idx, 0, len(t) - 1)
    P = S['perclos']
    F = {f'p{W}': np.nan_to_num(P[W][idx], nan=0.0) for W in [10, 30, 60, 180, 300]}
    for lam in [15, 30, 60, 120]:
        F[f'e{lam}'] = S['stea'][(lam, 0.2)][idx]
    F['cnt30'] = S['count'][(30, 0.15)][idx]
    F['cnt30all'] = S['count'][(30, 0.03)][idx]
    F['run'] = S['run'][idx]
    F['cusum'] = S['cusum'][np.log(0.15)][idx]
    F['ewma'] = S['ewma'][0.2][idx]
    X = pd.DataFrame(F)
    pre, neg, fD = S['masks'][(H, EP)]
    y = np.where(pre[idx], 1, np.where(neg[idx], 0, -1))
    return X, y, sec, idx


FEAT = {p: features(SIG[p]) for p in SUBS}


def fit(model, subs):
    X = pd.concat([FEAT[p][0][FEAT[p][1] >= 0] for p in subs]); y = np.concatenate([FEAT[p][1][FEAT[p][1] >= 0] for p in subs])
    if model == 'lr':
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
    else:
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=0)
    m.fit(X, y); return m


def score20(m, p):
    X, y, sec, idx = FEAT[p]; s = m.predict_proba(X)[:, 1]
    t = SIG[p]['t']; j = np.searchsorted(sec, t, side='right') - 1
    return s[np.clip(j, 0, len(s) - 1)]


def metrics_for(p, alarm):
    S = SIG[p]; pre, neg, fD = S['masks'][(H, EP)]
    return subject_metrics(S['t'], S['dt'], alarm, pre, neg, fD)


def choose_tau(scores, subs, b, use_base=True):
    taus = np.quantile(np.concatenate([scores[p] for p in subs]), np.linspace(0.5, 0.999, 80))
    best = None
    for tau in taus:
        rows = [metrics_for(p, (base_alarm(SIG[p]) if use_base else False) | (scores[p] >= tau)) for p in subs]
        r = pool(rows)
        if r['far'] <= b and r['act'] <= 0.10 and (best is None or r['cov'] > best[1] or (r['cov'] == best[1] and r['act'] < best[2])):
            best = (tau, r['cov'], r['act'])
    return best[0] if best else taus[-1]


def run(model, seed, k=5):
    fs = folds(seed, k)
    out = {b: [] for b in BUDGETS}
    for te in fs:
        tr = np.setdiff1d(np.arange(len(SUBS)), te)
        trs = [SUBS[i] for i in tr]; tes = [SUBS[i] for i in te]
        # inner OOF scores on training subjects for threshold selection
        oof = {}
        inner = [tr[i::4] for i in range(4)]
        for ite in inner:
            itr = [SUBS[i] for i in np.setdiff1d(tr, ite)]
            m = fit(model, itr)
            for i in ite: oof[SUBS[i]] = score20(m, SUBS[i])
        m = fit(model, trs)
        tes_scores = {p: score20(m, p) for p in tes}
        for b in BUDGETS:
            tau = choose_tau(oof, trs, b)
            for p in tes:
                out[b].append(dict(metrics_for(p, base_alarm(SIG[p]) | (tes_scores[p] >= tau)), participant=p))
    return {b: (pool(v), pd.DataFrame(v)) for b, v in out.items()}


if __name__ == '__main__':
    part = CACHE / 'ml_partial.pkl'
    res = pickle.load(open(part, 'rb')) if part.exists() else {}   # resume after an interruption
    for model, nseed in [('lr', 10), ('gb', 3)]:
        for seed in range(nseed):
            if (model, seed) in res: continue
            res[(model, seed)] = run(model, seed)
            print(model, seed, {b: (r[0]['cov_n'], round(r[0]['far'], 2), round(r[0]['act'], 3)) for b, r in res[(model, seed)].items()}, flush=True)
            pickle.dump(res, open(CACHE / 'ml_partial.pkl', 'wb'))
    pickle.dump(res, open(CACHE / 'ml.pkl', 'wb'))
