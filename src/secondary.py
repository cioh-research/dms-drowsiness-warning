"""Step 4. Secondary analyses: FA-rate curves, area summary (2-7 FA/h), paired statistics, sensitivity to window
and onset definition, alarm-on cap, lambda profile, standard-criteria table and closure-rate comparison."""
import pickle, json
from config import CACHE, RESULTS
import numpy as np, pandas as pd
from scipy import stats
from core import *
from cv import table, run_cv, FAM, SUBS, BUDGETS, FAMILIES
from evaluate import alarm_for, base_alarm, SIG

res = pickle.load(open(CACHE / 'cv.pkl', 'rb'))
D = pickle.load(open(CACHE / 'metrics.pkl', 'rb')); C = D['configs']; M = D['metrics']
OUT = {}
ACAP = 0.10
R = res[(120, 'drowsy')]; cv = R[(R['mode'] == 'cv5') & (R.acap == ACAP)]
cv_all = R[R['mode'] == 'cv5']

# ---- 1. Pareto curves: mean over 50 seeds per budget
curve = cv.groupby(['fam', 'b']).agg(cov=('cov', 'mean'), cov_sd=('cov', 'std'), far=('far', 'mean'), far_sd=('far', 'std'), act=('act', 'mean')).reset_index()
curve.to_csv(RESULTS / 'out_pareto.csv', index=False)


def interp_cov(fam, far_targets):
    c = curve[curve.fam == fam].sort_values('far')
    # monotone envelope
    cov = np.maximum.accumulate(c['cov'].values)
    return np.interp(far_targets, c['far'].values, cov, left=np.nan, right=np.nan)


targets = [3, 4, 5, 6, 7]
matched = pd.DataFrame({f: interp_cov(f, targets) for f in FAMILIES}, index=targets)
matched.to_csv(RESULTS / 'out_matched_far.csv')
# normalised area under coverage-FAR curve for FAR in [2,7]
auc = {}
for f in FAMILIES:
    x = np.linspace(2, 7, 51); y = interp_cov(f, x); auc[f] = float(np.nanmean(y))
OUT['auc_2_7'] = auc

# ---- 2. Statistics at seed 0: paired comparisons SCA vs each simple family
s0 = cv[cv.seed == 0]
def hits(fam, b):
    r = s0[(s0.fam == fam) & (s0.b == b)].iloc[0]; h = r.hit; return h[~np.isnan(h)].astype(bool)
stat_rows = []
rng = np.random.RandomState(1)
for b in [2.5, 5.0, 8.0]:
    hs = hits('stea', b)
    for fam in ['relax_p1', 'relax_multi', 'relax_p30', 'relax_ecd', 'count30', 'cusum', 'ewma']:
        hf = hits(fam, b)
        n01 = int((hs & ~hf).sum()); n10 = int((~hs & hf).sum())
        p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
        diff = hs.mean() - hf.mean()
        boots = []
        for _ in range(5000):
            i = rng.randint(0, len(hs), len(hs)); boots.append(hs[i].mean() - hf[i].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        stat_rows.append(dict(b=b, fam=fam, stea=int(hs.sum()), other=int(hf.sum()), only_stea=n01, only_other=n10, diff=diff, ci_lo=lo, ci_hi=hi, mcnemar_p=p))
ST = pd.DataFrame(stat_rows); ST.to_csv(RESULTS / 'out_stats.csv', index=False)

# across 50 seeds: how often SCA coverage >= competitor at same budget
win = {}
for b in [2.5, 5.0, 8.0]:
    for fam in ['relax_p1', 'cusum', 'ewma', 'count30']:
        a = cv[(cv.fam == 'stea') & (cv.b == b)].sort_values('seed')['cov'].values
        c = cv[(cv.fam == fam) & (cv.b == b)].sort_values('seed')['cov'].values
        win[f'{fam}_b{b}'] = dict(mean_diff=float((a - c).mean()), sd=float((a - c).std()), frac_ge=float((a >= c).mean()), frac_gt=float((a > c).mean()))
OUT['seed_wins'] = win

# ---- 3. Sensitivity: horizon and endpoint (seed 0 + mean over seeds)
sens = []
for key, R2 in res.items():
    c2 = R2[(R2['mode'] == 'cv5') & (R2.acap == ACAP)]
    for fam in ['stea', 'relax_p1', 'cusum', 'count30']:
        for b in [2.5, 5.0, 8.0]:
            g = c2[(c2.fam == fam) & (c2.b == b)]
            r0 = g[g.seed == 0].iloc[0]
            sens.append(dict(H=key[0], ep=key[1], fam=fam, b=b, cov_n0=r0.cov_n, cov_d=r0.cov_d, far0=r0.far, act0=r0.act, cov_mean=g['cov'].mean(), cov_sd=g['cov'].std(), far_mean=g.far.mean()))
SE = pd.DataFrame(sens); SE.to_csv(RESULTS / 'out_sensitivity.csv', index=False)
# baseline coverage per setting
base_rows = []
for key in res:
    m = M[(M.H == key[0]) & (M.ep == key[1])]
    for cfg in [('perclos_union',), ('ecd', 1.0), ('base',)]:
        r = pool(m[m.cfg == C.index(cfg)].to_dict('records')); base_rows.append(dict(H=key[0], ep=key[1], cfg=str(cfg), **r))
pd.DataFrame(base_rows).to_csv(RESULTS / 'out_base_sensitivity.csv', index=False)

# ---- 3b. effect of activity cap
ac = cv_all.groupby(['acap', 'fam', 'b']).agg(cov=('cov', 'mean'), cov_sd=('cov', 'std'), far=('far', 'mean'), act=('act', 'mean')).reset_index()
ac[ac.b.isin([2.5, 5.0, 8.0])].to_csv(RESULTS / 'out_acap.csv', index=False)
lo_all = R[R['mode'] == 'loso']
lo_all[lo_all.b.isin([2.5, 5.0, 8.0])][['acap', 'fam', 'b', 'cov_n', 'cov_d', 'far', 'act', 'lead_med']].to_csv(RESULTS / 'out_loso.csv', index=False)
# ---- 4. Parameter stability: configs chosen across folds/seeds for SCA
ch = []
for b in [2.5, 5.0, 8.0]:
    for r in cv[(cv.fam == 'stea') & (cv.b == b)].itertuples():
        for c in r.chosen: ch.append(dict(b=b, lam=C[c][1], d0=C[c][2], p=C[c][3], h=C[c][4]))
CH = pd.DataFrame(ch)
CH.to_csv(RESULTS / 'out_chosen.csv', index=False)
OUT['chosen_summary'] = {str(b): dict(lam=CH[CH.b == b].lam.value_counts(normalize=True).round(3).to_dict(), d0=CH[CH.b == b].d0.value_counts(normalize=True).round(3).to_dict(), p=CH[CH.b == b].p.value_counts(normalize=True).round(3).to_dict(), h_med=float(CH[CH.b == b].h.median()), h_iqr=[float(x) for x in CH[CH.b == b].h.quantile([.25, .75])]) for b in [2.5, 5.0, 8.0]}

# ---- 4b. lambda profile under ACAP
from cv import folds, select
T = table(120, 'drowsy'); lp = []
for lam in [15, 30, 60, 120, 180, 300, 600, 900]:
    idx = np.where([c[0] == 'stea' and c[1] == lam for c in C])[0]
    for b in [2.5, 5.0, 8.0]:
        for acap in [ACAP, 1.0]:
            covs = []; fars = []; acts = []
            for seed in range(50):
                fs = folds(seed); n = len(SUBS)
                hit = np.full(n, np.nan); fa = np.zeros(n); neg = np.zeros(n); act = np.zeros(n)
                for te in fs:
                    tr = np.setdiff1d(np.arange(n), te); c = select(T, idx, tr, b, acap)
                    hit[te] = T['hit'][c, te]; fa[te] = T['fa'][c, te]; neg[te] = T['neg'][c, te]; act[te] = T['act'][c, te]
                covs.append(np.nansum(hit) / np.sum(~np.isnan(hit))); fars.append(fa.sum() / (neg.sum() / 3600)); acts.append(act.sum() / neg.sum())
            lp.append(dict(lam=lam, b=b, acap=acap, cov=np.mean(covs), cov_sd=np.std(covs), far=np.mean(fars), act=np.mean(acts)))
pd.DataFrame(lp).to_csv(RESULTS / 'out_lambda_profile.csv', index=False)
# ---- 5. Detection matrix (seed 0) + onset-within-window
mat = pd.DataFrame(index=SUBS)
m = M[(M.H == 120) & (M.ep == 'drowsy')]
for name, cfg in [('P1', ('perclos', 60, 8)), ('P3', ('perclos', 180, 9)), ('P5', ('perclos', 300, 10)), ('ECD1', ('ecd', 1.0)), ('ECD2', ('ecd', 2.0)), ('ECD3', ('ecd', 3.0)), ('Base', ('base',))]:
    mm = m[m.cfg == C.index(cfg)].set_index('participant'); mat[name] = mm.hit.reindex(SUBS)
for b, name in [(2.5, 'STEA_c'), (5.0, 'STEA_b'), (8.0, 'STEA_h')]:
    r = s0[(s0.fam == 'stea') & (s0.b == b)].iloc[0]; mat[name] = r.hit
    mat[name + '_lead'] = r.lead
lab = load_labels()
fd = lab[lab.state == 'drowsy'].groupby('participant').t.min()
mat['firstD'] = fd.reindex(SUBS)
mat.to_csv(RESULTS / 'out_matrix.csv')
# among SCA hits: fraction where alarm already on at window start (lead >= 119.5)
for name in ['STEA_c', 'STEA_b', 'STEA_h']:
    l = mat[name + '_lead'].dropna(); OUT[f'{name}_already_on'] = dict(n=int(len(l)), at_start=int((l >= 119.5).sum()), median_lead_new=float(l[l < 119.5].median()) if (l < 119.5).any() else None)

# ---- 6. Descriptive mechanism: sub-second event statistics, pre-D vs explicit-active baseline (paired)
desc = []
for p in SUBS:
    S = SIG[p]; ev = S['events']; pre, neg, fD = S['masks'][(120, 'drowsy')]
    t = S['t']; dt = S['dt']
    if fD is None or len(ev) == 0: continue
    negt = dt[neg].sum(); pret = dt[pre].sum()
    if negt < 60: continue
    e = ev[:, 1]; d = ev[:, 2]
    in_pre = (e >= fD - 120) & (e <= fD)
    # negatives: event end in neg mask
    j = np.clip(np.searchsorted(t, e, side='right') - 1, 0, len(t) - 1); in_neg = neg[j]
    sub = (d < 1.0)
    desc.append(dict(participant=p,
        rate_all_pre=in_pre.sum() / pret * 60, rate_all_neg=in_neg.sum() / negt * 60,
        rate_sub_pre=(in_pre & sub & (d >= 0.2)).sum() / pret * 60, rate_sub_neg=(in_neg & sub & (d >= 0.2)).sum() / negt * 60,
        rate_long_pre=(in_pre & ~sub).sum() / pret * 60, rate_long_neg=(in_neg & ~sub).sum() / negt * 60,
        med_d_pre=np.median(d[in_pre]) if in_pre.any() else np.nan, med_d_neg=np.median(d[in_neg]) if in_neg.any() else np.nan))
DE = pd.DataFrame(desc)
DE.to_csv(RESULTS / 'out_mechanism.csv', index=False)
mech = {}
for k in ['rate_all', 'rate_sub', 'rate_long', 'med_d']:
    a = DE[f'{k}_pre'].values; bb = DE[f'{k}_neg'].values; ok = ~np.isnan(a) & ~np.isnan(bb)
    w = stats.wilcoxon(a[ok], bb[ok]) if ok.sum() > 5 and np.any(a[ok] != bb[ok]) else None
    mech[k] = dict(n=int(ok.sum()), pre_med=float(np.median(a[ok])), neg_med=float(np.median(bb[ok])), p=float(w.pvalue) if w else None, frac_increase=float((a[ok] > bb[ok]).mean()))
OUT['mechanism'] = mech

# ---- 7. Standards-based table + short-window table (H=120)
def P(cfg): return pool(m[m.cfg == C.index(cfg)].to_dict('records'))
tab5 = []
for name, cfg in [('PERCLOS 1 min >=8%', ('perclos', 60, 8)), ('PERCLOS 3 min >=9%', ('perclos', 180, 9)), ('PERCLOS 5 min >=10%', ('perclos', 300, 10)), ('PERCLOS early warnings (any)', ('perclos_union',)), ('PERCLOS >=12% (any window)', ('perclos_full',)), ('ECD >=1 s', ('ecd', 1.0)), ('ECD >=2 s', ('ecd', 2.0)), ('ECD >=3 s', ('ecd', 3.0)), ('Combined standard criteria', ('base',))]:
    r = P(cfg); lo, hi = wilson(r['cov_n'], r['cov_d']); tab5.append(dict(name=name, **r, ci_lo=lo, ci_hi=hi))
pd.DataFrame(tab5).to_csv(RESULTS / 'out_table_standards.csv', index=False)
sw = []
for W in [5, 10, 15, 20, 30, 60]:
    for th in [1, 2, 3, 4, 6, 8, 10, 12]:
        r = P(('perclos', W, th)); sw.append(dict(W=W, th=th, **r))
pd.DataFrame(sw).to_csv(RESULTS / 'out_shortwindow.csv', index=False)
# SCA alone (no base) selected points
json.dump(OUT, open(RESULTS / 'out_summary.json', 'w'), indent=1, ensure_ascii=False, default=float)
print(json.dumps(OUT, indent=1, ensure_ascii=False, default=float))
print(matched.round(3)); print(ST.round(3).to_string()); print(SE.round(3).to_string())
