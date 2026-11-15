"""Compare the reproduced results with the values reported in the manuscript -> results/comparison_with_manuscript.csv.
results/manuscript_values.csv holds the manuscript values (computed with the original fold assignment)."""
import json
import numpy as np, pandas as pd
from pathlib import Path
from config import RESULTS

FAMS = ['stea', 'relax_p1', 'relax_multi', 'relax_p30', 'relax_ecd', 'count30', 'cusum', 'ewma']
BUD = [2.5, 5.0, 8.0]


def collect(R):
    R = Path(R); out = {}
    T = pd.read_csv(R / 'out_table_standards.csv')
    for i, r in T.iterrows():
        out[f'T2 row{i + 1} warned_n'] = r.cov_n; out[f'T2 row{i + 1} FA/h'] = r.far
    P = pd.read_csv(R / 'out_pareto.csv')
    for f in FAMS:
        for b in BUD:
            out[f'T3 {f} {b} rate%'] = 100 * P[(P.fam == f) & (P.b == b)]['cov'].iloc[0]
    J = json.load(open(R / 'results.json'))
    for r in J['ml']:
        if r['b'] in BUD: out[f"T3 {r['model']} {r['b']} rate%"] = 100 * r['cov']
    A = pd.read_csv(R / 'out_ablation.csv'); A['variant'] = A.variant.str.replace('S-TEA', 'SCA')
    for _, r in A.iterrows(): out[f'T4 {r.variant} {r.b} rate%'] = 100 * r['cov']
    L = pd.read_csv(R / 'out_lite.csv'); L = L[L.lam == 120]
    for _, r in L.iterrows(): out[f'T4 simplified SCA {r.b} rate%'] = 100 * r['cov']
    M = json.load(open(R / 'out_microsleep.json'))
    for th in ['1.0', '2.0']:
        out[f'T5 ms>={th} standard rate%'] = 100 * M[th]['standard']['k'] / M[th]['n']
        for f in ['stea', 'relax_p1', 'relax_ecd', 'count30', 'cusum', 'ewma']:
            for b in BUD: out[f'T5 ms>={th} {f} {b} rate%'] = 100 * M[th][f'{f}_{b}']['mean']
    S = pd.read_csv(R / 'out_sensitivity.csv')
    for _, r in S[S.fam.isin(['stea', 'relax_p1'])].iterrows():
        out[f'T6 H{r.H} {r.ep} {r.fam} {r.b} warned_n (split 0)'] = r.cov_n0
    RB = pd.read_csv(R / 'out_robust.csv'); RB = RB[RB['mode'].isin(['clean', 'fps'])]
    for _, r in RB[RB.fam.isin(['standard', 'stea', 'relax_p1'])].iterrows():
        out[f'T7 {r["mode"]}{int(r.level)} {r.fam} {r.b} rate%'] = 100 * r['cov']; out[f'T7 {r["mode"]}{int(r.level)} {r.fam} {r.b} FA/h'] = r.far
    RR = pd.read_csv(R / 'out_robust_recal.csv').groupby(['level', 'fam', 'b'])[['cov', 'far']].mean().reset_index()
    for _, r in RR.iterrows():
        out[f'T7 spurious{r.level} {r.fam} {r.b} rate%'] = 100 * r['cov']; out[f'T7 spurious{r.level} {r.fam} {r.b} FA/h'] = r.far
    return out


if __name__ == '__main__':
    man = pd.read_csv(RESULTS / 'manuscript_values.csv').set_index('quantity').value
    rep = pd.Series(collect(RESULTS))
    C = pd.DataFrame(dict(manuscript=man, reproduced=rep.reindex(man.index)))
    C['difference'] = C.reproduced - C.manuscript
    C.round(3).to_csv(RESULTS / 'comparison_with_manuscript.csv')
    print(C.round(2).to_string())
    print('max |difference|:', C.difference.abs().max().round(2))
