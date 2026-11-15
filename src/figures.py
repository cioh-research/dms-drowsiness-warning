"""Step 13. Figures 2-8 of the manuscript and the graphical abstract -> figures/.
(Figure 1 shows the experimental setup and annotation software and is not generated here.)"""
import pickle
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CACHE, RESULTS, FIGURES
from core import stea_signal, load_labels, load_participant

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans', 'Arial', 'DejaVu Sans'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False})
C1, C2, C3, C4, C5, C6, C7, C8 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'
GRAY = '#9a9992'
EXAMPLE = 'S04'   # participant shown in Fig. 8 (random code)


def save(fig, name, **kw):
    fig.savefig(FIGURES / name, dpi=300, bbox_inches='tight', **kw); plt.close(fig)


# ---------- Fig. 2: same PERCLOS, different SCA evidence ----------
def fig2_concept():
    t = np.arange(0, 60, 0.01)
    def trace(cl):
        o = np.ones_like(t)
        for c, d in cl: o[(t >= c) & (t < c + d)] = 0.05
        return o
    dist = [(4 + 4.6 * k, 0.4) for k in range(12)]
    clus = [(42 + 1.45 * k, 0.4) for k in range(12)]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 5.6), sharex=True, gridspec_kw=dict(hspace=0.45, height_ratios=[1, 1, 1.1, 1.3]))
    for ax, cl, col, lab in [(axes[0], dist, C1, 'Distributed events'), (axes[1], clus, C2, 'Late event cluster')]:
        ax.plot(t, trace(cl), color=col, lw=1.0); ax.set_yticks([0, 1]); ax.set_ylabel('Eye opening'); ax.set_ylim(-0.1, 1.2)
        ax.text(0.5, 0.25, lab, color=col, fontsize=8.5, fontweight='bold')
    ax = axes[2]
    for cl, col in [(dist, C1), (clus, C2)]:
        closed = trace(cl) < 0.2; cs = np.cumsum(closed) * 0.01
        ax.plot(t, 100 * cs / 60, color=col, lw=1.6)
    ax.set_ylabel('PERCLOS 60 s (%)'); ax.text(1, 6.5, 'identical at t = 60 s (8.0 %)', fontsize=8, color='#52514e'); ax.set_ylim(0, 10)
    ax = axes[3]
    for cl, col, lab in [(dist, C1, 'Distributed'), (clus, C2, 'Clustered')]:
        ev = np.array([(c, c + d, d) for c, d in cl]); E = stea_signal(t, ev, 30, 0.2)
        ax.plot(t, E, color=col, lw=1.8, label=lab)
    ax.set_ylabel('Evidence E (s)'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=8, loc='upper left')
    ax.text(59, 0.05, 'SCA evidence, λ = 30 s', fontsize=7.5, color='#52514e', ha='right')
    save(fig, 'fig2_concept.png')


# ---------- Fig. 3: exceedance vs timely warning ----------
def fig3_exceedance():
    SIG = pickle.load(open(CACHE / 'signals.pkl', 'rb'))
    WS = [30, 60, 120, 180, 300]; WL = ['30 s', '1 min', '2 min', '3 min', '5 min']; TH = [8, 9, 10, 12]
    V = np.array([[sum(np.nanmax(S['perclos'][w]) > th for S in SIG.values()) for th in TH] for w in WS])
    T = pd.read_csv(RESULTS / 'out_table_standards.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.4))
    ax = axes[0]; ax.imshow(V, cmap='Blues', vmin=0, vmax=len(SIG), aspect='auto'); ax.grid(False)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'>{x}%' for x in TH]); ax.set_yticks(range(5)); ax.set_yticklabels(WL)
    ax.set_xlabel('PERCLOS threshold'); ax.set_ylabel('Window'); ax.set_title(f'(a) Exceeded at any time (n = {len(SIG)})', fontsize=8.5, loc='left')
    for i in range(5):
        for j in range(4): ax.text(j, i, V[i, j], ha='center', va='center', fontsize=8, color='white' if V[i, j] > 20 else '#0b0b0b')
    ax = axes[1]
    sel = [0, 1, 2, 5, 6, 7, 8]; names = ['P 1 min\n≥8%', 'P 3 min\n≥9%', 'P 5 min\n≥10%', 'ECD\n≥1 s', 'ECD\n≥2 s', 'ECD\n≥3 s', 'P or\nECD 1 s']
    cols = [C1, C1, C1, C2, C2, C2, C7]
    cov = 100 * T['cov'].values[sel]; lo = 100 * T.ci_lo.values[sel]; hi = 100 * T.ci_hi.values[sel]
    ax.bar(range(7), cov, color=cols, width=0.62); ax.errorbar(range(7), cov, yerr=[cov - lo, hi - cov], fmt='none', ecolor='#52514e', lw=0.8, capsize=2)
    for i, k in enumerate(sel): ax.text(i, hi[i] + 1.5, f"{int(T.cov_n[k])}/{int(T.cov_d[k])}", ha='center', fontsize=7)
    ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=7); ax.set_ylabel('Pre-onset warning rate (%)'); ax.set_ylim(0, 65)
    ax.set_title('(b) Warned within 2 min before onset (n = 38)', fontsize=8.5, loc='left')
    save(fig, 'fig3_exceedance.png')


def km_cuminc(lat, fu, grid):
    """Cumulative proportion warned (Kaplan-Meier complement); recordings ending before a warning are censored."""
    ev = np.where(np.isnan(lat), np.inf, lat); cens = fu
    times = np.sort(np.unique(ev[np.isfinite(ev)]))
    S = 1.0; out = []; k = 0
    for g in grid:
        while k < len(times) and times[k] <= g:
            tt = times[k]; d = np.sum(ev == tt); n = np.sum((ev >= tt) & (cens >= tt) | (ev == tt))
            if n > 0: S *= (1 - d / n)
            k += 1
        out.append(1 - S)
    return np.array(out)


# ---------- Fig. 4: cumulative proportion warned ----------
def fig4_latency():
    L = pd.read_csv(RESULTS / 'out_latency.csv'); grid = np.linspace(-120, 300, 841)
    ms = [('PERCLOS 1 min >=8%', C1, '-', 'PERCLOS 1 min ≥8%'), ('PERCLOS 3 min >=9%', C7, '-', 'PERCLOS 3 min ≥9%'), ('PERCLOS 5 min >=10%', C3, '-', 'PERCLOS 5 min ≥10%'),
          ('ECD >=1 s', GRAY, '-', 'ECD ≥1 s'), ('Standards OR', '#0b0b0b', '--', 'PERCLOS or ECD ≥1 s'), ('PERCLOS 1-min relax 5/h', C4, '-', 'Lower 1-min threshold (5 FA/h)'), ('SCA 5.0/h', C2, '-', 'SCA (5 FA/h)')]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for key, col, ls, lab in ms:
        g = L[L.method == key]; y = km_cuminc(g.latency.values, g.followup.values, grid)
        ax.step(grid, 100 * y, where='post', color=col, ls=ls, lw=1.7 if 'SCA' in key else 1.3, label=lab)
    ax.axvline(0, color=C8, lw=1, ls=':'); ax.text(2, 96, 'onset', color=C8, fontsize=7.5, va='top')
    ax.axvspan(-120, 0, color='#f3f2ee', zorder=0); ax.text(-118, 96, '2 min before onset', fontsize=7.5, color='#52514e', va='top')
    ax.set_xlim(-120, 300); ax.set_ylim(0, 100); ax.set_xlabel('Time relative to onset of visible drowsiness (s)'); ax.set_ylabel('Participants warned (%)')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    save(fig, 'fig4_latency.png')


# ---------- Fig. 5: closure-event rates ----------
def fig5_mechanism():
    D = pd.read_csv(RESULTS / 'out_mechanism.csv')
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.45))
    for ax, k, lab in [(axes[0], 'rate_all', 'All closure events (/min)'), (axes[1], 'rate_sub', 'Events 0.2–1 s (/min)'), (axes[2], 'rate_long', 'Events ≥1 s (/min)')]:
        a = D[f'{k}_neg'].values; b = D[f'{k}_pre'].values
        for i in range(len(a)): ax.plot([0, 1], [a[i], b[i]], color=GRAY, lw=0.6, alpha=0.7)
        ax.plot([0] * len(a), a, 'o', color=C1, ms=3.5); ax.plot([1] * len(b), b, 'o', color=C2, ms=3.5)
        ax.plot([-0.15, 0.15], [np.median(a)] * 2, color=C1, lw=2.2); ax.plot([0.85, 1.15], [np.median(b)] * 2, color=C2, lw=2.2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Alert', 'Before\nonset']); ax.set_xlim(-0.5, 1.5); ax.set_title(lab, fontsize=8.5)
    save(fig, 'fig5_mechanism.png')


# ---------- Fig. 6: warning rate vs false-alarm rate ----------
def fig6_far_curves():
    P = pd.read_csv(RESULTS / 'out_pareto.csv')
    fams = [('stea', 'SCA (proposed)', C2, 'o'), ('relax_p1', 'Lower 1-min PERCLOS threshold', C1, 's'), ('relax_p30', 'Lower 30-s PERCLOS threshold', C7, 'v'),
            ('relax_ecd', 'Lower ECD threshold', GRAY, 'D'), ('count30', 'Event count (30 s)', C4, '^'), ('cusum', 'CUSUM', C3, 'P'), ('ewma', 'EWMA', C5, 'X')]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for f, lab, col, mk in fams:
        d = P[P.fam == f].sort_values('b')
        ax.errorbar(d.far, 100 * d['cov'], yerr=100 * d.cov_sd, xerr=d.far_sd, fmt=mk + '-', color=col, ms=5, lw=1.3, elinewidth=0.6, capsize=0, label=lab, alpha=0.95)
    ax.set_xlabel('False alarms per hour'); ax.set_ylabel('Pre-onset warning rate (%)')
    ax.set_ylim(30, 100); ax.legend(fontsize=7.5, loc='lower right')
    save(fig, 'fig6_far_curves.png')


# ---------- Fig. 7: decay time constant and alarm-on cap ----------
def fig7_lambda():
    L = pd.read_csv(RESULTS / 'out_lambda_profile.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.35))
    for ax, acap, title in [(axes[0], 1.0, 'No limit on alarm-on time'), (axes[1], 0.1, 'Alarm-on time ≤ 10%')]:
        for b, col, mk in [(2.5, C1, 'o'), (5.0, C2, 's'), (8.0, C7, '^')]:
            d = L[(L.acap == acap) & (L.b == b)].sort_values('lam')
            ax.plot(d.lam, 100 * d['cov'], mk + '-', color=col, ms=4.5, lw=1.3, label=f'{b:g} false alarms/h')
            ax.plot(d.lam, 100 * d.act, mk + '--', color=col, ms=3.5, lw=0.9, alpha=0.6)
        ax.set_xscale('log'); ax.set_xticks([15, 30, 60, 120, 300, 900]); ax.set_xticklabels([15, 30, 60, 120, 300, 900]); ax.set_xlabel('Decay time constant λ (s)')
        ax.set_ylabel('Warning rate (solid), alarm-on time (dashed), %'); ax.set_title(title, fontsize=9); ax.set_ylim(0, 100)
    h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=7.5, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.16), frameon=False)
    save(fig, 'fig7_lambda.png')


# ---------- Fig. 8: example participant ----------
def fig8_case(key=EXAMPLE):
    SIG = pickle.load(open(CACHE / 'signals.pkl', 'rb')); S = SIG[key]; t = S['t']
    lab = load_labels(); L = lab[lab.participant == key]
    fD = L[L.state == 'drowsy'].t.min(); amb = L[L.state == 'ambiguous'].t.min()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw=dict(hspace=0.35))
    m = (t >= 0) & (t <= fD + 60)
    ax = axes[0]; ax.plot(t[m], load_participant(key).open.values[m], color='#0b0b0b', lw=0.5); ax.set_ylabel('Eye opening'); ax.set_ylim(0, 1.05)
    ax = axes[1]
    for W, col, nm in [(60, C1, '1 min'), (180, C7, '3 min'), (300, C3, '5 min')]: ax.plot(t[m], S['perclos'][W][m], color=col, lw=1.4, label=f'PERCLOS {nm}')
    for v, col in [(8, C1), (9, C7), (10, C3)]: ax.axhline(v, color=col, ls=':', lw=0.9)
    ax.set_ylabel('PERCLOS (%)'); ax.legend(fontsize=7.5, ncol=3, loc='upper left', frameon=False); ax.set_ylim(0, 14)
    ax = axes[2]; E = S['stea'][(120, 0.2)]
    run = np.minimum(S['run'], 3.0)
    ax.plot(t[m], run[m], color=GRAY, lw=0.9, label='ECD: duration of ongoing closure (s)')
    ax.plot(t[m], E[m], color=C2, lw=1.6, label='SCA evidence (λ = 120 s)')
    ax.axhline(1.0, color=GRAY, ls=':', lw=0.9); ax.text(20, 1.04, 'ECD 1 s threshold', fontsize=7.5, color='#52514e', ha='left', va='bottom')
    h = 0.5; ax.axhline(h, color=C2, ls='--', lw=0.9); ax.text(20, h + 0.04, 'example SCA threshold', fontsize=7.5, color=C2, ha='left', va='bottom')
    i_long = np.where(m & (S['run'] >= 1.0))[0]
    if len(i_long):
        x0 = t[i_long[0]]
        ax.annotate('long closure after onset\n(clipped at 3 s)', xy=(x0 + 1.5, 2.6), xytext=(x0 + 12, 2.45), fontsize=7.5, color='#52514e',
                    arrowprops=dict(arrowstyle='-', color='#9a9992', lw=0.8))
    ax.set_ylabel('Seconds / evidence'); ax.set_xlabel('Time (s)'); ax.legend(fontsize=7.5, loc='upper left', frameon=False); ax.set_ylim(0, 3.3)
    for a in axes:
        a.axvline(fD, color=C8, ls='--', lw=1); a.axvline(amb, color=C4, ls='--', lw=1)
    axes[0].text(fD + 2, 0.05, 'onset', color=C8, fontsize=7.5, bbox=dict(fc='white', ec='none', pad=1)); axes[0].text(amb - 2, 0.1, 'ambiguous', color=C4, fontsize=7.5, ha='right')
    save(fig, 'fig8_case.png')
    cross = t[m & (E >= h)]
    print('Fig. 8: SCA evidence crosses', h, 'at', round(fD - cross[0], 1) if len(cross) else None, 's before onset')


# ---------- Graphical abstract ----------
def graphical_abstract():
    L = pd.read_csv(RESULTS / 'out_latency.csv')
    BLUE, ORANGE, INK, INK2, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#fcfcfb'
    def km(g, end=120):
        t = g.latency.values; f = g.followup.values
        ev = ~np.isnan(t); tt = np.where(ev, t, f)
        order = np.argsort(tt); tt, ev = tt[order], ev[order]
        n = len(tt); S = 1.0; xs = [-120]; ys = [0]
        for i, (x, e) in enumerate(zip(tt, ev)):
            if x > end: break
            if e:
                S *= 1 - 1 / (n - i); xs += [x, x]; ys += [ys[-1], 1 - S]
        xs.append(end); ys.append(ys[-1]); return np.array(xs), np.array(ys) * 100
    fig = plt.figure(figsize=(13.28, 5.31), dpi=200, facecolor=SURF)
    ax0 = fig.add_axes([0.02, 0.05, 0.42, 0.9]); ax0.axis('off')
    ax0.text(0, 0.97, 'Short eyelid closures complement PERCLOS and\neye-closure duration to warn before microsleeps', fontsize=16, weight='bold', color=INK, va='top')
    items = [('Problem', 'PERCLOS never warned in the 2 min before the first microsleep;\nthe ≥ 1-s eye-closure rule fires only after one occurs.'),
             ('Method', 'Short-closure accumulation (SCA): a time-decayed sum of the\nshort (< 1 s) closures that rise before visible drowsiness.'),
             ('Result', 'Warned before onset: 37% → 74% at 5 false alarms/h;\nhalf of drivers warned ~90 s earlier;\n59% warned before the first microsleep.'),
             ('Use', 'First stage of a staged warning, added without changing\nexisting PERCLOS and eye-closure warnings.')]
    y = 0.70
    for hd, body in items:
        ax0.text(0, y, hd, fontsize=12, weight='bold', color=INK, va='top')
        ax0.text(0.17, y, body, fontsize=11.5, color=INK2, va='top', linespacing=1.4); y -= 0.185 if hd != 'Result' else 0.225
    ax = fig.add_axes([0.53, 0.15, 0.44, 0.72], facecolor=SURF)
    ax.axvspan(-120, 0, color='#f0efec', zorder=0)
    ax.text(-60, 96, 'Before visible\ndrowsiness', ha='center', va='top', fontsize=11, color=INK2)
    for name, col, lab in [('Standards OR', BLUE, 'Standard criteria'), ('SCA 5.0/h', ORANGE, 'Standard criteria + SCA')]:
        x, yv = km(L[L.method == name]); ax.step(x, yv, where='post', color=col, lw=2.5, label=lab)
    ax.axvline(0, color=INK2, lw=1, ls='--'); ax.text(4, 97, 'Onset', fontsize=10.5, color=INK2, va='top')
    ax.set_xlim(-120, 120); ax.set_ylim(0, 100); ax.grid(axis='y', color='#e4e3df', lw=0.8); ax.grid(axis='x', visible=False)
    ax.set_xlabel('Time relative to visible drowsiness onset (s)', fontsize=11, color=INK); ax.set_ylabel('Drivers warned (%)', fontsize=11, color=INK)
    for s in ['left', 'bottom']: ax.spines[s].set_color('#b5b4ad')
    ax.tick_params(colors=INK2, labelsize=10); ax.legend(loc='lower right', frameon=False, fontsize=10.5)
    ax.set_title('Cumulative share of drivers warned (n = 38)', fontsize=11, color=INK, loc='left')
    fig.savefig(FIGURES / 'graphical_abstract.png', dpi=200, facecolor=SURF); plt.close(fig)


if __name__ == '__main__':
    fig2_concept(); fig3_exceedance(); fig4_latency(); fig5_mechanism(); fig6_far_curves(); fig7_lambda(); fig8_case(); graphical_abstract()
    print('figures done')
