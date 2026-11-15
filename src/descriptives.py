"""Descriptive statistics quoted in Sections 2.2, 2.3 and 2.6 -> results/descriptives.json."""
import json, pickle
import numpy as np, pandas as pd
from config import CACHE, RESULTS, PROCESSED
from core import load_labels, gap_threshold

SIG = pickle.load(open(CACHE / 'signals.pkl', 'rb'))
lab = load_labels()
out = {'participants': len(SIG)}

# recording duration
dur = np.array([S['t'][-1] for S in SIG.values()]) / 60
out['recording_min_mean_sd'] = [round(dur.mean(), 1), round(dur.std(ddof=1), 1)]

# onset and ambiguous state
on = lab[lab.state == 'drowsy'].groupby('participant').t.min()
amb = lab[lab.state == 'ambiguous'].groupby('participant').t.min()
out['onset_n'] = int(len(on))
out['onset_min_median_range'] = [round(on.median() / 60, 1), round(on.min() / 60, 1), round(on.max() / 60, 1)]
lead = (on - amb.reindex(on.index)).dropna(); lead = lead[lead > 0]
out['ambiguous_before_onset_n'] = int(len(lead))
out['ambiguous_lead_s_median_iqr'] = [round(lead.median()), round(lead.quantile(.25)), round(lead.quantile(.75))]
out['ambiguous_within_2min_n'] = int((lead <= 120).sum())
post = np.array([SIG[p]['t'][-1] - on[p] for p in on.index]) / 60
out['post_onset_min_median'] = round(float(np.median(post)), 1)

# false-alarm exposure (explicit alert periods > 2 min before onset, or without onset)
neg_t = []
for p, S in SIG.items():
    pre, neg, fD = S['masks'][(120, 'drowsy')]
    dt = S['dt']; g = gap_threshold(dt); dtc = np.where((dt > 0) & (dt <= g), dt, 0)
    neg_t.append(dtc[neg].sum())
neg_t = np.array(neg_t)
out['fa_exposure_records'] = int((neg_t > 0).sum()); out['fa_exposure_hours'] = round(neg_t.sum() / 3600, 2)

# eye opening and closure duration during alert ('active') periods
op_all, med_d = [], []
for p, S in SIG.items():
    d = pd.read_csv(PROCESSED / f'{p}.csv.gz')
    t = S['t']; act = np.zeros(len(t), bool)
    for s0, s1, st in S['segs']:
        if st == 'active': act |= (t >= s0) & (t < s1)
    v = d.valid.values.astype(bool) & act
    op_all.append(d.eye_opening.values[v])
    ev = S['events']
    if len(ev):
        j = np.clip(np.searchsorted(t, ev[:, 1], side='right') - 1, 0, len(t) - 1)
        da = ev[act[j], 2]
        if len(da): med_d.append(np.median(da))
out['alert_eye_opening_median'] = round(float(np.nanmedian(np.concatenate(op_all))), 2)
out['alert_closure_duration_s_median_of_medians'] = round(float(np.median(med_d)), 3)

# samples whose value came from the eye-tracker-assisted correction (either eye)
allp = pd.concat([pd.read_csv(f, usecols=['valid', 'source']) for f in sorted(PROCESSED.glob('S*.csv.gz'))])
src = allp.source
out['samples_total'] = int(len(src))
out['samples_unclassified_missing'] = int((allp.valid == 0).sum())
out['samples_eyetracker_corrected_pct'] = round(100 * (src == 1).mean(), 2)
out['samples_manual_alert_pct'] = round(100 * (src == 2).mean(), 2)

json.dump(out, open(RESULTS / 'descriptives.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
