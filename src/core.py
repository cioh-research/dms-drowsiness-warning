"""Core signal computation and evaluation functions.

Signals: time-weighted PERCLOS, hysteresis closure events, short-closure accumulation (SCA; called
`stea` in the code), closure counts, CUSUM and EWMA. Metrics: pre-onset warning, false-alarm
episodes and alarm-on time in explicit alert periods.
"""
import numpy as np, pandas as pd
from config import PROCESSED, ANNOTATIONS, CACHE

H_DEFAULT = 120.0   # pre-onset evaluation window (s)
MERGE_GAP = 30.0    # alarms re-triggered within 30 s count as one episode


def load_participant(code):
    """Return the processed time series of one participant as a DataFrame with columns
    participant, t, dt, open, valid, closed80 (eye-tracker correction already applied)."""
    d = pd.read_csv(PROCESSED / f'{code}.csv.gz')
    df = pd.DataFrame(dict(participant=code, t=d.t.values, open=d.eye_opening.values,
                           valid=d.valid.values.astype(bool), closed80=d.closed.values.astype(bool)))
    df['dt'] = df.t.diff().fillna(0)
    return df


def participant_codes():
    return sorted([f.name.split('.')[0] for f in PROCESSED.glob('S*.csv.gz')], key=lambda x: int(x[1:]))


def load_all():
    return {c: load_participant(c) for c in participant_codes()}


def load_labels():
    lab = pd.read_csv(ANNOTATIONS).rename(columns={'code': 'participant'})
    return lab.sort_values(['participant', 't'])


def label_segments(lab, p, dur):
    """Return list of (start, end, state) for participant p."""
    L = lab[lab.participant == p]
    segs = []
    rows = list(L.itertuples())
    for i, r in enumerate(rows):
        end = rows[i + 1].t if i + 1 < len(rows) else dur
        segs.append((r.t, end, r.state))
    return segs


def eval_masks(t, segs, H=H_DEFAULT, endpoint='drowsy'):
    """Pre-event window mask, negative (explicit active) mask, firstD time."""
    dstates = ['drowsy'] if endpoint == 'drowsy' else ['drowsy', 'ambiguous']
    d_on = [s for s in segs if s[2] in dstates]
    firstD = d_on[0][0] if d_on else None
    pre = np.zeros(len(t), bool)
    neg = np.zeros(len(t), bool)
    for s0, s1, st in segs:
        if st == 'active':
            if firstD is None:
                neg |= (t >= s0) & (t < s1)
            else:
                neg |= (t >= s0) & (t < min(s1, firstD - H))
    if firstD is not None:
        pre = (t >= firstD - H) & (t <= firstD)
    return pre, neg, firstD


# ---------------- signals ----------------
def gap_threshold(dt):
    med = np.median(dt[dt > 0])
    return max(0.10, 3 * med)


def perclos_time(t, dt, valid, closed, W, min_valid=0.8):
    """Time-based PERCLOS (%) over trailing window W. Sample state applies from previous
    timestamp to current; gaps > threshold excluded. Returns NaN if valid fraction < min_valid."""
    g = gap_threshold(dt)
    ok = (dt > 0) & (dt <= g) & valid
    vt = np.where(ok, dt, 0.0)
    ct = np.where(ok & closed, dt, 0.0)
    cv = np.concatenate([[0], np.cumsum(vt)])
    cc = np.concatenate([[0], np.cumsum(ct)])
    # index of first sample with time > t-W  (window (t-W, t])
    idx = np.searchsorted(t, t - W, side='right')
    # partial handling of boundary sample: approximate by excluding boundary sample fully
    V = cv[1:] - cv[idx]
    C = cc[1:] - cc[idx]
    avail = np.minimum(W, t - t[0])
    frac = np.where(avail > 0, V / np.maximum(avail, 1e-9), 0)
    out = np.where(V > 0, 100 * C / np.maximum(V, 1e-9), 0.0)
    out = np.where(frac < min_valid, np.nan, out)
    out[avail <= 0] = 0.0
    return out


def closure_runs(t, dt, valid, open_val, th_start=0.35, th_reopen=0.3675, min_d=0.03):
    """Hysteresis closure detector on the (mean) eye-opening value, matching the dataset
    definition (start < 0.35, reopen >= 0.3675, events not bridged across gaps/invalid).
    Returns online running closure duration per sample and completed events (start,end,dur)."""
    g = gap_threshold(dt)
    n = len(t)
    run = np.zeros(n)
    events = []
    cur_start = None
    for i in range(n):
        ok = valid[i] and not np.isnan(open_val[i]) and (i == 0 or dt[i] <= g)
        if not ok:
            cur_start = None  # break: unknown end -> discard
            continue
        v = open_val[i]
        if cur_start is None:
            if v < th_start:
                cur_start = t[i]
        else:
            if v >= th_reopen:
                d = t[i] - cur_start
                if d >= min_d:
                    events.append((cur_start, t[i], d))
                cur_start = None
            else:
                run[i] = t[i] - cur_start
    return run, np.array(events).reshape(-1, 3)


def stea_signal(t, events, lam, d0, dmax=1.0, dmu=1.0):
    """Exponentially decaying accumulation of sub-threshold completed events (exact superposition)."""
    E = np.zeros(len(t))
    if len(events) == 0:
        return E
    d = events[:, 2]; e = events[:, 1]
    keep = d < dmu
    w = np.clip(np.minimum(d, dmax) - d0, 0, None)
    w = w[keep]; e = e[keep]
    for wj, ej in zip(w, e):
        if wj <= 0:
            continue
        i = np.searchsorted(t, ej, side='left')
        if i >= len(t):
            continue
        E[i:] += wj * np.exp(-(t[i:] - ej) / lam)
    return E


def count_window(t, events, W, dmin):
    e = events[:, 1][events[:, 2] >= dmin] if len(events) else np.array([])
    e = np.sort(e)
    hi = np.searchsorted(e, t, side='right')
    lo = np.searchsorted(e, t - W, side='right')
    return (hi - lo).astype(float)


def cusum_signal(t, events, k, dmin=0.03, dmu=1.0):
    """Event-indexed CUSUM on log-duration of sub-threshold events; value held between events."""
    S = np.zeros(len(t))
    if len(events) == 0:
        return S
    m = (events[:, 2] >= dmin) & (events[:, 2] < dmu)
    e = events[m, 1]; x = np.log(events[m, 2])
    s = 0.0; vals = []
    for xi in x:
        s = max(0.0, s + (xi - k)); vals.append(s)
    vals = np.array(vals)
    idx = np.searchsorted(e, t, side='right') - 1
    S = np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0.0)
    return S


def ewma_signal(t, events, alpha, dmin=0.03, dmu=1.0):
    S = np.zeros(len(t))
    if len(events) == 0:
        return S
    m = (events[:, 2] >= dmin) & (events[:, 2] < dmu)
    e = events[m, 1]; x = events[m, 2]
    s = None; vals = []
    for xi in x:
        s = xi if s is None else alpha * xi + (1 - alpha) * s
        vals.append(s)
    vals = np.array(vals)
    idx = np.searchsorted(e, t, side='right') - 1
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], 0.0)


def persist(alarm, t, p):
    """Alarm must be continuously true for p seconds (vectorised)."""
    if p <= 0:
        return alarm
    n = len(alarm); idx = np.arange(n)
    last_false = np.maximum.accumulate(np.where(~alarm, idx, -1))
    start = t[np.clip(last_false + 1, 0, n - 1)]
    return alarm & ((t - start) >= p)


# ---------------- metrics ----------------
def episodes(t, mask, merge_gap=MERGE_GAP):
    """Count alarm episodes in mask (bool), merging gaps <= merge_gap seconds."""
    if not mask.any():
        return 0
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = t[np.where(d == 1)[0]]
    ends_idx = np.where(d == -1)[0]
    ends = t[np.clip(ends_idx - 1, 0, len(t) - 1)]
    n = 1
    for i in range(1, len(starts)):
        if starts[i] - ends[i - 1] > merge_gap:
            n += 1
    return n


def subject_metrics(t, dt, alarm, pre, neg, firstD):
    """Return dict: hit, lead, fa_episodes, neg_hours, active_time, neg_time."""
    g = gap_threshold(dt)
    dtc = np.where((dt > 0) & (dt <= g), dt, 0)
    neg_time = dtc[neg].sum()
    act_time = dtc[neg & alarm].sum()
    fa = episodes(t, alarm & neg)
    hit = np.nan; lead = np.nan
    if firstD is not None:
        a = alarm & pre
        hit = bool(a.any())
        lead = (firstD - t[a][0]) if hit else np.nan
    return dict(hit=hit, lead=lead, fa=fa, neg_time=neg_time, act_time=act_time)


def pool(mrows):
    """Aggregate list of subject metric dicts."""
    df = pd.DataFrame(mrows)
    d = df[df.hit.notna()]
    cov_n = int(d.hit.sum()); cov_d = len(d)
    neg_h = df.neg_time.sum() / 3600
    return dict(cov_n=cov_n, cov_d=cov_d, cov=cov_n / cov_d if cov_d else np.nan,
                far=df.fa.sum() / neg_h if neg_h > 0 else np.nan,
                act=df.act_time.sum() / df.neg_time.sum() if df.neg_time.sum() > 0 else np.nan,
                lead_med=np.nanmedian(d.lead) if cov_n else np.nan)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
