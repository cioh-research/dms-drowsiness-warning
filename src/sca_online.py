"""Minimal streaming implementation of the short-closure accumulation (SCA) rule.

    w_j  = max(d_j - d0, 0) * 1[d_j < 1 s]                     (contribution of a completed closure j)
    E(t) = sum_{e_j <= t} w_j * exp(-(t - e_j) / lam)           (decaying evidence)
    SCA alarm on when E >= h has held for p seconds; final alarm = standard alarm OR SCA alarm.

The state is a single number (E) and the time of the last update, so the rule can run on an embedded DMS.
Example (paper settings are selected per false-alarm budget; the simplified rule uses equal weights,
d0 = 0.2 s, lam = 120 s, p = 0 and h of about 2.5 at 5 false alarms per hour):

    sca = SCA(d0=0.2, lam=120.0, h=2.5, equal_weight=True)
    for t, closure_duration in stream:          # closure_duration is None unless a closure just ended at t
        sca_on = sca.update(t, closure_duration)
        alarm = standard_alarm(t) or sca_on
"""
import math


class SCA:
    def __init__(self, d0=0.2, lam=120.0, h=2.5, p=0.0, dmax=1.0, equal_weight=False):
        self.d0, self.lam, self.h, self.p, self.dmax, self.equal = d0, lam, h, p, dmax, equal_weight
        self.E = 0.0; self.t_last = None; self.above_since = None

    def weight(self, d):
        if d >= self.dmax:            # closures >= 1 s are left to the eye-closure-duration (ECD) criterion
            return 0.0
        if self.equal:
            return 1.0 if d >= self.d0 else 0.0
        return max(d - self.d0, 0.0)

    def update(self, t, closure_duration=None):
        """Advance to time t (s); pass the duration of a closure that ended at t, if any. Returns the SCA alarm state."""
        if self.t_last is not None:
            self.E *= math.exp(-(t - self.t_last) / self.lam)
        self.t_last = t
        if closure_duration is not None:
            self.E += self.weight(closure_duration)
        if self.E >= self.h:
            if self.above_since is None:
                self.above_since = t
            return (t - self.above_since) >= self.p
        self.above_since = None
        return False


if __name__ == '__main__':
    # consistency check against the batch implementation used in the analyses
    import numpy as np
    from core import load_participant, closure_runs, stea_signal, participant_codes
    code = participant_codes()[0]; df = load_participant(code)
    t = df.t.values; _, ev = closure_runs(t, df.dt.values, df.valid.values, df.open.values)
    batch = stea_signal(t, ev, 120.0, 0.2)
    sca = SCA(d0=0.2, lam=120.0, h=np.inf); ends = {}
    for s, e, d in ev:
        i = int(np.searchsorted(t, e, side='left')); ends.setdefault(i, []).append(d)
    online = np.zeros(len(t))
    for i, ti in enumerate(t):
        ds = ends.get(i, [None])
        for d in ds: sca.update(ti, d)
        online[i] = sca.E
    print(code, 'max |online - batch| =', float(np.max(np.abs(online - batch))))
