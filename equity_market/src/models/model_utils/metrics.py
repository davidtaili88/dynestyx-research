"""Shared scoring metrics for the regime models (the common yardstick).

_standard_metrics / _crossings are used across the models AND the parameter sweeps to
score a P(bear) curve against the ground-truth labels -- importable WITHOUT running or
persisting a fit, which is why they live apart from persistence.py.
"""

from __future__ import annotations

import numpy as _np


def _crossings(p_bear, idx, lo, hi=None):
    """Count 0.5-threshold sign flips of P(bear) in [lo, hi] -- the whipsaw proxy."""
    import pandas as pd

    s = pd.Series(_np.asarray(p_bear), index=idx)
    s = s.loc[lo:hi] if hi is not None else s.loc[lo:]
    if len(s) < 2:
        return 0
    b = (s.values > 0.5).astype(int)
    return int(_np.abs(_np.diff(b)).sum())


def _standard_metrics(p_bear, idx, label):
    """The same yardstick used across experiments: whipsaw crossings (total + the
    dotcom and 1970s windows) plus recall / false-alarm. Computed only where the
    label is available; windows outside the data just return 0 crossings.
    """
    p = _np.asarray(p_bear, dtype=float)
    lab = _np.asarray(label, dtype=float)
    is_bear = lab == 1
    is_bull = lab == 0
    start = idx[0]
    m = {
        "cx_total": _crossings(p, idx, start),
        "cx_dotcom": _crossings(p, idx, "2000-01-01", "2003-06-30"),
        "cx_1970s": _crossings(p, idx, "1970-01-01", "1983-01-01"),
        # recall here = mean P(bear) over true-bear weeks (soft recall); false-alarm =
        # mean P(bear) over true-bull weeks. Read the two TOGETHER (see the metric note
        # in the model docs): a whipsaw drop that also tanks recall is not a real fix.
        "recall": float(p[is_bear].mean()) if is_bear.any() else float("nan"),
        "false_alarm": float(p[is_bull].mean()) if is_bull.any() else float("nan"),
        "n_weeks": int(len(p)),
    }
    return m
