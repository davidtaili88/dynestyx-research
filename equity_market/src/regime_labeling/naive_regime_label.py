"""
Plain 20% drawdown/rally rule, applied to the weekly price series so the
label lines up with the model's weekly filtering cadence.

This is a *smoothed* (hindsight) verdict, used only as an answer key to score
the nowcast against -- never as a model input. The whole evaluation is "how
well does the filtered nowcast anticipate this smoothed truth."

The output is a plain 0/1 Series (label_t, indexed like the price series),
so this can be swapped for a different labeling scheme (e.g. Pagan-Sossounov)
later without touching the evaluation code that consumes it -- the evaluator
should only ever depend on a generic bear/bull Series, not on how it was derived.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401  (puts sibling src/ subfolders on sys.path)

import numpy as np
import pandas as pd

BEAR = 1
BULL = 0


def drawdown_rally_label(price: pd.Series, threshold: float = 0.20) -> pd.Series:
    """Label each period BEAR (1) or BULL (0) using a running peak/trough rule.

    - Track the running peak of `price`. A bear phase begins the first period
      price closes >= `threshold` below that peak.
    - Within a bear phase, track the running trough. A bull phase begins the
      first period price closes >= `threshold` above that trough.
    - The series starts BULL (matches convention: you need a >=20% drop to
      declare a bear, so the initial state can't be bear before any drop is
      observed).

    Returns a Series of 0/1 int labels, same index as `price`.
    """
    values = price.to_numpy(dtype=float)
    n = len(values)
    labels = np.empty(n, dtype=int)

    state = BULL
    peak = values[0]
    trough = values[0]

    for i in range(n):
        p = values[i]
        if state == BULL:
            peak = max(peak, p)
            if p <= peak * (1.0 - threshold):
                state = BEAR
                trough = p
        else:  # state == BEAR
            trough = min(trough, p)
            if p >= trough * (1.0 + threshold):
                state = BULL
                peak = p
        labels[i] = state

    return pd.Series(labels, index=price.index, name="label_t")


def turning_points(label: pd.Series) -> pd.DataFrame:
    """Extract the timestamps where the label switches, for use in the
    timeliness evaluation.

    Returns a DataFrame with columns [time, from_state, to_state].
    """
    changes = label[label != label.shift(1)].iloc[1:]
    return pd.DataFrame(
        {
            "time": changes.index,
            "from_state": label.shift(1).loc[changes.index].to_numpy(),
            "to_state": changes.to_numpy(),
        }
    ).reset_index(drop=True)
