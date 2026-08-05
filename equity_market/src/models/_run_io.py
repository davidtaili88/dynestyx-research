"""Shared run-output persistence for the regime models.

Every model's main() calls save_run(...) right before plotting, so each fit's
P(bear) curve, ground-truth labels, dates, and a SPEC of exactly what model /
channels produced it land in equity_market/outputs/ automatically. This means a run
is reproducible-from-disk (reload the pkl, re-plot, or diff two models) without
re-running the ~8-minute NUTS fit.

Layout in outputs/:
  <run_name>.pkl   -- pickled dict (arrays + spec), the machine-readable record
  <run_name>.json  -- the spec ALONE, human-readable (open it to see what ran)
  runs_index.jsonl -- one line appended per save: {name, model, timestamp, metrics}
                      so `outputs/` is self-describing at a glance.

`outputs/` is git-ignored (see .gitignore: outputs/, *.pkl), so these are local
artifacts, never committed.
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import pathlib as _pathlib

import numpy as _np

_OUTPUTS_DIR = _pathlib.Path(__file__).resolve().parents[2] / "outputs"


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


def save_run(
    run_name: str,
    *,
    model: str,
    obs_cols,
    dates,
    p_bear,
    label,
    price=None,
    extra_spec: dict | None = None,
    mcmc=None,
) -> _pathlib.Path:
    """Persist one model run's output + a self-describing spec to outputs/.

    Args:
      run_name   : file stem, e.g. "regime_3state_rvdd_credit_curve". Overwrites
                   same-named prior runs (so re-running a model refreshes its record).
      model      : which model module produced this, e.g. "regime_model_3state".
      obs_cols   : the emission channels used, e.g. ["r_t","v_t","dd","cs_chg","inv"]
                   -- THE key part of the spec (what information the fit saw).
      dates      : the weekly DatetimeIndex for p_bear / label (one value per week).
      p_bear     : posterior-averaged filtered P(bear) array.
      label      : 0/1 ground-truth (Pagan-Sossounov) array aligned to dates.
      price      : optional weekly price array (handy for re-plotting).
      extra_spec : optional dict of anything else worth recording (horizons, window
                   sizes, target_accept, K, seed, ...). Merged into the saved spec.
      mcmc       : optional fitted MCMC -- if given, its posterior SAMPLES are saved
                   too, so the fit can be re-filtered later without re-running NUTS.

    Returns the path to the written .pkl.
    """
    import pickle

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    idx = _np.asarray(dates)
    p_bear = _np.asarray(p_bear, dtype=float)
    label = _np.asarray(label, dtype=float)

    metrics = _standard_metrics(p_bear, dates, label)

    spec = {
        "run_name": run_name,
        "model": model,
        "obs_cols": list(obs_cols),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    if extra_spec:
        spec.update(extra_spec)

    payload = {
        "spec": spec,
        "dates": idx,
        "p_bear": p_bear,
        "label": label,
    }
    if price is not None:
        payload["price"] = _np.asarray(price, dtype=float)
    if mcmc is not None:
        payload["samples"] = {k: _np.asarray(v) for k, v in mcmc.get_samples().items()}

    pkl_path = _OUTPUTS_DIR / f"{run_name}.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(payload, fh)

    # Human-readable spec sidecar.
    with open(_OUTPUTS_DIR / f"{run_name}.json", "w") as fh:
        _json.dump(spec, fh, indent=2)

    # Append a one-line index entry so outputs/ is browsable at a glance.
    with open(_OUTPUTS_DIR / "runs_index.jsonl", "a") as fh:
        fh.write(_json.dumps({
            "run_name": run_name,
            "model": model,
            "obs_cols": list(obs_cols),
            "timestamp": spec["timestamp"],
            "metrics": metrics,
        }) + "\n")

    print(
        f"[save_run] wrote {pkl_path.name} "
        f"(model={model}, channels={list(obs_cols)})\n"
        f"           metrics: cx_total={metrics['cx_total']} cx_dotcom={metrics['cx_dotcom']} "
        f"cx_1970s={metrics['cx_1970s']} recall={metrics['recall']:.3f} "
        f"false_alarm={metrics['false_alarm']:.3f}"
    )
    return pkl_path


def load_run(run_name: str) -> dict:
    """Reload a run saved by save_run -> the payload dict (spec, dates, p_bear, ...)."""
    import pickle

    with open(_OUTPUTS_DIR / f"{run_name}.pkl", "rb") as fh:
        return pickle.load(fh)
