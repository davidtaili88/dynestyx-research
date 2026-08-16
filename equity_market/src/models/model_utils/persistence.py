"""Persistence for the regime models: save/load run records and cached fits.

Two families, both writing to equity_market/outputs/ (git-ignored -> local artifacts):

  * save_run / load_run  -- a full model RUN record: the P(bear) curve, labels, dates and a
    self-describing spec (model, channels, metrics). Called by fit_mode_processor after each
    fit so a run is reproducible-from-disk without re-running the ~8-minute NUTS fit.
  * save_fit / load_fit  -- a lighter CACHE of a fit's posterior samples (+ optional extras
    like the P(bear) curve), used by the trading strategy / sweeps to skip a refit. These are
    MODEL-AGNOSTIC pickle helpers: obs_cols is passed IN (not read from a model module), so
    nothing has to import a specific model just to persist a fit.

Layout in outputs/:
  <run_name>.pkl   -- pickled dict (arrays + spec), the machine-readable record
  <run_name>.json  -- the spec ALONE, human-readable (open it to see what ran)
  runs_index.jsonl -- one line appended per save_run: {name, model, timestamp, metrics}

Metrics used in the run spec come from metrics.py (the shared yardstick).
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import pathlib as _pathlib

import numpy as _np

from metrics import _standard_metrics

_OUTPUTS_DIR = _pathlib.Path(__file__).resolve().parents[3] / "outputs"


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
      run_name   : file stem, e.g. "regime_3state_strategy_pbear". Overwrites
                   same-named prior runs (so re-running a model refreshes its record).
      model      : which model module produced this, e.g. "regime_model_3state".
      obs_cols   : the emission channels used, e.g. ["r_t","v_t","dd"]
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


def save_fit(mcmc, name: str, obs_cols, extra: dict | None = None) -> _pathlib.Path:
    """Persist a fitted model's POSTERIOR SAMPLES (+ optional extras) to outputs/<name>.pkl
    so it can be reloaded without a fresh NUTS run.

    MODEL-AGNOSTIC: obs_cols (the channels the fit used) is passed IN, not read from a model
    module -- so any caller can cache a fit without importing a specific model. We save
    mcmc.get_samples() (a plain dict of numpy/jax arrays, incl. the recorded
    f_filtered_states) rather than the live MCMC object -- portable and reload-safe. `extra`
    can carry the channel config / P(bear) curve / dates so a saved fit is self-describing.
    Returns the written path.
    """
    import pickle

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = {k: _np.asarray(v) for k, v in mcmc.get_samples().items()}
    payload = {"samples": samples, "obs_cols": list(obs_cols)}
    if extra:
        payload["extra"] = extra
    path = _OUTPUTS_DIR / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_fit(name: str) -> dict:
    """Reload a fit saved by save_fit -> {"samples": {...}, "obs_cols": [...], ...}.

    The returned "samples" dict can be fed to filtered_p_bear_over via a thin Predictive
    wrapper, or inspected directly for posterior parameter values. Raises FileNotFoundError
    if outputs/<name>.pkl is absent (nothing cached yet).
    """
    import pickle

    path = _OUTPUTS_DIR / f"{name}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)
