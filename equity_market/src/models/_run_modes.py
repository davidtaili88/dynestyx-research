"""Shared fit-mode runner for every regime model (2/3/4-state, hsmm).

ONE implementation of the global-vs-walkforward evaluation, imported by each model
instead of copy-pasted -- so "run the file" offers the same explicit mode choice
everywhere and there is never again a walk-forward path hiding in a function the
__main__ block doesn't reach.

Each model module M must provide (the uniform bits every model already has):
  M.fit(train_obs, num_warmup=, num_samples=) -> mcmc
  M.filtered_p_bear_over(mcmc, obs_frame)      -> np.ndarray P(bear) per week
  M.plot_regime_fit(...)                       -> the standard 3-panel figure
  M.K                                          -> number of states (for the spec)
  M.__name__ / module file                     -> used to name the model

Optional hooks (read via getattr with safe defaults, so simple bivariate models and
channel-augmented ones both work):
  M.obs_kwargs()  -> dict passed to ds.observations()/ds.split(); default {} (bivariate)
  M.needs_macro   -> bool, load the macro CSVs (credit/curve); default False
  M.obs_cols()    -> list[str] channel names for the run_name/spec; default from a
                     probe of ds.observations(**obs_kwargs).columns
  M.extra_spec()  -> dict of extra config to record; default {}
  M.walk_train_years / M.walk_step_years -> walk-forward window/step; default 8 / 1
"""
from __future__ import annotations


def _cfg(model, name, default):
    """getattr that also calls zero-arg callables, so a model can expose config as
    either an attribute or a function."""
    val = getattr(model, name, default)
    return val() if callable(val) else val


def _load_ds(model):
    from data import load_regime_dataset

    needs_macro = _cfg(model, "needs_macro", False)
    return load_regime_dataset(start="1957-03-01", include_vix=False, include_macro=needs_macro)


def _obs_cols(model, obs_frame):
    cols = _cfg(model, "obs_cols", None)
    if cols is not None:
        return list(cols)
    # Fall back to the frame's own columns (drops any non-observation columns just in case).
    return [c for c in obs_frame.columns]


def _short_model_name(model) -> str:
    # regime_model_3state -> 3state ; regime_model_2state_hsmm -> 2state_hsmm
    n = model.__name__.split(".")[-1]
    return n.replace("regime_model_", "")


def _run_global(model) -> None:
    """Single 80/20 fit, filtered forward over all history (train/test generalization)."""
    from labels import pagan_sossounov_label
    from pagan_sossounov import _T_CENSOR_MONTHS, _months_to_weeks
    from _run_io import save_run

    ds = _load_ds(model)
    kw = _cfg(model, "obs_kwargs", {})
    train_obs, test_obs = ds.split(**kw)
    full_obs = ds.observations(**kw)
    split_date = test_obs.index[0]
    idx = full_obs.index

    mcmc = model.fit(train_obs)
    mcmc.print_summary()

    p_bear = model.filtered_p_bear_over(mcmc, full_obs)
    label = pagan_sossounov_label(ds.weekly_price.loc[idx]).reindex(idx).ffill()

    cols = _obs_cols(model, full_obs)
    extra = dict(_cfg(model, "extra_spec", {}))
    extra.update(K=getattr(model, "K", None), fit="global 80/20", split_date=str(split_date.date()))
    save_run(
        f"regime_{_short_model_name(model)}_{'_'.join(cols)}_global",
        model=model.__name__.split(".")[-1],
        obs_cols=cols,
        dates=idx, p_bear=p_bear, label=label,
        price=ds.weekly_price.loc[idx].to_numpy(),
        extra_spec=extra, mcmc=mcmc,
    )

    model.plot_regime_fit(
        dates=idx, price=ds.weekly_price.loc[idx], r_t=full_obs["r_t"],
        p_bear=p_bear, bear_label=label, split_date=split_date,
        provisional_weeks=_months_to_weeks(_T_CENSOR_MONTHS),
        short_shocks=[("COVID", "2020-02-19", "2020-04-01")],
    )


def _run_walkforward(model) -> None:
    """Rolling trailing-window refits, stitched OOS P(bear) (non-stationarity-robust)."""
    from labels import pagan_sossounov_label
    from pagan_sossounov import _T_CENSOR_MONTHS, _months_to_weeks
    from _run_io import save_run

    ds = _load_ds(model)
    kw = _cfg(model, "obs_kwargs", {})
    full_obs = ds.observations(**kw)
    idx = full_obs.index

    train_years = _cfg(model, "walk_train_years", 8)
    step_years = _cfg(model, "walk_step_years", 1)
    print(f"Walk-forward ({train_years}yr trailing train, {step_years}yr OOS steps):")
    p_bear = model.walk_forward_p_bear(full_obs, train_years=train_years, step_years=step_years)
    label = pagan_sossounov_label(ds.weekly_price.loc[idx]).reindex(idx).ffill()
    oos_start = p_bear.first_valid_index()

    cols = _obs_cols(model, full_obs)
    extra = dict(_cfg(model, "extra_spec", {}))
    extra.update(K=getattr(model, "K", None), fit=f"walk-forward {train_years}yr/{step_years}yr",
                 oos_start=str(oos_start.date()) if oos_start is not None else None)
    save_run(
        f"regime_{_short_model_name(model)}_{'_'.join(cols)}_walkforward",
        model=model.__name__.split(".")[-1],
        obs_cols=cols,
        dates=idx, p_bear=p_bear.to_numpy(), label=label,
        price=ds.weekly_price.loc[idx].to_numpy(),
        extra_spec=extra,  # one fit per fold -> no single mcmc
    )

    model.plot_regime_fit(
        dates=idx, price=ds.weekly_price.loc[idx], r_t=full_obs["r_t"],
        p_bear=p_bear.to_numpy(), bear_label=label, split_date=oos_start,
        split_label="dotted line = OOS coverage begins; ALL of it right is walk-forward out-of-sample (rolling refit)",
        provisional_weeks=_months_to_weeks(_T_CENSOR_MONTHS),
        short_shocks=[("COVID", "2020-02-19", "2020-04-01")],
    )


def make_walk_forward_p_bear(model):
    """Build a walk_forward_p_bear bound to `model`'s fit/filtered_p_bear_over.

    Identical rolling-window logic for every model (it only needs those two calls),
    so each model gets it via `walk_forward_p_bear = make_walk_forward_p_bear(sys.modules[__name__])`
    instead of copy-pasting the loop. Warm-started, causal (see the returned fn docstring).
    """
    import numpy as np
    import pandas as pd

    def walk_forward_p_bear(full_obs, train_years: int = 8, step_years: int = 1,
                            num_warmup: int = 1000, num_samples: int = 1000):
        """Rolling-window OOS P(bear): refit on each trailing `train_years` window, emit
        OOS P(bear) for the next `step_years`, stitch. Params come only from each fold's
        trailing window (non-stationarity fix); the filter pass spans [train..test] so it
        is warm at the test block, but P(bear_t) uses only y_1:t so every kept value is OOS.
        First `train_years` are the initial window (NaN)."""
        idx = full_obs.index
        n = len(full_obs)
        w = int(train_years * 52)
        step = int(step_years * 52)
        p_bear = np.full(n, np.nan)
        fold = 0
        test_start = w
        while test_start < n:
            test_end = min(test_start + step, n)
            train_slice = full_obs.iloc[test_start - w:test_start]
            filter_slice = full_obs.iloc[test_start - w:test_end]
            fold += 1
            print(f"  fold {fold}: train {train_slice.index[0].date()}..{train_slice.index[-1].date()}"
                  f"  -> test {idx[test_start].date()}..{idx[test_end - 1].date()}")
            mcmc = model.fit(train_slice, num_warmup=num_warmup, num_samples=num_samples)
            pb_filter = np.asarray(model.filtered_p_bear_over(mcmc, filter_slice))
            keep = test_end - test_start
            p_bear[test_start:test_end] = pb_filter[-keep:]
            test_start = test_end
        return pd.Series(p_bear, index=idx, name="p_bear_oos")

    return walk_forward_p_bear


_FIT_MODES = {"global": _run_global, "walkforward": _run_walkforward}


def run_main(model, mode: str = "global") -> None:
    """Entry point every model's main() delegates to.
      'global'      -> single 80/20 fit (fast; train/test). DEFAULT.
      'walkforward' -> rolling refits (slow; non-stationarity-robust).
    CLI: `python regime_model_Xstate.py [global|walkforward]`.
    """
    if mode not in _FIT_MODES:
        raise ValueError(f"mode must be one of {sorted(_FIT_MODES)}, got {mode!r}")
    _FIT_MODES[mode](model)
