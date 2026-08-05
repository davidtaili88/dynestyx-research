"""OOS-robustness run: HOLD OUT 1999-2013 as a contiguous middle block.

This does NOT mutate the shipped strategy. It fits the regime model on the
"rest" of history (1957-1998 UNION 2013-present) so the model's emission /
transition parameters NEVER see 1999-2013, then filters P(bear) forward over
ALL history and backtests the *unchanged* strategy (build_positions) over the
held-out 1999-2013 window -- reporting Sharpe / maxDD vs buy&hold there.

Why this is a legitimate OOS test for a filtered state-space model: the forward
filter that produces P(bear) INSIDE 1999-2013 is seeded by pre-1999 data, but the
PARAMETERS it uses were learned only from data outside the window. So the curve on
1999-2013 is genuinely out-of-sample.

One caveat, made explicit: fit() assumes evenly-spaced weekly rows (obs_times =
arange, and the transition matrix is one Markov step per row). Concatenating the
1998 tail directly onto the 2013 head creates exactly ONE bogus one-week
"transition" at the seam -- negligible in a ~3600-week series, and it only affects
the fit, never the filtered curve (which runs over the real contiguous history).

Run:  python trading_strategies/oos_1999_2013.py            # 3-state (default)
      python trading_strategies/oos_1999_2013.py --model 4state
      python trading_strategies/oos_1999_2013.py --refit     # force refit
"""
from __future__ import annotations

import sys as _sys
import pathlib as _pl

_SRC = _pl.Path(__file__).resolve().parents[1] / "src"
_sys.path.insert(0, str(_SRC))
import _syspath  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from data import load_regime_dataset  # noqa: E402

import regime_pbear_strategy as strat  # noqa: E402  (reuse its legs + backtest verbatim)

HOLDOUT_START = "1999-01-01"
HOLDOUT_END = "2013-01-01"   # exclusive upper edge (1999-01-01 .. 2012-12-31)


def load_pbear_holdout(model_key: str = "3state", refit: bool = False):
    """Fit on history OUTSIDE [1999,2013), filter P(bear) over ALL weeks.

    Mirrors strat.load_pbear but replaces the 80/20 chronological train split with
    a middle-block holdout. Caches to a DISTINCT pkl so it never clobbers the
    shipped strategy's fit.
    """
    mod, save_fit, load_fit, needs_macro = strat._load_model(model_key)
    cache_name = f"regime_{model_key}_oos_9913_pbear"

    ds = load_regime_dataset(start=strat.DATA_START, include_vix=False,
                             include_macro=needs_macro)
    kw = mod.obs_kwargs()
    full_obs = ds.observations(**kw)
    idx = full_obs.index
    price = ds.weekly_price.loc[idx]
    r_t = full_obs["r_t"]

    if not refit:
        try:
            cached = load_fit(cache_name)
            pb = cached["extra"]["p_bear"]
            if len(pb) == len(idx):
                print(f"[cache] loaded holdout P(bear) from outputs/{cache_name}.pkl")
                return price, r_t, pd.Series(np.asarray(pb), index=idx, name="p_bear")
            print("[cache] stale -> refitting")
        except FileNotFoundError:
            print(f"[cache] none -> fitting {model_key} on the 1957-98 U 2013-now train set")

    # TRAIN = everything OUTSIDE the holdout window (non-contiguous).
    in_holdout = (idx >= HOLDOUT_START) & (idx < HOLDOUT_END)
    train_obs = full_obs.loc[~in_holdout]
    n_hold = int(in_holdout.sum())
    print(f"Fitting {model_key}: train={len(train_obs)} wk (pre-1999 + post-2012), "
          f"HELD OUT {n_hold} wk in [{HOLDOUT_START}, {HOLDOUT_END}). "
          f"channels={list(full_obs.columns)}")
    mcmc = mod.fit(train_obs)
    p_bear = np.asarray(mod.filtered_p_bear_over(mcmc, full_obs))

    save_fit(mcmc, cache_name, extra={
        "p_bear": p_bear,
        "dates": [str(d.date()) for d in idx],
        "obs_cols": list(full_obs.columns),
    })
    return price, r_t, pd.Series(p_bear, index=idx, name="p_bear")


def _report(name, sub, positions_sub):
    d = strat.backtest(sub["price"], sub["r_t"], positions_sub)
    s = strat.performance_stats(d["strat_logret"], d["strat_equity"], positions=d["pos_traded"])
    b = strat.performance_stats(d["bh_logret"], d["bh_equity"])
    print(f"\n{name}  ({d.index[0].date()} -> {d.index[-1].date()}, {len(d)} wk)")
    print(f"  STRATEGY : Sharpe {s['sharpe']:6.2f}  ann {s['ann_return']*100:6.2f}%  "
          f"vol {s['ann_vol']*100:5.2f}%  maxDD {s['max_drawdown']*100:6.1f}%  "
          f"tot {s['total_return']*100:7.1f}%")
    print(f"  BUY&HOLD : Sharpe {b['sharpe']:6.2f}  ann {b['ann_return']*100:6.2f}%  "
          f"vol {b['ann_vol']*100:5.2f}%  maxDD {b['max_drawdown']*100:6.1f}%  "
          f"tot {b['total_return']*100:7.1f}%")
    return s, b


def main(model_key: str = "3state", refit: bool = False) -> None:
    price, r_t, p_bear = load_pbear_holdout(model_key=model_key, refit=refit)

    # Positions from the UNCHANGED strategy legs, over full history (causal sweep).
    positions = strat.build_positions(p_bear)
    frame = pd.DataFrame({"price": price, "r_t": r_t, "position": positions})

    print("\n" + "=" * 74)
    print(f"OOS HOLDOUT TEST [{model_key}]  --  test window = [{HOLDOUT_START}, {HOLDOUT_END})")
    print("model fit on 1957-1998 + 2013-now; strategy params UNCHANGED")
    print("=" * 74)

    mask = (frame.index >= HOLDOUT_START) & (frame.index < HOLDOUT_END)
    _report("HELD-OUT TEST  1999-2013", frame.loc[mask], positions.loc[mask])
    _report("TRAIN (rest of history)", frame.loc[~mask], positions.loc[~mask])
    _report("FULL SAMPLE", frame, positions)
    print("=" * 74)


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    main(
        model_key=strat._flag(argv, "--model", str, "3state"),
        refit="--refit" in argv,
    )
