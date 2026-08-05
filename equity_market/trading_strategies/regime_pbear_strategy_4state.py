"""Regime-switching S&P 500 strategy driven by the 4-STATE / 3-EMISSION nowcast.

Dedicated mirror of regime_pbear_strategy.py, HARDWIRED to the 4-state model
(src/models/regime_model_4state.py, channels [r_t, v_t, dd], global fit). The 4-state
splits the bear regime into two flavors -- TURBULENT_BEAR (violent crash) and
CALM_BEAR (calm grind) -- and its filtered_p_bear_over already SUMS both, so P(bear)
here means the same thing (total bear probability) as in the 3-state twin. The blockier
4-state P(bear) is what produced the cleaner regime plot; this file trades exactly that
curve. Same strategy logic, thresholds, backtest, neutral analysis, sweep, and plot as
the shared script -- only the model wiring and the output/cache names differ.

RULES (4-state variant -- LONG same as 3-state, SHORT is PERSISTENCE-gated):
  * P(bear) < P_LONG                    -> LONG the S&P (+1)
  * P(bear) > P_SHORT_ENTER for
    SHORT_DWELL_WEEKS consecutive weeks  -> SHORT the S&P (-SHORT_SIZE), held until
                                            P(bear) falls back below P_SHORT_EXIT
  * otherwise                           -> FLAT (0)
The persistence gate is the KEY difference: the 4-state's blocky P(bear) hits ~1 on
brief scares AND real bears, and shorting both loses money (the scares' -104% drowns
the sustained bears' +49%). Requiring the bear to PERSIST a full quarter before shorting
keeps the dotcom/GFC edge and drops the scares. See the parameter block for the full
investigation + the anti-overfit note on why the dwell is 13 (a plateau midpoint, not
the argmax). Shorting is a DRAWDOWN HEDGE here, not a Sharpe win.

NO-LOOKAHEAD CONVENTION. P(bear_t) is the CAUSAL filtered probability using data
through week t's Friday close (the model's forward filter, filtered_p_bear_over).
The position implied by P(bear_t) is only tradeable at the t close, so it earns the
return of week t+1. We shift the position forward one week before applying returns --
so no bar's return is ever earned by a signal that peeked at it.

PnL / Sharpe / drawdown reporting mirrors the trading-project template style used for
the short-bonds strategy: an equity curve, annualized return / vol / Sharpe, max
drawdown, hit rate, and a strategy-vs-buy&hold plot.

Run:  python trading_strategies/regime_pbear_strategy_4state.py
      python trading_strategies/regime_pbear_strategy_4state.py --sweep       # dwell/size plateau check
      python trading_strategies/regime_pbear_strategy_4state.py --short-size 0 # long/flat only
      python trading_strategies/regime_pbear_strategy_4state.py --refit        # ignore cache, refit NUTS
      python trading_strategies/regime_pbear_strategy_4state.py --short-dwell 10 --short-size 1.0
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

# This file lives in equity_market/trading_strategies/, a SIBLING of src/. Put the
# src/ subfolders (dataset, models, ...) on sys.path so the flat imports below work,
# reusing the same _syspath machinery every src/ script uses.
_SRC = _pl.Path(__file__).resolve().parents[1] / "src"
_sys.path.insert(0, str(_SRC))
import _syspath  # noqa: E402,F401  (adds src/ subfolders: dataset, models, ...)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import regime_model_4state as rm  # noqa: E402  (the model this file is hardwired to)
from data import load_regime_dataset  # noqa: E402

# This strategy is pinned to the 4-state model. Its cache/outputs carry this key so they
# never collide with the shared script's 3-state artifacts.
MODEL_KEY = "4state"


# ----------------------------------------------------------------------------
# Strategy parameters
# ----------------------------------------------------------------------------
# LONG side: same as the 3-state (P_LONG=0.60 -- stay long through the rising 0.10-0.75
# P(bear) band; the calibration table showed forward returns stay positive up to ~0.75).
#
# SHORT side -- PERSISTENCE-GATED, and this is where the 4-state genuinely differs.
# We investigated why the 4-state "nails dotcom/GFC on paper but loses money short". The
# short PnL splits cleanly by BEAR DURATION:
#   * SUSTAINED bears (>=~1 quarter): +49% short PnL total -- dotcom (+28%, both legs),
#     GFC (+23%), 1974 (+12%), 1970 (+9%). This is the real edge.
#   * SHORT SCARES (<~1 quarter): -104% across 25 episodes -- brief P(bear)>0.97 blips
#     around dips that V-recover (2010, 2011, 1998, 2020, 2025). These drown the edge.
# So we short ONLY AFTER P(bear) has been PERSISTENTLY high -- i.e. only bears that have
# already lasted, filtering out the scares. Rule: enter short after P(bear) > P_SHORT_ENTER
# for SHORT_DWELL_WEEKS CONSECUTIVE weeks; hold the short while P(bear) > P_SHORT_EXIT.
#
# SHORT_DWELL_WEEKS = 13 (one quarter). *** DELIBERATELY NOT the argmax. *** A dwell scan
# (4..28wk) showed a BROAD, FLAT plateau: any dwell in ~10-18wk gives Sharpe ~0.47-0.49
# and maxDD ~-27 to -30% (half-size). There is NO sharp peak -- picking the single best
# (16) would be overfitting to this one sample. 13wk = one calendar quarter is a round,
# economically-meaningful unit ("short a bear that has persisted a full quarter") sitting
# in the middle of the plateau, chosen for robustness NOT for the top backtest number.
#
# SHORT_SIZE = 0.5 (half a unit). The short is a DRAWDOWN HEDGE, not a return driver:
# even the best short-inclusive config (~0.49) does NOT beat plain long/flat on Sharpe
# (~0.52) -- what shorting buys is a shallower max drawdown (-26.5% vs -28.5%). Half-size
# keeps the hedge from diluting the higher-Sharpe long book (a risk-budgeting choice, not
# a fitted one). Set SHORT_SIZE=0 to recover pure long/flat.
# ALL of these are OVERRIDABLE from the CLI (--long/--short-entry/--short-exit/
# --short-dwell/--short-size) so the defaults are a starting point, not hard-coded.
P_LONG = 0.60           # P(bear) below this -> long
P_SHORT_ENTER = 0.97    # P(bear) above this (persistently) -> arm the short
P_SHORT_EXIT = 0.90     # once short, cover when P(bear) falls back below this
SHORT_DWELL_WEEKS = 13  # consecutive weeks above P_SHORT_ENTER before shorting (1 quarter)
SHORT_SIZE = 0.5        # short position size (hedge sleeve; 0 = long/flat only)
WEEKS_PER_YEAR = 52.0   # weekly (W-FRI) bars -> annualization factor
DATA_START = "1957-03-01"  # same history the model runs are fit/evaluated on (_run_modes)


# ----------------------------------------------------------------------------
# 1. Get the causal filtered P(bear) curve from the 4-state / 3-emission model
# ----------------------------------------------------------------------------
def load_pbear(refit: bool = False):
    """Return (price, r_t, p_bear) as pandas objects on the weekly index.

    p_bear is the GLOBAL-fit causal filtered P(bear_t | y_1:t) from the 4-state model
    with its current channel config (r_t, v_t, dd). We fit ONCE on the 80/20 train split
    (rm.fit) and filter forward over ALL history (rm.filtered_p_bear_over) -- identical
    to _run_modes._run_global, so the curve the strategy trades is exactly the one the
    nowcast plots. rm.filtered_p_bear_over SUMS both bear flavors (turbulent + calm), so
    P(bear) is total bear probability.

    The fit is cached to outputs/regime_4state_strategy_pbear.pkl (NUTS is expensive);
    pass refit=True to force a fresh fit. save_fit/load_fit live on the 3-state module
    (generic pickle helpers), so we borrow them; the 4-state never needs the macro CSVs.
    """
    import regime_model_3state as _base  # generic save_fit/load_fit pickle helpers
    save_fit, load_fit = _base.save_fit, _base.load_fit
    cache_name = f"regime_{MODEL_KEY}_strategy_pbear"

    ds = load_regime_dataset(start=DATA_START, include_vix=False, include_macro=False)
    kw = rm.obs_kwargs()
    full_obs = ds.observations(**kw)
    idx = full_obs.index
    price = ds.weekly_price.loc[idx]
    r_t = full_obs["r_t"]

    # Try the cache first: it stores the P(bear) curve keyed to the obs index so we can
    # skip the NUTS fit on re-runs (strategy tuning is cheap; the fit is not).
    if not refit:
        try:
            cached = load_fit(cache_name)
            pb = cached["extra"]["p_bear"]
            if len(pb) == len(idx):
                print(f"[cache] loaded P(bear) from outputs/{cache_name}.pkl")
                return price, r_t, pd.Series(np.asarray(pb), index=idx, name="p_bear")
            print("[cache] stale (length changed) -> refitting")
        except FileNotFoundError:
            print(f"[cache] none found -> fitting the {MODEL_KEY} model (this is the slow step)")

    train_obs, _test_obs = ds.split(**kw)
    print(f"Fitting {MODEL_KEY} model on {len(train_obs)} train weeks, channels={list(full_obs.columns)} ...")
    mcmc = rm.fit(train_obs)
    p_bear = np.asarray(rm.filtered_p_bear_over(mcmc, full_obs))

    save_fit(mcmc, cache_name, extra={
        "p_bear": p_bear,
        "dates": [str(d.date()) for d in idx],
        "obs_cols": list(full_obs.columns),
    })
    return price, r_t, pd.Series(p_bear, index=idx, name="p_bear")


# ----------------------------------------------------------------------------
# 2. Turn P(bear) into a position: long / short / hold-band / whipsaw-neutral
# ----------------------------------------------------------------------------
def build_positions(p_bear: pd.Series, p_long: float = P_LONG,
                    p_short_enter: float = P_SHORT_ENTER, p_short_exit: float = P_SHORT_EXIT,
                    short_dwell_weeks: int = SHORT_DWELL_WEEKS,
                    short_size: float = SHORT_SIZE) -> pd.Series:
    """Map the P(bear) curve to a position per week: +1 long, -short_size short, 0 flat.

    Stateful sweep left-to-right (using ONLY p_bear up to t -- causal):
      * SHORT (PERSISTENCE-GATED): keep a run-length counter of consecutive weeks with
        p_bear > p_short_enter. Enter a short only when that counter reaches
        short_dwell_weeks (the bear has PERSISTED, not a brief scare). Once short, stay
        short until p_bear falls back below p_short_exit, then cover. This is the fix for
        the 4-state's false-alarm bleed -- see the module notes.
      * LONG: if not short and p_bear < p_long -> +1.
      * FLAT otherwise (the ambiguous middle, or a bear that hasn't persisted yet).

    short_size scales the short leg (0 -> pure long/flat). Position at t is decided at
    the t close; the backtest shifts it +1 week before applying returns (no lookahead).
    """
    pb = p_bear.to_numpy()
    n = len(pb)
    pos = np.zeros(n, dtype=float)
    run = 0          # consecutive weeks with p_bear > p_short_enter
    in_short = False
    for t in range(n):
        run = run + 1 if pb[t] > p_short_enter else 0
        if in_short:
            if pb[t] < p_short_exit:   # bear resolving -> cover
                in_short = False
        elif short_size > 0 and run >= short_dwell_weeks:
            in_short = True            # persistence confirmed -> arm the short
        if in_short:
            pos[t] = -short_size
        elif pb[t] < p_long:
            pos[t] = 1.0
        # else: flat (0.0)
    return pd.Series(pos, index=p_bear.index, name="position")


# ----------------------------------------------------------------------------
# 3. Backtest + performance stats (short-bonds-template style)
# ----------------------------------------------------------------------------
def backtest(price: pd.Series, r_t: pd.Series, positions: pd.Series) -> pd.DataFrame:
    """Apply positions to next-week returns and build the strategy P&L frame.

    r_t is the weekly LOG return. Position at week t (decided at the t close) earns
    week t+1's return, so we use positions.shift(1). Strategy log return_t =
    pos_{t-1} * r_t. We also track buy&hold (always +1) for comparison.
    """
    r = r_t.reindex(positions.index)
    pos_lag = positions.shift(1).fillna(0.0)  # no position before the first signal

    strat_logret = pos_lag * r
    bh_logret = r  # buy & hold the S&P

    df = pd.DataFrame({
        "price": price.reindex(positions.index),
        "p_bear": None,  # filled by caller if wanted; kept for a tidy single frame
        "position": positions,
        "pos_traded": pos_lag,
        "r_t": r,
        "strat_logret": strat_logret,
        "bh_logret": bh_logret,
    })
    df["strat_equity"] = np.exp(strat_logret.cumsum())
    df["bh_equity"] = np.exp(bh_logret.cumsum())
    return df


def _max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fractional drop of an equity curve (a negative number)."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def performance_stats(logret: pd.Series, equity: pd.Series, positions: pd.Series | None = None) -> dict:
    """Annualized return / vol / Sharpe, total return, max drawdown, hit rate, exposure.

    Sharpe here is EXCESS-of-zero (risk-free ~0), matching the simple short-bonds
    template. Returns are weekly LOG returns; we annualize by *52 (mean) and
    *sqrt(52) (vol), the standard weekly convention.
    """
    lr = logret.dropna()
    n = len(lr)
    ann_ret = float(lr.mean() * WEEKS_PER_YEAR)
    ann_vol = float(lr.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    total_ret = float(equity.iloc[-1] - 1.0)
    # Hit rate over ACTIVE weeks only (a flat week is neither a win nor a loss).
    active = lr[lr != 0.0]
    hit_rate = float((active > 0).mean()) if len(active) else float("nan")
    stats = {
        "years": n / WEEKS_PER_YEAR,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "total_return": total_ret,
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": hit_rate,
    }
    if positions is not None:
        p = positions.reindex(lr.index).fillna(0.0)
        stats["pct_long"] = float((p > 0).mean())
        stats["pct_short"] = float((p < 0).mean())
        stats["pct_neutral"] = float((p == 0).mean())
        # Count round-trip trades = number of position CHANGES.
        stats["n_trades"] = int((p.diff().fillna(p.iloc[0]) != 0).sum())
    return stats


def _fmt_stats(name: str, s: dict) -> str:
    lines = [f"  {name}"]
    lines.append(f"    years            : {s['years']:.1f}")
    lines.append(f"    total return     : {s['total_return']*100:8.1f}%")
    lines.append(f"    ann. return      : {s['ann_return']*100:8.2f}%")
    lines.append(f"    ann. vol         : {s['ann_vol']*100:8.2f}%")
    lines.append(f"    Sharpe           : {s['sharpe']:8.2f}")
    lines.append(f"    max drawdown     : {s['max_drawdown']*100:8.1f}%")
    lines.append(f"    hit rate (active): {s['hit_rate']*100:8.1f}%")
    if "pct_long" in s:
        lines.append(f"    exposure L/S/flat: {s['pct_long']*100:.0f}% / {s['pct_short']*100:.0f}% / {s['pct_neutral']*100:.0f}%")
        lines.append(f"    trades           : {s['n_trades']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 4. Plot: equity curves + position/P(bear) context
# ----------------------------------------------------------------------------
def plot_strategy(df: pd.DataFrame, p_bear: pd.Series, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             height_ratios=[3, 1.0, 1.0])
    dates = df.index

    ax = axes[0]
    ax.plot(dates, df["strat_equity"], color="darkgreen", lw=1.4,
            label="P(bear) long/short/neutral strategy")
    ax.plot(dates, df["bh_equity"], color="gray", lw=1.1, label="buy & hold S&P")
    ax.set_yscale("log")
    ax.set_ylabel("equity (log, start=1)")
    ax.set_title(f"Regime P(bear) strategy vs buy & hold  [{MODEL_KEY} nowcast, channels r_t/v_t/dd]")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.plot(dates, p_bear.reindex(dates), color="darkorange", lw=1)
    ax.axhline(P_LONG, color="green", lw=0.7, ls="--", label=f"long<{P_LONG:.0%}")
    ax.axhline(P_SHORT_ENTER, color="red", lw=0.7, ls="--",
               label=f"short-arm>{P_SHORT_ENTER:.0%} (after {SHORT_DWELL_WEEKS}wk)")
    ax.axhline(P_SHORT_EXIT, color="red", lw=0.7, ls=":", label=f"short-cover<{P_SHORT_EXIT:.0%}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(bear)")
    ax.legend(loc="center left", fontsize=7, ncol=3)

    # If the strategy never actually shorted, say so on the position panel (drawn next).
    _n_short = int((df["position"] < 0).sum())

    ax = axes[2]
    pos = df["position"].reindex(dates)
    ax.fill_between(dates, 0, pos, step="pre",
                    where=(pos > 0), color="green", alpha=0.5, label="long")
    ax.fill_between(dates, 0, pos, step="pre",
                    where=(pos < 0), color="red", alpha=0.5, label="short")
    ax.set_ylim(-1.3, 1.3)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel("position")
    ax.set_xlabel("date")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    if _n_short == 0:
        ax.text(0.5, -1.05, "no short trades (SHORT_SIZE=0 or no bear persisted "
                f"{SHORT_DWELL_WEEKS}wk) -- long/flat overlay",
                transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7, color="red", alpha=0.8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved plot -> {save_path}")
    plt.show()
    return fig, axes


# ----------------------------------------------------------------------------
# 5. Neutral-band analysis: is the threshold wrong?
# ----------------------------------------------------------------------------
def analyse_neutral(df: pd.DataFrame, p_bear: pd.Series, p_long: float = P_LONG,
                    p_short_enter: float = P_SHORT_ENTER) -> dict:
    """Decompose the weeks the strategy sits FLAT and ask what we gave up.

    The question: when we hold flat (position 0), what did the S&P do the NEXT week
    (the return we forwent)? If those forgone returns are systematically positive we
    were too cautious; if ~0 or negative, standing aside was correct. Two flat causes,
    which matter for THIS persistence-gated design:
      * MIDDLE-BAND flat: p_bear in [p_long, p_short_enter] -- the genuinely ambiguous
        middle where we simply have no conviction either way.
      * UNPERSISTED-BEAR flat: p_bear > p_short_enter but the bear has NOT yet lasted
        SHORT_DWELL_WEEKS, so we are deliberately standing aside from a possible scare.
        This bucket's forward return tells us whether the persistence gate is skipping
        weeks we SHOULD have shorted (very negative) or correctly avoiding scares (~0/+).
    ALSO reports the P(bear) bin -> mean-next-week-return calibration table.
    """
    pos_traded = df["pos_traded"]              # already lagged (what earned this week's r)
    r = df["r_t"]
    pb_lag = p_bear.shift(1).reindex(df.index)  # the P(bear) that DECIDED pos_traded

    flat = pos_traded == 0.0
    high = (pb_lag > p_short_enter).fillna(False)   # bear-strength weeks
    flat_unpersisted = flat & high                  # flat despite high p_bear (gate held us out)
    flat_band = flat & ~high                        # flat in the ambiguous middle

    def _bucket(mask):
        rr = r[mask].dropna()
        return {
            "weeks": int(len(rr)),
            "pct_of_all": float(len(rr) / len(r)),
            "mean_fwd_ret_bps": float(rr.mean() * 1e4) if len(rr) else float("nan"),
            "ann_fwd_ret": float(rr.mean() * WEEKS_PER_YEAR) if len(rr) else float("nan"),
            "share_up": float((rr > 0).mean()) if len(rr) else float("nan"),
        }

    out = {
        "all_neutral": _bucket(flat),
        "unpersisted_bear": _bucket(flat_unpersisted),
        "band_neutral": _bucket(flat_band),
    }

    # P(bear) bin -> mean NEXT-week return. Uses pb_lag (the deciding prob) vs r.
    edges = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0001]
    bins = pd.cut(pb_lag, edges, right=False, include_lowest=True)
    grp = pd.DataFrame({"r": r, "bin": bins}).dropna().groupby("bin", observed=True)["r"]
    out["pbear_bins"] = pd.DataFrame({
        "weeks": grp.size(),
        "mean_fwd_ret_bps": grp.mean() * 1e4,
        "ann_fwd_ret_pct": grp.mean() * WEEKS_PER_YEAR * 100,
        "share_up": grp.apply(lambda s: (s > 0).mean()),
    })
    return out


def _print_neutral(a: dict) -> None:
    def _row(name, b):
        return (f"    {name:18s}: {b['weeks']:5d} wk ({b['pct_of_all']*100:4.1f}%)  "
                f"fwd {b['mean_fwd_ret_bps']:+7.1f} bps/wk ({b['ann_fwd_ret']*100:+6.2f}%/yr)  "
                f"up {b['share_up']*100:4.1f}%")

    print("\n" + "-" * 72)
    print("  FLAT-WEEK ANALYSIS  (forgone NEXT-week S&P return while flat)")
    print("-" * 72)
    print(_row("all flat", a["all_neutral"]))
    print(_row("unpersisted-bear", a["unpersisted_bear"]))
    print(_row("middle-band", a["band_neutral"]))
    print("\n  P(bear) bin -> mean NEXT-week S&P return (the threshold calibration test):")
    b = a["pbear_bins"].copy()
    b["mean_fwd_ret_bps"] = b["mean_fwd_ret_bps"].round(1)
    b["ann_fwd_ret_pct"] = b["ann_fwd_ret_pct"].round(1)
    b["share_up"] = (b["share_up"] * 100).round(1)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(b.to_string())
    print("-" * 72)


# ----------------------------------------------------------------------------
# 6. Threshold sweep -- pick cut points from the (4-state) data, not by eye
# ----------------------------------------------------------------------------
def sweep_dwell(price, r_t, p_bear,
                dwell_grid=range(4, 27, 2), size_grid=(0.5, 1.0)) -> pd.DataFrame:
    """ROBUSTNESS check (NOT an optimizer) over the short persistence knobs
    (short_dwell_weeks, short_size), holding p_long/enter/exit at their defaults.

    Printed sorted BY DWELL (not by Sharpe) on purpose: the point is to SEE THE PLATEAU
    -- that a wide range of dwell values gives similar results -- so the shipped default
    (13wk = one quarter) is robust, not cherry-picked to the argmax. Re-uses the fitted
    P(bear); cheap (no NUTS).
    """
    rows = []
    for size in size_grid:
        for dw in dwell_grid:
            pos = build_positions(p_bear, short_dwell_weeks=dw, short_size=size)
            d = backtest(price, r_t, pos)
            s = performance_stats(d["strat_logret"], d["strat_equity"],
                                  positions=d["pos_traded"])
            rows.append({
                "short_size": size, "short_dwell": dw,
                "sharpe": s["sharpe"], "ann_return": s["ann_return"],
                "ann_vol": s["ann_vol"], "max_dd": s["max_drawdown"],
                "pct_short": s["pct_short"], "n_trades": s["n_trades"],
            })
    # Sort by (size, dwell) so the plateau is visible as a smooth column, not ranked.
    return pd.DataFrame(rows).sort_values(["short_size", "short_dwell"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(refit: bool = False,
         p_long: float = P_LONG, p_short_enter: float = P_SHORT_ENTER,
         p_short_exit: float = P_SHORT_EXIT, short_dwell_weeks: int = SHORT_DWELL_WEEKS,
         short_size: float = SHORT_SIZE, do_sweep: bool = False) -> None:
    price, r_t, p_bear = load_pbear(refit=refit)

    if do_sweep:
        print(f"\nPersistence robustness sweep [{MODEL_KEY}] (re-uses the fitted P(bear); no refit).")
        print("Read this as a PLATEAU check, not an optimizer: a wide dwell range should give")
        print("similar Sharpe/drawdown, which is why the default (13wk) is robust, not fitted.")
        sw = sweep_dwell(price, r_t, p_bear)
        with pd.option_context("display.width", 140, "display.max_rows", 40):
            print(sw.to_string(index=False,
                  formatters={c: "{:.3f}".format for c in
                              ["short_size", "sharpe", "ann_return", "ann_vol", "max_dd", "pct_short"]}))
        out_dir = _pl.Path(__file__).resolve().parents[1] / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        sw.to_csv(out_dir / f"regime_pbear_strategy_{MODEL_KEY}_sweep.csv", index=False)
        print(f"\n(full grid saved -> outputs/regime_pbear_strategy_{MODEL_KEY}_sweep.csv)")
        return

    positions = build_positions(p_bear, p_long=p_long, p_short_enter=p_short_enter,
                                p_short_exit=p_short_exit, short_dwell_weeks=short_dwell_weeks,
                                short_size=short_size)
    df = backtest(price, r_t, positions)
    df["p_bear"] = p_bear

    strat = performance_stats(df["strat_logret"], df["strat_equity"], positions=df["pos_traded"])
    bh = performance_stats(df["bh_logret"], df["bh_equity"])

    short_note = (f"short {short_size:g}x after p>{p_short_enter:.0%} for {short_dwell_weeks}wk"
                  if short_size > 0 else "long/flat (short OFF)")
    print("\n" + "=" * 60)
    print(f"P(bear) STRATEGY [{MODEL_KEY}]  (long<{p_long:.0%}, {short_note})")
    print(f"period: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} weekly bars)")
    print("=" * 60)
    print(_fmt_stats("STRATEGY", strat))
    print(_fmt_stats("BUY & HOLD", bh))
    print("=" * 60)

    # SHORT-SIDE DIAGNOSTIC: how many persistent-bear episodes actually triggered a short,
    # and how many high-p(bear) weeks were correctly SKIPPED as unpersisted scares.
    pb_arr = p_bear.to_numpy()
    n_high = int((pb_arr > p_short_enter).sum())
    n_short = int((df["position"] < 0).sum())
    # count short EPISODES (runs of consecutive short weeks)
    is_short = (df["position"] < 0).to_numpy()
    n_episodes = int(np.sum(is_short[1:] & ~is_short[:-1]) + (1 if is_short[0] else 0))
    print(f"  SHORT-SIDE: {n_high} wk with p>{p_short_enter:.0%}; shorted {n_short} of them "
          f"across {n_episodes} PERSISTENT-bear episodes ({n_high - n_short} wk skipped as "
          f"un-persisted scares / cover lag).")
    if short_size == 0:
        print("  -> SHORT_SIZE=0: pure long/flat overlay.")
    print("=" * 60)

    neutral = analyse_neutral(df, p_bear, p_long=p_long, p_short_enter=p_short_enter)
    _print_neutral(neutral)

    out_dir = _pl.Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"regime_pbear_strategy_{MODEL_KEY}_backtest.csv")
    neutral["pbear_bins"].to_csv(out_dir / f"regime_pbear_strategy_{MODEL_KEY}_neutral_bins.csv")
    plot_strategy(df, p_bear, save_path=out_dir / f"regime_pbear_strategy_{MODEL_KEY}.png")


def _flag(argv, name, cast, default):
    """Tiny CLI helper: --name VALUE -> cast(VALUE), else default."""
    return cast(argv[argv.index(name) + 1]) if name in argv else default


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    main(
        refit="--refit" in argv,
        do_sweep="--sweep" in argv,
        p_long=_flag(argv, "--long", float, P_LONG),
        p_short_enter=_flag(argv, "--short-enter", float, P_SHORT_ENTER),
        p_short_exit=_flag(argv, "--short-exit", float, P_SHORT_EXIT),
        short_dwell_weeks=_flag(argv, "--short-dwell", int, SHORT_DWELL_WEEKS),
        short_size=_flag(argv, "--short-size", float, SHORT_SIZE),
    )
