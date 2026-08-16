"""Parameter sweep: the RECOVERY-LEVERAGE multiplier (LEV_MULT) of the P(bear) strategy.

WHAT THIS ANSWERS
-----------------
The P(bear) strategy levers its long to LEV_MULT while riding a bear's recovery (armed when
a seen bear resolves back down; see regime_pbear_strategy). The shipped value is
LEV_MULT = 1.25, justified as "the conservative edge of a Sharpe plateau". This script
MEASURES that plateau: it re-runs the strategy across a grid of leverage values and reports
Sharpe per value, then hands the Sharpe curve to the overfitting screen
(sweep_utils.classify_plateau) to check the live 1.25 sits on a FLAT neighbourhood (nothing
to overfit) rather than on a cliff (a lucky point that lurches if nudged).

NO NUTS -- this is the CHEAP kind of sweep. Leverage only changes position sizing, which is
pure post-processing on a FIXED P(bear) curve. So we load P(bear) ONCE from the strategy's
cache (regime_3state_strategy_pbear_*.pkl) and re-run only build_positions -> backtest ->
performance_stats per leverage value. ~21 backtests, milliseconds each. (Contrast a
model-input sweep, e.g. an emission/window sweep, which needs a NUTS refit per value.)

The strategy's own functions are reused VERBATIM (build_positions / backtest /
performance_stats), so the swept P&L is the shipped backtest, not a parallel reimpl -- only
lev_mult varies; every other knob follows the strategy module's live defaults.

HOW TO RUN
----------
    python parameter_sweeps/sweep_lev_mult.py
    python parameter_sweeps/sweep_lev_mult.py --grid 1.0,2.0,0.05   # lo,hi,step
    python parameter_sweeps/sweep_lev_mult.py --refit               # force a fresh NUTS fit of P(bear)

Output: a printed table + plateau verdict, outputs/sweep_lev_mult.csv, and
outputs/sweep_lev_mult.png (Sharpe vs leverage with the live value + plateau band marked).
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

# This file lives in equity_market/parameter_sweeps/, a SIBLING of src/ and trading_strategies/.
# Put src/ subfolders on sys.path (like every script) AND trading_strategies/ so we can import
# the strategy module whose backtest we reuse.
_ROOT = _pl.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(_ROOT / "src"))
import _syspath  # noqa: E402,F401  (adds src/ subfolders: dataset, models, ...)
_sys.path.insert(0, str(_ROOT / "trading_strategies"))
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # this dir, for sweep_utils

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import regime_pbear_strategy as strat  # noqa: E402  (the 3-state P(bear) strategy)
from sweep_utils import classify_plateau  # noqa: E402  (the overfitting plateau/cliff screen)


DEFAULT_GRID = (1.0, 2.0, 0.05)  # (lo, hi, step) -- 1.0x .. 2.0x in 0.05 steps
LIVE_LEV = strat.LEV_MULT        # the shipped value we're defending (1.25)


def _leverage_grid(lo: float, hi: float, step: float) -> np.ndarray:
    """Inclusive grid lo, lo+step, ..., hi (rounded to avoid float drift)."""
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 4)


def score_leverage(lev: float, price, r_t, p_bear) -> dict:
    """Re-run the shipped strategy at recovery-leverage = `lev` and return its stats.

    Reuses the strategy's OWN build_positions / backtest / performance_stats verbatim, so
    this is the shipped backtest with only lev_mult overridden (every other knob stays at
    the strategy module's live default). No NUTS -- p_bear is fixed input.
    """
    positions = strat.build_positions(p_bear, lev_mult=float(lev))
    df = strat.backtest(price, r_t, positions)
    s = strat.performance_stats(df["strat_logret"], df["strat_equity"], positions)
    return {
        "lev_mult": float(lev),
        "sharpe": s["sharpe"],
        "ann_return": s["ann_return"],
        "ann_vol": s["ann_vol"],
        "total_return": s["total_return"],
        "max_drawdown": s["max_drawdown"],
    }


def run_sweep(grid: np.ndarray, refit: bool) -> pd.DataFrame:
    """Load P(bear) ONCE, then score every leverage on the grid."""
    print(f"Loading P(bear) from the 3-state strategy cache (refit={refit}) ...")
    price, r_t, p_bear = strat.load_pbear(refit=refit)
    print(f"  P(bear): {len(p_bear)} weekly obs, {p_bear.index[0].date()} -> {p_bear.index[-1].date()}\n")

    rows = [score_leverage(lev, price, r_t, p_bear) for lev in grid]
    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("RECOVERY-LEVERAGE SWEEP  (3-state P(bear) strategy; only LEV_MULT varies)")
    print("  Sharpe is the headline (the shipped 1.25 was justified on a Sharpe plateau).")
    print("=" * 78)
    show = df.copy()
    fmt = {c: "{:.3f}".format for c in ("sharpe", "ann_return", "ann_vol",
                                        "total_return", "max_drawdown")}
    fmt["lev_mult"] = "{:.2f}".format
    with pd.option_context("display.width", 120, "display.max_rows", 60):
        print(show.to_string(index=False, formatters=fmt))
    print("=" * 78)


def _report_plateau(df: pd.DataFrame) -> dict:
    """Run the overfitting screen on the Sharpe curve at the live leverage value."""
    sharpe = df["sharpe"].to_numpy()
    levs = df["lev_mult"].to_numpy()
    # index of the live value (nearest grid point to LIVE_LEV)
    live_idx = int(np.argmin(np.abs(levs - LIVE_LEV)))

    # flat_floor: a Sharpe jump below this is economically trivial. Sharpe steps here are
    # small (~0.01-0.05), so use a modest absolute floor -- 0.02 Sharpe between adjacent
    # 0.05-leverage steps is noise, not a cliff.
    flat_floor = 0.02
    # pooled_pnl for the ABSOLUTE material backstop: the strategy's total return at the live
    # value, in the same (fractional) units the backstop compares against.
    pooled = float(df.loc[live_idx, "total_return"])

    verdict = classify_plateau(sharpe, live_idx=live_idx, flat_floor=flat_floor,
                               pooled_pnl=pooled)

    print("\nOVERFITTING SCREEN (plateau vs cliff at the live leverage):")
    print(f"  live value        : LEV_MULT = {levs[live_idx]:.2f}  "
          f"(Sharpe {sharpe[live_idx]:.3f})")
    print(f"  verdict           : {verdict['verdict']}")
    print(f"  reason            : {verdict['reason']}")
    print(f"  adjacent jumps    : {[round(j, 4) for j in verdict['adj_jumps']]}  (Sharpe)")
    print(f"  max jump in sweep : {verdict['max_jump']:.4f}   "
          f"median jump: {verdict['median_jump']:.4f}   robust_sigma: {verdict['robust_sigma']:.4f}")
    if verdict["verdict"] == "PLATEAU":
        print("  => the shipped 1.25 sits on a FLAT Sharpe neighbourhood -- nothing to overfit.")
    else:
        print("  => FLAGGED: a human should eyeball the curve around the live value.")
    return verdict


def _plot(df: pd.DataFrame, verdict: dict, save_path: _pl.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["lev_mult"], df["sharpe"], "-o", color="#1f77b4", lw=1.4, ms=4,
            label="Sharpe")
    ax.axvline(LIVE_LEV, color="firebrick", lw=1.4, ls="--",
               label=f"live LEV_MULT = {LIVE_LEV:g}")
    ax.set_xlabel("recovery leverage (LEV_MULT)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title(f"Recovery-leverage sweep -- 3-state strategy   "
                 f"[plateau screen: {verdict['verdict']}]")
    ax.legend()
    ax.grid(alpha=0.25)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved figure -> {save_path}")


def main(argv):
    refit = "--refit" in argv
    lo, hi, step = DEFAULT_GRID
    if "--grid" in argv:
        lo, hi, step = (float(x) for x in argv[argv.index("--grid") + 1].split(","))
    grid = _leverage_grid(lo, hi, step)

    print(f"Leverage grid: {grid[0]:.2f} .. {grid[-1]:.2f}  step {step:.2f}  "
          f"({len(grid)} points)")
    print(f"Live value being defended: LEV_MULT = {LIVE_LEV:g}   (no NUTS -- P(bear) is fixed)\n")

    df = run_sweep(grid, refit)
    _print_table(df)
    verdict = _report_plateau(df)

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sweep_lev_mult.csv", index=False)
    print(f"\nsaved -> {out_dir / 'sweep_lev_mult.csv'}")
    _plot(df, verdict, out_dir / "sweep_lev_mult.png")


if __name__ == "__main__":
    main(_sys.argv[1:])
