"""DESIGN-TIME diagnostic: WHAT PEAK is the drawdown window actually tracking?

WHY THIS EXISTS (economic logic, not metric-picking)
----------------------------------------------------
dd_t = price_t / max(price over trailing W weeks) - 1. The window W is the MEMORY of
"what top are we measured against." The window sweep gave a monotone recall/recovery-FA
tradeoff, but a leaderboard can't tell you the MECHANISM. This does: it plots, per window,
the trailing-max (the tracked "peak") on top of price, so you can SEE the two failure modes
as mechanical facts rather than infer them from a metric that won by 0.02.

  * TOO LONG (e.g. 156wk): the peak CLINGS to the pre-crash high for W weeks, so in a
    recovery dd stays deeply negative into the new bull -> false "still-bear."
  * TOO SHORT (e.g. 26wk): the peak RATCHETS DOWN to follow price during a slow/grinding
    bear -- after W weeks the trailing-max has drifted down with the decline, so dd reverts
    toward 0 and the feature goes BLIND to a bear that lasts longer than W. It also reacts
    to ordinary bull pullbacks (its reference peak is too local) -> false alarms.

The economic anchor: the RIGHT window is ~the duration of a typical bear-to-recovery
episode -- long enough to keep the reference pinned to the true cycle top for the whole
episode, short enough to let go once a genuine new bull is underway. This script lets you
check that against the historical episodes directly.

WHAT IT SHOWS
-------------
outputs/drawdown_peak_tracking.png:
  Top: full-history log price with the tracked peak for several windows overlaid.
  Then one panel PER key episode (2000-02 dotcom, 2007-09 GFC + recovery, 1970s) zoomed in,
  price + each window's tracked peak + shaded P&S bear bands, so you can watch the peak
  ratchet down (short) or cling (long) inside each real bear/recovery.
  Bottom: for each window, the "peak age" = how many weeks old the currently-tracked peak
  is, over time -- a direct readout of when a window is clinging (age pinned near W) vs
  forgetting (age small). A window that spends a whole bear with peak-age hitting the W
  ceiling is one whose memory is SATURATED = too short for that episode.

RUN (no NUTS, cheap):
    python diagnostics/drawdown_peak_tracking.py
    python diagnostics/drawdown_peak_tracking.py --windows 26,39,52,104
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
from labels import pagan_sossounov_label  # noqa: E402


DATA_START = "1957-03-01"
DEFAULT_WINDOWS = (26, 52, 104, 156)  # 0.5, 1, 2, 3 yr -- spread to make the mechanism obvious
COLORS = {26: "#d62728", 39: "#ff7f0e", 52: "#2ca02c", 65: "#17becf",
          78: "#9467bd", 104: "#1f77b4", 156: "#8c564b"}

# Zoomed episodes: the regimes where the two failure modes should be VISIBLE.
EPISODES = [
    ("Dotcom (slow rolling-top bear)", "1999-06-01", "2003-12-31"),
    ("GFC crash + 2009 recovery", "2007-06-01", "2011-12-31"),
    ("1970s stagflation bears", "1968-01-01", "1983-01-01"),
]


def _tracked_peak(price: pd.Series, W: int) -> pd.Series:
    """The trailing-max the dd channel tracks with window W (min_periods=1, exactly as
    data.drawdown does). This IS the 'peak' dd measures distance below."""
    return price.rolling(W, min_periods=1).max()


def _peak_age_weeks(price: pd.Series, W: int) -> pd.Series:
    """How many weeks ago the currently-tracked peak occurred.

    age pinned near W (the ceiling) => the window is CLINGING to an old high (its memory is
    saturated -- the true peak is >=W weeks back). age small and rising slowly => the peak
    is recent (fresh highs, healthy bull) OR ratcheting down with a decline. Read alongside
    price: high age DURING a bear = clinging (long window); age hitting the W ceiling and
    the peak visibly tracking price DOWN = ratcheting (short window blind spot).
    """
    p = price.to_numpy()
    n = len(p)
    age = np.zeros(n)
    for t in range(n):
        lo = max(0, t - W + 1)
        window = p[lo:t + 1]
        argmax_local = int(np.argmax(window))
        peak_pos = lo + argmax_local
        age[t] = t - peak_pos
    return pd.Series(age, index=price.index)


def make_figure(windows, save_path: _pl.Path) -> None:
    import matplotlib.pyplot as plt

    ds = load_regime_dataset(start=DATA_START, include_vix=False, include_macro=False)
    price = ds.weekly_price.dropna()
    lab = pagan_sossounov_label(price).reindex(price.index).ffill()

    peaks = {W: _tracked_peak(price, W) for W in windows}
    ages = {W: _peak_age_weeks(price, W) for W in windows}

    n_ep = len(EPISODES)
    fig, axes = plt.subplots(2 + n_ep, 1, figsize=(13, 4 + 2.6 * (1 + n_ep)))

    def _shade_bears(ax, seg_idx):
        """Shade P&S bear weeks on a price/peak panel."""
        b = (lab.reindex(seg_idx) == 1).to_numpy()
        x = seg_idx
        in_bear = False
        for i, flag in enumerate(b):
            if flag and not in_bear:
                start = x[i]; in_bear = True
            elif not flag and in_bear:
                ax.axvspan(start, x[i], color="firebrick", alpha=0.10)
                in_bear = False
        if in_bear:
            ax.axvspan(start, x[-1], color="firebrick", alpha=0.10)

    # ---- panel 0: full history, price + tracked peaks ----
    ax = axes[0]
    ax.plot(price.index, price.to_numpy(), color="k", lw=0.8, label="S&P (weekly)")
    for W in windows:
        ax.plot(peaks[W].index, peaks[W].to_numpy(), color=COLORS.get(W), lw=0.9,
                alpha=0.8, label=f"tracked peak, W={W}wk")
    _shade_bears(ax, price.index)
    ax.set_yscale("log"); ax.set_ylabel("price (log)")
    ax.set_title("Tracked peak (trailing-max) per window, full history  "
                 "-- long W clings to old highs; short W ratchets down with price")
    ax.legend(fontsize=7, ncol=len(windows) + 1, loc="upper left")

    # ---- per-episode zooms ----
    for k, (name, a, b) in enumerate(EPISODES):
        ax = axes[1 + k]
        seg = price.loc[a:b]
        ax.plot(seg.index, seg.to_numpy(), color="k", lw=1.1, label="S&P")
        for W in windows:
            pk = peaks[W].loc[a:b]
            ax.plot(pk.index, pk.to_numpy(), color=COLORS.get(W), lw=1.3, alpha=0.85,
                    label=f"peak W={W}")
        _shade_bears(ax, seg.index)
        ax.set_ylabel("price")
        ax.set_title(f"{name}  ({a} -> {b})   [shaded = P&S bear]", fontsize=10)
        ax.legend(fontsize=7, ncol=len(windows) + 1, loc="best")

    # ---- bottom: peak AGE over time (clinging vs forgetting) ----
    ax = axes[-1]
    for W in windows:
        ax.plot(ages[W].index, ages[W].to_numpy(), color=COLORS.get(W), lw=0.8,
                alpha=0.85, label=f"peak age, W={W}")
        ax.axhline(W, color=COLORS.get(W), lw=0.6, ls=":", alpha=0.6)  # the W ceiling
    _shade_bears(ax, price.index)
    ax.set_ylabel("peak age (weeks)")
    ax.set_xlabel("date")
    ax.set_title("Peak AGE = weeks since the tracked peak. Pinned at the W ceiling (dotted) "
                 "during a bear = memory SATURATED (window too short for that episode).",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=len(windows), loc="upper left")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {save_path}")

    # ---- printed economic readout: how often is each window's memory SATURATED
    #      (peak age at the W ceiling) DURING true bears? that's the blind-spot rate. ----
    print("\nBLIND-SPOT CHECK: fraction of true-bear weeks where the tracked peak is >= (W-2)")
    print("weeks old (memory saturated -> peak ratcheting down with the decline, dd reverts to 0):")
    is_bear = (lab == 1).reindex(price.index).fillna(False).to_numpy()
    for W in windows:
        a = ages[W].to_numpy()
        saturated = a >= (W - 2)
        frac = float(np.mean(saturated[is_bear])) if is_bear.any() else float("nan")
        # also: mean dd during bears (how underwater the feature THINKS we are)
        dd = (price / peaks[W] - 1.0).to_numpy()
        mean_dd_bear = float(np.mean(dd[is_bear]))
        print(f"  W={W:3d}wk: saturated {frac*100:4.1f}% of bear weeks   "
              f"mean dd in bears = {mean_dd_bear:+.3f}  "
              f"({'BLIND: peak follows price down' if frac > 0.5 else 'holds a real reference'})")


def main(argv):
    if "--windows" in argv:
        windows = tuple(int(x) for x in argv[argv.index("--windows") + 1].split(","))
    else:
        windows = DEFAULT_WINDOWS
    out = _pl.Path(__file__).resolve().parents[1] / "outputs" / "drawdown_peak_tracking.png"
    make_figure(windows, out)


if __name__ == "__main__":
    main(_sys.argv[1:])
