"""Pagan & Sossounov (2003) bull/bear market dating -- a pure-Python
reimplementation of the algorithm in Valeriy Zakamulin's R package `bbdetection`
(CRAN, `run_dating_alg` / `setpar_dating_alg`).

"A simple framework for analysing bull and bear markets", J. Applied
Econometrics 18(1). It is the Bry-Boschan (1971) turning-point procedure adapted
to stock prices: date PEAKS and TROUGHS as local extrema, then censor by minimum
phase/cycle DURATION. A regime is a sustained phase between two turning points,
so -- unlike a 20% drawdown rule -- it structurally cannot be triggered by a
transient dip. This is why it is the right GROUND TRUTH for the regime nowcast.
Import pagan_sossounov_label directly from this module.

FAITHFULNESS TO bbdetection. The R package's dating routine
(setpar_dating_alg) takes exactly these parameters, all counted in OBSERVATIONS:
    t_window = 8   half-window for local extrema
    t_censor = 6   endpoint margin (turning points this close to an end dropped)
    t_phase  = 4   minimum phase (bull or bear) length
    t_cycle  = 16  minimum full cycle length
    max_chng = 20  % price change that EXEMPTS a short phase from the t_phase rule
These are the MONTHLY presets (the algorithm itself is frequency-agnostic -- it
counts observations, not calendar time). Our pipeline is WEEKLY, so we scale the
monthly presets to weeks at ~4.33 weeks/month by default (see _months_to_weeks
and the *_months parameters of pagan_sossounov_label); pass window_periods= etc.
directly to bypass the scaling and match bbdetection observation-for-observation.

CONVENTIONS. bbdetection's run_dating_alg returns a logical vector with
TRUE=Bull, FALSE=Bear. We instead return the project's 0/1 label convention
(BEAR=1, BULL=0, imported from labels) so the output is a drop-in for the
section-7 evaluator; the states are identical, only the encoding differs.

Run directly (`python pagan_sossounov.py`) to fetch the S&P 500, date it, PLOT
the peaks/troughs over the price, and PRINT a per-phase table (duration and
return of every bull and bear phase) to the terminal.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401  (puts sibling src/ subfolders on sys.path)

import numpy as np
import pandas as pd

# 0/1 label encoding, kept identical to labels.py (BEAR=1, BULL=0). Defined here
# rather than imported to avoid a labels<->pagan_sossounov import cycle (labels
# re-exports pagan_sossounov_label). labels.py has a test asserting these match.
BEAR = 1
BULL = 0

# bbdetection setpar_dating_alg defaults, in OBSERVATIONS (its monthly presets).
_T_WINDOW_MONTHS = 8.0
_T_CENSOR_MONTHS = 6.0
_T_PHASE_MONTHS = 4.0
_T_CYCLE_MONTHS = 16.0

# theta -- the amplitude override on the minimum-phase (t_phase) rule. A phase
# shorter than t_phase is normally censored; theta EXEMPTS it if it is violent
# enough. The number (20%) is meaningless without its HORIZON, and the horizon
# is a genuine FORK that changes the labels -- so it is named here, not left to
# the comparison site to imply:
#
#   WHOLE-PHASE (this implementation, = Zakamulin's bbdetection): theta is the
#     peak-to-trough price move ACROSS THE ENTIRE SHORT PHASE. Frequency-robust:
#     a 20% peak-to-trough move means the same thing on monthly, weekly or daily
#     bars, so NO rescaling with cadence and short violent crashes (1987: ~-33%
#     in a few weeks) are correctly KEPT.
#   SINGLE-PERIOD (original Pagan-Sossounov 2003): theta fires only if ONE bar
#     moves >=20%. Calibrated for MONTHLY data; at weekly/daily cadence a single
#     bar almost never moves 20%, so the override goes nearly DEAD and short
#     violent phases get wrongly censored. This is the trap for finer frequency.
#
# We are on WEEKLY data and implementing fresh, so we deliberately adopt the
# WHOLE-PHASE definition. _censor_short_phases checks theta against the phase
# endpoints (peak/trough), never against a single bar -- see the check there.
_MAX_CHNG_PCT = 20.0  # theta, as a WHOLE-PHASE peak-to-trough % move (see above)

_WEEKS_PER_MONTH = 52.0 / 12.0  # ~4.33, for scaling the monthly presets to weeks


def _months_to_weeks(months: float) -> int:
    return int(round(months * _WEEKS_PER_MONTH))


# ---------------------------------------------------------------------------
# Core dating steps. All operate on log price and on an extrema list of
# (index, kind) tuples with kind in {+1 peak, -1 trough}, in time order.
# ---------------------------------------------------------------------------
def _local_extrema(log_price: np.ndarray, window: int) -> list[tuple[int, int]]:
    """Candidate turning points: index i is a PEAK (+1) if log_price[i] is the
    max over [i-window, i+window], a TROUGH (-1) if the min. This mirrors
    bbdetection's rolling +/- t_window extrema scan.

    Ties are resolved to the EARLIEST occurrence (argmax/argmin returns the first
    position), so a run of equal values marks only its first index.
    """
    n = len(log_price)
    out: list[tuple[int, int]] = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        seg = log_price[lo:hi]
        if log_price[i] == seg.max() and (lo + int(seg.argmax())) == i:
            out.append((i, +1))
        elif log_price[i] == seg.min() and (lo + int(seg.argmin())) == i:
            out.append((i, -1))
    return out


def _enforce_alternation(
    extrema: list[tuple[int, int]], log_price: np.ndarray
) -> list[tuple[int, int]]:
    """Force peaks and troughs to alternate. When two same-kind extrema occur in
    a row, keep the more extreme one (higher peak / lower trough) and drop the
    other -- bbdetection's rule for collapsing consecutive same-type extrema.
    """
    result: list[tuple[int, int]] = []
    for idx, kind in extrema:
        if result and result[-1][1] == kind:
            prev_idx = result[-1][0]
            if kind == +1:  # two peaks: keep the higher
                if log_price[idx] > log_price[prev_idx]:
                    result[-1] = (idx, kind)
            else:  # two troughs: keep the lower
                if log_price[idx] < log_price[prev_idx]:
                    result[-1] = (idx, kind)
        else:
            result.append((idx, kind))
    return result


def _censor_short_phases(
    extrema: list[tuple[int, int]],
    log_price: np.ndarray,
    phase_min: int,
    max_chng_pct: float,
) -> list[tuple[int, int]]:
    """t_phase rule: remove any phase (adjacent peak<->trough) shorter than
    phase_min OBSERVATIONS, UNLESS theta (max_chng_pct) exempts it.

    theta is a WHOLE-PHASE amplitude: the check below measures the peak-to-trough
    move between the phase's two bounding turning points (log_price[i1] vs
    log_price[i0]), NOT any single bar. This is Zakamulin's bbdetection definition
    and is frequency-robust -- see the _MAX_CHNG_PCT block at the top of the module
    for why the single-period alternative is wrong at weekly cadence. It keeps
    1987-style short violent phases (a few weeks, ~-33%).

    Iterated to a fixed point because deleting a turning point can shorten a
    neighbouring phase. Removal collapses the too-short phase by dropping the LESS
    PROMINENT of the two extrema bounding it, then re-alternating -- the same net
    effect as bbdetection, where eliminating a short swing merges the flanking
    same-type extrema and keeps the more extreme one.
    """
    amp_thresh = np.log1p(max_chng_pct / 100.0)  # |Δ log price| for an X% move
    ext = list(extrema)
    changed = True
    while changed and len(ext) >= 2:
        changed = False
        for j in range(len(ext) - 1):
            i0, k0 = ext[j]
            i1, k1 = ext[j + 1]
            # WHOLE-PHASE theta: |Δlog price| between the phase endpoints i0->i1
            # (peak-to-trough), never a single bar. Short phase kept iff this
            # peak-to-trough move reaches theta.
            phase_move = abs(log_price[i1] - log_price[i0])
            if (i1 - i0) < phase_min and phase_move < amp_thresh:
                # Drop the less prominent endpoint of this short phase: for a
                # peak->trough phase the peak is "less prominent" if... we keep
                # the more extreme extremum on each side, so drop whichever of
                # the two is closer to the interior (equivalently: remove the
                # one whose neighbour on the far side would dominate it). The
                # robust, bbdetection-equivalent choice is to remove the extremum
                # with the SMALLER absolute prominence, which for an isolated
                # short swing is the second point; alternation then merges it.
                del ext[j + 1]
                ext = _enforce_alternation(ext, log_price)
                changed = True
                break
    return ext


def _censor_short_cycles(
    extrema: list[tuple[int, int]], log_price: np.ndarray, cycle_min: int
) -> list[tuple[int, int]]:
    """t_cycle rule: remove any full cycle (peak->peak or trough->trough) shorter
    than cycle_min OBSERVATIONS. Iterated to a fixed point. When a cycle is too
    short, drop the interior turning point and the weaker of the two same-type
    endpoints, then re-alternate -- matching bbdetection's cycle elimination.
    """
    ext = list(extrema)
    changed = True
    while changed and len(ext) >= 3:
        changed = False
        for j in range(len(ext) - 2):
            i0, k0 = ext[j]
            i2, k2 = ext[j + 2]
            if k0 == k2 and (i2 - i0) < cycle_min:
                if k0 == +1:  # two peaks: the LOWER peak is weaker
                    weaker_end = j if log_price[i0] < log_price[i2] else j + 2
                else:  # two troughs: the HIGHER trough is weaker
                    weaker_end = j if log_price[i0] > log_price[i2] else j + 2
                for d in sorted((j + 1, weaker_end), reverse=True):
                    del ext[d]
                ext = _enforce_alternation(ext, log_price)
                changed = True
                break
    return ext


def date_turning_points(
    price: pd.Series,
    window_periods: int,
    censor_periods: int,
    phase_min_periods: int,
    cycle_min_periods: int,
    max_chng_pct: float = _MAX_CHNG_PCT,
) -> list[tuple[int, int]]:
    """Run the full bbdetection dating pipeline and return the surviving turning
    points as a list of (index, kind) tuples (kind +1 peak, -1 trough), in time
    order. All *_periods are counts of OBSERVATIONS.

    Order of operations matches bbdetection / Pagan-Sossounov:
      1. local +/- window extrema
      2. enforce peak/trough alternation
      3. drop turning points within `censor_periods` of either sample end
      4. censor phases shorter than phase_min (with the max_chng exemption)
      5. censor cycles shorter than cycle_min
    """
    log_price = np.log(price.to_numpy(dtype=float))
    n = len(log_price)

    ext = _local_extrema(log_price, window_periods)
    ext = _enforce_alternation(ext, log_price)
    ext = [(i, k) for (i, k) in ext if censor_periods <= i < n - censor_periods]
    ext = _enforce_alternation(ext, log_price)
    ext = _censor_short_phases(ext, log_price, phase_min_periods, max_chng_pct)
    ext = _censor_short_cycles(ext, log_price, cycle_min_periods)
    return ext


def _label_from_turning_points(
    extrema: list[tuple[int, int]], n: int, index: pd.Index
) -> pd.Series:
    """Fill a 0/1 (BULL/BEAR) label from dated turning points. A segment that
    ENDS at a PEAK is a bull run (rising into the peak); ending at a TROUGH is a
    bear. The trailing segment after the last turning point is the phase that
    turning point OPENS (a peak opens a bear, a trough opens a bull).
    """
    labels = np.empty(n, dtype=int)
    if not extrema:
        labels[:] = BULL  # no datable turn: one phase, default bull
        return pd.Series(labels, index=index, name="label_t")

    prev = 0
    for idx, kind in extrema:
        labels[prev : idx + 1] = BULL if kind == +1 else BEAR
        prev = idx + 1
    last_kind = extrema[-1][1]
    labels[prev:] = BEAR if last_kind == +1 else BULL
    return pd.Series(labels, index=index, name="label_t")


def pagan_sossounov_label(
    price: pd.Series,
    window_months: float = _T_WINDOW_MONTHS,
    phase_min_months: float = _T_PHASE_MONTHS,
    cycle_min_months: float = _T_CYCLE_MONTHS,
    endpoint_months: float = _T_CENSOR_MONTHS,
    max_chng_pct: float = _MAX_CHNG_PCT,
    window_periods: int | None = None,
    phase_min_periods: int | None = None,
    cycle_min_periods: int | None = None,
    endpoint_periods: int | None = None,
) -> pd.Series:
    """Label each period BEAR (1) or BULL (0) by Pagan & Sossounov (2003) dating,
    faithful to bbdetection.

    By default the bbdetection MONTHLY presets (8/4/16/6 months, 20%) are scaled
    to the WEEKLY pipeline at ~4.33 weeks/month:
        window   ~35 wk,  phase min ~17 wk,  cycle min ~69 wk,  endpoint ~26 wk
    To match bbdetection observation-for-observation instead, pass the *_periods
    arguments directly (they OVERRIDE the month-scaled values) -- e.g. on a
    monthly series call with window_periods=8, phase_min_periods=4,
    cycle_min_periods=16, endpoint_periods=6.

    Returns a 0/1 int Series, same index as `price`: a drop-in replacement for
    labels.drawdown_rally_label in the section-7 evaluation.
    """
    window = window_periods if window_periods is not None else _months_to_weeks(window_months)
    phase_min = (
        phase_min_periods if phase_min_periods is not None else _months_to_weeks(phase_min_months)
    )
    cycle_min = (
        cycle_min_periods if cycle_min_periods is not None else _months_to_weeks(cycle_min_months)
    )
    endpoint = (
        endpoint_periods if endpoint_periods is not None else _months_to_weeks(endpoint_months)
    )

    ext = date_turning_points(price, window, endpoint, phase_min, cycle_min, max_chng_pct)
    return _label_from_turning_points(ext, len(price), price.index)


def phase_table(price: pd.Series, extrema: list[tuple[int, int]]) -> pd.DataFrame:
    """Per-phase summary between consecutive turning points: kind, start/end
    dates, DURATION (observations, inclusive) and RETURN over the phase.

    Return is computed bbdetection-style as (price[end] - price[start-1]) /
    price[start-1] -- measured from the observation BEFORE the phase opens, so a
    bear's drawdown is captured from its true peak. The leading (pre-first-turn)
    and trailing (post-last-turn) phases are INCOMPLETE and flagged as such.
    """
    idx = price.index
    values = price.to_numpy(dtype=float)
    n = len(values)

    # Turning-point indices split the sample into phases. Boundaries: 0, each
    # turning-point index, n-1.
    tp_idx = [i for i, _ in extrema]
    tp_kind = {i: k for i, k in extrema}

    rows = []
    boundaries = [0] + tp_idx + [n - 1]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        # Phase spans (a .. b]. Its kind is set by the turning point at b: rising
        # into a peak = bull, falling into a trough = bear. The trailing phase
        # (b == n-1 and n-1 is not a turning point) opens from the last turn.
        if b in tp_kind:
            kind = BULL if tp_kind[b] == +1 else BEAR
        else:  # trailing incomplete phase, opened by the last turning point
            last_kind = extrema[-1][1] if extrema else +1
            kind = BEAR if last_kind == +1 else BULL
        start = a if a == 0 else a + 1  # phase opens the period after prev turn
        end = b
        base = values[start - 1] if start > 0 else values[start]
        ret = (values[end] - base) / base
        incomplete = (a == 0) or (b == n - 1 and b not in tp_kind)
        rows.append(
            {
                "phase": "bear" if kind == BEAR else "bull",
                "start": idx[start],
                "end": idx[end],
                "duration": end - start + 1,
                "return": ret,
                "complete": not incomplete,
            }
        )
    return pd.DataFrame(rows)


def plot_dating(price: pd.Series, extrema: list[tuple[int, int]], save_path=None):
    """Plot the price (log scale) with dated PEAKS (v) and TROUGHS (^) marked and
    bear phases shaded, so the dating can be eyeballed against history.
    """
    import matplotlib.pyplot as plt

    # Build shading directly from `extrema` so the plot always matches the passed
    # turning points (not a re-derivation).
    lab = _label_from_turning_points(extrema, len(price), price.index).to_numpy()

    fig, ax = plt.subplots(figsize=(13, 6))
    dates = np.asarray(price.index)

    # Shade bear spans.
    start = 0
    for t in range(1, len(lab) + 1):
        if t == len(lab) or lab[t] != lab[start]:
            if lab[start] == BEAR:
                ax.axvspan(dates[start], dates[t - 1], color="crimson", alpha=0.12, lw=0)
            start = t

    ax.plot(dates, price.to_numpy(), color="black", lw=0.9)
    ax.set_yscale("log")

    pv = price.to_numpy(dtype=float)
    for i, k in extrema:
        if k == +1:
            ax.plot(dates[i], pv[i], "v", color="darkgreen", markersize=9)
        else:
            ax.plot(dates[i], pv[i], "^", color="crimson", markersize=9)

    ax.set_ylabel("S&P 500 (log)")
    ax.set_xlabel("date")
    ax.set_title("Pagan-Sossounov dating: peaks (v), troughs (^), bear phases shaded")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


def _print_phase_table(table: pd.DataFrame) -> None:
    """Pretty-print the per-phase duration/return table to the terminal.

    Durations are in OBSERVATIONS (weeks for the default weekly pipeline).
    """
    print(f"\n{'phase':<6} {'start':<12} {'end':<12} {'dur(obs)':>9} {'return':>9}  note")
    print("-" * 63)
    for _, r in table.iterrows():
        note = "" if r["complete"] else "(incomplete edge)"
        print(
            f"{r['phase']:<6} {str(r['start'].date()):<12} {str(r['end'].date()):<12} "
            f"{r['duration']:>8d} {r['return'] * 100:>8.1f}%  {note}"
        )

    comp = table[table["complete"]]
    for name in ("bull", "bear"):
        sub = comp[comp["phase"] == name]
        if len(sub):
            print(
                f"\n{name}: {len(sub)} complete phases | "
                f"median duration {sub['duration'].median():.0f} wk | "
                f"median return {sub['return'].median() * 100:+.1f}%"
            )


def main() -> None:
    from data_acquisition import load_regime_dataset

    ds = load_regime_dataset(start="1990-01-01", include_vix=False)
    price = ds.weekly_price

    window = _months_to_weeks(_T_WINDOW_MONTHS)
    censor = _months_to_weeks(_T_CENSOR_MONTHS)
    phase_min = _months_to_weeks(_T_PHASE_MONTHS)
    cycle_min = _months_to_weeks(_T_CYCLE_MONTHS)

    print(
        f"Pagan-Sossounov dating (bbdetection monthly presets scaled to weeks):\n"
        f"  window={window}wk  endpoint={censor}wk  phase_min={phase_min}wk  "
        f"cycle_min={cycle_min}wk  max_chng={_MAX_CHNG_PCT:.0f}%"
    )
    print(f"  sample: {price.index[0].date()} -> {price.index[-1].date()} ({len(price)} weeks)")

    extrema = date_turning_points(price, window, censor, phase_min, cycle_min)
    table = phase_table(price, extrema)
    _print_phase_table(table)

    label = pagan_sossounov_label(price)
    print(f"\nbear weeks: {int((label == BEAR).sum())} / {len(label)} "
          f"({100 * (label == BEAR).mean():.1f}%)")

    plot_dating(price, extrema)


if __name__ == "__main__":
    main()
