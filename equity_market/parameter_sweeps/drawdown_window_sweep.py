"""Parameter sweep: the DRAWDOWN-channel trailing-window length (weeks).

WHAT THIS ANSWERS
-----------------
The drawdown channel is  dd_t = price_t / max(price over trailing W weeks) - 1
(causal, price-only; 0 at a fresh W-week high, negative below the recent peak).
The current shipped default is W = 52 (1 year). This script refits the 3-state model
once per candidate W and scores each fit, so the window can be chosen by evidence.

WHY THE WINDOW IS A REAL KNOB (not "timeframe doesn't matter")
--------------------------------------------------------------
dd's reference peak is the trailing-max, and SPX's upward drift means in a healthy
bull the trailing-max IS ~today -> dd~0. That much is window-INSENSITIVE, and is the
intuition that says "any window that reaches the last bull peak works." BUT the window
also sets TWO things that DO depend on W, in OPPOSITE directions -- this is the tradeoff
the sweep has to measure, so we score BOTH sides, not just overall fit quality:

  * ONSET LAG (short W better).  dd only moves once price falls below the peak. A SHORT
    window's peak is recent, so an early dip shows up in dd immediately -> the bear is
    flagged sooner. A long window's peak may be a stale, far-above value, so small early
    declines barely dent dd -> slower onset.

  * RECOVERY FALSE-ALARM (long W worse).  After a crash the old high stays inside a long
    window for W weeks, so dd stays NEGATIVE deep into the new bull -> the model reads a
    healthy recovery as "still underwater / still bear" -> false alarms. A short window
    "forgets" the old peak after W weeks, so dd snaps back to 0 and clears the recovery.

So W trades onset responsiveness against recovery false-alarms. The prior 1/2/3yr sweep
found 52wk the only one that keeps false-alarm flat; this generalises that to a finer,
denser grid and reports the tradeoff explicitly so the pick is defensible.

WHAT IT DOES NOT SWEEP
----------------------
The EMISSION (Normal family, single shared dd_scale) is held FIXED at the shipped spec
so W is the only thing changing -- a clean one-variable sweep. The emission mis-spec
(dd is bounded at 0, ~20% point-mass spike at 0, hard left skew -> a single-scale Normal
is the wrong shape) is a SEPARATE axis; don't conflate the two. Sweep W here first.

HOW TO RUN (NUTS is the slow part -- one fit per window)
--------------------------------------------------------
    python parameter_sweeps/drawdown_window_sweep.py                 # default grid
    python parameter_sweeps/drawdown_window_sweep.py --windows 26,39,52,78,104
    python parameter_sweeps/drawdown_window_sweep.py --quick         # 400/400 draws (fast, rough)

Each fit reuses the EXACT shipped pipeline: RegimeDataset.observations(include_drawdown=
True, drawdown_window_weeks=W) -> model.fit(train split) -> filtered_p_bear_over(full) ->
_standard_metrics vs Pagan-Sossounov. The model reads the 'dd' column BY NAME and is
otherwise window-agnostic, so swapping W only changes the obs frame -- no model edits.

Output: a printed table + outputs/drawdown_window_sweep.csv, columns per window:
  recall / false_alarm  -- the two headline soft metrics (read TOGETHER)
  cx_total/cx_dotcom/cx_1970s -- whipsaw crossings (lower = steadier), overall + 2 eras
  gfc_rally_hold        -- min P(bear) during the spring-2008 GFC relief rally
                           (the whipsaw the channel exists to fix; want it HIGH, ~held)
  recovery_false_alarm  -- mean P(bear) over the 12mo AFTER each P&S bear ends
                           (the long-window failure mode: still-underwater -> false bear)
  oos_recall/oos_false_alarm -- same two, but only on the post-split (out-of-sample) tail
  dd_bull/dd_bear_gap/dd_scale -- fitted emission params (sanity: is the channel USED?)
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

# This file lives in equity_market/parameter_sweeps/, a SIBLING of src/. Put the src/
# subfolders (dataset, models, ...) on sys.path exactly like every src/ script does.
_SRC = _pl.Path(__file__).resolve().parents[1] / "src"
_sys.path.insert(0, str(_SRC))
import _syspath  # noqa: E402,F401  (adds src/ subfolders: dataset, models, ...)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import regime_model_3state as model  # noqa: E402
from data import load_regime_dataset  # noqa: E402
from labels import pagan_sossounov_label  # noqa: E402
from _run_io import _standard_metrics  # noqa: E402


DATA_START = "1957-03-01"          # same history the nowcast is fit/evaluated on
DEFAULT_WINDOWS = (26, 39, 52, 65, 78, 104, 156)  # 0.5,0.75,1,1.25,1.5,2,3 yr
GFC_RALLY = ("2008-03-01", "2008-05-31")  # the spring-2008 relief rally (the whipsaw fix target)


# ----------------------------------------------------------------------------
# extra, window-SPECIFIC diagnostics (beyond _standard_metrics)
# ----------------------------------------------------------------------------
def _gfc_rally_hold(p_bear: pd.Series) -> float:
    """MIN P(bear) during the spring-2008 GFC relief rally.

    This is THE thing the drawdown channel was added to fix: without it P(bear)
    collapsed 0.98 -> 0.00 mid-bear on that rally (the whipsaw). We want this HIGH
    (the model HELD the bear through the rally). NaN if the window isn't covered.
    """
    seg = p_bear.loc[GFC_RALLY[0]:GFC_RALLY[1]]
    return float(seg.min()) if len(seg) else float("nan")


def _recovery_false_alarm(p_bear: pd.Series, label: pd.Series,
                          months_after: int = 12) -> float:
    """Mean P(bear) over the `months_after` AFTER each P&S bear ENDS (bear->bull edge).

    This isolates the LONG-WINDOW failure mode: a long trailing window keeps the pre-crash
    peak in view, so dd stays negative into the new bull and the model calls the recovery
    a bear. Averaged over every recovery window in the sample. Want this LOW.
    """
    lab = label.reindex(p_bear.index).ffill()
    bear = (lab == 1).to_numpy()
    # bear->bull transitions: was bear at t-1, bull at t.
    ends = np.where((~bear[1:]) & bear[:-1])[0] + 1
    weeks_after = int(round(months_after * 52 / 12))
    mask = np.zeros(len(p_bear), dtype=bool)
    for e in ends:
        mask[e:e + weeks_after] = True
    # only count weeks that are actually BULL (a new bear starting resets the clock).
    mask &= ~bear
    return float(p_bear.to_numpy()[mask].mean()) if mask.any() else float("nan")


def _fitted_dd_params(mcmc) -> dict:
    """Posterior-mean drawdown emission params, so we can confirm the channel is USED
    (dd_bear_gap ~ 0 would mean the window makes dd useless -> the fit ignores it)."""
    s = mcmc.get_samples()
    out = {}
    for k in ("dd_bull", "dd_bear_gap", "dd_tbull_gap", "dd_scale"):
        if k in s:
            out[k] = float(np.asarray(s[k]).mean())
    return out


# ----------------------------------------------------------------------------
# one window -> one fit -> one scored row
# ----------------------------------------------------------------------------
def score_window(window_weeks: int, ds, num_warmup: int, num_samples: int) -> dict:
    """Fit the 3-state model with drawdown window = `window_weeks` and score it.

    Reuses the shipped pipeline verbatim except the dd window: build the obs frame with
    this window, fit on the 80/20 train split, filter forward over ALL history (causal,
    genuine OOS on the tail), score vs Pagan-Sossounov. Returns a flat dict (one CSV row).
    """
    # obs frame for THIS window. include_drawdown=True is fixed; only the window changes.
    # (Everything else -- credit/curve toggles -- follows the model's own obs_kwargs so we
    #  match the shipped emission set exactly, then override just the window.)
    kw = dict(model.obs_kwargs())
    kw["include_drawdown"] = True
    kw["drawdown_window_weeks"] = int(window_weeks)

    full_obs = ds.observations(**kw)
    train_obs, test_obs = ds.split(**kw)
    idx = full_obs.index
    split_date = test_obs.index[0]

    print(f"\n[W={window_weeks:3d}wk] fitting on {len(train_obs)} train weeks "
          f"(channels={list(full_obs.columns)}, warmup/samples={num_warmup}/{num_samples}) ...")
    mcmc = model.fit(train_obs, num_warmup=num_warmup, num_samples=num_samples)
    p_bear = np.asarray(model.filtered_p_bear_over(mcmc, full_obs))
    pb = pd.Series(p_bear, index=idx, name="p_bear")

    label = pagan_sossounov_label(ds.weekly_price.loc[idx]).reindex(idx).ffill()

    # headline metrics on the FULL sample, then the OOS (post-split) tail alone.
    m_full = _standard_metrics(p_bear, idx, label.to_numpy())
    oos_mask = idx >= split_date
    m_oos = _standard_metrics(p_bear[oos_mask], idx[oos_mask], label.to_numpy()[oos_mask])

    row = {
        "window_weeks": int(window_weeks),
        "window_years": round(window_weeks / 52.0, 2),
        "recall": m_full["recall"],
        "false_alarm": m_full["false_alarm"],
        "cx_total": m_full["cx_total"],
        "cx_dotcom": m_full["cx_dotcom"],
        "cx_1970s": m_full["cx_1970s"],
        "gfc_rally_hold": _gfc_rally_hold(pb),
        "recovery_false_alarm": _recovery_false_alarm(pb, label),
        "oos_recall": m_oos["recall"],
        "oos_false_alarm": m_oos["false_alarm"],
    }
    row.update(_fitted_dd_params(mcmc))

    # Persist the FULL P(bear) curve (not just the summary metrics) so downstream
    # diagnostics -- e.g. locating WHERE the whipsaw crossings fall per window -- can
    # run WITHOUT a NUTS refit. One CSV per window: date, p_bear, label.
    curve_dir = _pl.Path(__file__).resolve().parents[1] / "outputs" / "drawdown_window_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": idx, "p_bear": p_bear, "label": label.to_numpy()}).to_csv(
        curve_dir / f"W{int(window_weeks)}.csv", index=False)

    # where the crossings fall (context + episode buckets) -- returned for the report.
    cx_loc = _crossing_locations(pb, label)
    return row, cx_loc


# ----------------------------------------------------------------------------
# WHERE do the whipsaw crossings fall? (locate them, don't just count them)
# ----------------------------------------------------------------------------
# The sweep showed 26wk has ~2x the crossings of 65wk. A COUNT can't say whether that
# is real mid-bear whipsaw (bad) or harmless calm-bull chop -- only LOCATION can. This
# buckets every 0.5-crossing of P(bear) by its regime context and by episode, per window,
# and prints it as a SELF-CONTAINED, COPY-PASTEABLE block (so running locally -> pasting
# the terminal hands the full picture back with no files needed).
_EDGE_WEEKS = 8   # a crossing within +-this many weeks of a P&S bear boundary = "edge"
_CX_EPISODES = [  # label -> (start, end); crossings are tallied per episode too
    ("1970s (68-83)", "1968-01-01", "1983-01-01"),
    ("dotcom (99-03)", "1999-06-01", "2003-12-31"),
    ("GFC (07-11)", "2007-06-01", "2011-12-31"),
    ("covid (20)", "2020-01-01", "2020-12-31"),
]


def _crossing_locations(p_bear: pd.Series, label: pd.Series) -> dict:
    """Bucket every 0.5-crossing of P(bear) by regime CONTEXT and by episode.

    A crossing at week t is where (p>0.5) flips vs t-1 -- the exact definition cx_total
    counts. Context of each crossing (using the P&S label, causal-free -- this is
    diagnosis, not a model input):
      in_bear   : the crossing sits inside a true P&S bear (real whipsaw -- the model
                  abandoning/re-entering an established bear = the costly kind)
      near_edge : within _EDGE_WEEKS of a bear<->bull boundary (onset/exit timing churn
                  -- expected, mostly benign)
      calm_bull : deep in a true bull, far from any bear (the feature twitching on
                  ordinary price texture -- pollutes false-alarm/turnover, not recall)
    Also returns per-episode crossing counts, so a CLUSTERED failure (all extra crossings
    in the slow-bear episodes) is distinguishable from a diffuse one.
    """
    p = p_bear.to_numpy()
    lab = label.reindex(p_bear.index).ffill().to_numpy()
    idx = p_bear.index

    b = (p > 0.5).astype(int)
    cx = np.where(np.abs(np.diff(b)) == 1)[0] + 1  # week positions of crossings
    is_bear = lab == 1
    # distance (in weeks) from each week to the nearest bear<->bull boundary
    edges = np.where(np.abs(np.diff(is_bear.astype(int))) == 1)[0] + 1
    near_edge = np.zeros(len(p), dtype=bool)
    for e in edges:
        near_edge[max(0, e - _EDGE_WEEKS):e + _EDGE_WEEKS + 1] = True

    ctx = {"in_bear": 0, "near_edge": 0, "calm_bull": 0}
    for t in cx:
        if is_bear[t] and not near_edge[t]:
            ctx["in_bear"] += 1
        elif near_edge[t]:
            ctx["near_edge"] += 1
        else:
            ctx["calm_bull"] += 1

    ep_counts = {}
    for name, a, c in _CX_EPISODES:
        seg = pd.Series(b, index=idx).loc[a:c].to_numpy()
        ep_counts[name] = int(np.abs(np.diff(seg)).sum()) if len(seg) > 1 else 0

    return {"total": int(len(cx)), "ctx": ctx, "episodes": ep_counts}


def _print_crossing_report(per_window: dict) -> None:
    """Copy-pasteable crossing-location report across all swept windows.

    per_window: {W -> _crossing_locations(...) result}. Prints two tables:
    (1) crossing CONTEXT mix per window (in_bear / near_edge / calm_bull),
    (2) crossing counts per EPISODE per window (to see clustering).
    """
    Ws = sorted(per_window)
    print("\n" + "#" * 100)
    print("# CROSSING-LOCATION REPORT  (copy-paste this whole block back)")
    print("# where do P(bear) 0.5-crossings fall? in_bear = real whipsaw (bad);")
    print("# near_edge = onset/exit churn (benign); calm_bull = twitch on chop (false-alarm).")
    print("#" * 100)
    # (1) context mix
    print("\n[context mix per window]")
    print(f"{'W(wk)':>6} {'total':>6} {'in_bear':>8} {'near_edge':>10} {'calm_bull':>10}")
    for W in Ws:
        c = per_window[W]["ctx"]; tot = per_window[W]["total"]
        print(f"{W:>6} {tot:>6} {c['in_bear']:>8} {c['near_edge']:>10} {c['calm_bull']:>10}")
    # (2) per-episode
    print("\n[crossings per episode per window]")
    ep_names = [n for n, _, _ in _CX_EPISODES]
    hdr = f"{'W(wk)':>6} " + " ".join(f"{n:>16}" for n in ep_names)
    print(hdr)
    for W in Ws:
        ep = per_window[W]["episodes"]
        print(f"{W:>6} " + " ".join(f"{ep.get(n, 0):>16}" for n in ep_names))
    print("#" * 100)
    print("# READ: if the extra crossings at short W concentrate in in_bear + the slow-bear")
    print("# episodes (1970s/dotcom), short windows fail STRUCTURALLY where it matters ->")
    print("# avoid short W there (motivates the variable/event-reset peak). If they're mostly")
    print("# calm_bull/diffuse, it's benign chop and the penalty is milder.")
    print("#" * 100)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run_sweep(windows, num_warmup: int, num_samples: int) -> pd.DataFrame:
    ds = load_regime_dataset(start=DATA_START, include_vix=False,
                             include_macro=model.needs_macro if hasattr(model, "needs_macro")
                             else getattr(model, "_NEEDS_MACRO", False))
    rows = []
    cx_by_window = {}
    for w in windows:
        try:
            row, cx_loc = score_window(w, ds, num_warmup, num_samples)
            rows.append(row)
            cx_by_window[int(w)] = cx_loc
        except Exception as e:  # keep the sweep alive if one window blows up
            print(f"[W={w}] FAILED: {e!r}")
            rows.append({"window_weeks": int(w), "window_years": round(w / 52.0, 2),
                         "error": repr(e)})
    return pd.DataFrame(rows), cx_by_window


def _print_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("DRAWDOWN WINDOW SWEEP  (3-state, emission fixed; ONLY the dd trailing window varies)")
    print("  read recall & false_alarm TOGETHER; gfc_rally_hold HIGH = whipsaw held;")
    print("  recovery_false_alarm LOW = short-enough window forgets old peak into the recovery.")
    print("=" * 100)
    cols = ["window_weeks", "window_years", "recall", "false_alarm", "cx_total",
            "gfc_rally_hold", "recovery_false_alarm", "oos_recall", "oos_false_alarm",
            "dd_bear_gap", "dd_scale"]
    show = df[[c for c in cols if c in df.columns]].copy()
    fmt = {c: "{:.3f}".format for c in show.columns
           if c not in ("window_weeks", "cx_total")}
    with pd.option_context("display.width", 160, "display.max_rows", 60):
        print(show.to_string(index=False, formatters=fmt))
    print("=" * 100)
    print("Choosing: want HIGH recall + LOW/flat false_alarm + HIGH gfc_rally_hold + LOW")
    print("recovery_false_alarm. The pick is where recall/rally-hold saturate while")
    print("false_alarm is still flat -- the shortest such window (least recovery lag).")


def main(argv):
    quick = "--quick" in argv
    num_warmup = 400 if quick else 1000
    num_samples = 400 if quick else 1000
    if "--windows" in argv:
        windows = tuple(int(x) for x in argv[argv.index("--windows") + 1].split(","))
    else:
        windows = DEFAULT_WINDOWS

    print(f"Windows to sweep (weeks): {windows}")
    print(f"NUTS draws per fit: warmup={num_warmup}, samples={num_samples}"
          f"{'  [--quick: rough, for a fast read]' if quick else ''}")
    print(f"That is {len(windows)} NUTS fits total -- this is the slow part.\n")

    df, cx_by_window = run_sweep(windows, num_warmup, num_samples)
    _print_table(df)
    _print_crossing_report(cx_by_window)

    out_dir = _pl.Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drawdown_window_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}  (+ per-window P(bear) curves in outputs/drawdown_window_curves/)")


if __name__ == "__main__":
    main(_sys.argv[1:])
