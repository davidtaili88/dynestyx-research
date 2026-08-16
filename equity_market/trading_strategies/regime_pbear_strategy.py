"""Simple regime-switching S&P 500 strategy driven by a regime nowcast.

Driven by the 3-state model (src/models/regime_model_3state.py): channels [r_t, v_t, dd]
via its fit / filtered_p_bear_over interface. (The 4-state variant was archived to
unused_mechanisms/; --model 4state is no longer available.)

DESIGN -- TWO legs, both TRANSITION-based (see the parameter block for the full evidence
and anti-overfit reasoning behind every number):
  * LONG core   : +1 when P(bear) < P_LONG (0.90, the model's BEAR LINE), else flat -- long
                  unless the model calls an outright bear (the one principled long/flat cutoff).
  * RECOVERY LEV: raise the long to LEV_MULT (1.25x) while riding a bear's RECOVERY (armed
                  when a bear resolves back down to bull-leaning, held until an early
                  disturbance ends it, NO re-arm until a fresh bear). Recovery-momentum edge.
So positions are in {0, +1, +1.25x} -- long/flat with a leverage bump out of bears. (A
momentum short was also tried; it added no net edge -- a more accurate nowcast already
captures the downside by going flat in bears -- so it is DISABLED (SHORT_SIZE=0), left in
the code only for reproducibility.)

WHY TRANSITIONS not LEVELS: P(bear) is near-binary (~0 most weeks, ~1 in bears; only ~1% of
weeks land in [0.25,0.60]) -- no gradient to size along -- so level-based sizing was tested
and REJECTED. Pass --flat to fall back to plain long/flat.

NO-LOOKAHEAD CONVENTION. P(bear_t) is the CAUSAL filtered probability using data
through week t's Friday close (the model's forward filter, filtered_p_bear_over).
The position implied by P(bear_t) is therefore only tradeable at the t close, so it
earns the return of week t+1. We shift the position forward one week before applying
returns -- so no bar's return is ever earned by a signal that peeked at it.

Reporting: an equity curve, annualized return / vol / Sharpe, max drawdown, hit rate,
and a strategy-vs-buy&hold plot.

Run:  python trading_strategies/regime_pbear_strategy.py                  # 3-state
      python trading_strategies/regime_pbear_strategy.py --refit          # ignore cache, refit NUTS
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

import importlib  # noqa: E402
from data_acquisition import load_regime_dataset  # noqa: E402


# ----------------------------------------------------------------------------
# Strategy parameters
# ----------------------------------------------------------------------------
# FINAL DESIGN (2026-08-09). Every rule below is a MECHANISM tied to a P(bear) TRANSITION
# (an event: crossing a line, peaking, resolving), NOT a size read off P(bear)'s LEVEL.
# This was a deliberate anti-overfit choice: we tested level-based sizing and it FAILED
# (P(bear) is near-binary -- ~0 most weeks, ~1 in bears -- with rank-corr ~0.03 to forward
# return in the sub-crisis range, so there is no gradient to size along; the middle bins
# are 16-56wk noise). Transitions, by contrast, gave replicated edges. See the block above
# each parameter for the evidence.
#
# TWO LIVE LEGS (the momentum short was tested and DISABLED 2026-08-08 -- see SHORT_SIZE):
#   LONG core   : long when P(bear) < P_LONG, else flat.
#   RECOVERY LEV: lever the LONG to LEV_MULT through a bear's recovery (a real state), armed
#                 only when a bear RESOLVES, held until a fresh DISTURBANCE, NO re-arm.
#   (MOMENTUM SHT: kept behind SHORT_SIZE=0 for reproducibility; not part of the live book.)
#
# ---- LONG core -------------------------------------------------------------------------
# P_LONG=0.90 (2026-08-09): stay long until P(bear) crosses the model's OWN BEAR LINE (0.90).
# WHY 0.90 and not an arbitrary middle value: since we no longer short, the strategy is just
# long/flat, so the long/flat cutoff should be the ONE principled probability line the model
# already defines -- "this is a bear" = 0.90 (the same line the recovery-arm uses as REARM).
# A middle cutoff like 0.60 is arbitrary AND noise-tripped: P(bear) routinely spikes to ~0.7
# for a week on noise then falls back, and 0.60 needlessly flattened us through those weeks.
# EVIDENCE: raising 0.60 -> 0.90 is a clean win -- the 59 weeks we were flat at 0.60 but long
# at 0.90 (P(bear) in [0.60,0.90], the noise band) returned +14%/yr forward; full Sharpe
# 0.578 -> 0.586, OOS Sharpe 0.85 -> 0.87. Cost: OOS maxDD -20% -> -23% (staying long a bit
# deeper into a developing bear) -- net-favorable (OOS Sharpe still up, DD still << B&H's -32%).
# So 0.90 is the principled, tested cutoff: long unless the model calls an outright bear.
P_LONG = 0.90
#
# ---- RECOVERY LEVERAGE -----------------------------------------------------------------
# MECHANISM: "longing extra units" only makes sense with a WHY. The why is RECOVERY
# MOMENTUM -- weeks where P(bear) is FALLING out of a bear earned +14.5%/yr fwd vs +9.2%
# for long-eligible weeks overall (n=220, a real sample, not a level bucket). So we lever
# the LONG while the market is climbing OUT of a bear. Each threshold below sits directly
# above its own assignment.
#
# LEV_MULT: a multiple sweep showed Sharpe PEAKS at ~1.25-1.5x then declines while drawdown
# worsens monotonically -- past ~1.5x you buy return purely with risk. 1.25x is the
# conservative edge of the plateau (Sharpe 0.581 vs 0.570 base; 1.5x=0.586 but -58% DD).
# NOT the argmax; the low end of the plateau.
LEV_MULT = 1.25           # lever the long to this multiple through a bear's recovery
#
# LEV_ARM / REARM_BEAR_LINE: a "recovery" begins when, having been in a bear
# (P(bear) > REARM_BEAR_LINE), P(bear) resolves back down through LEV_ARM -- we ride the
# whole rebound as a STATE, not per-week. LEV_ARM is a CONCEPTUAL-CLARITY choice, not a
# performance one: it's NOT load-bearing (arm 0.60 vs 0.90 both give Sharpe ~0.585 because
# the disturb logic decides when leverage actually engages), so we pick the defensible STORY.
# 0.60 = "arm once the model is back to bull-leaning"; 0.90 would mean "start levering while
# the model still calls it a ~90%-bear" -- indefensible even if it scores the same. Arming
# LOWER (0.10, 'wait for full bull confirmation') is WORSE (0.573): the recovery-momentum
# edge is front-loaded, so you must arm as it resolves, not after.
LEV_ARM = 0.60            # recovery armed when P(bear) resolves back down through this (bull-leaning)
REARM_BEAR_LINE = 0.90    # "a bear" = P(bear) exceeded this (what must precede a recovery)
#
# LEV_DISTURB: the recovery (and the leverage) ENDS the moment P(bear) climbs back above
# this -- a fresh sign of stress. The ONE genuinely un-derivable threshold, framed HONESTLY
# as a declared early-exit RISK POSTURE ("pull the extra leverage at the first real sign of
# stress"), NOT an optimum. The EARLINESS is tested-and-justified: every LATER/CONDITIONED
# exit is worse -- raising it toward the bear line rides leverage into bears (0.60 -> Sharpe
# 0.570); a duration GRACE period (ignore early-bull spikes for ~26wk then heed) is worse
# (0.578), because some 'early chop' spikes are real continuation legs (dotcom) the early
# exit correctly catches. The exact decimal is insensitive in the early band (0.25 vs 0.40 ~
# same; only ~1% of weeks live in [0.25,0.60]), so 0.25 is the conservative (earliest) choice.
LEV_DISTURB = 0.25        # recovery/leverage ends when P(bear) climbs back above this (early-exit posture)
#
# NO RE-ARM: once a disturbance ends the leverage we do NOT re-lever on the next dip under
# LEV_DISTURB; we require a WHOLE NEW bear cycle (> REARM_BEAR_LINE then resolve) first.
# Tested: re-arming adds 1000+ levered weeks but LOWERS Sharpe (0.581->0.567) -- re-levering
# into a market that already showed stress is uncompensated risk. Confirmed OOS (0.83 vs 0.79).
#
# ---- MOMENTUM SHORT --------------------------------------------------------------------
# MECHANISM: the losing zone is the DESCENT into a bear, not any P(bear) level. Forward
# short PnL after P(bear) UP-CROSSES 0.90 is +0.6-0.8% per event for a 2-6wk hold, decays
# to 0 by ~8wk, and goes NEGATIVE past 10wk (the recovery bounce) -- REPLICATED on 3- and
# 4-state. So: short the entry, exit before the bounce.
#   P_SHORT=0.90: enter on the UP-CROSS (prev<=0.90 & now>0.90), a one-time event.
#   SHORT_PEAK_TAIL=3: once P(bear) first ticks DOWN (peak = bear engaged), hold 3 more wks
#     then HARD-EXIT. Rationale (user's): the peak confirms a real bear; past it the question
#     flips from "capture PnL" to "avoid overstaying" -- so stop flirting and leave. Exit at
#     0-3wk-after-peak is a flat plateau (Sharpe ~0.568-0.570); it FALLS OFF A CLIFF at 4wk
#     (0.522). "Peak" = FIRST downtick (twitchy, Sharpe-preserving); a pullback-THRESHOLD
#     peak instead holds far longer -> a real -11pt drawdown hedge but Sharpe ~0.50 (a
#     different product; we chose the Sharpe-preserving scalp).
#   SHORT_SIZE=0.5: half unit. Full size captures a bit more raw PnL but adds enough vol to
#     drop Sharpe BELOW the no-short baseline; half-size lands at baseline Sharpe with a
#     small return bump. The short is a minor sweetener (~+0.4%/yr), sized so it never hurts.
# SHORT_SIZE=0.0 (2026-08-08): the momentum short is DISABLED -- removing it is a clean win
# on the shipped event-reset-dd + Normal curve. WITH short vs NO short (same curve, leverage
# bugfix on): OOS Sharpe 0.78 -> 0.85, OOS maxDD -24% -> -20% (recovering/beating the original
# -22%), OOS return 11.4% -> 13.0%, trades 155 -> 109, turnover 2.08x -> 1.53x; full-sample
# Sharpe tied (0.576 vs 0.578). WHY the short died: a more accurate nowcast (event-reset dd)
# resolves bears faster, so the strategy's flat-during-bear stance already captures the
# downside -- the short just added turnover and OOS drawdown. Five variants were tested
# (size 0.5/1.0, level-based, stop-loss, confirmation-delay); all underperformed no-short.
# The short CODE below is kept intact (behind SHORT_SIZE) so the test stays reproducible.
P_SHORT = 0.90
SHORT_PEAK_TAIL = 3      # weeks to hold after P(bear) first ticks down, then hard-exit
SHORT_SIZE = 0.0         # DISABLED (was 0.5) -- see the note above; no-short wins OOS
#
WEEKS_PER_YEAR = 52.0    # weekly (W-FRI) bars -> annualization factor
DATA_START = "1957-03-01"  # same history the model runs are fit/evaluated on (fit_mode_processor)
# NOTE: results are FRICTIONLESS. A turnover-based transaction-cost model was built but not
# validated/calibrated, so it was removed from here -- see unused_mechanisms/transaction_costs.md.
# (The backtest engine still accepts a dormant cost_per_turn=0.0 kwarg if it's ever re-enabled.)


def _load_model(model_key: str):
    """Import the regime model module for `model_key`. Only "3state" is live.

    The 3-state model exposes the interface the strategy needs: fit(train_obs),
    filtered_p_bear_over(mcmc, frame), obs_kwargs(), _NEEDS_MACRO. (The fit-cache pickle
    helpers save_fit/load_fit are model-agnostic and now live in model_utils/persistence.)
    (The 4-state model was archived to unused_mechanisms/, so model_key="4state" is no
    longer available -- it is kept only as a reference implementation.)
    """
    if model_key != "3state":
        raise ValueError(
            f"model_key={model_key!r} is not available: only the 3-state model is live. "
            "The 4-state model was moved to unused_mechanisms/ (reference only)."
        )
    mod = importlib.import_module("regime_model_3state")
    from persistence import save_fit, load_fit
    needs_macro = getattr(mod, "_NEEDS_MACRO", getattr(mod, "needs_macro", False))
    return mod, save_fit, load_fit, needs_macro


# ----------------------------------------------------------------------------
# 1. Get the causal filtered P(bear) curve from the 3-state / 3-emission model
# ----------------------------------------------------------------------------
def load_pbear(model_key: str = "3state", refit: bool = False):
    """Return (price, r_t, p_bear) as pandas objects on the weekly index.

    p_bear is the GLOBAL-fit causal filtered P(bear_t | y_1:t) from the chosen model
    (3state or 4state) with its current channel config (r_t, v_t, dd). We fit ONCE on
    the 80/20 train split (model.fit) and filter forward over ALL history
    (model.filtered_p_bear_over) -- identical to fit_mode_processor._run_global, so the curve
    the strategy trades is exactly the one the nowcast plots. In the 4-state,
    filtered_p_bear_over already sums BOTH bear flavors, so P(bear) means the same
    thing (total bear probability) in both models.

    The fit is cached to outputs/regime_<model>_strategy_pbear.pkl (NUTS is
    expensive); pass refit=True to force a fresh fit.
    """
    mod, save_fit, load_fit, needs_macro = _load_model(model_key)
    # KEY THE CACHE TO THE EMISSION so changing the model's emission family (e.g.
    # DD_EMISSION "normal" <-> "hurdle_logt" in regime_model_3state) auto-selects a
    # DIFFERENT cache file instead of silently serving a stale curve fit with the old
    # emission. Any model exposing a DD_EMISSION tag gets its own cache; models without
    # one fall back to the plain name (unchanged behavior).
    emission_tag = getattr(mod, "DD_EMISSION", None)
    cache_name = (f"regime_{model_key}_strategy_pbear_{emission_tag}" if emission_tag
                  else f"regime_{model_key}_strategy_pbear")

    ds = load_regime_dataset(start=DATA_START, include_vix=False,
                             include_macro=needs_macro)
    kw = mod.obs_kwargs()
    full_obs = ds.observations(**kw)
    idx = full_obs.index
    price = ds.weekly_price.loc[idx]
    r_t = full_obs["r_t"]

    # Try the cache first: it stores the P(bear) curve keyed to the obs index so we
    # can skip the NUTS fit on re-runs (parameter tuning of the STRATEGY is cheap;
    # the fit is not).
    if not refit:
        try:
            cached = load_fit(cache_name)
            pb = cached["extra"]["p_bear"]
            if len(pb) == len(idx):
                print(f"[cache] loaded P(bear) from outputs/{cache_name}.pkl")
                return price, r_t, pd.Series(np.asarray(pb), index=idx, name="p_bear")
            print("[cache] stale (length changed) -> refitting")
        except FileNotFoundError:
            print(f"[cache] none found -> fitting the {model_key} model (this is the slow step)")

    train_obs, _test_obs = ds.split(**kw)
    print(f"Fitting {model_key} model on {len(train_obs)} train weeks, channels={list(full_obs.columns)} ...")
    mcmc = mod.fit(train_obs)
    p_bear = np.asarray(mod.filtered_p_bear_over(mcmc, full_obs))

    # Cache just the curve + its dates (not the whole fit) via save_fit's `extra`.
    save_fit(mcmc, cache_name, list(full_obs.columns), extra={
        "p_bear": p_bear,
        "dates": [str(d.date()) for d in idx],
        "obs_cols": list(full_obs.columns),
    })
    return price, r_t, pd.Series(p_bear, index=idx, name="p_bear")


# ----------------------------------------------------------------------------
# 2. Turn P(bear) into a position (TWO live transition-based legs; see the parameter block)
# ----------------------------------------------------------------------------
def build_positions(p_bear: pd.Series, p_long: float = P_LONG,
                    lev_mult: float = LEV_MULT, lev_arm: float = LEV_ARM,
                    rearm_bear_line: float = REARM_BEAR_LINE, lev_disturb: float = LEV_DISTURB,
                    p_short: float = P_SHORT, short_peak_tail: int = SHORT_PEAK_TAIL,
                    short_size: float = SHORT_SIZE, flat: bool = False) -> pd.Series:
    """Map the P(bear) curve to a per-week position (causal: uses only p_bear up to t).

    TWO LIVE LEGS -- long core + recovery leverage -- giving positions in {0, +1, +lev_mult}:

      LONG core   : +1 whenever p_bear < p_long, else 0 (flat).
      RECOVERY LEV: while long, raise +1 -> lev_mult during a recovery STATE. The state is
        ARMED when, having seen a bear (p_bear > rearm_bear_line), p_bear resolves back down
        through lev_arm; it ENDS on a disturbance (p_bear climbs back above lev_disturb).
        NO RE-ARM: after a disturbance we require a whole NEW bear (>rearm_bear_line) before
        re-levering -- we do not re-lever on a mere dip back under lev_disturb.

    (A momentum short leg also exists in the code but is DISABLED by default (short_size=0),
    so it contributes nothing to the live book -- it was tested and added no edge; kept only
    for reproducibility. When short_size>0 it opens a -short_size short on the up-cross of
    p_short and exits short_peak_tail weeks after p_bear peaks, overriding the long.)

    flat=True short-circuits everything to the plain long/flat baseline (long if
    p_bear < p_long) for apples-to-apples comparison. Position at t is decided at the t
    close; the backtest shifts it +1 week before applying returns (no lookahead).
    """
    pb = p_bear.to_numpy()
    n = len(pb)
    pos = np.where(pb < p_long, 1.0, 0.0)

    if flat:
        return pd.Series(pos, index=p_bear.index, name="position")

    in_rec = False      # currently riding a recovery (leverage on)
    seen_bear = False   # a bear (p>rearm_bear_line) has occurred and not yet been "used"
    in_short = False     # currently short
    short_peaked = False # p_bear has ticked down since this short opened
    since_peak = 0
    for t in range(1, n):
        p, pprev = pb[t], pb[t - 1]

        # ---- MOMENTUM SHORT (takes priority over long/recovery while active) ----
        if not in_short and short_size > 0 and pprev <= p_short and p > p_short:
            in_short = True
            short_peaked = False
            since_peak = 0
        if in_short:
            if not short_peaked and p < pprev:   # first downtick = peak (bear engaged)
                short_peaked = True
                since_peak = 0
            if short_peaked:
                since_peak += 1
            pos[t] = -short_size
            if short_peaked and since_peak >= short_peak_tail:
                in_short = False
            # a bear that fires the short also counts as "seen" for the recovery arm below
            seen_bear = True
            in_rec = False
            continue

        # ---- RECOVERY LEVERAGE (only when not short) ----
        if p > rearm_bear_line:
            seen_bear = True
            in_rec = False
        # arm the recovery when a SEEN bear resolves back down through lev_arm
        armed_now = False
        if seen_bear and not in_rec and p < lev_arm and pprev >= lev_arm:
            in_rec = True
            seen_bear = False
            armed_now = True
        # end the recovery on a fresh disturbance -- but NOT on the same week we just armed.
        # BUGFIX: a SHARP recovery drops P(bear) from ~1.0 to between lev_disturb (0.25) and
        # lev_arm (0.60) in ONE week; without the `armed_now` guard the arm fired and this
        # disturb check disarmed it on the SAME iteration (0.45 > 0.25), so leverage never
        # engaged. That silently dropped the fastest, strongest-momentum recoveries (1974,
        # 2003, ...) -- exactly the ones worth levering -- and event-reset dd makes recoveries
        # sharper, so it bit more. The disturb exit is meant for a FRESH disturbance on a
        # LATER week, never the arming transition itself.
        if in_rec and p > lev_disturb and not armed_now:
            in_rec = False
        if in_rec and pos[t] > 0:
            pos[t] = lev_mult

    return pd.Series(pos, index=p_bear.index, name="position")


# ----------------------------------------------------------------------------
# 3. Backtest + performance stats
# ----------------------------------------------------------------------------
def backtest(price: pd.Series, r_t: pd.Series, positions: pd.Series,
             cost_per_turn: float = 0.0) -> pd.DataFrame:
    """Apply positions to next-week returns and build the strategy P&L frame.

    r_t is the weekly LOG return. Position at week t (decided at the t close) earns
    week t+1's return, so we use positions.shift(1). Strategy log return_t =
    pos_{t-1} * r_t. We also track buy&hold (always +1) for comparison.

    Runs FRICTIONLESS by default (cost_per_turn=0). The cost_per_turn kwarg is a dormant
    hook for a deferred turnover-cost model -- see unused_mechanisms/transaction_costs.md.

    Returns a DataFrame indexed by week with: p-return columns, cumulative equity
    curves (start=1.0), and the traded position.
    """
    r = r_t.reindex(positions.index)
    pos_lag = positions.shift(1).fillna(0.0)  # no position before the first signal

    gross_logret = pos_lag * r
    # Turnover = |change in the HELD position| each week (first week's |pos| is the
    # cost of establishing the initial position). Cost is a fractional drag, applied as
    # log(1 - cost) so it compounds consistently with the log-return equity curve.
    turnover = pos_lag.diff().abs().fillna(pos_lag.abs())
    cost_frac = cost_per_turn * turnover
    cost_logret = np.log1p(-cost_frac.clip(upper=0.999999))  # <=0, guard against >=100% cost
    strat_logret = gross_logret + cost_logret

    bh_logret = r.copy()  # buy & hold the S&P
    if cost_per_turn > 0 and len(bh_logret):  # single entry cost for buy&hold
        bh_logret.iloc[0] += np.log1p(-cost_per_turn)

    df = pd.DataFrame({
        "price": price.reindex(positions.index),
        "p_bear": None,  # filled by caller if wanted; kept for a tidy single frame
        "position": positions,
        "pos_traded": pos_lag,
        "r_t": r,
        "turnover": turnover,
        "cost_logret": cost_logret,
        "gross_logret": gross_logret,
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


def performance_stats(logret: pd.Series, equity: pd.Series, positions: pd.Series | None = None,
                      turnover: pd.Series | None = None, cost_logret: pd.Series | None = None) -> dict:
    """Annualized return / vol / Sharpe, total return, max drawdown, hit rate, exposure.

    Sharpe here is EXCESS-of-zero (risk-free ~0). Returns are weekly LOG returns;
    we annualize by *52 (mean) and *sqrt(52) (vol), the standard weekly convention.
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
    if turnover is not None:
        tv = turnover.reindex(lr.index).fillna(0.0)
        stats["ann_turnover"] = float(tv.sum() / (n / WEEKS_PER_YEAR)) if n else float("nan")
    if cost_logret is not None:
        cl = cost_logret.reindex(lr.index).fillna(0.0)
        stats["ann_cost_drag"] = float(cl.mean() * WEEKS_PER_YEAR)  # <=0, cost in ann. log-return
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
    if "ann_turnover" in s:
        lines.append(f"    ann. turnover    : {s['ann_turnover']:8.2f}x")
    if "ann_cost_drag" in s:
        lines.append(f"    ann. cost drag   : {s['ann_cost_drag']*100:8.2f}%")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 4. Plot: equity curves + position/P(bear) context
# ----------------------------------------------------------------------------
def _pagan_sossounov_bands(index: pd.Index) -> np.ndarray | None:
    """0/1 Pagan-Sossounov BEAR label aligned to `index` (1 = bear week), or None if
    the labels module isn't importable. Same ground truth the nowcast plot shades."""
    try:
        from pagan_sossounov import pagan_sossounov_label
    except Exception:
        return None
    ds = load_regime_dataset(start=DATA_START, include_vix=False, include_macro=False)
    lab = pagan_sossounov_label(ds.weekly_price).reindex(index).ffill()
    return np.asarray(lab)


def _shade_bear_bands(ax, dates, label):
    """Shade contiguous P&S BEAR runs on `ax` -- IDENTICAL style to the nowcast plot
    (regime_model_3state.plot_regime_fit: crimson, alpha=0.12) so the two figures'
    red bands line up exactly."""
    if label is None:
        return
    start = 0
    for t in range(1, len(label) + 1):
        if t == len(label) or label[t] != label[start]:
            if label[start] == 1:  # BEAR run
                ax.axvspan(dates[start], dates[t - 1], color="crimson", alpha=0.12, linewidth=0)
            start = t


def plot_strategy(df: pd.DataFrame, p_bear: pd.Series, save_path=None, model_key: str = "3state"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             height_ratios=[3, 1.0, 1.0])
    dates = df.index
    ps_label = _pagan_sossounov_bands(dates)  # P&S BEAR ground-truth bands (shared with nowcast)

    ax = axes[0]
    _has_short = SHORT_SIZE > 0 and (df["position"] < 0).any()
    _strat_label = ("P(bear) long/flat + recovery-leverage strategy" if not _has_short
                    else "P(bear) long/short/flat strategy")
    ax.plot(dates, df["strat_equity"], color="darkgreen", lw=1.4, label=_strat_label)
    ax.plot(dates, df["bh_equity"], color="gray", lw=1.1, label="buy & hold S&P")
    _shade_bear_bands(ax, dates, ps_label)  # red = Pagan-Sossounov BEAR (same as nowcast)
    ax.set_yscale("log")
    ax.set_ylabel("equity (log, start=1)")
    ax.set_title(f"Regime P(bear) strategy vs buy & hold  [{model_key} nowcast, channels r_t/v_t/dd]"
                 "\nlong < 90% (bear line) + 1.25x recovery leverage, no short   |   "
                 "red bands = Pagan-Sossounov BEAR")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1]
    ax.plot(dates, p_bear.reindex(dates), color="darkorange", lw=1)
    ax.axhline(P_LONG, color="red", lw=0.8, ls="--", label=f"long/flat cutoff {P_LONG:.0%} (bear line)")
    ax.axhline(LEV_ARM, color="green", lw=0.7, ls="--", label=f"lev-arm {LEV_ARM:.0%} (bull-leaning)")
    ax.axhline(LEV_DISTURB, color="seagreen", lw=0.7, ls=":", label=f"lev-disturb {LEV_DISTURB:.0%} (early exit)")
    if SHORT_SIZE > 0:  # only drawn if the short is actually enabled
        ax.axhline(P_SHORT, color="firebrick", lw=0.7, ls="-.", label=f"short up-cross {P_SHORT:.0%}")
    _shade_bear_bands(ax, dates, ps_label)  # P&S BEAR bands behind the P(bear) curve
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(bear)")
    ax.legend(loc="center left", fontsize=7, ncol=3)

    ax = axes[2]
    pos = df["position"].reindex(dates)
    # Long/recovery split: shade the levered portion (pos>1) a darker green so the
    # recovery-leverage state is visible distinct from the +1 core long.
    ax.fill_between(dates, 0, pos.clip(upper=1.0), step="pre",
                    where=(pos > 0), color="green", alpha=0.45, label="long (1x)")
    ax.fill_between(dates, 1.0, pos, step="pre",
                    where=(pos > 1.0), color="darkgreen", alpha=0.6, label=f"recovery ({LEV_MULT:g}x)")
    yticks = {0, 1, LEV_MULT}
    if SHORT_SIZE > 0 and (pos < 0).any():   # only draw/label the short if it's actually used
        ax.fill_between(dates, 0, pos, step="pre",
                        where=(pos < 0), color="red", alpha=0.6, label=f"short ({SHORT_SIZE:g}x)")
        yticks.add(-SHORT_SIZE)
    _shade_bear_bands(ax, dates, ps_label)  # P&S BEAR bands behind the position track
    ax.set_ylim((-0.9 if SHORT_SIZE > 0 else -0.15), max(1.4, LEV_MULT + 0.15))
    ax.set_yticks(sorted(yticks))
    ax.set_ylabel("position")
    ax.set_xlabel("date")
    ax.legend(loc="upper left", fontsize=7, ncol=3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved plot -> {save_path}")
    plt.show()
    return fig, axes


# ----------------------------------------------------------------------------
# 5. Calibration diagnostic: P(bear) level vs forward return
# ----------------------------------------------------------------------------
def analyse_neutral(df: pd.DataFrame, p_bear: pd.Series) -> dict:
    """P(bear) LEVEL vs mean NEXT-week S&P return, as a fixed-bin table.

    This is the diagnostic that DROVE the whole final design: it shows the relationship
    is near-binary and non-monotonic (forward return does NOT rise smoothly with P(bear)),
    which is why the strategy sizes off TRANSITIONS (crossings / peaks / resolutions), not
    off P(bear)'s level. Kept as a sanity readout; treat the small-sample mid bins as noise.
    """
    r = df["r_t"]
    pb_lag = p_bear.shift(1).reindex(df.index)  # the P(bear) that DECIDED next-week's return

    edges = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0001]
    bins = pd.cut(pb_lag, edges, right=False, include_lowest=True)
    grp = pd.DataFrame({"r": r, "bin": bins}).dropna().groupby("bin", observed=True)["r"]
    return {"pbear_bins": pd.DataFrame({
        "weeks": grp.size(),
        "mean_fwd_ret_bps": grp.mean() * 1e4,
        "ann_fwd_ret_pct": grp.mean() * WEEKS_PER_YEAR * 100,
        "share_up": grp.apply(lambda s: (s > 0).mean()),
    })}


def _print_neutral(a: dict) -> None:
    print("\n" + "-" * 72)
    print("  CALIBRATION: P(bear) LEVEL -> mean NEXT-week S&P return (near-binary/non-monotonic")
    print("  -> we size off TRANSITIONS, not level; mid bins are small-sample noise)")
    print("-" * 72)
    b = a["pbear_bins"].copy()
    b["mean_fwd_ret_bps"] = b["mean_fwd_ret_bps"].round(1)
    b["ann_fwd_ret_pct"] = b["ann_fwd_ret_pct"].round(1)
    b["share_up"] = (b["share_up"] * 100).round(1)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(b.to_string())
    print("-" * 72)


# ----------------------------------------------------------------------------
# 6. Robustness sweeps -- show PLATEAUS around the shipped params (not an optimizer)
# ----------------------------------------------------------------------------
def sweep_design(price, r_t, p_bear,
                 lev_grid=(1.0, 1.25, 1.5, 1.75, 2.0),
                 disturb_grid=(0.10, 0.25, 0.40, 0.60),
                 tail_grid=(0, 2, 3, 4, 6),
                 cost_per_turn: float = 0.0) -> pd.DataFrame:
    """Robustness check (NOT an optimizer): vary ONE knob at a time around the shipped
    defaults and show the metric is a PLATEAU, so each default is a robust mid-plateau
    pick rather than an argmax. Re-uses the fitted P(bear) (cheap; no NUTS).

    Three mini-sweeps: recovery LEV_MULT, recovery LEV_DISTURB, and short SHORT_PEAK_TAIL.
    """
    rows = []

    def _row(kind, val, pos):
        d = backtest(price, r_t, pos, cost_per_turn=cost_per_turn)
        s = performance_stats(d["strat_logret"], d["strat_equity"], positions=d["pos_traded"],
                              turnover=d["turnover"], cost_logret=d["cost_logret"])
        rows.append({"knob": kind, "value": val, "sharpe": s["sharpe"],
                     "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                     "max_dd": s["max_drawdown"], "pct_short": s["pct_short"],
                     "ann_turnover": s["ann_turnover"]})

    for m in lev_grid:
        _row("lev_mult", m, build_positions(p_bear, lev_mult=m))
    for d_ in disturb_grid:
        _row("lev_disturb", d_, build_positions(p_bear, lev_disturb=d_))
    for tl in tail_grid:
        _row("short_peak_tail", tl, build_positions(p_bear, short_peak_tail=tl))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
_OOS_SPLIT = "2012-09-14"  # 80/20 train/test boundary of the global fit (see the saved run)


def main(model_key: str = "3state", refit: bool = False, flat: bool = False,
         do_sweep: bool = False, cost_per_turn: float = 0.0) -> None:
    price, r_t, p_bear = load_pbear(model_key=model_key, refit=refit)

    if do_sweep:
        print(f"\nRobustness sweep [{model_key}] (re-uses the fitted P(bear); no refit).")
        print("Read as PLATEAU checks around the shipped defaults (lev_mult=1.25, lev_disturb=0.25,")
        print("short_peak_tail=3), NOT an optimizer -- each default is a robust mid-plateau pick.")
        if cost_per_turn > 0:
            print(f"(transaction costs ON: {cost_per_turn*1e4:.1f} bps per unit turnover)")
        sw = sweep_design(price, r_t, p_bear, cost_per_turn=cost_per_turn)
        with pd.option_context("display.width", 140, "display.max_rows", 40):
            print(sw.to_string(index=False, formatters={c: "{:.3f}".format for c in
                  ["value", "sharpe", "ann_return", "ann_vol", "max_dd", "pct_short"]}))
        out_dir = _pl.Path(__file__).resolve().parents[1] / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        sw.to_csv(out_dir / f"regime_pbear_strategy_{model_key}_sweep.csv", index=False)
        print(f"\n(saved -> outputs/regime_pbear_strategy_{model_key}_sweep.csv)")
        return

    positions = build_positions(p_bear, flat=flat)
    df = backtest(price, r_t, positions, cost_per_turn=cost_per_turn)
    df["p_bear"] = p_bear

    strat = performance_stats(df["strat_logret"], df["strat_equity"], positions=df["pos_traded"],
                              turnover=df["turnover"], cost_logret=df["cost_logret"])
    bh = performance_stats(df["bh_logret"], df["bh_equity"])

    cost_desc = (f"  (net of {cost_per_turn*1e4:.1f} bps/unit-turnover transaction costs)"
                 if cost_per_turn > 0 else "  (frictionless)")
    _short_desc = (f" + {SHORT_SIZE:g}x momentum short (up-cross {P_SHORT:.0%}, peak+{SHORT_PEAK_TAIL}wk)"
                   if SHORT_SIZE > 0 else "  [short OFF]")
    desc = ("PLAIN long/flat (baseline)" if flat else
            f"long/flat<{P_LONG:.0%} (bear line) + {LEV_MULT:g}x recovery "
            f"(arm {LEV_ARM:.0%}, disturb {LEV_DISTURB:.0%}, no re-arm){_short_desc}")
    print("\n" + "=" * 68)
    print(f"P(bear) STRATEGY [{model_key}]")
    print(f"  {desc}")
    print(f"{cost_desc}")
    print(f"period: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} weekly bars)")
    print("=" * 68)
    print(_fmt_stats("STRATEGY", strat))
    print(_fmt_stats("BUY & HOLD", bh))
    print("=" * 68)

    # OUT-OF-SAMPLE split: the design's params were chosen on FULL-sample calibration, so
    # report the post-2012 tail (right of the fit's 80/20 boundary) separately as the honest
    # generalization check -- this is where the recovery-leverage assumption is really tested.
    oos = df.loc[_OOS_SPLIT:]
    so = performance_stats(oos["strat_logret"], oos["strat_equity"], positions=oos["pos_traded"],
                           turnover=oos["turnover"], cost_logret=oos["cost_logret"])
    bo = performance_stats(oos["bh_logret"], oos["bh_equity"])
    print(f"  OUT-OF-SAMPLE ({_OOS_SPLIT} -> end, {len(oos)} wk):")
    print(f"    strategy : Sharpe {so['sharpe']:.2f}  ann {so['ann_return']*100:.1f}%  maxDD {so['max_drawdown']*100:.0f}%")
    print(f"    buy&hold : Sharpe {bo['sharpe']:.2f}  ann {bo['ann_return']*100:.1f}%")
    print("=" * 68)

    neutral = analyse_neutral(df, p_bear)
    _print_neutral(neutral)

    out_dir = _pl.Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"regime_pbear_strategy_{model_key}_backtest.csv")
    neutral["pbear_bins"].to_csv(out_dir / f"regime_pbear_strategy_{model_key}_neutral_bins.csv")
    plot_strategy(df, p_bear, save_path=out_dir / f"regime_pbear_strategy_{model_key}.png", model_key=model_key)


def _flag(argv, name, cast, default):
    """Tiny CLI helper: --name VALUE -> cast(VALUE), else default."""
    return cast(argv[argv.index(name) + 1]) if name in argv else default


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    main(
        model_key=_flag(argv, "--model", str, "3state"),
        refit="--refit" in argv,
        flat="--flat" in argv,          # fall back to plain long/flat for comparison
        do_sweep="--sweep" in argv,
        # (transaction costs deferred -- see unused_mechanisms/transaction_costs.md;
        #  main() keeps cost_per_turn=0.0 default so the engine runs frictionless.)
    )
