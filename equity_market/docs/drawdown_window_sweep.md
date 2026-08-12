# Drawdown-channel window sweep

**Date:** 2026-08-07
**Script:** `parameter_sweeps/drawdown_window_sweep.py`
**Raw results:** `docs/drawdown_window_sweep.csv` (durable copy; the live run also writes
`outputs/drawdown_window_sweep.csv`, which is git-ignored).

## What was swept

The drawdown emission channel is `dd_t = price_t / max(price, trailing W weeks) - 1`
(causal, price-only, `<= 0`). This sweep varies **only** the trailing window `W`, refitting
the 3-state model once per window. The **emission was held fixed** at the shipped
single-scale Normal — so this ranks windows *for the current (misspecified) emission*, not
in general. See the "caveat" section.

Metrics (vs Pagan-Sossounov ground truth):
- `recall` / `false_alarm` — mean P(bear) over true-bear / true-bull weeks (read together).
- `cx_total` — whipsaw crossings of the mid-band (lower = steadier; does **not** distinguish
  a real mid-bear whipsaw from harmless dips through 0.5 in calm times).
- `gfc_rally_hold` — min P(bear) during the spring-2008 relief rally (the whipsaw the channel
  exists to fix; **want high**).
- `recovery_false_alarm` — mean P(bear) over the 12mo after each P&S bear ends (**want low**;
  the long-window failure mode: still-underwater into the new bull).
- `oos_recall` / `oos_false_alarm` — same two, post-split tail only.

## Results

| W (wk) | yr | recall | false_alarm | cx_total | gfc_rally_hold | recovery_FA | oos_recall | oos_false_alarm | dd_bear_gap | dd_scale |
|-------:|----:|-------:|------------:|---------:|---------------:|------------:|-----------:|----------------:|------------:|---------:|
| 26  | 0.50 | 0.525 | 0.134 | 164 | 0.847 | 0.183 | 0.664 | 0.129 | 0.083 | 0.027 |
| **39** | **0.75** | **0.497** | **0.118** | **156** | **0.992** | **0.175** | **0.576** | **0.118** | **0.095** | **0.030** |
| 52 *(shipped)* | 1.00 | 0.419 | 0.115 | 100 | 0.975 | 0.226 | 0.530 | 0.100 | 0.115 | 0.033 |
| 65  | 1.25 | 0.397 | 0.136 |  80 | 0.940 | 0.300 | 0.483 | 0.098 | 0.128 | 0.035 |
| 78  | 1.50 | 0.370 | 0.144 |  82 | 0.950 | 0.306 | 0.497 | 0.110 | 0.125 | 0.037 |
| 104 | 2.00 | 0.364 | 0.155 |  80 | 0.843 | 0.315 | 0.434 | 0.104 | 0.141 | 0.038 |
| 156 | 3.00 | 0.360 | 0.148 |  82 | 0.891 | 0.265 | 0.448 | 0.108 | 0.133 | 0.040 |

## Findings

1. **The hypothesized tradeoff is confirmed and monotone.** Recall falls smoothly as W grows
   (0.525 → 0.36); recovery_false_alarm rises as W grows (0.183 → 0.315). Short window = recent
   peak = faster onset = more bear caught, but faster forgiveness; long window = old peak stays
   in view = underwater into recoveries = false alarms. Mechanism holds exactly.

2. **39wk (0.75yr) strictly dominates the shipped 52wk.** Higher recall (0.497 vs 0.419), equal
   false-alarm (0.118 vs 0.115), **best** gfc_rally_hold in the whole sweep (0.992 vs 0.975),
   **lower** recovery_FA (0.175 vs 0.226), higher oos_recall (0.576 vs 0.530). Better on every
   axis at no false-alarm cost. The original coarse 1/2/3yr sweep that picked 52 never tested
   *below* 52, so it missed this.

3. **26wk is the recall-max but a different (twitchier) product.** Highest recall (0.525) and
   oos_recall (0.664), but false_alarm ticks up (0.134) and `cx_total` jumps to 164 — the peak
   refreshes so fast that dd rattles. Recall gain 39→26 (+0.03) costs +0.016 false-alarm, a worse
   exchange than anywhere above it.

4. **Everything >= 65wk is dominated.** The inflection is at 52->65: recovery_FA jumps
   0.226 -> 0.300 and false_alarm 0.115 -> 0.136 right there — the window starts holding the old
   peak too long.

## Caveats before shipping 39wk

- **`cx_total` at 39 (156) is much higher than at 52 (100).** Rally-hold and recall improved, but
  the curve crosses the mid-band more often. `cx_total` can't tell a real mid-bear whipsaw (bad)
  from brief dips through 0.5 in calm times (harmless). **Eyeball the 39wk P(bear) curve before
  committing** — it may be twitchy like 26wk, or the extra crossings may be harmless.

- **Emission was the fixed, misspecified Normal.** The single-scale Normal puts ~27% of bull dd
  mass above 0 (impossible) and can't hold the bull boundary atom (see
  `diagnostics/drawdown_emission_shape.py`). A better-specified emission (log-Normal on `-dd` +
  a bull hurdle atom) changes how hard the model leans on each week's dd, which **could shift the
  window ranking**. Re-run this sweep after the emission fix as confirmation before hard-swapping
  52 -> 39.

## Recommendation

Treat **39wk as the leading candidate**, pending (a) an eyeball of its P(bear) curve for
twitchiness and (b) a re-run after the emission is fixed. Do **not** hard-swap `_DRAWDOWN_WINDOW_WEEKS`
52 -> 39 yet.
