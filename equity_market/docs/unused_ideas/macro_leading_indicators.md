# Unused idea: macro leading-indicator emission channels (credit spread + yield-curve inversion)

**Status: tried, not shipped. Removed from the live codebase 2026-08-16 for a clean
presentation. This file preserves the idea and its rationale; the fragmented code was
deleted rather than moved (see "Where it lived" below to reconstruct if ever revisited).**

## The idea

On top of the price-only emission channels (`r_t` weekly log return, `v_t` log realized
vol, `dd` drawdown), add two *macro* channels sourced from monthly FRED series, on the
theory that credit and rates *lead* equity regimes:

1. **`cs_chg` — credit-spread WIDENING momentum.**
   `cs_chg_t = (BAA − AAA) − (BAA − AAA) from `horizon_months` ago`, i.e. the multi-month
   *change* in the Moody's Baa−Aaa investment-grade quality spread.
   - Why the CHANGE, not the level: the raw BAA−AAA *level* is blind at a shallow bear
     onset (sits in the normal bull band; onset separation ≈ −0.09sd). Its multi-month
     *change* carries the signal — spreads widen as a default-driven bear develops
     (true-bear separation ≈ +0.5sd), and it is nearly uncorrelated with the
     price/vol/drawdown channels, so it adds new information rather than re-encoding price.
   - Horizon = 5 months: peak of a broad, smooth 1–24mo plateau (h=3..8 all ≈0.46–0.55sd;
     no lone spike), so robust, not overfit to one window.

2. **`inv` — yield-curve inversion DEPTH.**
   `inv_t = max(0, TB3MS − GS10)` — how far the 3mo T-bill sits above the 10yr Treasury,
   0 when the curve is normal.
   - Why one-sided (clamped at 0), not the raw slope or its change: the raw slope level and
     its change both *flip sign across eras* (yc-level separation was −0.58 in 1957–69 but
     +1.13 in 1983–99), so neither can be a single global emission mean. Clamping to
     inversion *depth* keeps only the half of the range that consistently means "stress":
     non-negative in every era it fires (1957–69 +0.35, 1970–82 +0.74), silent (0) when the
     curve is normal.

## Why it was interesting: era complementarity

The two channels are complementary *by era*, with corr(`cs_chg`, `inv`) ≈ 0.05
(independent information):

- **Credit** is strong in DEFAULT-driven bears (dotcom +0.95, GFC, post-2010 +1.05) but
  DEAD in the rate-driven 1970–82 stagflation bears (≈ −0.09) — there credit decoupled
  from equities.
- **Inversion** is strong exactly where credit is dead — the rate-driven 1970–82 bears
  (+0.74) — and silent in the default-driven dotcom/GFC bears that credit covers.
- Accepted limitation of `inv`: it fires at/before a rate-driven top then fades once the
  Fed pivots to cutting and the curve re-steepens mid-bear — an early *pulse*, not a
  sustained hold. Credit carries the bear BODY; inversion carries the rate-driven ONSET.

## Why it wasn't shipped

The shipped nowcast is price-only (`r_t`, `v_t`, `dd` with the event-reset peak). The
macro channels stayed behind `INCLUDE_CREDIT` / `INCLUDE_CURVE` toggles that were left
`False` in the 3-state model, and the shipped 4-state model never wired them in at all
(`needs_macro = False`). They were an exploration that didn't earn a place in the final
model — the price-level drawdown channel (event-reset) carried the regime signal the
project needed without pulling in monthly macro data, an offline CSV dependency, and two
more emission channels to specify and defend.

## Data it used (also removed)

Static monthly FRED CSVs committed under `src/dataset/` (deleted with this cleanup):
`BAA.csv`, `AAA.csv` (Moody's corporate yields, 1919+), `GS10.csv` (10yr Treasury, 1953+),
`TB3MS.csv` (3mo T-bill, 1934+). Plus `BAMLH0A0HYM2.csv` (HY OAS) — considered but
truncated to 3yr by FRED in Apr-2026, so never usable full-history.

## Where it lived (to reconstruct)

- `src/dataset/data.py`: `_MACRO_SERIES`, `_load_macro_series`, `RegimeDataset.macro`,
  `credit_spread_change()`, `curve_inversion()`, and the `include_credit` /
  `credit_horizon_months` / `include_curve` params on `observations()` / `split()`.
- `src/models/regime_model_3state.py`: `INCLUDE_CREDIT` / `INCLUDE_CURVE` toggles and the
  two extra emission blocks they gated.
- Recover the exact code from git history at commit `6301504` (or earlier).
