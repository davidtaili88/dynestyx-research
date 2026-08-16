# Unused idea: transaction costs for the P(bear) strategy

**Status: built but NOT validated. Removed from the live strategy's surface 2026-08-16
(decluttered; the cost model was never tested/calibrated, so it shouldn't imply the
reported Sharpe/return figures are net-of-cost). The backtest ENGINE still accepts a
dormant `cost_per_turn=0.0` keyword (defaults to frictionless), so this can be re-enabled
without re-plumbing — see "How to re-enable" below.**

## The idea

Charge a friction per unit of position **turnover**. A trade of size `|Δposition|` (e.g.
0 → +1 is 1 unit; +1 → +1.25x recovery is 0.25 unit; +1 → −0.5 short is 1.5 units) costs
`cost_per_turn * |Δposition|` in return terms, subtracted from that week's return.
`cost_per_turn` bundles commission + spread + slippage as a fraction of notional traded.

Proposed value: **5 bps (0.0005)** per unit one-way — a round-ish estimate for a liquid S&P
proxy (ETF / futures); a full round trip (in and out) then costs ~10 bps.

## Why it was deferred

The 5 bps figure is a plausible guess, **not calibrated or validated** against real fills for
the actual instrument/size this would trade. Shipping it half-validated risked two bad
outcomes: (a) reporting "net-of-cost" numbers that aren't trustworthy, or (b) leaving cost
machinery visible in the live file implying a rigor that isn't there. So the live strategy
reports **frictionless** results (clearly labelled as such), and this cost model waits until
someone calibrates `cost_per_turn` to a real cost curve.

## The formula (as it was implemented)

```python
# in backtest(), per week:
turnover    = pos_lag.diff().abs().fillna(pos_lag.abs())   # |change in held position|
cost_frac   = cost_per_turn * turnover
cost_logret = np.log1p(-cost_frac.clip(upper=0.999999))    # <= 0; guard against >=100% cost
strat_logret = gross_logret + cost_logret
# buy&hold pays a single entry cost of cost_per_turn (bought once at the start):
if cost_per_turn > 0 and len(bh_logret):
    bh_logret.iloc[0] += np.log1p(-cost_per_turn)
```

`performance_stats` then reported `ann_turnover` and a net-of-cost Sharpe when `cost_logret`
was supplied.

## How to re-enable (no re-plumbing needed)

The engine functions (`backtest`, `performance_stats`, `sweep_design`) still take a
`cost_per_turn` keyword defaulting to `0.0`. To turn costs back on:

1. Pass a non-zero `cost_per_turn` (in return units, e.g. `0.0005` for 5 bps) into `backtest`
   / `sweep_design` — ideally after calibrating it to a real cost curve.
2. Re-add a CLI flag in `main()` if you want it toggleable (the old one was
   `--cost-bps N`, converting `N * 1e-4` → `cost_per_turn`).

The turnover/cost computation inside `backtest` was never removed — only the constant, the
CLI flag, and the surface comments were. So step 1 alone reactivates the feature.
