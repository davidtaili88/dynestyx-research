# Bayesian Regime Detection in Equity Markets

*Summer research project (with Professor Bou-Rabee).*

A research project on **Bayesian state-space modelling of market regimes**, which grew into
a trading strategy as its applied direction. Starting from probabilistic filtering, it builds
a **regime nowcast**: a hidden Markov model that estimates, causally and week by week, the
probability the market is in a bear regime, implemented with
[`dynestyx`](https://pypi.org/project/dynestyx/) (a probabilistic state-space / filtering
library on top of NumPyro + JAX).

The core object is a latent **P(bear)** inferred each week from price-only signals. To test
whether the nowcast carries a real, tradeable edge, it is turned into a simple exposure rule
on the S&P 500 — long in bull regimes, flat in bears, with a modest leverage bump while
riding a bear's recovery — and backtested out of sample.

## Does the nowcast carry an edge?

The exposure rule, backtested on 69 years of weekly S&P 500 data (1957–2026), is the test.
The number that matters is **out-of-sample** — the 2012–2026 tail (~14 years) that the model
never saw during training:

| Out-of-sample (2012–2026) | Strategy | Buy & hold SPX |
|---|---:|---:|
| CAGR | **14.5%** | 12.7% |
| Sharpe | **0.87** | 0.75 |
| Max drawdown | **−22.7%** | −31.8% |

The strategy **beats buy-and-hold on return, risk-adjusted return, and drawdown out of
sample** — higher CAGR and Sharpe with a materially shallower max drawdown, by cutting
exposure ahead of sustained bears rather than by leverage alone.

Full-sample figures (1957–2026, in-sample) for reference:

| Full-sample (1957–2026) | Strategy | Buy & hold SPX |
|---|---:|---:|
| CAGR | **9.8%** | 7.7% |
| Sharpe | **0.59** | 0.48 |
| Max drawdown | **−50.5%** | −56.2% |

<sub>Results are frictionless; a transaction-cost model was scoped but not yet validated
(see `unused_mechanisms/`).</sub>

## How it works

1. **Data** (`equity_market/src/dataset/`) — weekly S&P 500 log returns (`r_t`), log
   realized volatility (`v_t`), and an event-reset **drawdown** channel (`dd`, price vs the
   running cycle peak) — all causal, price-only, no look-ahead.

2. **Model** (`equity_market/src/models/regime_model_3state.py`) — a 3-state discrete-time
   HMM (BULL / TURBULENT-BULL / BEAR) with per-state Student-t emissions, fit by NUTS via
   dynestyx's filtering machinery. Its causal forward filter produces the weekly **P(bear)**
   nowcast. Priors are grounded from label-free training statistics and screened for
   sensitivity (`equity_market/src/param_analysis/`).

3. **Ground truth** (`equity_market/src/regime_labeling/`) — a Pagan–Sossounov bull/bear
   dating of the price series, used *only* to score the nowcast, never as a model input.

4. **Strategy** (`equity_market/trading_strategies/regime_pbear_strategy.py`) — maps P(bear)
   to a position via two transition-based rules: **long/flat** at the model's bear line, plus
   **1.25× recovery leverage** while climbing out of a confirmed bear. Sizing is by *regime
   transition*, not by P(bear) level (a deliberate anti-overfit choice — P(bear) is
   near-binary, so there is no gradient to size along).

## Repository layout

```
tutorial_notebooks/   Notebooks worked through to learn dynestyx (NumPyro -> filtering ->
                      NUTS/SVI -> HMMs); background, not part of the strategy pipeline.
equity_market/
  src/dataset/        data acquisition + feature channels
  src/models/         the 3-state HMM regime nowcast (+ shared fit/persistence harness)
  src/regime_labeling/  Pagan-Sossounov ground-truth dating
  src/param_analysis/   prior grounding + inertness screen
  trading_strategies/ the P(bear) -> position strategy + backtest
  parameter_sweeps/   robustness / overfitting checks (e.g. the leverage plateau)
  unused_mechanisms/  explored-but-not-shipped variants (4-state model, macro channels, ...)
outputs/              cached fits, backtests, figures (git-ignored)
```

## Running it

```bash
pip install -r requirements.txt

# fit the regime model (one NUTS fit; caches P(bear) for the strategy)
python equity_market/src/models/regime_model_3state.py

# run the strategy backtest against buy-and-hold
python equity_market/trading_strategies/regime_pbear_strategy.py
```
