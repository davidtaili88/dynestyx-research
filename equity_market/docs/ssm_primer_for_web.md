# Help me understand this state-space model (SSM / HMM)

I'm a quantitative researcher. I've built a Bayesian hidden Markov model to
"nowcast" whether the S&P 500 is currently in a bull or bear market, and I want
to gain a **concrete, first-principles understanding of how this kind of
state-space model actually works** — not a code review. Please teach me.

Assume I know: basic probability, Bayes' theorem, what a Gaussian/Student-t is,
and how to read Python. Assume I do NOT have deep prior knowledge of: HMM
filtering recursions, the forward algorithm, MCMC/NUTS internals, or
label-switching. Explain those from the ground up as they come up.

## What the model does (plain-language problem statement)

- **Goal:** at each week `t`, output `P(bear_t)` — the probability the market is
  in a "bear" regime right now, given only data observed up to and including week
  `t` (a real-time *nowcast*, so no peeking at the future).
- **Data:** weekly S&P 500 observations from 1990 on (~1500 weeks in the training
  split). Each week has two observed numbers:
  - `r_t` = the weekly log return (log of this Friday's close / last Friday's close).
  - `v_t` = log of intra-week "realized volatility" = log of the standard
    deviation of the ~5 daily log returns *within* that week.
- **Hidden state:** an unobserved regime label per week. The 3-state version uses
  BEAR / TURBULENT_BULL / BULL. We never observe the regime directly; we infer it
  from `(r_t, v_t)`.
- **Framework:** the model is written in NumPyro (a probabilistic programming
  library) on top of a **proprietary library called `dynestyx`** that provides the
  SSM/HMM machinery: `DynamicalModel` (bundles the initial condition, the state
  transition, and the observation model), `Filter(HMMConfig())` (runs the HMM
  filtering recursion — the forward algorithm — to produce filtered state
  probabilities), and `dsx.sample(...)`. **You will not know dynestyx's internals
  and should not guess them** — explain the SSM *concepts* from how these pieces
  are used and named (`initial_condition`, `state_evolution`, `observation_model`,
  filtered states), not from any assumed source code.

## The four things I most want to understand (please cover all four, in order)

1. **What are the "hidden state" and the "observations" here, precisely**, and
   what makes this a *state-space model* / HMM rather than just a classifier that
   labels each week independently? What does the Markov assumption buy us?

2. **The transition matrix vs. the belief `P(bear_t)`.** These confuse me. The
   `state_evolution` returns a Categorical driven by a 3x3 matrix `A` (rows =
   "given I'm in state i now, prob of each state next week"). Separately, the
   filter produces `P(bear_t | data up to t)`. Walk me through the **filtering
   recursion** — the predict step (push last week's belief through `A`) and the
   update step (correct it with this week's `(r_t, v_t)` likelihood) — and show me
   how `A` and the emission *together* produce the `P(bear_t)` curve. This is the
   heart of what I want to internalize.

3. **Why is this *Bayesian*, and what is NUTS doing?** The parameters (`A` via
   `p_self`/`off_split`, the per-state drift means, the volatilities, the
   Student-t dof) all get priors and are sampled with NUTS/MCMC. Explain the two
   nested layers: (a) for a *fixed* set of parameters, the filter computes
   `P(bear_t)`; (b) NUTS explores the *posterior over parameters*, and the final
   `P(bear_t)` is averaged over that posterior. How do these two layers fit
   together? What is NUTS actually sampling, and what does "the likelihood" even
   mean for an HMM (hint: the forward algorithm also yields the marginal
   likelihood of the observations)?

4. **Identification & label-switching.** The code goes to a lot of trouble to
   *order* the states: the drift means are built as `mu_bear < mu_tbull < mu_bull`
   via positive gaps, and the volatilities are structured so BEAR and
   TURBULENT_BULL share a "high vol" while BULL is "low vol." Explain the
   label-switching problem in mixture/HMM models (why, without these constraints,
   the sampler can permute the state labels and make the posterior meaningless),
   and how these ordering tricks fix it.

## A specific conceptual problem I'm trying to build intuition for

My real motivation: the model currently raises `P(bear)` on almost **any weekly
dip**, even a one-week drop in the *middle of a healthy multi-year bull market*.
I suspect this is fundamental: the observations (`r_t`, `v_t`) are both
**short-horizon** (one week), but a bull/bear *regime* is a **long-horizon,
sustained** property. So it feels like a short-memory model is being asked a
long-memory question. As you explain the mechanics above, please help me see:
**where in the SSM structure does the "memory" / persistence actually live**, and
why can't a strong transition-persistence prior alone make the model ignore
transient dips? (I want to understand this deeply before I change how "bear" is
defined.) I do NOT need you to rewrite the model — I want the conceptual
understanding first.

---

## CODE FILE 1 of 2: `regime_model_3state.py` (the SSM itself)

```python
"""Bayesian regime nowcast (spec sections 2, 5, 6): 3-STATE discrete-time HMM
fit with dynestyx/NumPyro, BIVARIATE emission on (r_t, v_t): weekly log return
and log intra-week realized vol.

This is the 3-state variant. Its 2-state predecessor lives in
regime_model_2state.py; the third state (TURBULENT_BULL) was added to fix the
filtered-P(bear) whipsaw in bull markets that the 2-state model could not solve
even with a strong persistence prior and fat emission tails. See the module
constants and regime_model() docstring for the full rationale.

Follows the notebooks/07_hidden_markov_model.ipynb pattern (Categorical
state_evolution driven by a learned transition matrix, Filter(HMMConfig())
+ NUTS), swapped from a toy categorical emission to a per-state Student-t
emission on r_t, with priors from section 5 instead of a flat Dirichlet.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import dynestyx as dsx
from dynestyx import DynamicalModel, Filter
from dynestyx.inference.filters import HMMConfig

# 3-state layout. BEAR is kept at index 0 so filtered_p_bear / the plot need no
# change. The middle TURBULENT_BULL state is the structural fix for the whipsaw:
# it gives transient 1-2 week bull-market vol spikes (which the diagnostic showed
# are ~40% of all turbulent weeks) a HOME that is NOT bear, so the filter stops
# raising P(bear) every time vol pops for a fortnight.
#   BEAR (0):           drift < 0, HIGH vol   -- the real bear regime
#   TURBULENT_BULL (1): drift ~ 0, HIGH vol   -- transient turbulence, not bear
#   BULL (2):           drift > 0, LOW  vol   -- the calm uptrend
# States are ordered by drift (bear < tbull < bull); vol separates the two
# high-vol states {BEAR, TURBULENT_BULL} from the low-vol BULL.
BEAR = 0
TURBULENT_BULL = 1
BULL = 2
K = 3

# Empirical drift split from the sklearn sanity-check baseline (baseline_hmm.py),
# an independent, non-Bayesian fit -- NOT the section 4 drawdown label -- used
# only to center these priors at a realistic gap. calm/turbulent state means
# there were +0.00306 (bull-like) and -0.00288 (bear-like) weekly log return.
_EMPIRICAL_BULL_DRIFT = 0.003
_EMPIRICAL_BEAR_DRIFT = -0.003
_EMPIRICAL_DRIFT_GAP = _EMPIRICAL_BULL_DRIFT - _EMPIRICAL_BEAR_DRIFT

# Empirical v_t (LOG intra-week realized vol) split, from prior_analysis.py's
# 70/30 percentile split of the TRAINING weeks on v_t itself (calm-proxy vs.
# turbulent-proxy -- again NOT the drawdown label). These anchor the priors for
# the SECOND emission dimension (v_t), the analogue of the drift constants above
# for the return dimension:
#   calm-proxy      weeks: v_t mean ~ -5.34, within-group std ~ 0.45
#   turbulent-proxy weeks: v_t mean ~ -4.30, within-group std ~ 0.35
# so the calm->turbulent gap in log-vol is ~ 1.04, and it co-moves with the
# drift sign (calm weeks are the +drift ones), which is exactly why v_t can do
# the state SEPARATION while drift keeps the LABELS -- see regime_model docstring.
# NOTE: v_t is already log(realized_vol) (data.py), so these are on the log
# scale and a Normal emission on v_t == LogNormal on raw realized vol.
_EMPIRICAL_CALM_LOG_VOL = -5.34  # bull-leaning (low intra-week realized vol)
_EMPIRICAL_TURBULENT_LOG_VOL = -4.30  # bear-leaning (high intra-week realized vol)
_EMPIRICAL_LOG_VOL_GAP = _EMPIRICAL_TURBULENT_LOG_VOL - _EMPIRICAL_CALM_LOG_VOL
_EMPIRICAL_LOG_VOL_WITHIN_STD = 0.40  # typical within-group spread of v_t


class _JointRV(dist.Distribution):
    """Conditional-independence joint of the two emission dimensions [r_t, v_t].

    Deliberately NOT a numpyro MultivariateNormal or a `.to_event(1)` wrap of one
    family, because the two dimensions use DIFFERENT families (StudentT for the
    return, Normal for log realized vol). Instead this holds one distribution per
    dimension and defines log_prob(y) to return the PER-DIMENSION log-probs as a
    length-2 vector [log p(r|x), log p(v|x)].

    The dynestyx HMM filter then does `jnp.sum(dist.log_prob(y))`
    (inference/hmm_filters.py, the line commented "critical for vector-valued
    observations"), so summing the two entries reconstructs exactly the
    conditional-independence factorization
        log p(r, v | x) = log p(r | x) + log p(v | x).
    We do NOT sum inside log_prob -- returning the vector and letting the filter
    sum keeps the per-dimension structure the framework expects.

    event_shape is (2,): one "event" is the pair (r_t, v_t) at a single week.
    """

    # Both emitted quantities are continuous (real-valued: r_t is a log return,
    # v_t is a log realized vol), so the support is the real line per component.
    support = dist.constraints.real_vector

    def __init__(self, r_dist, v_dist):
        self.r_dist = r_dist
        self.v_dist = v_dist
        # batch_shape empty (scalar per component), event_shape (2,) = the pair.
        super().__init__(batch_shape=(), event_shape=(2,))

    def log_prob(self, value):
        # value is [..., 2] = [r_t, v_t]; return [..., 2] per-dim log-probs so the
        # filter's jnp.sum turns it into the joint (see class docstring). Do NOT
        # sum here.
        r = value[..., 0]
        v = value[..., 1]
        return jnp.stack(
            [self.r_dist.log_prob(r), self.v_dist.log_prob(v)], axis=-1
        )

    def sample(self, key, sample_shape=()):
        # Only needed for prior/predictive sampling, not for filtering/NUTS on the
        # conditioned model, but implemented for completeness. Split the key so the
        # two dimensions draw independently (matching the conditional-independence
        # assumption), then stack into the [..., 2] event layout.
        kr, kv = jr.split(key)
        r = self.r_dist.sample(kr, sample_shape)
        v = self.v_dist.sample(kv, sample_shape)
        return jnp.stack([r, v], axis=-1)


def regime_model(obs_times=None, obs_values=None, predict_times=None):
    """3-state HMM: Categorical regime, bivariate (r_t, v_t) emission.

    States (see module constants): BEAR / TURBULENT_BULL / BULL. The middle state
    exists to absorb transient bull-market turbulence (the ~40% of turbulent weeks
    that come in 1-2 week bursts) so it is not misread as bear -- the structural
    fix for the filtered-P(bear) whipsaw that a stronger persistence prior and
    fatter emission tails could not solve on their own.

    State evolution: first-order Markov, 3x3 transition matrix. Strong persistence
    prior per state (Beta(500,3)); the off-diagonal leaving-mass is split between
    the two destinations by a learned Dirichlet.

    Identification (two ordered axes, both label-switching guards):
      * DRIFT ladder, bear < tbull < bull (positive gaps) -- separates the two
        high-vol states (BEAR vs TURBULENT_BULL) from each other and pins labels.
      * VOL level, {BEAR, TURBULENT_BULL} high vs BULL low -- separates turbulent
        weeks from calm ones. Both r_t's inferred scale and the observed v_t carry
        this. So: vol says turbulent-vs-calm, drift says bear-vs-turbulent-bull.

    P(bear) reads off the BEAR state alone; TURBULENT_BULL is explicitly not bear.
    """
    # Persistence prior: Beta(500, 3) per state, mean p_self ~ 0.994. DELIBERATELY
    # strong (see the 2-state history: even this was dragged to ~0.986 by the noisy
    # weekly likelihood, which is why the 3rd state -- not just a stronger prior --
    # is needed). Now one self-transition per state (K=3). Each row's remaining
    # (1 - p_self) mass is split over the OTHER two states by a learned Dirichlet,
    # rather than assumed equal: e.g. from BULL you are far more likely to slip into
    # TURBULENT_BULL than jump straight to BEAR, and we let the data express that.
    p_self = numpyro.sample("p_self", dist.Beta(500.0, 3.0).expand([K]).to_event(1))
    # Off-diagonal split: for each state, how the (1 - p_self) leaving-mass divides
    # between the two destinations. Dirichlet(1,1) = uniform prior over that split.
    off_split = numpyro.sample(
        "off_split", dist.Dirichlet(jnp.ones(2)).expand([K]).to_event(1)
    )

    def _row(i):
        # Build transition row i: p_self on the diagonal, (1-p_self)*off_split on
        # the two off-diagonal entries (in ascending state-index order).
        others = [j for j in range(K) if j != i]
        row = [None] * K
        row[i] = p_self[i]
        row[others[0]] = (1.0 - p_self[i]) * off_split[i, 0]
        row[others[1]] = (1.0 - p_self[i]) * off_split[i, 1]
        return jnp.stack(row)

    A = jnp.stack([_row(i) for i in range(K)])

    # Drift identification (label-switching guard), now a 3-rung ladder ordered
    # bear < tbull < bull, each rung a POSITIVE gap above the previous so the
    # ordering is enforced by construction (NUTS never sees the mislabeled modes):
    #   mu_bear   = base (near empirical bear drift, negative)
    #   mu_tbull  = mu_bear + gap1   (transient-turbulence drift, ~0)
    #   mu_bull   = mu_tbull + gap2  (calm uptrend drift, positive)
    # HalfNormal gap scales are a few x the empirical bull-bear gap -- wide enough
    # for NUTS to move, not so tight they funnel. Splitting the full bear->bull
    # gap into two rungs lets the middle state sit at ~0 drift on its own.
    mean_return_bear = numpyro.sample("mean_return_bear", dist.Normal(_EMPIRICAL_BEAR_DRIFT, 0.01))
    drift_gap1 = numpyro.sample("drift_gap1", dist.HalfNormal(2.0 * _EMPIRICAL_DRIFT_GAP))
    drift_gap2 = numpyro.sample("drift_gap2", dist.HalfNormal(2.0 * _EMPIRICAL_DRIFT_GAP))
    mean_return_tbull = mean_return_bear + drift_gap1
    mean_return_bull = mean_return_tbull + drift_gap2
    # Indexed [BEAR, TURBULENT_BULL, BULL] = [0, 1, 2].
    mean_return = jnp.stack([mean_return_bear, mean_return_tbull, mean_return_bull])

    # Per-state WEEKLY-RETURN volatility. DECISION / definition, to avoid
    # conflation with the v_t emission channel:
    #   * This is the `scale` PARAMETER of the r_t (weekly log return) emission
    #     -- an INFERRED std-like quantity NUTS learns, one per state, NOT a
    #     statistic computed from the data over any window. There is no lookback
    #     length here; it is pinned down jointly by every week the filter
    #     attributes to that state.
    #   * It is a STANDARD DEVIATION (scale), not a variance: it must share
    #     mean_return's units to be the `scale=` of the StudentT below. (For a
    #     StudentT the true variance is scale**2 * df/(df-2); `scale` is the
    #     std only in the df->inf limit, but it is std-DIMENSIONED regardless.)
    #   * It is the spread of the WEEKLY return distribution -- distinct from
    #     v_t, which is the observed INTRA-week std of ~5 daily returns. Both
    #     are "volatility" but on different observables (r_t vs. daily-within-t)
    #     and different roles (inferred parameter vs. observed data), so they
    #     are kept as separate quantities, not merged.
    #
    # 3 states: BEAR and TURBULENT_BULL are both HIGH return-vol, BULL is LOW.
    # Parameterize as a low (calm/BULL) base plus a positive offset shared by the
    # two high-vol states, so we assert "the two turbulent states are more volatile
    # than the calm one" without creating an unordered per-state vol free-for-all
    # (which is what previously slowed sampling). BEAR and TURBULENT_BULL are NOT
    # separated by return-vol here -- they are separated by DRIFT (the ladder
    # above). That is the whole point: vol tells turbulent-from-calm, drift tells
    # bear-from-turbulent-bull.
    return_vol_low = numpyro.sample("return_vol_low", dist.HalfNormal(0.02))  # BULL
    return_vol_high_offset = numpyro.sample(
        "return_vol_high_offset", dist.HalfNormal(0.02)
    )
    return_vol_high = return_vol_low + return_vol_high_offset  # BEAR & TURBULENT_BULL
    # Indexed [BEAR, TURBULENT_BULL, BULL].
    return_vols = jnp.stack([return_vol_high, return_vol_high, return_vol_low])

    # Student-t tail-thickness (degrees of freedom), shared across all states --
    # a single global "how outlier-prone are weekly returns in general" knob, not
    # a per-regime property. Shift inside the model (not a post-hoc clip) so the
    # tail_dof > 2 floor is a smooth reparameterization NUTS can differentiate
    # through, not a flat-gradient clip.
    tail_dof = 2.0 + numpyro.sample("tail_dof_raw", dist.Gamma(2.0, 0.1))

    # SECOND EMISSION DIMENSION: v_t = observed LOG intra-week realized vol.
    # DECISION / definition, and how it differs from return_vols above:
    #   * v_t is OBSERVED DATA (computed once per week: log-std of that week's
    #     ~5 daily returns, data.py's weekly_log_realized_vol). return_vols was
    #     an INFERRED scale parameter of the r_t distribution. Different objects,
    #     different observables -- so this is a genuine extra channel, not a
    #     reparameterization of the return volatility.
    #   * Because v_t is ALREADY on the log scale, a Normal emission on v_t is a
    #     LogNormal on raw realized vol -- the right choice: log-realized-vol is
    #     roughly Gaussian/symmetric (prior_analysis.py checks this), whereas raw
    #     vol is right-skewed and bounded at 0. So we emit v_t ~ Normal per state.
    #
    # ROLE OF THIS CHANNEL: it is the SEPARATION axis. v_t is strongly bimodal
    # (calm vs. turbulent, ~1.04 log-units apart, see constants above), a far
    # stronger signal than the tiny drift gap. On weeks where r_t is ambiguous,
    # v_t decisively moves the filter's P(bear_t) update. Drift still does
    # IDENTIFICATION (the drift ladder pins the labels); v_t does the heavy
    # lifting of telling turbulent from calm weeks apart.
    #
    # ORDERING (label-switching guard): the turbulent mean = calm mean + a POSITIVE
    # gap, and BEAR + TURBULENT_BULL SHARE that turbulent mean while BULL gets the
    # calm mean. So the v_t channel groups {BEAR, TURBULENT_BULL} together (both
    # high-vol) and the drift ladder then splits them -- the two ordered channels
    # reinforce the same labeling rather than competing for it.
    log_vol_calm = numpyro.sample(
        "log_vol_calm", dist.Normal(_EMPIRICAL_CALM_LOG_VOL, 0.3)
    )
    log_vol_gap = numpyro.sample(
        "log_vol_gap", dist.HalfNormal(_EMPIRICAL_LOG_VOL_GAP)
    )
    log_vol_turbulent = log_vol_calm + log_vol_gap
    # Indexed [BEAR, TURBULENT_BULL, BULL]: BEAR and TURBULENT_BULL SHARE the
    # turbulent v_t mean (both are high-realized-vol weeks -- that is exactly what
    # makes transient spikes look "turbulent"); BULL gets the calm mean. So the v_t
    # channel groups {BEAR, TURBULENT_BULL} together and drift then splits them.
    v_loc = jnp.stack([log_vol_turbulent, log_vol_turbulent, log_vol_calm])

    # Per-state spread of v_t around its state mean (the within-group std from
    # the empirical split, ~0.40). Shared across states as a single scale: the
    # separation already comes from the ordered means above, so we do NOT need a
    # per-state, ordered spread parameter here -- that would just add another
    # weakly-identified quantity. One shared HalfNormal centered near the
    # empirical within-group std is enough.
    v_scale = numpyro.sample(
        "v_scale", dist.HalfNormal(_EMPIRICAL_LOG_VOL_WITHIN_STD)
    )
    # NOTE: v_t emission is Normal (not StudentT). A StudentT-on-v_t experiment was
    # tried (in the 2-state model) to absorb bull-market vol spikes as tail events;
    # NUTS drove its df to ~12 (effectively Gaussian), i.e. it judged the spikes to
    # be REAL recurring signal, not statistical outliers. That finding is exactly
    # what motivated this 3-state model instead: ~40% of turbulent weeks are
    # transient 1-2 week bursts, which fat tails cannot re-home but a dedicated
    # TURBULENT_BULL state can. So v_t stays Normal here.

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    # What we should observe under each regime. BIVARIATE: obs y is a length-2
    # vector [r_t, v_t]. We return a per-dimension emission whose log_prob([r, v])
    # yields the two-vector [log p(r|x), log p(v|x)]; the HMM filter SUMS these
    # (hmm_filters.py: `jnp.sum(lp)`), which is exactly the conditional-independence
    # factorization  log p(r,v | x) = log p(r|x) + log p(v|x).
    #
    # CONDITIONAL-INDEPENDENCE ASSUMPTION (stated explicitly): given the state x,
    # r_t and v_t are treated as independent. They are correlated in reality
    # (leverage effect: down weeks are turbulent weeks), and conditioning on x
    # absorbs much of that co-movement but not all. This is the standard, usually-
    # fine simplification; a genuinely joint bivariate emission would be the next
    # step only if this proves inadequate.
    #
    # r_t emits StudentT (fat tails absorb return crashes as outliers); v_t emits
    # Normal (see note above -- its StudentT df collapsed to ~Gaussian, so no
    # benefit). Different families per dimension, so we build the joint via
    # per-dimension log_prob rather than a single vector dist.
    def observation_model(x, u, t):
        return _JointRV(
            r_dist=dist.StudentT(df=tail_dof, loc=mean_return[x], scale=return_vols[x]),
            v_dist=dist.Normal(loc=v_loc[x], scale=v_scale),
        )

    dynamics = DynamicalModel(
        initial_condition=dist.Categorical(probs=jnp.ones(K) / K),
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def fit(
    train_obs,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    seed: int = 0,
    target_accept_prob: float = 0.95,
):
    """Fit the 3-state regime model on the training weekly frame via NUTS.

    target_accept_prob is raised from NUTS's 0.8 default to 0.95: it forces a
    SMALLER leapfrog step size, so the integrator can thread the tight funnel the
    drift ladder creates (bear/tbull/bull rungs fighting over a weak drift signal)
    without overshooting and diverging.

    `train_obs` is the BIVARIATE frame: a pandas DataFrame indexed by week with
    columns ["r_t", "v_t"]. Index position is used as obs_times (evenly spaced
    weekly cadence). obs_values is stacked to shape (T, 2) = [r_t, v_t] per week.
    """
    obs_times = jnp.arange(len(train_obs), dtype=jnp.float32)
    obs_values = jnp.asarray(
        train_obs[["r_t", "v_t"]].to_numpy(), dtype=jnp.float32
    )  # (T, 2)

    def conditioned_model():
        with Filter(filter_config=HMMConfig(record_filtered=True)):
            return regime_model(obs_times=obs_times, obs_values=obs_values)

    mcmc = MCMC(
        NUTS(conditioned_model, target_accept_prob=target_accept_prob),
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    mcmc.run(jr.PRNGKey(seed))
    return mcmc


def filtered_p_bear(mcmc):
    """Posterior-averaged filtered P(BEAR_t | y_1:t), one value per week.

    `f_filtered_states` has shape (num_samples, T, K); we take the BEAR column and
    average over the posterior draws to get a single filtered P(bear) curve. It is
    the BEAR state's prob ONLY -- TURBULENT_BULL is NOT counted as bear.
    """
    filtered_states = mcmc.get_samples()["f_filtered_states"]  # (num_samples, T, K)
    return filtered_states[:, :, BEAR].mean(axis=0)
```

## CODE FILE 2 of 2: `data.py` (what r_t and v_t are — the observations)

```python
"""Data pipeline for the S&P 500 regime nowcast.

Fetches S&P 500 (and VIX) daily closes, resamples to a weekly observation
series (r_t = weekly log return, v_t = log realized vol from daily returns
within the week), and produces a train/test split.

Nothing in this module looks past the data available at each row's own
week -- the weekly resampling only aggregates *within* that week.
"""

import numpy as np
import pandas as pd

WEEKLY_ANCHOR = "W-FRI"  # weekly bars anchored on Friday closes


def weekly_log_returns(daily_close, anchor=WEEKLY_ANCHOR):
    """r_t: log return of the last daily close in each calendar week vs. prior week."""
    weekly_close = daily_close.resample(anchor).last().dropna()
    log_price = np.log(weekly_close)
    r = log_price.diff().dropna()
    r.name = "r_t"
    return r


def weekly_log_realized_vol(daily_close, anchor=WEEKLY_ANCHOR):
    """v_t: log of realized vol, from daily log returns *within* each week only.

    Realized vol for week t uses only the daily closes belonging to week t, so this
    stays a same-week (not lookahead) aggregate, matching the weekly filtering
    cadence the model runs at.
    """
    daily_log_ret = np.log(daily_close).diff().dropna()
    weekly_groups = daily_log_ret.resample(anchor)
    # A week needs >=2 daily returns for realized vol to be defined; weeks with
    # exactly 1 give a degenerate std of 0 -> log(0) = -inf, so they're dropped.
    counts = weekly_groups.count()
    realized_vol = weekly_groups.std(ddof=0)  # std of the ~5 daily returns in-week
    realized_vol = realized_vol[counts >= 2].dropna()
    log_rv = np.log(realized_vol)
    log_rv.name = "v_t"
    return log_rv


# The model consumes a per-week bivariate observation frame with columns
# ["r_t", "v_t"], plus the weekly price series (last daily close per week) used
# only for plotting and for the separate "ground truth" drawdown label. r_t and
# v_t are both SHORT-HORIZON: each summarizes exactly one week, sharing no data
# with the next week's values.
```

---

Please start with concept #1 and work through all four, then address my specific
"where does the memory live / why can't persistence alone ignore transient dips"
question at the end. Use concrete numbers from the model where it helps (e.g. the
transition matrix has ~0.99 on the diagonal). Feel free to use small worked
numerical examples of the predict/update filtering step — that's exactly the kind
of thing that would help me internalize it.
```
