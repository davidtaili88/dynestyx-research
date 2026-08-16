"""Bayesian regime nowcast (spec sections 2, 5, 6): 2-STATE discrete-time HMM
fit with dynestyx/NumPyro, BIVARIATE emission on (r_t, v_t): weekly log return
and log intra-week realized vol.

This is the 2-state baseline. It works but exhibits a filtered-P(bear) WHIPSAW
in bull markets: ~40% of turbulent weeks come in transient 1-2 week bursts, and
a 2-state model has nowhere to put "turbulent" except the bear state, so P(bear)
flips on every passing vol spike. Neither a strong persistence prior (Beta(500,3)
here, still dragged to ~0.986 by the likelihood) nor fat emission tails
(StudentT-on-v_t collapsed to ~Gaussian) fixed it. The structural fix -- a third
TURBULENT_BULL state -- lives in regime_model_3state.py. This file is kept as the
comparison baseline.

Follows the notebooks/07_hidden_markov_model.ipynb pattern (Categorical
state_evolution driven by a learned transition matrix, Filter(HMMConfig())
+ NUTS), swapped from a toy categorical emission to a per-state Student-t
emission on r_t, with priors from section 5 instead of a flat Dirichlet.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401  (puts sibling src/ subfolders on sys.path)

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import DynamicalModel, Filter
from dynestyx.inference.filters import HMMConfig

# 2-state layout. BEAR is the turbulent/-drift state, BULL is the calm/+drift
# state. States are ORDERED by drift (bear < bull); volatility co-moves with the
# drift sign but is not the identification axis (see regime_model docstring).
BEAR = 0
BULL = 1
K = 2

# PRIOR CENTERS for the drift channel -- these are NOT hard constants the model uses;
# they are the LOCATIONS at which the learnable drift priors are centered (the model
# still learns the actual values around them, see regime_model). The seeds come from a
# since-removed sklearn 2-state Gaussian-HMM sanity-check baseline, an independent,
# non-Bayesian fit -- NOT the section 4 drawdown label -- used only to put the priors
# in a realistic neighborhood. That fit's calm/turbulent state means were +0.00306
# (bull-like) and -0.00288 (bear-like) weekly log return; rounded and frozen below.
# (Naming: _PRIOR_ not _EMPIRICAL_ precisely because these are prior centers, not
# measured constants the model is forced to use.)
_PRIOR_BULL_DRIFT_MEAN = 0.003
_PRIOR_BEAR_DRIFT_MEAN = -0.003
_PRIOR_DRIFT_GAP = _PRIOR_BULL_DRIFT_MEAN - _PRIOR_BEAR_DRIFT_MEAN

# PRIOR CENTERS for the v_t (LOG intra-week realized vol) channel -- again prior
# LOCATIONS, not hard constants. Seeds come from prior_analysis.py's 70/30 percentile
# split of the TRAINING weeks on v_t itself (calm-proxy vs. turbulent-proxy -- again
# NOT the drawdown label). These anchor the priors for the SECOND emission dimension
# (v_t), the analogue of the drift prior centers above for the return dimension:
#   calm-proxy      weeks: v_t mean ~ -5.34, within-group std ~ 0.45
#   turbulent-proxy weeks: v_t mean ~ -4.30, within-group std ~ 0.35
# so the calm->turbulent gap in log-vol is ~ 1.04, and it co-moves with the
# drift sign (calm weeks are the +drift ones), which is exactly why v_t can do
# the state SEPARATION while drift keeps the LABELS -- see regime_model docstring.
# NOTE: v_t is already log(realized_vol) (data_acquisition.py), so these are on the log
# scale and a Normal emission on v_t == LogNormal on raw realized vol.
_PRIOR_CALM_LOG_VOL_MEAN = -5.34  # bull-leaning (low intra-week realized vol)
_PRIOR_TURBULENT_LOG_VOL_MEAN = -4.30  # bear-leaning (high intra-week realized vol)
_PRIOR_LOG_VOL_GAP = _PRIOR_TURBULENT_LOG_VOL_MEAN - _PRIOR_CALM_LOG_VOL_MEAN
_PRIOR_LOG_VOL_WITHIN_STD = 0.40  # typical within-group spread of v_t


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
    (Note: the "2" here is the two OBSERVATION DIMENSIONS, unrelated to K=2 states.)
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
    """2-state HMM: Categorical regime, bivariate (r_t, v_t) emission.

    States: BEAR / BULL. State evolution: first-order Markov, 2x2 transition
    matrix. Strong persistence prior (Beta(500,3)); off-diagonal is the remainder.

    Identification axis: DRIFT, not volatility (deviates from section 2's "order
    by variance" default). Bull/bear is definitionally a drift distinction, so
    mean_return is the ordered, label-switching-preventing parameter -- mu_bull is
    forced > mu_bear by construction (sampling a positive gap rather than mu_bull
    freely), with the gap prior centered near the empirical split above. The v_t
    channel does the heavy SEPARATION work (turbulent vs calm), while drift keeps
    the LABELS.

    KNOWN LIMITATION: this 2-state model whipsaws in bull markets -- transient
    turbulent weeks have nowhere to go but the bear state. See regime_model_3state.py
    for the fix (a dedicated TURBULENT_BULL state).

    ============================================================================
    PRIOR vs STRUCTURAL -- read this to tell what NUTS can learn from what is baked
    in. (Reference: notebooks/07_hidden_markov_model.ipynb. There the loaded-die
    emission `probs=[[1/6..],[..,1/2]]` is a hard STRUCTURAL constant the model can
    never learn; the transition matrix `A = numpyro.sample("A", Dirichlet(...))` is
    a PRIOR the data updates. Same two categories here.)

      * PRIOR  = a `numpyro.sample("name", dist.Xxx(...))` on a LEARNABLE parameter.
                 NUTS updates it against the data; with ~3600 weeks the likelihood
                 can and does OVERRULE the prior's location/scale (this is why prior
                 tweaks alone -- e.g. cranking p_self -- barely move the posterior).
                 The dist.Xxx just seeds the search; it does not pin the value.
      * STRUCTURAL = a hard constant (a literal number / jnp.array) OR an arithmetic
                 CONSTRUCTION (e.g. mu_bull = mu_bear + gap). Structure changes what
                 the model CAN express; the likelihood works THROUGH it and cannot
                 undo it. Adding/removing structure is what actually changes behavior
                 (e.g. the 3rd state, or the 3-state vol-gap). A constraint built by
                 construction (mu_bull >= mu_bear) is STRUCTURAL even though the
                 magnitudes it combines are priors.
    Each definition below is tagged [PRIOR], [STRUCTURAL], or [PRIOR+STRUCTURAL].
    ============================================================================
    """
    # Persistence prior: Beta(500, 3), mean p_self ~ 0.994 -> expected regime run
    # ~168 weeks (~3.2 yr). DELIBERATELY strong. A weaker Beta(100,1) (mean 0.990)
    # let the POSTERIOR fall to ~0.98 (~1yr) because the noisy weekly emission kept
    # mis-reading bull-market dips as bear weeks and voting persistence DOWN --
    # visible as the P(bear) whipsaw. Even Beta(500,3) only holds the posterior at
    # ~0.986: the likelihood genuinely wants to switch, because ~40% of turbulent
    # weeks are transient and this 2-state model has no non-bear home for them.
    # (That unfixable-by-prior behaviour is what motivated the 3-state model.)
    # [PRIOR] p_self: learnable self-persistence, one per state. Beta(500,3) only
    # SEEDS it; the likelihood updates it (and, as noted, drags it to ~0.986).
    p_self = numpyro.sample("p_self", dist.Beta(500.0, 3.0).expand([K]).to_event(1))
    # [STRUCTURAL] A: the transition matrix is BUILT deterministically from p_self
    # (each row = [stay, leave]). No new randomness -- the 2x2 first-order Markov
    # SHAPE is baked in; only the p_self magnitudes inside it are learned.
    A = jnp.stack(
        [
            jnp.stack([p_self[0], 1.0 - p_self[0]]),
            jnp.stack([1.0 - p_self[1], p_self[1]]),
        ]
    )

    # Drift identification: mu_bull = mu_bear + positive gap. HalfNormal scale
    # is set several times the empirical gap (not equal to it) -- a scale
    # this close to a single week's return noise (std ~2-3%) starves NUTS of
    # room and creates a narrow, hard-to-sample funnel around the gap axis.
    # [PRIOR] mean_return_bear and drift_gap: both learnable (their MAGNITUDES are
    # inferred). [STRUCTURAL] the ORDERING mu_bull = mu_bear + (HalfNormal >= 0):
    # building bull as bear PLUS a non-negative gap makes mu_bull >= mu_bear hold BY
    # CONSTRUCTION -- the label-switching guard is structural, the values are priors.
    # -> [PRIOR+STRUCTURAL].
    mean_return_bear = numpyro.sample("mean_return_bear", dist.Normal(_PRIOR_BEAR_DRIFT_MEAN, 0.01))
    drift_gap = numpyro.sample("drift_gap", dist.HalfNormal(4.0 * _PRIOR_DRIFT_GAP))
    mean_return_bull = mean_return_bear + drift_gap
    # Indexed [BEAR, BULL] = [0, 1].
    mean_return = jnp.stack([mean_return_bear, mean_return_bull])

    # WEEKLY-RETURN volatility -- the `scale` PARAMETER of the r_t (weekly log
    # return) emission. DECISION / definition:
    #   * It is an INFERRED std-like quantity NUTS learns, NOT a statistic computed
    #     from the data over any window. It is pinned down jointly by every week.
    #   * It is a STANDARD DEVIATION (scale), not a variance: it must share
    #     mean_return's units to be the `scale=` of the StudentT below. (For a
    #     StudentT the true variance is scale**2 * df/(df-2); `scale` is the
    #     std only in the df->inf limit, but it is std-DIMENSIONED regardless.)
    #   * Distinct from v_t (the observed INTRA-week std of ~5 daily returns):
    #     different observable, different role (inferred param vs. observed data).
    #
    # SIMPLIFICATION: a single SHARED scale across both states, not per-state.
    # The old per-state version (return_vol_base + return_vol_offset) spent two
    # parameters asserting "bear weeks have different weekly-return spread than
    # bull weeks" -- a weak, poorly-identified claim that also created a second
    # label-switching axis on top of drift. It is largely redundant with the v_t
    # channel, which already tells turbulent-from-calm far more strongly. So the
    # r_t scale is now one shared number: r_t contributes drift-labeling signal
    # only, and stops spending parameters on per-state spread.
    # [PRIOR] return_vol: learnable r_t emission scale. [STRUCTURAL] the choice to
    # SHARE one scale across both states (return_vols = [x, x]) -- that both states
    # have the SAME return spread is baked in, not learned.
    return_vol = numpyro.sample("return_vol", dist.HalfNormal(0.02))
    # Same shared scale for both states, indexed [BEAR, BULL] for the emission.
    return_vols = jnp.stack([return_vol, return_vol])

    # Student-t tail-thickness (degrees of freedom) for the r_t emission, FIXED at
    # a constant rather than sampled. WHY Student-t at all (not Gaussian): weekly
    # equity returns have fat tails -- crashes (-10% weeks) occur far more often
    # than a Gaussian predicts, and under a Gaussian a single such week is so
    # improbable the model is forced to flip state to explain it. Fat tails let the
    # model absorb a crash as an in-regime OUTLIER instead of regime evidence,
    # which directly damps single-week whipsaw.
    #   WHY FIXED (not learned): when it was sampled it converged to ~10, i.e.
    #   already nearly Gaussian, so learning it bought almost nothing while costing
    #   a parameter and a sampling dimension. Fixing it at 5 keeps the crash-
    #   absorbing benefit (slightly fatter tails than the learned ~10, if anything
    #   BETTER for whipsaw resistance) for free. Shared across states -- outlier-
    #   proneness is a global property of weekly returns, not per-regime.
    # [STRUCTURAL] tail_dof = 5.0 is a HARD CONSTANT (like the notebook's die probs):
    # a literal, NOT a numpyro.sample, so the model can never learn it. It fixes the
    # r_t emission family's tail thickness by fiat.
    tail_dof = 5.0

    # SECOND EMISSION DIMENSION: v_t = observed LOG intra-week realized vol.
    # DECISION / definition, and how it differs from return_vols above:
    #   * v_t is OBSERVED DATA (computed once per week: log-std of that week's
    #     ~5 daily returns, data_acquisition.py's weekly_log_realized_vol). return_vols was
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
    # IDENTIFICATION (mu_bull > mu_bear pins the labels); v_t does the heavy
    # lifting of telling the two clusters apart.
    #
    # ORDERING (label-switching guard, mirrors the drift channel): we make the
    # turbulent mean = calm mean + a POSITIVE gap, so the higher-vol state is
    # tied to the SAME index that drift makes the bear state. Calm/low-vol -> the
    # +drift (BULL) state, turbulent/high-vol -> the -drift (BEAR) state. This is
    # consistent with the empirical split (calm weeks are the +drift weeks), so
    # the two ordered channels REINFORCE the same labeling rather than competing
    # for it -- no new uncontrolled label-switching axis is introduced.
    # [PRIOR] log_vol_calm, log_vol_gap: learnable magnitudes. [STRUCTURAL] the
    # ORDERING turbulent = calm + (HalfNormal >= 0), so the BEAR v_t mean >= BULL v_t
    # mean holds BY CONSTRUCTION -- same ordered-gap guard as the drift channel, and
    # deliberately aligned with it (turbulent state == bear state). -> [PRIOR+STRUCTURAL].
    log_vol_calm = numpyro.sample(
        "log_vol_calm", dist.Normal(_PRIOR_CALM_LOG_VOL_MEAN, 0.3)
    )
    log_vol_gap = numpyro.sample(
        "log_vol_gap", dist.HalfNormal(_PRIOR_LOG_VOL_GAP)
    )
    log_vol_turbulent = log_vol_calm + log_vol_gap
    # Indexed [BEAR, BULL]: BEAR is the turbulent (high-vol) state, BULL is the
    # calm (low-vol) state.
    v_loc = jnp.stack([log_vol_turbulent, log_vol_calm])

    # Spread (STANDARD DEVIATION) of v_t around its state mean -- hence the name
    # vol_stdev. Shared across states as a single number (~0.40, the within-group
    # std from the empirical split). It does NOT separate the states -- the per-
    # state MEANS (v_loc) do that. vol_stdev instead sets the SIGNAL-TO-NOISE of
    # the v_t channel: small -> two narrow, well-separated bells -> each v_t votes
    # decisively for one state; large -> wide overlapping bells -> mushy updates.
    # Kept shared (not per-state) because the empirical within-group spreads were
    # similar (~0.35 vs ~0.45), so one number loses almost nothing and avoids
    # another weakly-identified, label-switching-adjacent axis.
    # [PRIOR] vol_stdev: learnable v_t spread. [STRUCTURAL] shared across states
    # (one number for both), so the v_t signal-to-noise is baked as state-independent.
    vol_stdev = numpyro.sample(
        "vol_stdev", dist.HalfNormal(_PRIOR_LOG_VOL_WITHIN_STD)
    )
    # NOTE: v_t emission is Normal (not StudentT). A StudentT-on-v_t experiment was
    # tried here to absorb bull-market vol spikes as tail events; NUTS drove its df
    # to ~12 (effectively Gaussian), i.e. it judged the spikes to be REAL recurring
    # signal, not statistical outliers. The diagnostic bears this out: ~40% of
    # turbulent weeks are transient 1-2 week bursts, which a 2-state model has no
    # choice but to read as bear. Fat tails cannot fix that conflation -- a THIRD
    # (turbulent-bull) state is the structural fix (regime_model_3state.py). So v_t
    # stays Normal here.

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    # [STRUCTURAL] emission FAMILIES are hard choices (like the notebook's Categorical
    # die): r_t ~ StudentT, v_t ~ Normal, conditional-independence between them. NUTS
    # learns their loc/scale (priors above) but can NEVER change the family or the
    # independence assumption -- those are baked into observation_model / _JointRV.
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
            v_dist=dist.Normal(loc=v_loc[x], scale=vol_stdev),
        )

    dynamics = DynamicalModel(
        # [STRUCTURAL] initial_condition: a FIXED uniform 50/50 over states (a hard
        # jnp.ones(K)/K constant, like the notebook's `probs=jnp.ones(2)/2`), not
        # learned. It only sets the filter's t=0 belief and washes out fast.
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
    """Fit the 2-state regime model on the training weekly frame via NUTS.

    target_accept_prob is raised from NUTS's 0.8 default to 0.95 (smaller leapfrog
    step size -> fewer divergences in tight regions). Kept consistent with the
    3-state model, where the drift ladder makes this matter more.

    `train_obs` is the BIVARIATE frame from RegimeDataset.observations() (or its
    train split): a pandas DataFrame indexed by week with columns ["r_t", "v_t"].
    Index position is used as obs_times (evenly spaced weekly cadence, section 3).

    obs_values is stacked to shape (T, 2) = [r_t, v_t] per week -- the vector
    observation layout the dynestyx handler expects (handlers.py: "(..., T, D)
    for vector observations") and that _JointRV.log_prob consumes as [r, v].
    """
    obs_times = jnp.arange(len(train_obs), dtype=jnp.float32)
    # Column order MUST be [r_t, v_t] to match _JointRV.log_prob's value[...,0]=r,
    # value[...,1]=v indexing. Pull columns explicitly rather than relying on
    # frame column order.
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


def filtered_p_bear(mcmc) -> jnp.ndarray:
    """Posterior-averaged filtered P(BEAR_t | y_1:t), one value per
    training-week timestep, for use against section 7's evaluation.

    BEAR is identified directly by drift ordering (mu_bear < mu_bull, see
    regime_model docstring), so this is a real "bear" probability, not a
    volatility-state proxy.

    `f_filtered_states` has shape (num_samples, T, K); average over the
    posterior draws to get a single filtered P(bear) curve per timestep.
    """
    filtered_states = mcmc.get_samples()["f_filtered_states"]  # (num_samples, T, K)
    return filtered_states[:, :, BEAR].mean(axis=0)


def filtered_p_bear_over(mcmc, obs_frame, seed: int = 1) -> jnp.ndarray:
    """Filtered P(BEAR_t | y_1:t) over an ARBITRARY weekly frame (e.g. train+test
    combined), using the parameters LEARNED ON TRAIN.

    This is how we evaluate out-of-sample: we do NOT refit on the test tail. We
    take the train posterior draws and run the SAME HMM forward filter over the
    full observation series via Predictive. The forward filter is causal --
    P(bear_t) uses only y_1:t -- so the values over the test weeks are genuine
    out-of-sample filtered probabilities, not hindsight.

    `obs_frame` is a bivariate DataFrame with columns [r_t, v_t] (typically the
    full RegimeDataset.observations()). Returns one posterior-averaged P(bear)
    per row.
    """
    obs_times = jnp.arange(len(obs_frame), dtype=jnp.float32)
    obs_values = jnp.asarray(obs_frame[["r_t", "v_t"]].to_numpy(), dtype=jnp.float32)

    def conditioned_model():
        with Filter(filter_config=HMMConfig(record_filtered=True)):
            return regime_model(obs_times=obs_times, obs_values=obs_values)

    # Feed the train posterior samples through the model conditioned on the FULL
    # series; ask Predictive only for the recorded filtered states.
    predictive = Predictive(
        conditioned_model,
        posterior_samples=mcmc.get_samples(),
        return_sites=["f_filtered_states"],
    )
    filtered = predictive(jr.PRNGKey(seed))["f_filtered_states"]  # (num_samples, T, K)
    return filtered[:, :, BEAR].mean(axis=0)


def plot_regime_fit(
    dates,
    price,
    r_t,
    p_bear,
    bear_label,
    split_date=None,
    split_label="dotted line = train | test split; right of it is out-of-sample",
    provisional_weeks=0,
    short_shocks=None,
    save_path=None,
    title_prefix="Regime nowcast (2-state)",
):
    """Single combined plot: price with ground-truth bear bands in the
    background, weekly log return, and the model's filtered P(bear_t) --
    the section 7 "does the filtered nowcast anticipate the smoothed
    ground truth" comparison, in one figure.

    `bear_label` is any 0/1 (BULL/BEAR) ground-truth Series on the plot's index
    -- currently the Pagan-Sossounov dating (labels.pagan_sossounov_label); the
    model is agnostic to how it was derived.

    `split_date` (optional): if given, a vertical divider is drawn at that date on
    every panel. `split_label` sets the caption describing what the divider means.
    Default (single-split main()): the train/test boundary -- LEFT in-sample, RIGHT
    the held-out tail. For walk-forward (main_walkforward), it instead marks where
    OOS coverage begins, with EVERYTHING to the right being out-of-sample.

    Two annotations make P&S's KNOWN LIMITATIONS visible rather than silent
    (P&S is a long-horizon, hindsight-smoothed truth, kept canonical on purpose):

    `provisional_weeks` (optional): P&S cannot date a turning point within its
    endpoint-censor margin (~26 wk) of the sample end -- that trailing stretch is
    PROVISIONAL and will revise as data arrives. If >0, the last that many weeks
    are hatched to flag "dating not yet reliable here".

    `short_shocks` (optional): list of (name, start_date, end_date) for violent
    SHORT episodes P&S deliberately does NOT date as bears (shorter than its
    ~8-month extrema window -- e.g. the 2020 COVID crash). These are marked
    distinctly so a correct model P(bear) spike there is not misread as a false
    alarm against a truth that simply omits the event.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    dates = np.asarray(dates)
    label = np.asarray(bear_label)

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True, height_ratios=[3, 1.2, 1.2]
    )

    def shade_bear_bands(ax):
        start = 0
        for t in range(1, len(label) + 1):
            if t == len(label) or label[t] != label[start]:
                if label[start] == 1:  # BEAR
                    ax.axvspan(dates[start], dates[t - 1], color="crimson", alpha=0.12, linewidth=0)
                start = t

    def draw_split(ax):
        if split_date is not None:
            ax.axvline(np.asarray(split_date), color="navy", lw=1.2, linestyle=":")

    def shade_provisional(ax):
        # Hatch the trailing censor-margin weeks where P&S dating is not reliable.
        if provisional_weeks and provisional_weeks > 0:
            lo = dates[max(0, len(dates) - provisional_weeks)]
            ax.axvspan(lo, dates[-1], facecolor="none", edgecolor="gray",
                       hatch="///", alpha=0.5, linewidth=0)

    def mark_short_shocks(ax):
        # Vertical dashed markers over short violent episodes P&S omits.
        if short_shocks:
            for _name, s, e in short_shocks:
                ax.axvspan(np.asarray(s), np.asarray(e), color="purple",
                           alpha=0.10, linewidth=0)

    def plot_results(ax, dates, result_series, colour, ylabel):
        shade_bear_bands(ax)
        mark_short_shocks(ax)
        shade_provisional(ax)
        ax.plot(dates, result_series, color=colour, lw=1)
        draw_split(ax)
        ax.set_ylabel(ylabel)

    ax = axes[0]
    plot_results(ax, dates, price, "black", "S&P 500")
    ax.set_yscale("log")
    title = f"{title_prefix}: price, weekly return, filtered P(bear)"
    if split_date is not None:
        title += f"  [{split_label}]"
    ax.set_title(title)

    # Name the short-shock markers on the price panel so the purple bands read as
    # "P&S omits this violent-but-short event", not as bear bands.
    if short_shocks:
        for name, s, _e in short_shocks:
            ax.annotate(
                f"{name}\n(P&S blind spot)",
                xy=(np.asarray(s), ax.get_ylim()[1]),
                xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7, color="purple",
            )

    plot_results(axes[1], dates, r_t, "steelblue", "r_t")
    ax = axes[1]
    ax.axhline(0.0, color="gray", lw=0.5)

    plot_results(axes[2], dates, p_bear, "darkorange", "P(bear)")
    ax = axes[2]
    ax.axhline(0.5, color="gray", lw=0.5, linestyle="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("date")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, axes


# ---- fit-mode runner hooks (see fit_mode_processor.run_main) ----------------------------
# Walk-forward from the shared factory (model-agnostic; only needs fit/filtered_p_bear_over).
import sys as _sys_wf  # noqa: E402
from fit_mode_processor import make_walk_forward_p_bear as _make_wf  # noqa: E402

walk_forward_p_bear = _make_wf(_sys_wf.modules[__name__])

needs_macro = False   # bivariate (r_t, v_t) only
obs_cols = ["r_t", "v_t"]

# DEFAULT fit mode with no CLI arg: "global" (fast 80/20) | "walkforward" (slow rolling
# 8yr refits, non-stationarity-robust). HOW TO USE: edit this to change the default and
# run `python regime_model_2state.py`; OR override per-run: `... walkforward` (CLI wins).
FIT_MODE = "global"


def obs_kwargs():
    return {}          # bare observations()/split() -> bivariate


def extra_spec():
    return {}


def main(mode: str | None = None) -> None:
    """Fit + evaluate + plot + save. mode: 'global' (80/20, fast) | 'walkforward'
    (rolling 8yr refits, slow); None -> FIT_MODE. CLI arg overrides:
    `python regime_model_2state.py [global|walkforward]`."""
    from fit_mode_processor import run_main
    run_main(_sys_wf.modules[__name__], mode if mode is not None else FIT_MODE)


if __name__ == "__main__":
    import sys as _s
    main(_s.argv[1] if len(_s.argv) > 1 else None)
