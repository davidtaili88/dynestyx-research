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

# DRAWDOWN CHANNEL toggle. When True (default), the model uses a 3rd emission channel
# (dd = price / 1yr-trailing-max - 1) that fixes the relief-rally whipsaw + doubles bear
# recall (see the big note in regime_model and [[drawdown-channel-breakthrough]]). Set
# False to recover the original 2-channel [r_t, v_t] model (fit/filter auto-adapt).
INCLUDE_DRAWDOWN = True
_DRAWDOWN_WINDOW_WEEKS = 52  # 1yr; chosen by sweep (only window that keeps false-alarm flat)

# MACRO CHANNELS (two, COMPLEMENTARY BY ERA -- both target the residual mid-bear whipsaw
# in slow/rolling-top bears that dd alone doesn't fully damp; see the emission blocks):
#   INCLUDE_CREDIT: BAA-AAA multi-month WIDENING (cs_chg). Strong in DEFAULT-driven bears
#     (dotcom, GFC, post-2010), DEAD in the rate-driven 1970-82 stagflation bears.
#   INCLUDE_CURVE : yield-curve INVERSION DEPTH (inv = max(0, 3m-10y)). Strong exactly
#     where credit is dead (1970-82, +0.74sd), silent in default-driven bears.
# corr(cs_chg, inv) ~ 0.05: independent signals -> the two together span both bear TYPES.
# Both need the macro CSVs -> load_regime_dataset(..., include_macro=True). Off would
# recover the 3-channel [r,v,dd] model. See data.py credit_spread_change / curve_inversion.
# CREDIT is OFF by default (2026-07-27). The episode-level analysis showed credit alone is
# a net LIABILITY under the global fit: its bull-signal in the calm ~80% of history swamps
# the price channels and VETOES bear calls almost everywhere (bear recall 0.42 -> 0.08). Its
# bear-signal only appears in a few default-driven bears, so a single global bull/bear gap
# averages to a suppressor. CURVE (inversion depth) does the useful work on its own -- it
# CONFIRMS default bears (dotcom 0.63->0.75, GFC 0.52->0.82) AND covers the rate-driven
# 1970s bears -- so we ship dd+curve. Credit is left toggleable pending a WALK-FORWARD test
# (per-era refit is where credit is expected to stop being a global suppressor).


INCLUDE_CREDIT = False
_CREDIT_HORIZON_MONTHS = 5   # peak of a broad 3-8mo plateau in the separation sweep (not overfit)
INCLUDE_CURVE = False
print(INCLUDE_CREDIT, INCLUDE_CURVE)
# Observation column order MUST match this everywhere (fit, filter, _JointRV indexing).
# Order: r_t, v_t, then dd, then cs_chg, then inv -- append-only so earlier channels'
# indices never shift (mirrors observations() append order in data.py).
_OBS_COLS = ["r_t", "v_t"]
if INCLUDE_DRAWDOWN:
    _OBS_COLS.append("dd")
if INCLUDE_CREDIT:
    _OBS_COLS.append("cs_chg")
if INCLUDE_CURVE:
    _OBS_COLS.append("inv")

# One place that maps the module toggles to the data-layer channel kwargs, so every
# observations()/split() call site stays consistent with _OBS_COLS above.
_OBS_KWARGS = dict(
    include_drawdown=INCLUDE_DRAWDOWN,
    drawdown_window_weeks=_DRAWDOWN_WINDOW_WEEKS,
    include_credit=INCLUDE_CREDIT,
    credit_horizon_months=_CREDIT_HORIZON_MONTHS,
    include_curve=INCLUDE_CURVE,
)
# Whether any channel needs the macro CSVs loaded (credit/curve do; dd/r/v don't).
_NEEDS_MACRO = INCLUDE_CREDIT or INCLUDE_CURVE

# DEFAULT fit mode when you run the file with no CLI arg:
#   "global"      -> one 80/20 fit; fast; train/test generalization.
#   "walkforward" -> rolling 8yr refits; slow (one NUTS fit per fold); the
#                    non-stationarity-robust eval (each fold learns era-local params).
# HOW TO USE:
#   * edit this constant to change the DEFAULT, then just `python regime_model_3state.py`
#   * OR override per-run WITHOUT editing:  `python regime_model_3state.py walkforward`
#     (a CLI arg always wins over this constant).
FIT_MODE = "global"

# NOTE (2026-07-22): a BULL negative-binomial dwell tail (R_BULL chained sub-phases)
# was implemented here and DISPROVEN. Goal was mature-bull inertia to absorb the
# vol-gap fix's decisive single-week vol signal. It fails for the SAME reason the
# min-dwell HSMM failed: in a CONTINUOUSLY-HELD bull the belief drains to the terminal
# sub-phase (measured 98% occupancy) and the dwell collapses back to geometric exactly
# when a mature bull needs inertia. Proven independent of the advance rate: with an
# absorbing terminal a long-held bull ALWAYS pools at the last phase. So phase-chain
# augmentation shapes only the ENTRY transient, never mature-regime persistence. See
# docs/hsmm_plan.md. Kept 3-state PLAIN (geometric) + the vol-gap confidence fix.

# PRIOR CENTERS for the drift channel -- NOT hard constants the model uses; they are
# the LOCATIONS at which the learnable drift priors are centered (the model learns the
# actual values around them). DELIBERATELY ROUND BALLPARKS, not precise fitted numbers:
# a prior-sensitivity check (scratchpad/prior_sensitivity.py) showed the data tightens
# every one of these posteriors to <=18% (usually <7%) of its prior width, i.e. the
# likelihood overrules the center 5-50x -- so the EXACT value is inert. We therefore use
# round, obviously-approximate values (and center bear drift at 0, NOT presupposing a
# sign) rather than frozen digits from an old sklearn baseline, which merely LOOKED
# overfitted. (Named _PRIOR_ not _EMPIRICAL_ because these are prior centers, not
# measured constants the model is forced to use.)
_PRIOR_BEAR_DRIFT_MEAN = 0.0     # neutral center; data pulls it negative on its own
_PRIOR_DRIFT_GAP = 0.005         # round ballpark for the bull-vs-bear weekly drift gap
                                 # (~0.5%/wk); only sets the DRIFT-LADDER gap prior scale

# PRIOR CENTERS for the v_t (LOG intra-week realized vol) channel -- prior LOCATIONS,
# not hard constants; same "inert exact value" finding as above (tighten <=0.07). Round
# ballparks for where calm vs turbulent log-vol sit and how far apart:
#   calm (bull-leaning, low vol)      log-vol ~ -5.3
#   turbulent (bear-leaning, high vol) log-vol ~ -4.3   -> calm->turbulent gap ~ 1.0
# NOTE: v_t is already log(realized_vol) (data.py), so these are on the log scale.
_PRIOR_CALM_LOG_VOL_MEAN = -5.3   # bull-leaning (low intra-week realized vol), rounded
_PRIOR_LOG_VOL_GAP = 1.0          # round calm->turbulent log-vol gap (~ -4.3 minus -5.3)
_PRIOR_LOG_VOL_SPREAD = 0.4   # typical within-group spread of v_t


class _JointRV(dist.Distribution):
    """Conditional-independence joint of the two emission dimensions [r_t, v_t].

    Deliberately NOT a numpyro MultivariateNormal or a `.to_event(1)` wrap of one
    dist, because the two dimensions are PARAMETERIZED SEPARATELY: both are StudentT,
    but each has its OWN df (tail_dof for r_t, v_tail_dof for v_t) as well as its own
    loc/scale, and we keep the conditional-independence factorization explicit rather
    than forcing a shared/joint parameterization. So this holds one distribution per
    dimension and defines log_prob(y) to return the PER-DIMENSION log-probs as a
    length-2 vector [log p(r|x), log p(v|x)]. (Historically the two dimensions used
    DIFFERENT families -- StudentT return, Normal log-vol -- which is why a custom joint
    was needed; v_t is now also StudentT, but the per-dimension design still stands.)

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

    def __init__(self, r_dist, v_dist, d_dist=None, extra_dists=()):
        # r_dist, v_dist are the always-present [r_t, v_t] channels. d_dist is the
        # OPTIONAL drawdown channel (kept as a named arg for back-compat). extra_dists
        # is an ordered tuple of ANY FURTHER conditionally-independent channels
        # (credit cs_chg, curve inv, ...). The event is the concatenation
        # [r_t, v_t, (dd,) *extras] and log_prob returns the per-dim vector the filter
        # SUMS -- conditional independence extends to K channels identically.
        self.r_dist = r_dist
        self.v_dist = v_dist
        self.d_dist = d_dist
        self.extra_dists = tuple(extra_dists)
        # Ordered list of every channel dist; the [r, v, (d), *extra] column order here
        # MUST match _OBS_COLS in the data frame.
        self._all = [r_dist, v_dist]
        if d_dist is not None:
            self._all.append(d_dist)
        self._all.extend(self.extra_dists)
        super().__init__(batch_shape=(), event_shape=(len(self._all),))

    def log_prob(self, value):
        # value is [..., D]; return [..., D] per-dim log-probs so the filter's jnp.sum
        # turns it into the joint (see class docstring). Do NOT sum here.
        parts = [d.log_prob(value[..., i]) for i, d in enumerate(self._all)]
        return jnp.stack(parts, axis=-1)

    def sample(self, key, sample_shape=()):
        # Only needed for prior/predictive sampling, not for filtering/NUTS on the
        # conditioned model, but implemented for completeness. Independent per channel.
        keys = jr.split(key, len(self._all))
        parts = [d.sample(keys[i], sample_shape) for i, d in enumerate(self._all)]
        return jnp.stack(parts, axis=-1)


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
      * DRIFT ladder, bear < tbull < bull (positive gaps) -- pins labels.
      * v_t VOL ladder, calm(BULL) < turbulent(TBULL) < violent(BEAR) (positive gaps).
        BEAR now sits ABOVE TURBULENT_BULL in realized vol, NOT sharing its mean --
        this is the under-confidence fix (see the v_loc block below). So the STRONG
        v_t channel, not just the weak drift gap, separates bear from turbulent-bull:
        a deep/violent drawdown reads unambiguously as bear, letting P(bear) reach ~1.
      NOTE this is NOT a third constraint, just a consistency requirement BETWEEN the
      two ladders: both arrays are indexed with BEAR at position 0, so the drift ladder
      and the vol ladder call the SAME state "bear". "Bear = low-drift + high-vol corner"
      is therefore already IMPLIED by the two ladders above -- not an extra rule. (If the
      arrays were indexed inconsistently, the channels would label different states as
      bear and compete; keeping them aligned avoids that, by construction.)
      (The r_t RETURN-VOL scale is ALSO a 3-rung ladder bull<tbull<bear -- a third
      ordered channel that reinforces the same labels; see the return_vols block. Its
      bear rung learned ~2x tbull's, so it too separates bear from turbulent-bull.)

    P(bear) reads off the BEAR state alone; TURBULENT_BULL is explicitly not bear.

    ============================================================================
    PRIOR vs STRUCTURAL -- what NUTS can learn vs what is baked in. (Reference:
    notebooks/07_hidden_markov_model.ipynb -- the loaded-die emission `probs` is a
    hard STRUCTURAL constant; `A = numpyro.sample("A", Dirichlet(...))` is a PRIOR.)

      * PRIOR = a `numpyro.sample(...)` on a LEARNABLE parameter. NUTS updates it;
                with ~3600 weeks the likelihood OVERRULES the prior's location/scale
                (why prior-only tweaks -- e.g. cranking p_self -- barely move things).
      * STRUCTURAL = a hard constant OR an arithmetic CONSTRUCTION. Changes what the
                model CAN express; the likelihood works THROUGH it, cannot undo it.
                Adding/removing structure is what actually moves behavior here:
                  - the 3rd state (TURBULENT_BULL) itself -> STRUCTURAL, and it worked.
                  - the vol-gap fix (BEAR v_t mean > TBULL, log_vol_bear_extra) ->
                    STRUCTURAL (a NEW parameter that didn't exist before), and it
                    lifted P(bear) 0.71 -> 1.0. Note it LOOKS like "just a prior" but
                    the point is the parameter now EXISTS, not its HalfNormal scale.
                  - a neg-binom / min-dwell duration tail -> STRUCTURAL but DISPROVEN
                    (belief drains to the terminal phase; see module NOTE at top).
    Each definition below is tagged [PRIOR], [STRUCTURAL], or [PRIOR+STRUCTURAL].
    ============================================================================
    """
    # Persistence prior: Beta(500, 3) per state, mean p_self ~ 0.994. DELIBERATELY
    # strong (see the 2-state history: even this was dragged to ~0.986 by the noisy
    # weekly likelihood, which is why the 3rd state -- not just a stronger prior --
    # is needed). Now one self-transition per state (K=3). Each row's remaining
    # (1 - p_self) mass is split over the OTHER two states by a learned Dirichlet,
    # rather than assumed equal: e.g. from BULL you are far more likely to slip into
    # TURBULENT_BULL than jump straight to BEAR, and we let the data express that.
    # [PRIOR] p_self: learnable per-state self-persistence (Beta only seeds it).
    p_self = numpyro.sample("p_self", dist.Beta(500.0, 3.0).expand([K]).to_event(1))
    # [PRIOR] off_split: learnable direction of the leaving-mass (which OTHER state
    # you slip into). Dirichlet(1,1) seeds it uniform; the data moves it.
    off_split = numpyro.sample(
        "off_split", dist.Dirichlet(jnp.ones(2)).expand([K]).to_event(1)
    )

    # [STRUCTURAL] A: the 3x3 first-order Markov SHAPE (each row = self + split of the
    # rest) is baked in; only the p_self / off_split magnitudes inside are learned.
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
    # [PRIOR] mean_return_bear, drift_gap1, drift_gap2: learnable MAGNITUDES.
    # [STRUCTURAL] the LADDER mu_bear < mu_tbull < mu_bull built via non-negative
    # HalfNormal gaps -> the ordering holds BY CONSTRUCTION (label-switching guard).
    # -> [PRIOR+STRUCTURAL].
    mean_return_bear = numpyro.sample("mean_return_bear", dist.Normal(_PRIOR_BEAR_DRIFT_MEAN, 0.01))
    drift_gap1 = numpyro.sample("drift_gap1", dist.HalfNormal(2.0 * _PRIOR_DRIFT_GAP))
    drift_gap2 = numpyro.sample("drift_gap2", dist.HalfNormal(2.0 * _PRIOR_DRIFT_GAP))
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
    # 3 states, ORDERED by return-vol: BULL lowest, TURBULENT_BULL middle, BEAR highest.
    # (Earlier this was a two-level [high, high, low] with bear & tbull SHARING one high
    # value; the fitted 3-way ladder showed bear's return-vol is ~2x tbull's, so sharing
    # was wrong -- see the WHY 3-WAY note just below.) Built as an ordered ladder (not an
    # unordered per-state free-for-all) to keep the label-switching guard and avoid the
    # slow sampling that the unordered version caused.
    # Return-vol is a THREE-RUNG ordered ladder: bull (lowest) < tbull < bear (highest),
    # each rung a non-negative HalfNormal gap -> ordering by construction (label-switching
    # guard), mirroring the drift and v_t ladders. So EACH of the three states has its
    # own weekly-return spread; BEAR gets the widest.
    #
    # WHY 3-WAY AND NOT SHARED (settled 2026-07-23, keep this rationale):
    # A prior worry was that per-state return-vol is redundant with the v_t ladder --
    # an 8wk-return-std vs 8wk-mean-v_t correlation of ~0.76 SUGGESTED so. It is NOT:
    # fitting a 3-way ladder, BEAR learned a return-vol ~2x TBULL's (bear ~0.038 vs
    # tbull ~0.019 vs bull ~0.011). The 0.76 was measured on SMOOTHED 8-week windows,
    # which averages away exactly the per-week crash tails (-8%, -12% weeks) that make
    # bear's WEEKLY-return spread much wider than tbull's. v_t measures intra-week
    # choppiness; return-vol measures weekly-return magnitude -- genuinely distinct, so
    # both channels separate bear from tbull and both are kept. (Making it 3-way did
    # NOT reduce the P(bear) whipsaw -- like the v_t gap, more separation = a bit more
    # single-week jumpiness -- but the whipsaw is a separate, emission-side problem;
    # return-vol being 3-way is justified on its own merits.)
    # [PRIOR] the three magnitudes; [STRUCTURAL] the ordered ladder. -> [PRIOR+STRUCTURAL].
    return_vol_bull = numpyro.sample("return_vol_bull", dist.HalfNormal(0.02))  # lowest
    return_vol_gap_tbull = numpyro.sample("return_vol_gap_tbull", dist.HalfNormal(0.02))
    return_vol_gap_bear = numpyro.sample("return_vol_gap_bear", dist.HalfNormal(0.02))
    return_vol_tbull = return_vol_bull + return_vol_gap_tbull
    return_vol_bear = return_vol_tbull + return_vol_gap_bear  # highest
    # Indexed [BEAR, TURBULENT_BULL, BULL]: now THREE distinct return-vols.
    return_vols = jnp.stack([return_vol_bear, return_vol_tbull, return_vol_bull])

    # Student-t tail-thickness (degrees of freedom) for r_t, shared across states.
    # df = the "tail-fatness dial": low (~3-5) = fat tails, high (30+) ~ Gaussian.
    #
    # DOES r_t ACTUALLY NEED FAT TAILS? (tested 2026-07-23) -- weaker than you'd think.
    # Marginal r_t is fat-tailed (excess kurtosis ~6.2), BUT the 3-WAY return-vol ladder
    # already absorbs most of that: a -12% week is a normal draw from the WIDE bear-state
    # distribution, not a tail outlier. So the WITHIN-state residual is near-Gaussian and
    # df floats HIGH: full-sample df ~ 26.5 (~Gaussian), and across five 8-year windows
    # df was 25-30 in FOUR (incl. windows containing the 1987, 1973-74, 2008 crashes);
    # only 2001-2009 (dotcom AND 2008 in one window) dropped to ~13.8. df is also WEAKLY
    # identified (90% CIs ~10..50), i.e. the data barely distinguishes df=15 from Gaussian.
    # Forcing r_t Gaussian gave nearly identical P(bear) (mean |dP|~0.006) and marginally
    # FEWER crossings (100 vs 108). So Gaussian would be fine ~4/5 of the time -- because
    # LEVEL 1 (the vol ladder) already did the tail work.
    # WHY KEEP StudentT anyway: as the LEVEL-2 backstop (see the tail-handling note below
    # the v_scale block). It is one weakly-identified, self-adjusting param that floats to
    # ~Gaussian when the ladder already absorbed the tails and engages (df~14) in unusually
    # crash-heavy windows (2001-2009) the ladder can't fully span. Applied to BOTH channels
    # for consistency. (Evidence: scratchpad/rt_gaussian.py, rt_df_windows.py.)
    # [PRIOR+STRUCTURAL] tail_dof_raw is LEARNED (unlike the 2-state, which fixes dof
    # at 5); the `2.0 +` is a STRUCTURAL floor guaranteeing dof > 2 (finite variance).
    tail_dof = 2.0 + numpyro.sample("tail_dof_raw", dist.Gamma(2.0, 0.1))

    # SECOND EMISSION DIMENSION: v_t = observed LOG intra-week realized vol.
    # DECISION / definition, and how it differs from return_vols above:
    #   * v_t is OBSERVED DATA (computed once per week: log-std of that week's
    #     ~5 daily returns, data.py's weekly_log_realized_vol). return_vols was
    #     an INFERRED scale parameter of the r_t distribution. Different objects,
    #     different observables -- so this is a genuine extra channel, not a
    #     reparameterization of the return volatility.
    #   * Because v_t is ALREADY on the log scale, emitting on v_t is a LOG-domain
    #     emission on raw realized vol -- the right choice: log-realized-vol is roughly
    #     symmetric (prior_analysis.py checks this), whereas raw vol is right-skewed and
    #     bounded at 0. The log is what tames raw vol's huge tails (see the tail-handling
    #     note below). We emit v_t ~ StudentT (symmetric, fat-tail backstop on top of the
    #     log); a StudentT on log-vol == a log-StudentT on raw vol.
    #
    # ROLE OF THIS CHANNEL: it is the SEPARATION axis. v_t is strongly bimodal
    # (calm vs. turbulent, ~1.04 log-units apart, see constants above), a far
    # stronger signal than the tiny drift gap. On weeks where r_t is ambiguous,
    # v_t decisively moves the filter's P(bear_t) update. Drift still does
    # IDENTIFICATION (the drift ladder pins the labels); v_t does the heavy
    # lifting of telling turbulent from calm weeks apart.
    #
    # ORDERING: a THREE-RUNG v_t ladder (calm < turbulent-bull < bear), each rung a
    # POSITIVE gap above the previous, so the whole ordering is enforced BY
    # CONSTRUCTION (no label-switching mode for NUTS to fall into) -- exactly like the
    # drift ladder above.
    #
    # WHY BEAR IS NOW A SEPARATE (HIGHER) VOL RUNG, not sharing TBULL's mean:
    # previously BEAR and TURBULENT_BULL SHARED one turbulent v_t mean, so the ONLY
    # thing separating them was the tiny weekly drift_gap1. When a real bear arrived,
    # the STRONG channel (v_t) said "turbulent" for BOTH states equally and could not
    # break the tie; the filter split the mass ~50/50 and P(bear) CAPPED near 0.5-0.7
    # (the under-confidence). Giving BEAR a HIGHER realized-vol mean than TURBULENT_BULL
    # lets the strong v_t channel itself distinguish them: a deep, violent drawdown
    # (high v_t) reads unambiguously as BEAR, so P(bear) can reach ~1. This encodes the
    # real distinction -- bears are more VIOLENT than transient bull-market vol spikes
    # -- and makes drawdown MAGNITUDE the bear-vs-tbull discriminator (the "large
    # enough drawdown" idea). The gaps are HalfNormal so ordering holds by construction
    # and no new label-switching axis is introduced: calm/tbull/bear vol reinforce the
    # same labels the drift ladder assigns (bear = lowest drift AND highest vol).
    # [PRIOR] log_vol_calm, log_vol_gap, log_vol_bear_extra: learnable magnitudes.
    # [STRUCTURAL] the THREE-RUNG v_t ladder calm < tbull < bear, each rung a
    # non-negative HalfNormal gap above the previous -> ordering by construction.
    #   *** THE UNDER-CONFIDENCE FIX lives here: log_vol_bear_extra is a NEW PARAMETER
    #   that gives BEAR its own v_t mean ABOVE TBULL. Before it existed, BEAR and TBULL
    #   were FORCED to share one v_t mean (v_loc=[turb,turb,calm]) -- a STRUCTURAL
    #   constraint that capped P(bear) at ~0.71. Adding this parameter (not changing a
    #   prior!) let the strong v_t channel separate them and lifted P(bear) to ~1.0.
    #   This is the textbook PRIOR-vs-STRUCTURAL point: the HalfNormal(0.5*gap) is a
    #   prior, but what changed behavior is that the PARAMETER now EXISTS. ***
    # -> [PRIOR+STRUCTURAL].
    log_vol_calm = numpyro.sample(
        "log_vol_calm", dist.Normal(_PRIOR_CALM_LOG_VOL_MEAN, 0.3)
    )
    log_vol_gap = numpyro.sample(
        "log_vol_gap", dist.HalfNormal(_PRIOR_LOG_VOL_GAP)
    )
    log_vol_tbull = log_vol_calm + log_vol_gap  # TURBULENT_BULL: transient-turbulence vol
    # Extra rung lifting BEAR above TBULL. Prior scale = half the calm->turbulent gap:
    # bears are meaningfully more violent than transient spikes, but not another full
    # turbulent-gap higher. Wide enough for NUTS to move; ordered so bear >= tbull.
    log_vol_bear_extra = numpyro.sample(
        "log_vol_bear_extra", dist.HalfNormal(0.5 * _PRIOR_LOG_VOL_GAP)
    )
    log_vol_bear = log_vol_tbull + log_vol_bear_extra  # BEAR: the most violent rung
    # Indexed [BEAR, TURBULENT_BULL, BULL]: now THREE distinct v_t means (bear highest,
    # then tbull, then calm). BEAR and TBULL are no longer emission-identical, so the
    # strong v_t channel -- not just the weak drift gap -- separates them.
    v_loc = jnp.stack([log_vol_bear, log_vol_tbull, log_vol_calm])

    # Per-state spread of v_t around its state mean (the within-group std from
    # the empirical split, ~0.40). Shared across states as a single scale: the
    # separation already comes from the ordered means above, so we do NOT need a
    # per-state, ordered spread parameter here -- that would just add another
    # weakly-identified quantity. One shared HalfNormal centered near the
    # empirical within-group std is enough.
    # [PRIOR] v_scale learnable; [STRUCTURAL] shared across all states (one number).
    v_scale = numpyro.sample(
        "v_scale", dist.HalfNormal(_PRIOR_LOG_VOL_SPREAD)
    )

    # ========================================================================
    # HOW TAIL EVENTS ARE HANDLED (both emission channels), and WHY BOTH USE StudentT
    # ------------------------------------------------------------------------
    # A "tail event" is a violent week (a crash, a vol explosion) far out in the
    # distribution. If the emission cannot accommodate it, the filter is forced to flip
    # state to explain it -> spurious P(bear) jumps. This model handles tails at TWO
    # levels; the StudentT df here is the SECOND, backstop level.
    #
    # LEVEL 1 -- STRUCTURAL tail handling (does most of the work):
    #   * r_t: the 3-WAY RETURN-VOL LADDER. Each state has its own return spread
    #     (bull ~0.011 < tbull ~0.019 < bear ~0.038). A -12% week is NOT a tail outlier
    #     in the raw pooled sense -- it is a NORMAL-sized draw from the WIDE bear-state
    #     distribution. So conditioning on the state already absorbs the crash: the
    #     WITHIN-state residual is near-Gaussian (this is why r_t's df floats high, ~26).
    #     The ladder converts "market-wide fat tails" into "per-state ordinary spread."
    #   * v_t: the LOG TRANSFORM. We emit v_t = log(realized vol). Raw realized vol has
    #     enormous right-skewed tails (excess kurtosis ~52); the log compresses them to
    #     near-Gaussian (excess kurtosis ~1.1). So v_t's big spikes are pre-tamed by the
    #     transform before any emission family is chosen.
    #
    # LEVEL 2 -- StudentT df as a BACKSTOP (this block + the r_t tail_dof above):
    #   Level 1 handles the TYPICAL crash, but not every tail event is caught by the
    #   structure -- e.g. a window with TWO crashes and a recovery between (2001-2009)
    #   has within-state residuals the single vol level cannot fully span, and v_t's
    #   post-log residual is still MILDLY fat (fitted df ~7.5). A StudentT emission on
    #   BOTH channels is a cheap safeguard for exactly these leftover tail events the
    #   ladder/log don't fully absorb: the df is a self-adjusting dial that floats to
    #   ~Gaussian (high df) when the structure already handled the tails, and drops to
    #   fatten the tails when it didn't. It costs one weakly-identified param per channel
    #   and never hurts (df -> inf recovers Normal). Applied to BOTH channels for
    #   consistency and because v_t is actually the FATTER residual (df ~7.5 < r_t's ~26),
    #   so if either deserves the safeguard it is v_t. (P(bear) effect is small either
    #   way -- Level 1 is the real tail handler; this is insurance. Evidence:
    #   scratchpad/vt_tails.py, studentt_vt.py, rt_gaussian.py, rt_df_windows.py.)
    # ========================================================================
    # [PRIOR+STRUCTURAL] v_tail_dof_raw LEARNED; `2.0 +` is a STRUCTURAL floor (df>2,
    # finite variance), mirroring the r_t tail_dof construction above.
    v_tail_dof = 2.0 + numpyro.sample("v_tail_dof_raw", dist.Gamma(2.0, 0.1))

    # ========================================================================
    # THIRD EMISSION CHANNEL: DRAWDOWN (dd = price / 1yr-trailing-max - 1). ENABLED by
    # the INCLUDE_DRAWDOWN module flag (default True). This is the BREAKTHROUGH fix for
    # the relief-rally whipsaw (see [[drawdown-channel-breakthrough]]).
    #   WHY: r_t and v_t only see price CHANGE, never price LEVEL. So a multi-week relief
    #   RALLY inside a bear (+returns, calm vol) looks bull-like -> the model ABANDONS the
    #   bear (P(bear) 0.98->0.00 in spring-2008), then re-enters = the whipsaw. Drawdown
    #   carries the missing LEVEL info: a rally still ~11% below peak keeps dd strongly
    #   negative -> "still underwater, still a bear" -> the filter holds through the rally.
    #   Four prior fixes (duration, persistence, wider-vol, input-smoothing) all FAILED
    #   because they fought the filter with global params the likelihood overrules; this
    #   works because it adds genuinely NEW causal information instead.
    #   RESULT vs 2-channel: spring-2008 rally-hold 0.00 -> 0.97; whipsaw crossings
    #   122 -> 98; bear RECALL 0.22 -> 0.42 (drawdown ALSO catches calm grinding bears,
    #   which are still underwater though their vol looks bull-like -- the calm-bear fix);
    #   bull false-alarm 0.111 -> 0.114 (unchanged, so concern-2 stays safe).
    #   WINDOW = 1yr, chosen by a 1/2/3yr sweep: all fix the whipsaw, but 1yr uniquely
    #   keeps false-alarm flat (longer windows stay underwater into recoveries -> false
    #   bear). Using ALL 3 windows as separate channels was tested and BREAKS the model
    #   (0.84-0.96 correlated -> triple-counted -> swamps r/v). So: single 1yr channel.
    #
    # STRUCTURE: a 2-LEVEL dd-mean ladder -- BEAR is underwater (dd_bear = dd_bull -
    # bear_gap, learned ~-0.12), BULL ~0 (at/near peak), TURBULENT_BULL mildly underwater.
    # Normal emission on dd. [PRIOR] dd_bull, dd_bear_gap, dd_tbull_gap, dd_scale;
    # [STRUCTURAL] the ordered 2-level assignment. -> [PRIOR+STRUCTURAL].
    # ========================================================================
    if INCLUDE_DRAWDOWN:
        dd_bull = numpyro.sample("dd_bull", dist.Normal(0.0, 0.05))         # bull: ~0 (near peak)
        dd_bear_gap = numpyro.sample("dd_bear_gap", dist.HalfNormal(0.20))  # how far underwater bear is
        dd_tbull_gap = numpyro.sample("dd_tbull_gap", dist.HalfNormal(0.10))  # tbull mildly underwater
        dd_loc = jnp.stack([dd_bull - dd_bear_gap, dd_bull - dd_tbull_gap, dd_bull])  # [BEAR,TBULL,BULL]
        dd_scale = numpyro.sample("dd_scale", dist.HalfNormal(0.15))

    # ========================================================================
    # FOURTH EMISSION CHANNEL: CREDIT-SPREAD WIDENING (cs_chg = BAA-AAA minus its value
    # _CREDIT_HORIZON_MONTHS ago). ENABLED by INCLUDE_CREDIT. Targets the residual
    # mid-bear whipsaw in DEFAULT-driven bears (dotcom, GFC) that dd alone under-damps:
    # the spread keeps WIDENING through the bear even on relief-rally weeks (a level dd
    # can drift back toward 0 on), so cs_chg holds the bear when price momentarily doesn't.
    #   SEPARATION: ~+0.5sd on true bears, ~0 corr with r/v/dd (NEW info, not re-encoded
    #   price -- the bar dd had to clear). The LEVEL was blind at onset (~-0.09sd); the
    #   multi-month CHANGE carries the signal (spreads widen AS the bear develops).
    #   ERA CAVEAT: strong in default-driven bears, ~DEAD in the rate-driven 1970-82
    #   stagflation bears -- that era is covered by the curve-inversion channel instead
    #   (the two are complementary by era; see INCLUDE_CURVE). So under a single GLOBAL
    #   fit cs_chg's bear/bull gap is a blend the 1970s drags toward 0; it is most
    #   effective under the walk-forward fit, where a dotcom fold learns a strong gap and
    #   a 1970s fold learns ~none (same non-stationarity logic as walk_forward_p_bear).
    # STRUCTURE: 2-LEVEL ladder -- BEAR widening (cs_chg_bear = base + bear_gap > 0),
    # BULL ~0 (stable/tightening), TBULL mildly widening. Normal emission (cs_chg is a
    # roughly-symmetric monthly difference). [PRIOR] cs_base, cs_bear_gap, cs_tbull_gap,
    # cs_scale; [STRUCTURAL] the ordered assignment. -> [PRIOR+STRUCTURAL].
    # ========================================================================
    if INCLUDE_CREDIT:
        cs_base = numpyro.sample("cs_base", dist.Normal(0.0, 0.05))          # bull: ~0 (stable spread)
        cs_bear_gap = numpyro.sample("cs_bear_gap", dist.HalfNormal(0.30))   # bear: spread WIDENING (>0)
        cs_tbull_gap = numpyro.sample("cs_tbull_gap", dist.HalfNormal(0.15))  # tbull: mild widening
        cs_loc = jnp.stack([cs_base + cs_bear_gap, cs_base + cs_tbull_gap, cs_base])  # [BEAR,TBULL,BULL]
        cs_scale = numpyro.sample("cs_scale", dist.HalfNormal(0.25))

    # ========================================================================
    # FIFTH EMISSION CHANNEL: YIELD-CURVE INVERSION DEPTH (inv = max(0, 3m-10y), 0 when
    # the curve is normal). ENABLED by INCLUDE_CURVE. This is the COMPLEMENT to credit:
    # it carries bear signal exactly in the RATE-driven 1970-82 bears where credit is
    # dead (+0.74sd there), and is ~silent (inv=0) in the default-driven bears credit
    # covers. corr(cs_chg, inv) ~ 0.05: independent, so together they span both bear TYPES.
    #   WHY ONE-SIDED (depth, not slope): the raw slope level / change FLIP SIGN across
    #   eras (unusable as a global mean); clamping to inversion DEPTH keeps only the half
    #   that consistently means stress -- non-negative-signed wherever it fires, silent
    #   otherwise. The equity tell is the inversion itself, not the level (validated by era).
    #   LIMITATION (accepted): inv fires at/before a rate-driven top then FADES once the
    #   Fed cuts and the curve re-steepens mid-bear -- an early pulse, not a sustained hold
    #   (pure depth, no decay memory, is the version validated). It carries the rate-driven
    #   ONSET; credit / dd carry the body.
    # STRUCTURE: 2-LEVEL ladder -- BEAR inverted (inv_bear = inv_base + bear_gap > 0),
    # BULL ~0 (normal curve), TBULL mildly inverted. HalfNormal-gap ladder like the others.
    # NOTE inv >= 0 by construction, so a Normal emission puts a little mass below 0; that
    # is fine (it is just the calm-state noise floor around inv=0) and matches how dd's
    # Normal handles its bounded-at-0 top. [PRIOR] inv_base, inv_bear_gap, inv_tbull_gap,
    # inv_scale; [STRUCTURAL] the ordered assignment. -> [PRIOR+STRUCTURAL].
    # ========================================================================
    if INCLUDE_CURVE:
        inv_base = numpyro.sample("inv_base", dist.Normal(0.0, 0.05))         # bull: ~0 (normal curve)
        inv_bear_gap = numpyro.sample("inv_bear_gap", dist.HalfNormal(0.40))  # bear: curve INVERTED (>0)
        inv_tbull_gap = numpyro.sample("inv_tbull_gap", dist.HalfNormal(0.20))  # tbull: mild inversion
        inv_loc = jnp.stack([inv_base + inv_bear_gap, inv_base + inv_tbull_gap, inv_base])  # [BEAR,TBULL,BULL]
        inv_scale = numpyro.sample("inv_scale", dist.HalfNormal(0.30))

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    # [STRUCTURAL] emission FAMILIES (r_t ~ StudentT, v_t ~ StudentT) + conditional
    # independence are hard choices; their loc/scale AND now their df (priors above) are
    # learned. Both channels are StudentT as a tail-event backstop (see the big note).
    # What we should observe under each regime. BIVARIATE: obs y is a length-2
    # vector [r_t, v_t]. We return a per-dimension emission whose log_prob([r, v])
    # yields the two-vector [log p(r|x), log p(v|x)]; the HMM filter SUMS these
    # (hmm_filters.py: `jnp.sum(lp)`), which is exactly the conditional-independence
    # factorization  log p(r,v | x) = log p(r|x) + log p(v|x).
    #
    # CONDITIONAL-INDEPENDENCE ASSUMPTION (stated explicitly, DEFINITIVELY tested
    # 2026-07-24 -- this supersedes an earlier hand-wave that called it "weak, ~1% shared
    # variance"; that rule-of-thumb was wrong for the reason below):
    # Summing the two log-probs == assuming r ⊥ v | x. r and v ARE dependent (leverage
    # effect: down weeks are turbulent). Two measurements matter:
    #   1. TAIL DEPENDENCE (scratchpad/tail_dep.py): whole-sample corr(r,v) is only -0.10,
    #      BUT in the worst 5-10% of return weeks it is -0.44..-0.47, and P(v in top 5% |
    #      r in bottom 5%) = 0.26 (vs 0.05 under independence). So the dependence is
    #      CONCENTRATED IN THE CRASH TAIL -- exactly where a regime nowcast's calls matter.
    #      The "weak on average" defence is therefore NOT valid on its own.
    #   2. DEFINITIVE TEST (scratchpad/bivariate_t.py): fit a bivariate-t with a LEARNABLE
    #      r-v correlation (LKJ prior) and compare on HELD-OUT data. Result: the model
    #      learned a near-ZERO linear correlation (-0.03) -- because the 3-WAY RETURN-VOL
    #      LADDER already captures the co-movement (high-vol states carry both wider r
    #      spread and higher v), leaving almost no LINEAR dependence for a correlation
    #      term to add. Held-out P(bear) curves overlay (97.7% correlated, 82% of weeks
    #      identical within 0.05); held-out loglik moved +9.7, but that is attributable to
    #      the bivariate-t's SHARED df (the indep model uses separate r/v df ~26/~9), not
    #      to the ~0 correlation. So modeling the LINEAR dependence buys ~nothing here.
    # NET: conditional independence is empirically justified -- not because the dependence
    # is weak (it is strong in the tail), but because the vol ladder already absorbs the
    # LINEAR part, and the residual is ~0. CAVEAT: a single correlation term cannot model
    # TAIL dependence (joint-tail clustering) anyway; if that ever needs capturing, a
    # tail-dependent copula (not a bivariate-t correlation) is the tool -- a genuine
    # "last drop" upgrade, still not a priority vs the whipsaw.
    #
    # MECHANISM + A SUBTLE INTERACTION WITH THE WHIPSAW (scratchpad/doublecount.py):
    #   Independence DOUBLE-COUNTS a down-turbulent week: the drop already implies the
    #   elevated vol, but the model scores each channel's surprise separately, over-
    #   charging ~rho*z_r*z_v nats against calm states. Because rho<0 this ALWAYS fires
    #   on down-and-turbulent (bear-flavoured) weeks -- same direction as the whipsaw.
    #   The filter update is in LOG-ODDS, so an error of the SAME size on two states
    #   cancels in their comparison. In an EARLIER model BEAR and TBULL SHARED v_loc and
    #   return_vol, so their errors were near-identical (diff <0.01 nats) and cancelled --
    #   which is why the double-count barely reached P(bear). BUT the under-confidence
    #   fixes (v_t ladder + 3-way return-vol) UN-SHARED those params: bear vs tbull now
    #   differ ~2x in return_vol and ~0.7 in v_loc, so the per-state errors now differ by
    #   ~0.23 nats (measured), NOT <0.01 -- the cancellation is WEAKENED. Also, within-
    #   state tail corr survives only INSIDE bear (-0.35 in bear's bottom r-decile; ~0 in
    #   tbull/bull), concentrating the residual double-count on exactly the decisive weeks.
    #   So the SAME changes that lifted confidence (bear/tbull separation) slightly re-
    #   exposed the double-count on the BEAR-vs-TBULL split. That was a plausible whipsaw
    #   lead -- so we DIRECTLY TESTED it (scratchpad/option_a.py): add a learnable
    #   leverage slope so v_t's mean shifts with the return residual (v_loc + beta*z_r),
    #   removing the double-count at its source. RESULT: beta learned to -0.014 +/- 0.009
    #   (right sign, but NOT distinguishable from 0), held-out P(bear) curves were
    #   IDENTICAL (corr 1.000, zero weeks differ by >0.1), and whipsaw was UNCHANGED
    #   (122 vs 122 crossings). So the double-count, though real, does NOT reach the
    #   nowcast and is NOT the whipsaw cause -- confirmed from two angles (this + the
    #   bivariate-t correlation test, both ~0). WHY IT DOESN'T BITE: the crash weeks where
    #   r-v dependence is strongest are exactly the weeks P(bear) is already pinned at 1.0,
    #   so there is no room for a double-count to change the answer. CONCLUSION: conditional
    #   independence is not just documented-and-tolerated but TESTED-AND-VINDICATED here;
    #   the whipsaw lives elsewhere.
    #
    # BOTH channels emit StudentT with their own learnable df (see the tail-handling
    # note above): a self-adjusting safeguard for tail events the structural handlers
    # (r_t vol ladder, v_t log transform) don't fully absorb. df -> inf recovers Normal,
    # so this never hurts. Same StudentT FAMILY on both, but SEPARATE df per channel
    # (tail_dof for r_t, v_tail_dof for v_t) since their residual tail-fatness differs.
    # When INCLUDE_DRAWDOWN, a 3rd channel dd ~ Normal(dd_loc[x], dd_scale) is added, so
    # obs y is [r_t, v_t, dd] and the filter sums 3 log-probs (conditional independence
    # across all three -- dd is a different observable, price LEVEL not price CHANGE).
    def observation_model(x, u, t):
        d_dist = dist.Normal(loc=dd_loc[x], scale=dd_scale) if INCLUDE_DRAWDOWN else None
        # Extra macro channels, appended in the SAME order as _OBS_COLS: credit then curve.
        extra = []
        if INCLUDE_CREDIT:
            extra.append(dist.Normal(loc=cs_loc[x], scale=cs_scale))
        if INCLUDE_CURVE:
            extra.append(dist.Normal(loc=inv_loc[x], scale=inv_scale))
        return _JointRV(
            r_dist=dist.StudentT(df=tail_dof, loc=mean_return[x], scale=return_vols[x]),
            v_dist=dist.StudentT(df=v_tail_dof, loc=v_loc[x], scale=v_scale),
            d_dist=d_dist,
            extra_dists=extra,
        )

    dynamics = DynamicalModel(
        # [STRUCTURAL] fixed uniform initial belief over the 3 states (not learned).
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
    without overshooting and diverging. Cheaper first line of defence than
    reparameterizing; if divergences persist, non-center the drift ladder instead.

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
        train_obs[_OBS_COLS].to_numpy(), dtype=jnp.float32
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


# Durable artifacts live in equity_market/outputs/ (already git-ignored alongside
# *.pkl -- see .gitignore). NUTS fits are expensive, so we cache the posterior
# samples there and reload instead of refitting.
import pathlib as _pathlib

_OUTPUTS_DIR = _pathlib.Path(__file__).resolve().parents[2] / "outputs"


def save_fit(mcmc, name: str, extra: dict | None = None) -> _pathlib.Path:
    """Persist a fitted model's POSTERIOR SAMPLES (+ optional extras) to
    outputs/<name>.pkl so it can be reloaded without a fresh NUTS run.

    We save mcmc.get_samples() (a plain dict of numpy/jax arrays, including the
    recorded f_filtered_states) rather than the live MCMC object -- portable and
    reload-safe. `extra` can carry the channel config / P(bear) curve / dates so a
    saved fit is self-describing. Records _OBS_COLS so a reload knows which channels
    the fit used. Returns the written path.
    """
    import pickle
    import numpy as _np

    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = {k: _np.asarray(v) for k, v in mcmc.get_samples().items()}
    payload = {"samples": samples, "obs_cols": list(_OBS_COLS)}
    if extra:
        payload["extra"] = extra
    path = _OUTPUTS_DIR / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_fit(name: str) -> dict:
    """Reload a fit saved by save_fit -> {"samples": {...}, "obs_cols": [...], ...}.

    The returned "samples" dict can be fed to filtered_p_bear_over via a thin
    Predictive wrapper, or inspected directly for posterior parameter values. Raises
    FileNotFoundError if outputs/<name>.pkl is absent (nothing cached yet).
    """
    import pickle

    path = _OUTPUTS_DIR / f"{name}.pkl"
    with open(path, "rb") as fh:
        return pickle.load(fh)


def filtered_p_bear(mcmc) -> jnp.ndarray:
    """Posterior-averaged filtered P(BEAR_t | y_1:t), one value per
    training-week timestep, for use against section 7's evaluation.

    BEAR is identified by the drift ladder (mu_bear the lowest rung, see
    regime_model docstring), so this is a real "bear" probability. Crucially it is
    the BEAR state's prob ONLY -- the TURBULENT_BULL state is NOT counted as bear,
    which is the whole reason it was added.

    `f_filtered_states` has shape (num_samples, T, K); average over the
    posterior draws to get a single filtered P(bear) curve per timestep.
    """
    filtered_states = mcmc.get_samples()["f_filtered_states"]  # (num_samples, T, K)
    # P(bear) is the BEAR state's filtered prob ONLY -- the TURBULENT_BULL state
    # deliberately does NOT count as bear (that is the whole point of adding it).
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

    K-agnostic vs the 2-state twin only in that BEAR is one of three states here;
    the slice `[:, :, BEAR]` still picks the bear prob alone -- TURBULENT_BULL is
    NOT counted as bear.
    """
    obs_times = jnp.arange(len(obs_frame), dtype=jnp.float32)
    obs_values = jnp.asarray(obs_frame[_OBS_COLS].to_numpy(), dtype=jnp.float32)

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
    title_prefix="Regime nowcast (3-state)",
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


import sys as _sys_wf  # noqa: E402
from _run_modes import make_walk_forward_p_bear as _make_wf  # noqa: E402

# Rolling-window OOS P(bear) refit. The logic is model-agnostic (it only calls this
# module's fit / filtered_p_bear_over), so it comes from the shared factory in
# _run_modes rather than a copy pasted into every model. WHY walk-forward: the vol
# level v_bear drifts across eras (episodic/crisis-driven); a single global fit averages
# calm and crisis bears into one v_bear that fits neither. Walk-forward fits ONLY the
# trailing window per fold, so no parameter is a 70-year blend. WINDOW=8yr: shortest that
# always contains >=1 P&S bear (so v_bear stays identifiable) while still turning over
# within a vol regime. Warm-started + causal: filter spans [train..test] so belief is
# warm entering the test block, but params come only from the fold's window and
# P(bear_t) uses only y_1:t, so every kept value is genuine OOS.
walk_forward_p_bear = _make_wf(_sys_wf.modules[__name__])


# ---- fit-mode runner hooks (consumed by _run_modes.run_main) --------------------
# The global-vs-walkforward machinery + save_run wiring live once in _run_modes; each
# model just exposes its config here. See _run_modes for the runner contract.
needs_macro = _NEEDS_MACRO           # load the credit/curve macro CSVs when those channels are on
obs_cols = _OBS_COLS                 # channel names -> run_name + saved spec


def obs_kwargs():
    return dict(_OBS_KWARGS)


def extra_spec():
    return {
        "include_drawdown": INCLUDE_DRAWDOWN,
        "drawdown_window_weeks": _DRAWDOWN_WINDOW_WEEKS,
        "include_credit": INCLUDE_CREDIT,
        "credit_horizon_months": _CREDIT_HORIZON_MONTHS,
        "include_curve": INCLUDE_CURVE,
    }


def main(mode: str | None = None) -> None:
    """Fit + evaluate + plot + save. `mode` selects the evaluation:
      'global'      -> single 80/20 fit (fast; train/test generalization).
      'walkforward' -> rolling 8yr refits (slow; non-stationarity-robust).
    mode=None falls back to the FIT_MODE module constant. CLI arg overrides both:
    `python regime_model_3state.py [global|walkforward]`.
    """
    import sys as _s
    from _run_modes import run_main
    run_main(_s.modules[__name__], mode if mode is not None else FIT_MODE)


if __name__ == "__main__":
    import sys as _s
    main(_s.argv[1] if len(_s.argv) > 1 else None)
