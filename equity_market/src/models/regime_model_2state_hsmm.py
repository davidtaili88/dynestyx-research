"""Bayesian regime nowcast -- 2-STATE HIDDEN SEMI-MARKOV (HSMM) variant, expressed
as a plain HMM over a DURATION-AUGMENTED state space so it runs unchanged inside
the dynestyx HMM filter.

WHY THIS FILE EXISTS. The plain 2-state model (regime_model_2state.py) whipsaws:
P(bear) flips to ~1 on any large drawdown even when it is just a bull-market dip,
because a plain HMM's dwell time is GEOMETRIC -- P(leave a regime) is identical on
week 1 and week 100, so nothing stops it leaving a fresh bull the very next week.
No persistence prior can fix that; it is the model class. The fix is a MINIMUM-DWELL
floor: forbid leaving a regime until it has lasted N weeks. A standard HMM cannot
represent "time-in-state", so we AUGMENT the state with a dwell-phase counter and run
the ordinary filter over the enlarged space (see docs/hsmm_plan.md, and
docs/duration_dependence_investigation.md for why augmentation is the only route).

FLOORS (from the P&S phase-length histogram, 1957-2026):
  * BULL floor = 17 weeks (~4 months). The user's rule "a bull, once entered, cannot
    flip for 4 months". Safe: the shortest real P&S bull is 27 weeks, so a 17-wk
    floor never forbids a genuine transition.
  * BEAR floor = 5 weeks (small). "Bears however long IF the drawdown is large" is
    NOT a dwell rule -- "however long" means little floor, and "if large enough" is
    an EMISSION condition the return channel already handles. So the 5-wk floor only
    blocks 1-week bear blips; deep negative returns (the likelihood) are what hold a
    real bear and let P(bear) reach ~1, and a return recovery lets bear exit promptly.

STATE LAYOUT (K' = 22):
  bull_1..bull_17  (indices 0..16)   -- BULL dwell phases (floor 17)
  bear_1..bear_5   (indices 17..21)  -- BEAR dwell phases (floor 5)
Each augmented state (regime, phase) EMITS its regime's ordinary emission -- the
phase is a clock, not a new regime. P(bear_t) = filtered mass summed over ALL bear
phases (see filtered_p_bear).

TRANSITIONS (exact min-dwell / Erlang staircase, HARD exit):
  * Inside the floor the phase MUST advance: phase_i -> phase_{i+1} with prob 1 for
    i < floor. The chain cannot leave and cannot stall, so the minimum dwell is
    EXACTLY `floor` weeks -- P(switch before the floor) is literally 0.
  * At the TERMINAL phase (bull_17 / bear_5) the regime either self-loops (stay,
    resetting to... itself at the terminal phase -- see note) with prob p_self, or
    EXITS to the other regime's phase 1 with prob 1 - p_self.
  * "Self-loop at terminal" means: once past its floor, a regime persists geometrically
    (constant hazard) exactly like the plain HMM -- the ONLY change vs plain is the
    hard floor before that geometric tail. So dwell = floor + Geometric(1 - p_self):
    a minimum of `floor` weeks, then the usual memoryless persistence.

This is the 2-state HSMM. Its 3-state twin (regime_model_3state_hsmm.py) adds a
TURBULENT_BULL sub-state that shares the bull complex's 17-wk clock. Comparing the
two answers: does a 4-month bull FLOOR alone tame the whipsaw, or is the transient-
turbulence state ALSO needed?
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

# Regime labels (the ORIGINAL 2 regimes -- NOT the augmented states).
BEAR = 0
BULL = 1
K_REGIME = 2

# Dwell FLOORS (weeks). See module docstring for provenance.
# [STRUCTURAL] hard integer constants (like the notebook's die probs) -- NOT learned,
# NOT priors. They define the minimum dwell the augmented graph enforces; changing
# them changes the model's expressible dwell distributions.
BULL_FLOOR = 17  # ~4 months; user rule, safe vs 27-wk shortest real P&S bull
BEAR_FLOOR = 5   # small; emissions ("large drawdown") carry "however long"

# Augmented state layout. bull phases first (so BULL keeps a contiguous block), then
# bear phases. K' = BULL_FLOOR + BEAR_FLOOR.
_BULL_PHASES = list(range(0, BULL_FLOOR))                     # indices 0..16
_BEAR_PHASES = list(range(BULL_FLOOR, BULL_FLOOR + BEAR_FLOOR))  # indices 17..21
K = BULL_FLOOR + BEAR_FLOOR  # 22 augmented states

# Per-augmented-state REGIME id, used to (a) pick the emission and (b) sum P(bear).
# _REGIME_OF[k] in {BEAR, BULL} for augmented state k.
_REGIME_OF = jnp.array([BULL] * BULL_FLOOR + [BEAR] * BEAR_FLOOR)
# Index of each regime's phase-1 (entry) augmented state, and terminal phase.
_BULL_ENTRY, _BULL_TERMINAL = _BULL_PHASES[0], _BULL_PHASES[-1]
_BEAR_ENTRY, _BEAR_TERMINAL = _BEAR_PHASES[0], _BEAR_PHASES[-1]

# PRIOR CENTERS for the drift channel (prior locations, NOT hard constants; the model
# learns around them). See plain 2-state model for provenance.
_PRIOR_BULL_DRIFT_MEAN = 0.003
_PRIOR_BEAR_DRIFT_MEAN = -0.003
_PRIOR_DRIFT_GAP = _PRIOR_BULL_DRIFT_MEAN - _PRIOR_BEAR_DRIFT_MEAN

# PRIOR CENTERS for the v_t channel (prior locations, not hard constants). See plain
# 2-state model for provenance.
_PRIOR_CALM_LOG_VOL_MEAN = -5.34
_PRIOR_TURBULENT_LOG_VOL_MEAN = -4.30
_PRIOR_LOG_VOL_GAP = _PRIOR_TURBULENT_LOG_VOL_MEAN - _PRIOR_CALM_LOG_VOL_MEAN
_PRIOR_LOG_VOL_WITHIN_STD = 0.40


class _JointRV(dist.Distribution):
    """Conditional-independence joint of [r_t, v_t] -- identical to the plain
    2-state model's _JointRV (StudentT on r_t, Normal on v_t; log_prob returns the
    length-2 per-dimension vector the HMM filter sums). See regime_model_2state.py
    for the full rationale.
    """

    support = dist.constraints.real_vector

    def __init__(self, r_dist, v_dist):
        self.r_dist = r_dist
        self.v_dist = v_dist
        super().__init__(batch_shape=(), event_shape=(2,))

    def log_prob(self, value):
        r = value[..., 0]
        v = value[..., 1]
        return jnp.stack(
            [self.r_dist.log_prob(r), self.v_dist.log_prob(v)], axis=-1
        )

    def sample(self, key, sample_shape=()):
        kr, kv = jr.split(key)
        r = self.r_dist.sample(kr, sample_shape)
        v = self.v_dist.sample(kv, sample_shape)
        return jnp.stack([r, v], axis=-1)


def _build_augmented_A(p_self_bull, p_self_bear):
    """Assemble the K'xK' augmented transition matrix from the two regime-level
    self-persistences.

    Exact min-dwell staircase, HARD exit:
      * bull_i -> bull_{i+1} with prob 1 for i in 1..16  (forced advance in floor)
      * bull_17 -> bull_17 with p_self_bull  (stay: geometric tail past the floor)
                -> bear_1  with 1 - p_self_bull  (exit to the OTHER regime's phase 1)
      * bear_j -> bear_{j+1} with prob 1 for j in 1..4
      * bear_5 -> bear_5 with p_self_bear
                -> bull_1 with 1 - p_self_bear
    Every row sums to 1 by construction.

    NOTE on the terminal self-loop: staying at the terminal phase (rather than
    re-entering at phase 1) is what makes the post-floor dwell GEOMETRIC -- i.e.
    dwell = floor + Geometric(1 - p_self). If we instead looped terminal->phase_1 the
    dwell would be a multiple of the floor, which is not what we want.
    """
    # Build with a plain-Python nested list of scalars (constants where fixed, the
    # traced p_self where variable), then jnp.stack. Each row is a length-K list; the
    # single variable entry per terminal row is a traced scalar, all others are 0/1.
    rows = []
    for k in range(K):
        row = [jnp.float32(0.0)] * K
        if k < _BULL_TERMINAL:  # bull_i, i<17: forced advance
            row[k + 1] = jnp.float32(1.0)
        elif k == _BULL_TERMINAL:  # bull_17: stay (geom) or exit to bear_1
            row[_BULL_TERMINAL] = p_self_bull
            row[_BEAR_ENTRY] = 1.0 - p_self_bull
        elif k < _BEAR_TERMINAL:  # bear_j, j<5: forced advance
            row[k + 1] = jnp.float32(1.0)
        else:  # bear_5: stay (geom) or exit to bull_1
            row[_BEAR_TERMINAL] = p_self_bear
            row[_BULL_ENTRY] = 1.0 - p_self_bear
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def regime_model(obs_times=None, obs_values=None, predict_times=None):
    """2-state HSMM: duration-augmented HMM. Regime-level priors are IDENTICAL to
    the plain 2-state model; the only structural change is the augmented transition
    matrix (min-dwell floors) in place of the plain 2x2.

    Identification axis is DRIFT (mu_bull > mu_bear), same as the plain model; v_t
    does the turbulent/calm separation. Both act at the REGIME level and are then
    broadcast across that regime's dwell phases.

    ============================================================================
    PRIOR vs STRUCTURAL (see regime_model_2state.py for the full legend; reference
    notebooks/07_hidden_markov_model.ipynb). The instructive HSMM-specific point:
      * The DWELL FLOORS (BULL_FLOOR=17, BEAR_FLOOR=5) and the whole phase-augmented
        transition SHAPE built by _build_augmented_A are [STRUCTURAL] -- hard integer
        constants + a fixed graph of which augmented state can follow which. They are
        NOT learned and NOT priors; they change what dwell distributions the model can
        express. This IS the HSMM (vs the plain 2x2).
      * p_self here is a [PRIOR] but its MEANING changed structurally: it is now only
        the GEOMETRIC-TAIL hazard AFTER the floor, because the floor (structural) has
        already consumed the first BULL_FLOOR/BEAR_FLOOR weeks.
      * Emission params below are identical to the plain 2-state (same PRIOR/STRUCTURAL
        tags); they are broadcast across phases by _REGIME_OF ([STRUCTURAL] mapping).
    ============================================================================
    """
    # [PRIOR] p_self: learnable, but now = the geometric-tail hazard AFTER the floor
    # (the structural floor supplies the hard minimum; p_self the memoryless tail).
    p_self = numpyro.sample("p_self", dist.Beta(500.0, 3.0).expand([K_REGIME]).to_event(1))
    # [STRUCTURAL] A: the augmented (regime, phase) transition graph -- floors + gates.
    A = _build_augmented_A(p_self[BULL], p_self[BEAR])

    # --- Emission params: verbatim from the plain 2-state model, at REGIME level ---
    mean_return_bear = numpyro.sample("mean_return_bear", dist.Normal(_PRIOR_BEAR_DRIFT_MEAN, 0.01))
    drift_gap = numpyro.sample("drift_gap", dist.HalfNormal(4.0 * _PRIOR_DRIFT_GAP))
    mean_return_bull = mean_return_bear + drift_gap
    mean_return_regime = jnp.stack([mean_return_bear, mean_return_bull])  # [BEAR, BULL]

    return_vol = numpyro.sample("return_vol", dist.HalfNormal(0.02))
    return_vol_regime = jnp.stack([return_vol, return_vol])

    tail_dof = 5.0

    log_vol_calm = numpyro.sample("log_vol_calm", dist.Normal(_PRIOR_CALM_LOG_VOL_MEAN, 0.3))
    log_vol_gap = numpyro.sample("log_vol_gap", dist.HalfNormal(_PRIOR_LOG_VOL_GAP))
    log_vol_turbulent = log_vol_calm + log_vol_gap
    v_loc_regime = jnp.stack([log_vol_turbulent, log_vol_calm])  # [BEAR, BULL]

    vol_stdev = numpyro.sample("vol_stdev", dist.HalfNormal(_PRIOR_LOG_VOL_WITHIN_STD))

    # BROADCAST regime-level emission params across the augmented (regime, phase)
    # states via _REGIME_OF: augmented state k emits its regime's params.
    mean_return = mean_return_regime[_REGIME_OF]   # (K',)
    return_vols = return_vol_regime[_REGIME_OF]    # (K',)
    v_loc = v_loc_regime[_REGIME_OF]               # (K',)

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    def observation_model(x, u, t):
        return _JointRV(
            r_dist=dist.StudentT(df=tail_dof, loc=mean_return[x], scale=return_vols[x]),
            v_dist=dist.Normal(loc=v_loc[x], scale=vol_stdev),
        )

    dynamics = DynamicalModel(
        # Uniform over the 22 augmented states. (A regime-uniform init would also be
        # fine; uniform-over-phases is the neutral default and washes out fast.)
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
    """Fit the 2-state HSMM on the training weekly frame via NUTS. Interface is
    identical to the plain 2-state fit(); only the model differs.
    """
    obs_times = jnp.arange(len(train_obs), dtype=jnp.float32)
    obs_values = jnp.asarray(train_obs[["r_t", "v_t"]].to_numpy(), dtype=jnp.float32)  # (T,2)

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


def _sum_bear_phases(filtered_states):
    """Reduce (num_samples, T, K') augmented filtered probs to P(bear) by summing
    the filtered mass over ALL bear phases, then posterior-averaging.

    This is the key HSMM-specific readout: 'bear' is a REGIME spread over BEAR_FLOOR
    augmented phases, so P(bear_t) is the total mass on any bear phase -- not a single
    state index like in the plain model.
    """
    bear_mask = (_REGIME_OF == BEAR)  # (K',) boolean
    p_bear_per_sample = filtered_states[:, :, bear_mask].sum(axis=-1)  # (num_samples, T)
    return p_bear_per_sample.mean(axis=0)  # (T,)


def filtered_p_bear(mcmc) -> jnp.ndarray:
    """Posterior-averaged filtered P(BEAR_t | y_1:t) over the TRAINING weeks --
    summed over all bear dwell phases. See _sum_bear_phases.
    """
    return _sum_bear_phases(mcmc.get_samples()["f_filtered_states"])


def filtered_p_bear_over(mcmc, obs_frame, seed: int = 1) -> jnp.ndarray:
    """Filtered P(BEAR_t | y_1:t) over an ARBITRARY weekly frame using TRAIN-learned
    params (causal forward filter via Predictive -- genuine out-of-sample). Same
    mechanism as the plain 2-state twin, but reduces over bear phases.
    """
    obs_times = jnp.arange(len(obs_frame), dtype=jnp.float32)
    obs_values = jnp.asarray(obs_frame[["r_t", "v_t"]].to_numpy(), dtype=jnp.float32)

    def conditioned_model():
        with Filter(filter_config=HMMConfig(record_filtered=True)):
            return regime_model(obs_times=obs_times, obs_values=obs_values)

    predictive = Predictive(
        conditioned_model,
        posterior_samples=mcmc.get_samples(),
        return_sites=["f_filtered_states"],
    )
    filtered = predictive(jr.PRNGKey(seed))["f_filtered_states"]  # (num_samples, T, K')
    return _sum_bear_phases(filtered)


# plot_regime_fit is imported from the plain 2-state model -- identical figure, just a
# different P(bear) input. Wrapped so this model's title_prefix is applied by the shared
# runner (which calls model.plot_regime_fit without a title_prefix).
from regime_model_2state import plot_regime_fit as _plot_regime_fit_base  # noqa: E402


def plot_regime_fit(*args, **kwargs):
    kwargs.setdefault("title_prefix", "Regime nowcast (2-state HSMM, 17wk bull / 5wk bear floor)")
    return _plot_regime_fit_base(*args, **kwargs)


# ---- fit-mode runner hooks (see _run_modes.run_main) ----------------------------
import sys as _sys_wf  # noqa: E402
from _run_modes import make_walk_forward_p_bear as _make_wf  # noqa: E402

walk_forward_p_bear = _make_wf(_sys_wf.modules[__name__])

needs_macro = False
obs_cols = ["r_t", "v_t"]

# DEFAULT fit mode with no CLI arg: "global" (fast 80/20) | "walkforward" (slow rolling
# refits, non-stationarity-robust). HOW TO USE: edit this to change the default and run
# `python regime_model_2state_hsmm.py`; OR override per-run: `... walkforward` (CLI wins).
FIT_MODE = "global"


def obs_kwargs():
    return {}


def extra_spec():
    return {}


def main(mode: str | None = None) -> None:
    """Fit + evaluate + plot + save. mode: 'global' (80/20, fast) | 'walkforward'
    (rolling refits, slow); None -> FIT_MODE. CLI arg overrides:
    `python regime_model_2state_hsmm.py [global|walkforward]`."""
    from _run_modes import run_main
    run_main(_sys_wf.modules[__name__], mode if mode is not None else FIT_MODE)


if __name__ == "__main__":
    import sys as _s
    main(_s.argv[1] if len(_s.argv) > 1 else None)
