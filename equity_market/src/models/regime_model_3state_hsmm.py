"""Bayesian regime nowcast -- 3-STATE HIDDEN SEMI-MARKOV (HSMM) variant, expressed
as a plain HMM over a DURATION-AUGMENTED state space so it runs unchanged inside the
dynestyx HMM filter.

This is the 3-state HSMM. Its 2-state twin lives in regime_model_2state_hsmm.py; the
plain (non-HSMM) 3-state baseline is regime_model_3state.py. See docs/hsmm_plan.md.

WHAT THE THIRD STATE ADDS OVER THE 2-STATE HSMM. The 2-state HSMM already blocks the
whipsaw's TRANSITION half: a fresh bull cannot flip to bear for 17 weeks. But within
that protected window a turbulence spike has nowhere to live -- the 2-state bull
emission must stretch to cover both calm and turbulent weeks. The 3-state HSMM gives
that turbulence a HOME: a TURBULENT_BULL sub-state INSIDE the bull complex. The bull
and turbulent-bull sub-states SHARE ONE 17-week dwell clock (the "bull complex"), so:

  * During phases 1..16 (the protected window) the chain moves FREELY between BULL and
    TURBULENT_BULL, and the shared clock KEEPS COUNTING across those flips -- it does
    NOT reset when a turbulence spike happens (user decision). BEAR is unreachable.
  * At phase 17 (terminal) BEAR becomes reachable from EITHER BULL or TURBULENT_BULL
    (user decision -- no forced turbulence-first hop). Hard exit.

So a mid-recovery vol spike surfaces as TURBULENT_BULL rather than dragging P(bear)
up, and it still cannot escape to bear inside the 4-month floor. Comparing this to the
2-state HSMM answers: does a 4-month bull FLOOR alone tame the whipsaw, or is naming
the transient turbulence ALSO needed?

FLOORS (from the P&S phase-length histogram; see 2-state HSMM docstring):
  * BULL COMPLEX floor = 17 weeks (~4 months). Applies to the {BULL, TURBULENT_BULL}
    unit jointly (the shared clock), NOT to each sub-state separately.
  * BEAR floor = 5 weeks (small); emissions ("large drawdown") carry "however long".

STATE LAYOUT (K' = 39):
  (BULL,  phase 1..17)   indices  0..16   -- calm sub-state of the bull complex
  (TBULL, phase 1..17)   indices 17..33   -- turbulent sub-state, SAME shared clock
  bear_1..bear_5         indices 34..38   -- BEAR dwell phases (floor 5)
Each augmented state emits its REGIME's ordinary emission (BULL / TURBULENT_BULL /
BEAR). P(bear_t) = filtered mass summed over the 5 bear phases -- TURBULENT_BULL is
NOT counted as bear (the whole point of the third state).
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import DynamicalModel, Filter
from dynestyx.inference.filters import HMMConfig

# Regime labels (the 3 ORIGINAL regimes -- NOT the augmented states).
BEAR = 0
TURBULENT_BULL = 1
BULL = 2
K_REGIME = 3

# Dwell floors (weeks). BULL_FLOOR is the SHARED bull-complex clock length.
# [STRUCTURAL] hard integer constants -- NOT learned, NOT priors. They + the shared-
# clock graph below define the HSMM's dwell structure (see regime_model_2state.py for
# the PRIOR-vs-STRUCTURAL legend; notebooks/07 for the canonical example).
BULL_FLOOR = 17  # ~4 months; applies to {BULL, TURBULENT_BULL} jointly
BEAR_FLOOR = 5   # small; emissions carry "however long"

# Augmented state layout (K' = 2*BULL_FLOOR + BEAR_FLOOR = 39):
#   BULL  phases : indices [0, BULL_FLOOR)
#   TBULL phases : indices [BULL_FLOOR, 2*BULL_FLOOR)
#   BEAR  phases : indices [2*BULL_FLOOR, 2*BULL_FLOOR + BEAR_FLOOR)
_BULL_PHASES = list(range(0, BULL_FLOOR))
_TBULL_PHASES = list(range(BULL_FLOOR, 2 * BULL_FLOOR))
_BEAR_PHASES = list(range(2 * BULL_FLOOR, 2 * BULL_FLOOR + BEAR_FLOOR))
K = 2 * BULL_FLOOR + BEAR_FLOOR

# Per-augmented-state REGIME id (pick emission + sum P(bear)).
_REGIME_OF = jnp.array(
    [BULL] * BULL_FLOOR + [TURBULENT_BULL] * BULL_FLOOR + [BEAR] * BEAR_FLOOR
)
# Convenient endpoints.
_BULL_ENTRY = _BULL_PHASES[0]      # (BULL, phase 1) -- where bear exits INTO
_BULL_TERM = _BULL_PHASES[-1]      # (BULL, phase 17)
_TBULL_TERM = _TBULL_PHASES[-1]    # (TBULL, phase 17)
_BEAR_ENTRY = _BEAR_PHASES[0]      # bear_1
_BEAR_TERM = _BEAR_PHASES[-1]      # bear_5

# PRIOR CENTERS for the drift + v_t channels (prior locations, NOT hard constants; the
# model learns around them). See plain models for provenance.
_PRIOR_BULL_DRIFT_MEAN = 0.003
_PRIOR_BEAR_DRIFT_MEAN = -0.003
_PRIOR_DRIFT_GAP = _PRIOR_BULL_DRIFT_MEAN - _PRIOR_BEAR_DRIFT_MEAN
_PRIOR_CALM_LOG_VOL_MEAN = -5.34
_PRIOR_TURBULENT_LOG_VOL_MEAN = -4.30
_PRIOR_LOG_VOL_GAP = _PRIOR_TURBULENT_LOG_VOL_MEAN - _PRIOR_CALM_LOG_VOL_MEAN
_PRIOR_LOG_VOL_WITHIN_STD = 0.40


class _JointRV(dist.Distribution):
    """Conditional-independence joint of [r_t, v_t] (StudentT on r_t, Normal on v_t).
    Identical to the plain models' _JointRV. See regime_model_2state.py for rationale.
    """

    support = dist.constraints.real_vector

    def __init__(self, r_dist, v_dist):
        self.r_dist = r_dist
        self.v_dist = v_dist
        super().__init__(batch_shape=(), event_shape=(2,))

    def log_prob(self, value):
        r = value[..., 0]
        v = value[..., 1]
        return jnp.stack([self.r_dist.log_prob(r), self.v_dist.log_prob(v)], axis=-1)

    def sample(self, key, sample_shape=()):
        kr, kv = jr.split(key)
        r = self.r_dist.sample(kr, sample_shape)
        v = self.v_dist.sample(kv, sample_shape)
        return jnp.stack([r, v], axis=-1)


def _build_augmented_A(p_self_bull, p_self_bear, q_turb):
    """Assemble the K'xK' augmented transition matrix.

    Params (all regime-level, learned):
      p_self_bull : geometric-tail hazard of the BULL COMPLEX past its 17-wk floor --
                    at the terminal phase, prob of staying in the complex vs exiting
                    to bear.
      p_self_bear : geometric-tail hazard of BEAR past its 5-wk floor.
      q_turb      : within the bull complex, the probability that the NEXT week is
                    TURBULENT_BULL (vs BULL). One shared propensity-to-be-turbulent;
                    this is what lets a spike surface as TBULL and then subside back to
                    BULL, all on the SAME clock.

    Bull complex (SHARED clock, phases 1..17), for sub in {BULL, TBULL} at phase i:
      * i < 17 : advance to phase i+1, landing in TBULL w.p. q_turb else BULL. BEAR
                 unreachable. (Clock always advances; sub-state is redrawn each week.)
      * i = 17 : STAY in the complex w.p. p_self_bull -- landing at phase 17 again,
                 TBULL w.p. q_turb else BULL (geometric tail, sub-state still free) --
                 or EXIT to bear_1 w.p. 1 - p_self_bull.

    Bear (phases 1..5):
      * j < 5 : advance to bear_{j+1}.
      * j = 5 : stay bear_5 w.p. p_self_bear, or exit to (BULL, phase 1) w.p.
                1 - p_self_bear.  (Bear exits into the CALM sub-state at the start of a
                fresh bull complex -- the clock resets to 1, protecting the new bull.)

    Every row sums to 1 by construction.
    """
    zero = jnp.float32(0.0)
    one = jnp.float32(1.0)
    rows = []

    def bull_idx(i):   # augmented index of (BULL, phase i)  [i is 0-based]
        return _BULL_PHASES[i]

    def tbull_idx(i):  # augmented index of (TBULL, phase i)
        return _TBULL_PHASES[i]

    for k in range(K):
        row = [zero] * K
        if k in _BULL_PHASES or k in _TBULL_PHASES:
            # Position within the shared clock (0-based phase).
            i = (k - _BULL_ENTRY) if k in _BULL_PHASES else (k - _TBULL_PHASES[0])
            if i < BULL_FLOOR - 1:
                # Forced advance to phase i+1; sub-state redrawn (TBULL w.p. q_turb).
                row[tbull_idx(i + 1)] = q_turb
                row[bull_idx(i + 1)] = one - q_turb
            else:
                # Terminal phase 17: stay in complex (geom) w.p. p_self_bull, sub-state
                # free; else exit to bear_1.
                row[tbull_idx(BULL_FLOOR - 1)] = p_self_bull * q_turb
                row[bull_idx(BULL_FLOOR - 1)] = p_self_bull * (one - q_turb)
                row[_BEAR_ENTRY] = one - p_self_bull
        else:
            # Bear phase j (0-based within the bear block).
            j = k - _BEAR_ENTRY
            if j < BEAR_FLOOR - 1:
                row[k + 1] = one
            else:
                row[_BEAR_TERM] = p_self_bear
                row[_BULL_ENTRY] = one - p_self_bear  # exit into (BULL, phase 1)
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def regime_model(obs_times=None, obs_values=None, predict_times=None):
    """3-state HSMM: duration-augmented HMM with a shared bull-complex clock.

    Regime-level emission priors mirror the plain 3-state model AT THE TIME THIS FILE
    was written (drift ladder bear<tbull<bull; {BEAR,TBULL} SHARE the turbulent v_t
    mean, BULL calm). NOTE: this predates the plain 3-state's later VOL-GAP FIX (BEAR
    v_t mean > TBULL); this HSMM still shares the mean, so it would carry the same
    ~0.71 under-confidence -- irrelevant here because the HSMM is a documented negative
    result (see module NOTE / docs/hsmm_plan.md), not a shipped model.

    ============================================================================
    PRIOR vs STRUCTURAL (full legend in regime_model_2state.py; ref notebooks/07).
      * The SHARED BULL-COMPLEX CLOCK, the floors, and the whole augmented graph built
        by _build_augmented_A are [STRUCTURAL] -- fixed constants + a fixed transition
        graph (which augmented state may follow which). This is the HSMM.
      * p_self_bull, p_self_bear, q_turb are [PRIOR] (learnable), but they parameterize
        that structural graph: the p_self_* are geometric-tail hazards AFTER the floor,
        q_turb is the within-complex BULL-vs-TBULL split. The graph SHAPE around them
        is structural.
      * Emission params below carry the same PRIOR/STRUCTURAL tags as the plain model.
    ============================================================================
    """
    # [PRIOR] bull-complex & bear geometric-tail hazards (past their structural floors).
    p_self_bull = numpyro.sample("p_self_bull", dist.Beta(500.0, 3.0))
    p_self_bear = numpyro.sample("p_self_bear", dist.Beta(500.0, 3.0))
    # [PRIOR] q_turb: learnable within-complex turbulence propensity (BULL vs TBULL).
    q_turb = numpyro.sample("q_turb", dist.Beta(1.0, 4.0))
    # [STRUCTURAL] A: the shared-clock augmented transition graph (floors + gates +
    # the bull-complex sub-state coupling). This graph IS the model's novelty.
    A = _build_augmented_A(p_self_bull, p_self_bear, q_turb)

    # --- Emission params: verbatim from the plain 3-state model, at REGIME level ---
    # Drift ladder bear < tbull < bull (positive gaps -> label-switching guard).
    mean_return_bear = numpyro.sample("mean_return_bear", dist.Normal(_PRIOR_BEAR_DRIFT_MEAN, 0.01))
    drift_gap1 = numpyro.sample("drift_gap1", dist.HalfNormal(2.0 * _PRIOR_DRIFT_GAP))
    drift_gap2 = numpyro.sample("drift_gap2", dist.HalfNormal(2.0 * _PRIOR_DRIFT_GAP))
    mean_return_tbull = mean_return_bear + drift_gap1
    mean_return_bull = mean_return_tbull + drift_gap2
    # Regime-indexed [BEAR, TURBULENT_BULL, BULL].
    mean_return_regime = jnp.stack([mean_return_bear, mean_return_tbull, mean_return_bull])

    # Return-vol: BEAR & TBULL high (shared), BULL low.
    return_vol_low = numpyro.sample("return_vol_low", dist.HalfNormal(0.02))
    return_vol_high_offset = numpyro.sample("return_vol_high_offset", dist.HalfNormal(0.02))
    return_vol_high = return_vol_low + return_vol_high_offset
    return_vol_regime = jnp.stack([return_vol_high, return_vol_high, return_vol_low])

    tail_dof = 2.0 + numpyro.sample("tail_dof_raw", dist.Gamma(2.0, 0.1))

    # v_t: {BEAR, TBULL} share the turbulent mean; BULL calm.
    log_vol_calm = numpyro.sample("log_vol_calm", dist.Normal(_PRIOR_CALM_LOG_VOL_MEAN, 0.3))
    log_vol_gap = numpyro.sample("log_vol_gap", dist.HalfNormal(_PRIOR_LOG_VOL_GAP))
    log_vol_turbulent = log_vol_calm + log_vol_gap
    v_loc_regime = jnp.stack([log_vol_turbulent, log_vol_turbulent, log_vol_calm])

    v_scale = numpyro.sample("v_scale", dist.HalfNormal(_PRIOR_LOG_VOL_WITHIN_STD))

    # Broadcast regime-level params across augmented (regime, phase) states.
    mean_return = mean_return_regime[_REGIME_OF]  # (K',)
    return_vols = return_vol_regime[_REGIME_OF]   # (K',)
    v_loc = v_loc_regime[_REGIME_OF]              # (K',)

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

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
    """Fit the 3-state HSMM via NUTS. Same interface as the plain models."""
    obs_times = jnp.arange(len(train_obs), dtype=jnp.float32)
    obs_values = jnp.asarray(train_obs[["r_t", "v_t"]].to_numpy(), dtype=jnp.float32)

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
    """(num_samples, T, K') -> posterior-averaged P(bear_t) by summing over the 5
    BEAR phases. TURBULENT_BULL is NOT counted as bear (the point of the 3rd state).
    """
    bear_mask = (_REGIME_OF == BEAR)
    return filtered_states[:, :, bear_mask].sum(axis=-1).mean(axis=0)


def filtered_p_bear(mcmc) -> jnp.ndarray:
    """Posterior-averaged filtered P(BEAR_t | y_1:t) over TRAINING weeks."""
    return _sum_bear_phases(mcmc.get_samples()["f_filtered_states"])


def filtered_p_bear_over(mcmc, obs_frame, seed: int = 1) -> jnp.ndarray:
    """Filtered P(BEAR_t | y_1:t) over an arbitrary frame with TRAIN-learned params
    (causal Predictive filter -- genuine OOS). Sums over bear phases.
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
    filtered = predictive(jr.PRNGKey(seed))["f_filtered_states"]
    return _sum_bear_phases(filtered)


# Reuse the plain 2-state plot (identical figure, different P(bear) input). Wrapped so
# this model's title_prefix is applied by the shared runner.
from regime_model_2state import plot_regime_fit as _plot_regime_fit_base  # noqa: E402


def plot_regime_fit(*args, **kwargs):
    kwargs.setdefault("title_prefix", "Regime nowcast (3-state HSMM, shared 17wk bull-complex clock / 5wk bear floor)")
    return _plot_regime_fit_base(*args, **kwargs)


# ---- fit-mode runner hooks (see _run_modes.run_main) ----------------------------
import sys as _sys_wf  # noqa: E402
from _run_modes import make_walk_forward_p_bear as _make_wf  # noqa: E402

walk_forward_p_bear = _make_wf(_sys_wf.modules[__name__])

needs_macro = False
obs_cols = ["r_t", "v_t"]

# DEFAULT fit mode with no CLI arg: "global" (fast 80/20) | "walkforward" (slow rolling
# refits, non-stationarity-robust). HOW TO USE: edit this to change the default and run
# `python regime_model_3state_hsmm.py`; OR override per-run: `... walkforward` (CLI wins).
FIT_MODE = "global"


def obs_kwargs():
    return {}


def extra_spec():
    return {}


def main(mode: str | None = None) -> None:
    """Fit + evaluate + plot + save. mode: 'global' (80/20, fast) | 'walkforward'
    (rolling refits, slow); None -> FIT_MODE. CLI arg overrides:
    `python regime_model_3state_hsmm.py [global|walkforward]`."""
    from _run_modes import run_main
    run_main(_sys_wf.modules[__name__], mode if mode is not None else FIT_MODE)


if __name__ == "__main__":
    import sys as _s
    main(_s.argv[1] if len(_s.argv) > 1 else None)
