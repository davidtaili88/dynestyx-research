"""Bayesian regime nowcast -- 4-STATE discrete-time HMM, BIVARIATE emission on
(r_t, v_t): weekly log return and log intra-week realized vol.

WHY 4 STATES (the calm-bear fix, 2026-07-26). The 3-state model
(regime_model_3state.py) defines BEAR as HIGH-vol, so its "bear detector" is really a
TURBULENCE detector: it catches only the 4 violent P&S bears (1987, 2000-02, 2008,
2022; v_t ~ -4.3) and is BLIND to the 14 calm/grinding bears (1957, 1973-74(-48%!),
1977, 1980-82, ...; ~77% of P&S bear-weeks) whose v_t ~ -5.2 is INDISTINGUISHABLE from
bull markets (bull v_t ~ -5.2). Those weeks DO have negative drift, but vol is the
strong channel and shouts "calm" -> P(bear) ~ 0.03. (See calm_bears.py/calm_bears_v2.py
and the [[calm-bear-blindness]] memo.)

The fix is a dedicated CALM-BEAR state: negative drift + BULL-like LOW vol, so a bear
can be identified by DRIFT ALONE, not by vol. Bears now come in two flavors:
  TURB_BEAR : negative drift, HIGH vol  -- the violent crash bear (what 3-state caught)
  CALM_BEAR : negative drift, LOW  vol  -- the calm grind bear (what 3-state MISSED)
and P(bear) = P(TURB_BEAR) + P(CALM_BEAR).

DESIGN TENSION (accepted): calm-bear needs LOW vol, which softens the vol-based
separation the 3-state used for confidence. But that separation was helping 4 bears
while hiding 14, so the trade favors catching the 14; drift (not vol) becomes the
primary bear identifier.

IDENTIFICATION -- data-measured per-flavor means (P&S x vol-median split), which fix the
label-switching orderings below:
                drift r_t    v_t (log vol)   return-std
  TURB_BEAR     -0.0070      -4.57 (high)    0.0298 (widest)
  CALM_BEAR     -0.0032      -5.61 (low)     0.0174
  TURB_BULL     +0.0029      -4.68 (high)    0.0245
  CALM_BULL     +0.0048      -5.64 (low)     0.0133 (narrowest)
  * DRIFT is a clean 4-RUNG ladder: turb_bear < calm_bear < turb_bull < calm_bull.
    THIS is the primary bear/bull identifier -- it separates calm_bear from calm_bull
    (which vol CANNOT: their v_t are -5.61 vs -5.64, twins).
  * VOL is a 3-LEVEL ladder: {calm_bear = calm_bull} < turb_bull < turb_bear. The two
    calm states share the low vol level; the two turbulent states stack above (turb_bear
    highest). The tiny turb_bear/turb_bull vol gap (0.11) may learn ~0 and harmlessly
    collapse to 2-level. Applied to BOTH the r_t return-vol scale and the v_t emission
    (return-vol's two calm states are ~equal too, 0.013 vs 0.017 -- approximated shared).

Follows the notebooks/07_hidden_markov_model.ipynb pattern; see regime_model_3state.py
for the PRIOR-vs-STRUCTURAL legend, the tail-handling (both channels StudentT) rationale,
and the conditional-independence discussion (all carried over unchanged here).
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

# 4-state layout, ORDERED BY DRIFT (ascending) so the drift ladder is a simple cumsum.
# The two BEAR flavors are indices 0,1 so P(bear) = filtered[...,:2].sum().
TURB_BEAR = 0   # -drift, HIGH vol  -- violent crash bear
CALM_BEAR = 1   # -drift, LOW  vol  -- calm grind bear (the new state)
TURB_BULL = 2   # +drift, HIGH vol  -- transient turbulence
CALM_BULL = 3   # +drift, LOW  vol  -- calm uptrend
K = 4
# Which states count as BEAR (for the P(bear) readout).
_BEAR_STATES = (TURB_BEAR, CALM_BEAR)
# Vol MEMBERSHIP: is each state turbulent (high vol) or calm (low vol)?
_IS_TURBULENT = jnp.array([1, 0, 1, 0])  # [turb_bear, calm_bear, turb_bull, calm_bull]

# DRAWDOWN CHANNEL (see regime_model_3state for the full breakthrough rationale). In the
# 4-state, drawdown is an especially natural fit: it is a BEAR-vs-BULL (price-level) signal,
# so BOTH bear flavors sit underwater and BOTH bull flavors ~0 -- and it reinforces the
# calm-bear state (a calm grind is still underwater though its vol looks bull-like).
INCLUDE_DRAWDOWN = True
_DRAWDOWN_WINDOW_WEEKS = 52  # 1yr (sweep-chosen)
_OBS_COLS = ["r_t", "v_t", "dd"] if INCLUDE_DRAWDOWN else ["r_t", "v_t"]

# PRIOR CENTERS -- round ballparks, NOT hard constants (data overrules them; see
# regime_model_3state.py for the full inert-value argument). Drift centers span the
# 4-rung ladder; vol centers the calm level and the turbulent gap.
_PRIOR_BOTTOM_DRIFT_MEAN = -0.007   # turb_bear rung (most negative), neutral-ish ballpark
_PRIOR_DRIFT_GAP = 0.004            # typical gap between adjacent drift rungs (~0.4%/wk)
_PRIOR_CALM_LOG_VOL_MEAN = -5.6     # shared calm v_t level (calm_bull ~ calm_bear)
_PRIOR_TURB_LOG_VOL_GAP = 1.0       # calm -> turbulent v_t jump (~ -4.6 minus -5.6)
_PRIOR_LOG_VOL_SPREAD = 0.4         # within-group v_t spread
_PRIOR_RETURN_VOL_CALM = 0.013      # calm return-vol level ballpark
_PRIOR_RETURN_VOL_TURB_GAP = 0.012  # calm -> turbulent return-vol jump


class _JointRV(dist.Distribution):
    """Conditional-independence joint of [r_t, v_t], both StudentT with their own df.
    Identical to regime_model_3state._JointRV -- returns the length-2 per-dimension
    log-prob vector the HMM filter sums. See that file for the full rationale and the
    conditional-independence tests (tested-and-vindicated there).
    """

    support = dist.constraints.real_vector

    def __init__(self, r_dist, v_dist, d_dist=None):
        # d_dist = OPTIONAL 3rd channel: DRAWDOWN (see regime_model_3state for the full
        # rationale). When present, event is [r_t, v_t, dd] and log_prob returns length-3.
        self.r_dist = r_dist
        self.v_dist = v_dist
        self.d_dist = d_dist
        super().__init__(batch_shape=(), event_shape=(2 if d_dist is None else 3,))

    def log_prob(self, value):
        parts = [self.r_dist.log_prob(value[..., 0]), self.v_dist.log_prob(value[..., 1])]
        if self.d_dist is not None:
            parts.append(self.d_dist.log_prob(value[..., 2]))
        return jnp.stack(parts, axis=-1)

    def sample(self, key, sample_shape=()):
        keys = jr.split(key, 3 if self.d_dist is not None else 2)
        parts = [self.r_dist.sample(keys[0], sample_shape), self.v_dist.sample(keys[1], sample_shape)]
        if self.d_dist is not None:
            parts.append(self.d_dist.sample(keys[2], sample_shape))
        return jnp.stack(parts, axis=-1)


def regime_model(obs_times=None, obs_values=None, predict_times=None):
    """4-state HMM: Categorical regime, bivariate (r_t, v_t) StudentT emission.

    States (drift-ordered): TURB_BEAR < CALM_BEAR < TURB_BULL < CALM_BULL.
    See module docstring for the calm-bear rationale and the measured orderings.
    Tagging convention ([PRIOR]/[STRUCTURAL]) matches regime_model_3state.py.
    """
    # --- Transitions: full learnable 4x4 (self-persist + Dirichlet over 3 destinations)
    # [PRIOR] p_self per state; [PRIOR] off_split (Dirichlet over the OTHER 3 states).
    # [STRUCTURAL] the row shape (self on diagonal, (1-self)*split elsewhere).
    p_self = numpyro.sample("p_self", dist.Beta(500.0, 3.0).expand([K]).to_event(1))
    off_split = numpyro.sample(
        "off_split", dist.Dirichlet(jnp.ones(K - 1)).expand([K]).to_event(1)
    )

    def _row(i):
        others = [j for j in range(K) if j != i]
        row = [None] * K
        row[i] = p_self[i]
        for d, w in zip(others, off_split[i]):
            row[d] = (1.0 - p_self[i]) * w
        return jnp.stack(row)

    A = jnp.stack([_row(i) for i in range(K)])

    # --- Drift: 4-RUNG ladder turb_bear < calm_bear < turb_bull < calm_bull.
    # [PRIOR] the base + 3 gaps (magnitudes); [STRUCTURAL] the ordered ladder built via
    # non-negative HalfNormal gaps -> ordering by construction (label-switching guard).
    # This is the PRIMARY bear/bull identifier -- it alone separates calm_bear from
    # calm_bull (vol cannot; their v_t are twins). -> [PRIOR+STRUCTURAL].
    drift_base = numpyro.sample("drift_base", dist.Normal(_PRIOR_BOTTOM_DRIFT_MEAN, 0.01))
    drift_gaps = numpyro.sample(
        "drift_gaps", dist.HalfNormal(2.0 * _PRIOR_DRIFT_GAP).expand([K - 1]).to_event(1)
    )
    # cumulative sum: rung 0 = base, rung k = base + sum(gaps[:k]).
    mean_return = drift_base + jnp.concatenate([jnp.zeros(1), jnp.cumsum(drift_gaps)])
    # mean_return indexed [TURB_BEAR, CALM_BEAR, TURB_BULL, CALM_BULL].

    # --- RETURN-VOL (r_t StudentT scale): 3-LEVEL ladder calm < turb_bull < turb_bear.
    # [PRIOR] the calm base + 2 turbulent gaps; [STRUCTURAL] the 3-level assignment:
    # both CALM states share the calm level; TURB_BULL sits a gap above; TURB_BEAR the
    # highest. -> [PRIOR+STRUCTURAL]. (Return-vol's two calm states are ~equal in data,
    # 0.013 vs 0.017, so sharing the calm level is a mild, accepted approximation.)
    rv_calm = numpyro.sample("return_vol_calm", dist.HalfNormal(_PRIOR_RETURN_VOL_CALM * 2))
    rv_gap_tbull = numpyro.sample("return_vol_gap_tbull", dist.HalfNormal(_PRIOR_RETURN_VOL_TURB_GAP))
    rv_gap_tbear = numpyro.sample("return_vol_gap_tbear", dist.HalfNormal(_PRIOR_RETURN_VOL_TURB_GAP))
    rv_turb_bull = rv_calm + rv_gap_tbull
    rv_turb_bear = rv_turb_bull + rv_gap_tbear
    # indexed [TURB_BEAR, CALM_BEAR, TURB_BULL, CALM_BULL]:
    return_vols = jnp.stack([rv_turb_bear, rv_calm, rv_turb_bull, rv_calm])

    # --- r_t tail df: StudentT tail-fatness backstop (see 3-state tail-handling note).
    # [PRIOR+STRUCTURAL] learned dof; `2.0 +` is the structural finite-variance floor.
    tail_dof = 2.0 + numpyro.sample("tail_dof_raw", dist.Gamma(2.0, 0.1))

    # --- v_t emission mean: 3-LEVEL ladder calm < turb_bull < turb_bear (same shape as
    # return-vol). [PRIOR] calm base + 2 gaps; [STRUCTURAL] the 3-level assignment (both
    # calm states SHARE the calm v_t level -- this is what makes calm_bear look bull-like
    # in vol, so DRIFT must do the bear ID). The tiny turb_bear/turb_bull gap (~0.11) may
    # learn ~0. -> [PRIOR+STRUCTURAL].
    v_calm = numpyro.sample("log_vol_calm", dist.Normal(_PRIOR_CALM_LOG_VOL_MEAN, 0.3))
    v_gap_tbull = numpyro.sample("log_vol_gap_tbull", dist.HalfNormal(_PRIOR_TURB_LOG_VOL_GAP))
    v_gap_tbear = numpyro.sample("log_vol_gap_tbear", dist.HalfNormal(0.3 * _PRIOR_TURB_LOG_VOL_GAP))
    v_turb_bull = v_calm + v_gap_tbull
    v_turb_bear = v_turb_bull + v_gap_tbear
    v_loc = jnp.stack([v_turb_bear, v_calm, v_turb_bull, v_calm])
    # [PRIOR] v_scale learnable; [STRUCTURAL] shared across states (one number).
    v_scale = numpyro.sample("v_scale", dist.HalfNormal(_PRIOR_LOG_VOL_SPREAD))
    # v_t tail df -- StudentT backstop, own df (see 3-state note; v_t is the fatter channel).
    v_tail_dof = 2.0 + numpyro.sample("v_tail_dof_raw", dist.Gamma(2.0, 0.1))

    # DRAWDOWN channel (see regime_model_3state for the breakthrough rationale). 2-level
    # dd-mean by BEAR-vs-BULL: BOTH bear flavors underwater (dd_bull - bear_gap), BOTH bull
    # flavors ~0. This is a cleaner fit than the 3-state's per-state ladder because drawdown
    # is inherently a bear/bull (price-level) signal -- and it reinforces CALM_BEAR (a calm
    # grind is still underwater though its vol is bull-like). [PRIOR] the magnitudes;
    # [STRUCTURAL] the bear-underwater / bull-~0 assignment.
    if INCLUDE_DRAWDOWN:
        dd_bull = numpyro.sample("dd_bull", dist.Normal(0.0, 0.05))         # bulls ~0 (near peak)
        dd_bear_gap = numpyro.sample("dd_bear_gap", dist.HalfNormal(0.20))  # how far underwater bears are
        # [TURB_BEAR, CALM_BEAR, TURB_BULL, CALM_BULL]: both bears underwater, both bulls ~0.
        dd_loc = jnp.stack([dd_bull - dd_bear_gap, dd_bull - dd_bear_gap, dd_bull, dd_bull])
        dd_scale = numpyro.sample("dd_scale", dist.HalfNormal(0.15))

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    # [STRUCTURAL] emission families: r_t ~ StudentT, v_t ~ StudentT, (+ dd ~ Normal when
    # enabled), conditional independence (tested-and-vindicated in the 3-state model).
    def observation_model(x, u, t):
        d_dist = dist.Normal(loc=dd_loc[x], scale=dd_scale) if INCLUDE_DRAWDOWN else None
        return _JointRV(
            r_dist=dist.StudentT(df=tail_dof, loc=mean_return[x], scale=return_vols[x]),
            v_dist=dist.StudentT(df=v_tail_dof, loc=v_loc[x], scale=v_scale),
            d_dist=d_dist,
        )

    dynamics = DynamicalModel(
        # [STRUCTURAL] fixed uniform initial belief over the 4 states.
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
    """Fit the 4-state model via NUTS. Same interface as the 3-state fit()."""
    obs_times = jnp.arange(len(train_obs), dtype=jnp.float32)
    obs_values = jnp.asarray(train_obs[_OBS_COLS].to_numpy(), dtype=jnp.float32)

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


def _sum_bear(filtered_states):
    """(num_samples, T, K) -> posterior-averaged P(bear) by summing BOTH bear flavors
    (TURB_BEAR + CALM_BEAR). This is the whole point of the 4-state model: a calm bear
    counts as bear even though its vol is bull-like.
    """
    bear = filtered_states[:, :, TURB_BEAR] + filtered_states[:, :, CALM_BEAR]
    return bear.mean(axis=0)


def filtered_p_bear(mcmc) -> jnp.ndarray:
    """Posterior-averaged filtered P(bear_t | y_1:t) over TRAINING weeks (both flavors)."""
    return _sum_bear(mcmc.get_samples()["f_filtered_states"])


def filtered_p_bear_over(mcmc, obs_frame, seed: int = 1) -> jnp.ndarray:
    """Filtered P(bear_t) over an arbitrary frame with TRAIN-learned params (causal
    Predictive filter -- genuine OOS). Sums both bear flavors.
    """
    obs_times = jnp.arange(len(obs_frame), dtype=jnp.float32)
    obs_values = jnp.asarray(obs_frame[_OBS_COLS].to_numpy(), dtype=jnp.float32)

    def conditioned_model():
        with Filter(filter_config=HMMConfig(record_filtered=True)):
            return regime_model(obs_times=obs_times, obs_values=obs_values)

    predictive = Predictive(
        conditioned_model,
        posterior_samples=mcmc.get_samples(),
        return_sites=["f_filtered_states"],
    )
    filtered = predictive(jr.PRNGKey(seed))["f_filtered_states"]  # (num_samples, T, K)
    return _sum_bear(filtered)


# Reuse the 3-state plot (identical figure; different P(bear) input). Wrap it so this
# model's title_prefix is applied automatically by the shared runner (which calls
# model.plot_regime_fit without a title_prefix).
from regime_model_3state import plot_regime_fit as _plot_regime_fit_base  # noqa: E402


def plot_regime_fit(*args, **kwargs):
    kwargs.setdefault("title_prefix", "Regime nowcast (4-state: turb/calm x bear/bull)")
    return _plot_regime_fit_base(*args, **kwargs)


# ---- fit-mode runner hooks (see _run_modes.run_main) ----------------------------
# Walk-forward comes from the shared factory (logic only needs fit/filtered_p_bear_over).
import sys as _sys_wf  # noqa: E402
from _run_modes import make_walk_forward_p_bear as _make_wf  # noqa: E402

walk_forward_p_bear = _make_wf(_sys_wf.modules[__name__])

needs_macro = False          # bivariate + drawdown only; no macro CSVs needed
obs_cols = _OBS_COLS

# DEFAULT fit mode with no CLI arg: "global" (fast 80/20) | "walkforward" (slow rolling
# 8yr refits, non-stationarity-robust). HOW TO USE: edit this to change the default and
# run `python regime_model_4state.py`; OR override per-run: `... walkforward` (CLI wins).
FIT_MODE = "global"


def obs_kwargs():
    return dict(include_drawdown=INCLUDE_DRAWDOWN, drawdown_window_weeks=_DRAWDOWN_WINDOW_WEEKS)


def extra_spec():
    return {"include_drawdown": INCLUDE_DRAWDOWN, "drawdown_window_weeks": _DRAWDOWN_WINDOW_WEEKS}


def main(mode: str | None = None) -> None:
    """Fit + evaluate + plot + save. mode: 'global' (80/20, fast) | 'walkforward'
    (rolling 8yr refits, slow); None -> FIT_MODE. CLI arg overrides:
    `python regime_model_4state.py [global|walkforward]`."""
    from _run_modes import run_main
    run_main(_sys_wf.modules[__name__], mode if mode is not None else FIT_MODE)


if __name__ == "__main__":
    import sys as _s
    main(_s.argv[1] if len(_s.argv) > 1 else None)
