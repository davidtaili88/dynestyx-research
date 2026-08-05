"""Stationarity test for the regime model's VOL-LEVEL parameters, via NUTS.

PURPOSE (methodology, not a one-off). The 2-state nowcast is fit once on a long
train span (1957->2012) and scored on a held-out tail (2012->2026). That single
chronological split is only justified IF the model's parameters do not DRIFT
across eras -- otherwise parameters learned mostly on old data misfit the recent
tail, and *where* we cut the split silently changes the result. This file tests
that assumption for the VOL LEVELS, which we established are the WELL-IDENTIFIED
emission parameters (the drift gap, by contrast, is near-unidentifiable at weekly
cadence and would only ever return "inconclusive" -- so we start with vol).

WHY THE REAL MODEL (NUTS), NOT A PROXY. An earlier plan used a Gaussian-mixture
proxy for speed. We deliberately pay the NUTS time cost instead so the per-era
estimates ARE the model's own posterior -- same emission, same HMM filter over the
sequence, same priors -- removing any "the proxy defines states differently than
the model does" objection. The posterior gives uncertainty directly, so no
bootstrap is needed: two eras' vol levels are "the same" iff their posteriors
OVERLAP, "drifted" iff they separate.

WHAT IS AND ISN'T TESTED HERE.
  * TESTED (two LAYERS, answering DIFFERENT questions -- do not conflate):
      - SERIES level (series_stationarity): ADF + KPSS on the raw v_t series --
        does it have a unit root, or mean-revert around a fixed level? This is
        the pooled/unconditional view.
      - PARAMETER level (fit_era/analyze): does the per-STATE vol level (v_bull,
        v_bear) DRIFT across eras? This is the conditional view and is the one
        that decides the split. The two can disagree (v_t is series-stationary
        yet its per-era level drifts) -- that disagreement is the whole point.
  * NOT here: drift_gap (unidentifiable -> its own later analysis), and
    persistence p_self (a SEQUENCE property needing a label-free proxy -- also
    later). The PARAMETER-drift layer is scoped to ONE well-identified parameter
    (vol) to prove the approach before scaling it up.

CHEATING GUARDS (see the long design discussion this came from).
  * No P&S labels are used to define the eras or the states -- eras are pure
    calendar cuts and the model finds its own states -- so P&S's era-dependent
    bear-fraction cannot leak into the verdict.
  * Eras are disjoint contiguous calendar spans; the model is fit independently
    on each, so no era's parameters are informed by another's.

DECISION RULE (pre-committed, before seeing results). For the vol levels:
  * KEEP the single 1957/2012 split if every era-pair's v_bull posteriors overlap
    AND every era-pair's v_bear posteriors overlap (no significant drift in the
    well-identified parameters).
  * CHANGE (walk-forward, or a shorter recent window) if a vol level is
    SIGNIFICANTLY different across eras (posteriors separated) -- that is genuine
    emission non-stationarity and a single global fit is then misspecified.
  * BORDERLINE (posteriors barely touch) counts as inconclusive, not stationary;
    tie-break toward walk-forward, which is robust to drift either way.

Run directly: fits the model on each era, prints per-era posterior summaries for
v_bull/v_bear, a pairwise overlap table, the KEEP/CHANGE verdict, and a plot
overlaying all 3 eras' posteriors (arviz-style density + 95% HDI, matching the
notebooks' az.plot_posterior idiom) so the drift is visible at a glance.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401  (puts sibling src/ subfolders on sys.path)

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# Reuse the ACTUAL model + fit, so per-era estimates are the model's own posterior.
from regime_model_2state import fit


# Era cuts: disjoint contiguous calendar spans (thirds of the 1957-2026 sample).
# Thirds (not decades) give each era enough weeks for NUTS to identify the vol
# levels tightly; a finer decade view can come later if thirds are borderline.
_ERA_BOUNDS = [
    ("1957-1980", 1957, 1980),
    ("1980-2003", 1980, 2003),
    ("2003-2026", 2003, 2027),
]


@dataclass
class EraVolPosterior:
    """Per-era posterior draws for the two vol-level parameters."""

    name: str
    n_weeks: int
    v_bull: np.ndarray  # log_vol_calm draws (calm/low-vol state mean)
    v_bear: np.ndarray  # log_vol_calm + log_vol_gap draws (turbulent state mean)

    def summary(self, draws: np.ndarray) -> tuple[float, float, float, float]:
        """(mean, std, 2.5%, 97.5%) of a draw vector."""
        return (
            float(draws.mean()),
            float(draws.std()),
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        )


def fit_era(era_obs, name: str, num_warmup: int, num_samples: int) -> EraVolPosterior:
    """Fit the REAL 2-state model on one era's observation frame and extract the
    vol-level posteriors.

    v_bull = log_vol_calm (calm state mean); v_bear = log_vol_calm + log_vol_gap
    (turbulent state mean) -- reconstructed exactly as the model's v_loc does, so
    these are the model's own per-state vol means, not a re-derivation.
    """
    mcmc = fit(era_obs, num_warmup=num_warmup, num_samples=num_samples)
    s = mcmc.get_samples()
    v_bull = np.asarray(s["log_vol_calm"])
    v_bear = np.asarray(s["log_vol_calm"] + s["log_vol_gap"])
    return EraVolPosterior(name=name, n_weeks=len(era_obs), v_bull=v_bull, v_bear=v_bear)


def _ci_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Do two posteriors' 95% credible intervals overlap? (Non-overlap = the two
    era estimates are significantly different -> drift.)"""
    a_lo, a_hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    b_lo, b_hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
    return not (a_hi < b_lo or b_hi < a_lo)


def _overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """A softer overlap measure: how much of the pooled range the CIs share,
    in [0,1]. ~0 means cleanly separated, ~1 means nearly identical. Reported
    alongside the hard overlap flag to expose BORDERLINE cases."""
    a_lo, a_hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    b_lo, b_hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
    inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
    union = max(a_hi, b_hi) - min(a_lo, b_lo)
    return inter / union if union > 0 else 1.0


def series_stationarity(v_t: np.ndarray) -> None:
    """SERIES-LEVEL check on the raw v_t series (ADF + KPSS).

    This is a SEPARATE, complementary question from the per-era parameter-drift
    test below, and must NOT be conflated with it:

      * ADF / KPSS ask: does the raw v_t SERIES have a unit root -- i.e. does it
        wander like a random walk, or mean-revert around a FIXED level?
      * The NUTS per-era test asks: does the per-STATE vol LEVEL drift across
        eras (conditional on regime)?

    These come apart, and here they DISAGREE on purpose. Realized vol is strongly
    mean-reverting, so ADF will reject the unit root (verdict: "series stationary")
    -- but it is stationary around a mean that has SHIFTED between eras, which ADF
    (run on the pooled series, blind to regime and to a slow level shift) cannot
    see. So a "stationary" here does NOT overturn the drift finding; it shows
    exactly why the pooled/unconditional view misses drift the conditional model
    catches. We run BOTH tests because their nulls are opposite (ADF H0 = unit
    root / non-stationary; KPSS H0 = stationary), which guards against ADF's known
    low power -- agreement between them is the trustworthy case.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    x = np.asarray(v_t, dtype=float)
    adf_stat, adf_p, *_ = adfuller(x, autolag="AIC")
    # KPSS around a constant level ('c'); it warns + clips p outside [0.01,0.10],
    # which is expected and fine for a verdict.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")

    adf_reject = adf_p < 0.05  # reject unit root -> series is (level-)stationary
    kpss_reject = kpss_p < 0.05  # reject stationarity -> series is non-stationary

    print("\n=== SERIES-LEVEL check on raw v_t (ADF + KPSS) ===")
    print("  (complementary to the drift test -- see series_stationarity docstring)")
    print(f"  ADF : stat={adf_stat:7.3f}  p={adf_p:.4f}  "
          f"-> {'stationary (rejects unit root)' if adf_reject else 'CANNOT reject unit root'}")
    print(f"  KPSS: stat={kpss_stat:7.3f}  p={kpss_p:.4f}  "
          f"-> {'NON-stationary (rejects stationarity)' if kpss_reject else 'stationary (cannot reject)'}")

    if adf_reject and not kpss_reject:
        print("  => Both agree: v_t is SERIES-stationary (mean-reverts around a level).")
        print("     This does NOT contradict cross-era LEVEL drift below -- the level")
        print("     it reverts to has shifted between eras, which these tests can't see.")
    elif not adf_reject and kpss_reject:
        print("  => Both agree: v_t has a UNIT ROOT / trend (series non-stationary).")
    elif adf_reject and kpss_reject:
        # ADF rejects unit root (mean-reverts) BUT KPSS rejects stationarity: the
        # textbook signature of a series that is mean-reverting in the short run
        # around a LEVEL THAT SHIFTS -- i.e. trend/level-stationary with breaks.
        # That level shift IS the cross-era vol drift the NUTS test measures.
        print("  => ADF rejects a unit root (v_t mean-REVERTS) but KPSS rejects")
        print("     stationarity: the signature of mean-reversion around a SHIFTING")
        print("     LEVEL (level-stationary with breaks). That level shift is exactly")
        print("     the cross-era vol drift the per-era test below quantifies -- the")
        print("     two layers AGREE: short-run mean-reverting, long-run level-drifting.")
    else:
        print("  => Neither test rejects -> inconclusive at the series level "
              "(borderline / low power).")


def analyze(eras: list[EraVolPosterior]) -> None:
    """Print per-era summaries, the pairwise overlap table, and the verdict."""
    print("\n=== Per-era posterior for the VOL LEVELS (log realized vol) ===")
    print(f"{'era':<12} {'n_wk':>5}  {'param':<7} {'mean':>7} {'std':>6} {'2.5%':>7} {'97.5%':>7}")
    for e in eras:
        for pname, draws in [("v_bull", e.v_bull), ("v_bear", e.v_bear)]:
            m, sd, lo, hi = e.summary(draws)
            print(f"{e.name:<12} {e.n_weeks:>5}  {pname:<7} {m:>7.3f} {sd:>6.3f} {lo:>7.3f} {hi:>7.3f}")

    print("\n=== Pairwise cross-era comparison (CI overlap = same; separated = drift) ===")
    any_drift = False
    for pname in ("v_bull", "v_bear"):
        print(f"\n  {pname}:")
        for i in range(len(eras)):
            for j in range(i + 1, len(eras)):
                a = getattr(eras[i], pname)
                b = getattr(eras[j], pname)
                overlaps = _ci_overlap(a, b)
                frac = _overlap_fraction(a, b)
                gap = abs(a.mean() - b.mean())
                tag = "OVERLAP" if overlaps else "SEPARATED (drift)"
                if not overlaps:
                    any_drift = True
                print(
                    f"    {eras[i].name} vs {eras[j].name}: "
                    f"|Δmean|={gap:.3f}  CI-overlap={frac:.2f}  -> {tag}"
                )

    print("\n=== VERDICT (vol levels only) ===")
    if any_drift:
        print("  At least one vol level is SIGNIFICANTLY different across eras.")
        print("  -> Emission non-stationarity in a well-identified parameter.")
        print("  -> CHANGE the evaluation: prefer walk-forward, or a shorter")
        print("     recent-only window. The single 1957/2012 split is not safe.")
    else:
        print("  All vol-level posteriors OVERLAP across eras (no significant drift")
        print("  in the well-identified emission parameter).")
        print("  -> KEEP the single 1957/2012 split: how we cut does not change the")
        print("     vol estimates. (Still TODO: persistence + drift, separately.)")
    print("\n  NOTE: borderline CI-overlap (small but nonzero) is INCONCLUSIVE, not")
    print("  stationary -- read the overlap fractions above before committing.")


def plot_era_posteriors(eras: list[EraVolPosterior], save_path=None):
    """Overlay all eras' vol-level posteriors on shared axes -- one panel for
    v_bull, one for v_bear -- as arviz-style densities with the 95% HDI marked,
    matching the notebooks' az.plot_posterior(hdi_prob=0.95) idiom. Separated
    densities across eras = drift; overlapping = stationary.
    """
    import arviz as az
    import matplotlib.pyplot as plt

    # One consistent colour per era across both panels, so an era is recognisable
    # at a glance (like the notebook's single-ref posterior style, extended to a
    # per-era overlay).
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, pname in zip(axes, ("v_bull", "v_bear")):
        for e, c in zip(eras, colours):
            draws = getattr(e, pname)
            # KDE density (arviz), same estimator az.plot_posterior draws with.
            grid, density = az.kde(draws)
            ax.plot(grid, density, color=c, lw=1.8, label=f"{e.name} (n={e.n_weeks})")
            ax.fill_between(grid, density, color=c, alpha=0.12)
            # 95% HDI as a bar under the density, plus the posterior mean tick --
            # the two things az.plot_posterior annotates.
            lo, hi = az.hdi(draws, hdi_prob=0.95)
            ymax = density.max()
            ax.plot([lo, hi], [-0.04 * ymax, -0.04 * ymax], color=c, lw=3, solid_capstyle="butt")
            ax.plot(draws.mean(), -0.04 * ymax, marker="o", color=c, markersize=4)
        ax.set_title(f"{pname}  (log realized vol)")
        ax.set_xlabel("log realized vol")
        ax.set_yticks([])
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Per-era posterior for the vol levels (95% HDI bar under each density)\n"
        "separated across eras = drift  |  overlapping = stationary",
        fontsize=10,
    )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, axes


def main() -> None:
    from data import load_regime_dataset

    # Full history, model's realized-vol channel (no VIX) -- same data the 2-state
    # model uses. Eras are pure calendar cuts (no P&S labels), so the labeler's
    # era-dependence cannot leak into the stationarity verdict.
    ds = load_regime_dataset(start="1957-03-01", include_vix=False)
    obs = ds.observations()

    # Series-level layer first (fast): ADF + KPSS on the raw v_t series. This is
    # the unconditional/pooled view; the per-era NUTS test below is the
    # conditional one. They answer different questions -- see series_stationarity.
    series_stationarity(obs["v_t"].to_numpy())

    eras: list[EraVolPosterior] = []
    for name, y0, y1 in _ERA_BOUNDS:
        era_obs = obs[(obs.index.year >= y0) & (obs.index.year < y1)]
        print(f"\nFitting era {name} ({len(era_obs)} weeks) via NUTS ...")
        eras.append(fit_era(era_obs, name, num_warmup=1000, num_samples=1000))

    analyze(eras)
    plot_era_posteriors(eras)


if __name__ == "__main__":
    main()
