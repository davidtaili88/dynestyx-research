"""Justifying the regime model's priors: two complementary tools.

Every prior in the model should be defensible in ONE of two ways:
  (1) GROUNDED -- roughly derivable from label-free data (so the guess isn't arbitrary), OR
  (2) INERT    -- the data overrules it so hard the exact value doesn't change conclusions
                  (so an arbitrary round guess is safe).
A prior that is NEITHER is a red flag. A prior that is strongly informative BY DESIGN
(e.g. the persistence prior p_self) is a third, deliberate case -- flagged, not "fixed".

TOOL 1 -- grounding (the historical-stats functions below):
  Split weeks by v_t (log realized vol) percentile -- an observed INPUT the model already
  consumes -- as a calm/turbulent proxy, and read off r_t/v_t moments per group. This grounds
  the r_t drift and v_t LOCATION priors WITHOUT touching the section-4 drawdown label (the
  answer key, "never a model input"): conditioning on that label would leak ground truth into
  what the model is allowed to believe before it sees data. Note the scale/spread priors and
  the whole dd channel CANNOT be grounded this way -- there is no label-free per-state split
  for them -- which is exactly why tool 2 exists.

TOOL 2 -- inertness screen (prior_sensitivity(), below):
  For each prior, compare its PRIOR width to its POSTERIOR width from a cached fit. A large
  ratio = the likelihood overruled the prior = INERT (guess safely). See that function's
  docstring for precisely what the ratio does and DOESN'T prove.

Note: v_t from data_acquisition.py's RegimeDataset is already log(realized_vol) (see
weekly_log_realized_vol) -- NOT raw vol. Do not np.log() it again.

Grounding stats are scoped to the training split only, matching what the model is fit on.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401  (puts sibling src/ subfolders on sys.path)

import numpy as np
import pandas as pd

from data_acquisition import load_regime_dataset


def unconditional_stats(r_t: pd.Series, v_t: pd.Series) -> dict:
    """Whole-training-sample moments, no state conditioning at all."""
    return {
        "r_t_mean": r_t.mean(),
        "r_t_std": r_t.std(),
        "r_t_skew": r_t.skew(),
        "v_t_mean": v_t.mean(),  # v_t is already log(realized_vol)
        "v_t_std": v_t.std(),
        "realized_vol_mean": np.exp(v_t).mean(),
    }


def percentile_split_stats(r_t: pd.Series, v_t: pd.Series, split_pct: float = 70.0) -> dict:
    """Split weeks by v_t (log realized vol) percentile (NOT the drawdown
    label) into a 'calm-proxy' group (below split_pct) and 'turbulent-proxy'
    group (at/above split_pct), then report r_t / v_t moments within each.

    v_t and r_t share the same index by construction (data_acquisition.py's observations()
    frame is built from aligned, NaN-dropped r_t/v_t), so aligning here is a
    no-op sanity check, not a real join.
    """
    aligned = pd.concat([r_t, v_t], axis=1).dropna()
    aligned.columns = ["r_t", "v_t"]

    threshold = np.percentile(aligned["v_t"], split_pct)
    calm_proxy = aligned[aligned["v_t"] < threshold]
    turbulent_proxy = aligned[aligned["v_t"] >= threshold]

    def moments(df: pd.DataFrame) -> dict:
        return {
            "n_weeks": len(df),
            "r_t_mean": df["r_t"].mean(),
            "r_t_std": df["r_t"].std(),
            "v_t_mean": df["v_t"].mean(),
            "v_t_std": df["v_t"].std(),
            "realized_vol_mean": np.exp(df["v_t"]).mean(),
        }

    return {
        "split_pct": split_pct,
        "v_t_threshold": threshold,
        "calm_proxy": moments(calm_proxy),
        "turbulent_proxy": moments(turbulent_proxy),
    }


def recommend_priors(r_t: pd.Series, v_t: pd.Series, split_pct: float = 70.0,
                     round_to: dict | None = None) -> list[dict]:
    """DERIVE named prior centers from the label-free split -- closing the loop.

    This is the piece that turns the split MOMENTS into concrete prior RECOMMENDATIONS, so
    you don't eyeball the table and round by hand. It maps each GROUNDABLE prior to the exact
    statistic that grounds it, applies a round-to grid (priors should be round ballparks, not
    fitted digits -- see the inertness point), and returns (prior_name, raw_stat, rounded,
    grounding) rows. The dd/scale/tail priors are NOT here: they aren't groundable label-free
    (no per-state split), so they live in the inertness screen instead, not this tool.

    GROUNDING MAP (each prior <- one label-free statistic on the training split):
      _PRIOR_CALM_LOG_VOL_MEAN  <- calm-group v_t mean          (the low-vol level)
      _PRIOR_LOG_VOL_GAP        <- (turb v_t mean - calm v_t mean)  (calm->turbulent jump)
      _PRIOR_LOG_VOL_SPREAD     <- within-group v_t std          (spread around each level)
      _PRIOR_BEAR_DRIFT_MEAN    <- turbulent-group r_t mean, sign-forced NEGATIVE
                                   (bear leans turbulent; sign is STRUCTURE, magnitude grounded)
      _PRIOR_DRIFT_GAP          <- |turb r_t mean - calm r_t mean|  (bull<->bear drift gap)
    """
    round_to = round_to or {
        "_PRIOR_CALM_LOG_VOL_MEAN": 0.1, "_PRIOR_LOG_VOL_GAP": 0.1,
        "_PRIOR_LOG_VOL_SPREAD": 0.1, "_PRIOR_BEAR_DRIFT_MEAN": 0.005, "_PRIOR_DRIFT_GAP": 0.005,
    }
    s = percentile_split_stats(r_t, v_t, split_pct=split_pct)
    calm, turb = s["calm_proxy"], s["turbulent_proxy"]

    def _round(x, grid):
        return round(round(x / grid) * grid, 6)

    # within-group v_t spread: pooled std of the two groups (both ~equal by design).
    pooled_v_std = 0.5 * (calm["v_t_std"] + turb["v_t_std"])

    raw = {
        "_PRIOR_CALM_LOG_VOL_MEAN": (calm["v_t_mean"],           "calm-group v_t mean"),
        "_PRIOR_LOG_VOL_GAP":       (turb["v_t_mean"] - calm["v_t_mean"], "turb - calm v_t mean"),
        "_PRIOR_LOG_VOL_SPREAD":    (pooled_v_std,               "within-group v_t std (pooled)"),
        # bear leans turbulent -> use the turbulent group's drift, but FORCE the sign negative
        # (structure): a bear is defined by negative drift; the magnitude is what we ground.
        "_PRIOR_BEAR_DRIFT_MEAN":   (-abs(turb["r_t_mean"]),     "turb-group r_t mean, sign-forced -"),
        "_PRIOR_DRIFT_GAP":         (abs(turb["r_t_mean"] - calm["r_t_mean"]), "|turb - calm r_t mean|"),
    }
    rows = []
    for name, (val, grounding) in raw.items():
        rows.append(dict(prior=name, raw=val, rounded=_round(val, round_to[name]),
                         grounding=grounding))
    return rows


def check_v_t_distribution_shape(v_t: pd.Series) -> dict:
    """Descriptive check on whether v_t (log realized vol) looks like it has
    a natural break (motivating a percentile split) or is roughly unimodal
    (meaning any split point is somewhat arbitrary, and priors should stay
    wide).
    """
    return {
        "v_t_skew": v_t.skew(),
        "v_t_kurtosis": v_t.kurtosis(),
        "percentiles": {p: np.percentile(v_t, p) for p in [10, 25, 50, 70, 75, 90, 95, 99]},
    }


def plot_v_t_histogram(v_t: pd.Series, split_pct: float = 70.0, save_path: str | None = None):
    import matplotlib.pyplot as plt

    threshold = np.percentile(v_t, split_pct)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(v_t, bins=60, color="steelblue", alpha=0.8)
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"{split_pct:.0f}th pct split")
    ax.set_xlabel("v_t (log realized vol)")
    ax.set_ylabel("count")
    ax.set_title("Training-period log realized vol distribution")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# TOOL 2: prior INERTNESS screen (prior width vs posterior width)
# ---------------------------------------------------------------------------
# The PRIOR SPECS for the shipped 3-state model (DD_EMISSION="normal"), declared here so the
# screen knows each prior's TRUE width analytically (not guessed from samples). Each entry:
#   sample_name -> (human prior, prior_std, prior_mean, family)
# prior_std / prior_mean are the analytic moments of the declared distribution. Keep in sync
# with regime_model_3state.py (only priors that appear as posterior samples are screened).
def _halfnormal_moments(sigma):
    # HalfNormal(sigma): mean = sigma*sqrt(2/pi), std = sigma*sqrt(1 - 2/pi).
    return sigma * np.sqrt(2 / np.pi), sigma * np.sqrt(1 - 2 / np.pi)


def _shipped_prior_specs() -> dict:
    hn = _halfnormal_moments
    specs = {
        # --- persistence (INFORMATIVE BY DESIGN -- expect it NOT to tighten) ---
        "p_self":               ("Beta(500,3)",       0.00344, 0.9940, "beta"),
        # --- r_t drift (GROUNDABLE: location) ---
        "mean_return_bear":     ("Normal(-0.005,0.01)", 0.0100, -0.005, "normal"),
        "drift_gap1":           ("HalfNormal(0.010)", hn(0.010)[1], hn(0.010)[0], "halfnormal"),
        "drift_gap2":           ("HalfNormal(0.010)", hn(0.010)[1], hn(0.010)[0], "halfnormal"),
        # --- r_t return-vol (SCALES: expect inert) ---
        "return_vol_bull":      ("HalfNormal(0.02)",  hn(0.02)[1], hn(0.02)[0], "halfnormal"),
        "return_vol_gap_tbull": ("HalfNormal(0.02)",  hn(0.02)[1], hn(0.02)[0], "halfnormal"),
        "return_vol_gap_bear":  ("HalfNormal(0.02)",  hn(0.02)[1], hn(0.02)[0], "halfnormal"),
        # --- v_t (GROUNDABLE location: calm level + gap; SCALES: the rest) ---
        "log_vol_calm":         ("Normal(-5.3,0.3)",  0.3000, -5.3,   "normal"),
        "log_vol_gap":          ("HalfNormal(1.0)",   hn(1.0)[1], hn(1.0)[0], "halfnormal"),
        "log_vol_bear_extra":   ("HalfNormal(0.5)",   hn(0.5)[1], hn(0.5)[0], "halfnormal"),
        "v_scale":              ("HalfNormal(0.4)",   hn(0.4)[1], hn(0.4)[0], "halfnormal"),
        # --- dd channel (NOT groundable label-free; expect inert) ---
        # dd_bull is now sampled as dd_bull_depth ~ HalfNormal(0.05), negated to give the
        # correct dd<=0 support (the recorded sample is the POSITIVE depth; bull dd = -depth).
        "dd_bull_depth":        ("HalfNormal(0.05)",  hn(0.05)[1], hn(0.05)[0], "halfnormal"),
        "dd_bear_gap":          ("HalfNormal(0.20)",  hn(0.20)[1], hn(0.20)[0], "halfnormal"),
        "dd_tbull_gap":         ("HalfNormal(0.10)",  hn(0.10)[1], hn(0.10)[0], "halfnormal"),
        "dd_scale":             ("HalfNormal(0.15)",  hn(0.15)[1], hn(0.15)[0], "halfnormal"),
        # --- StudentT tail dof (SHAPE: expect inert) ---
        "tail_dof_raw":         ("Gamma(2,0.1)",      np.sqrt(2) / 0.1, 2 / 0.1, "gamma"),
        "v_tail_dof_raw":       ("Gamma(2,0.1)",      np.sqrt(2) / 0.1, 2 / 0.1, "gamma"),
    }
    return specs


def prior_sensitivity(fit_pkl: str | None = None, tighten_inert: float = 10.0) -> list[dict]:
    """INERTNESS SCREEN: for each prior, prior-width vs posterior-width from a cached fit.

    WHAT "INERT" MEANS (and what this screen actually measures):
      A prior is INERT if changing it (within reason) does NOT change your conclusions --
      i.e. the likelihood, not the prior, is setting the answer. This function is a CHEAP
      PROXY for that, computed from a SINGLE existing fit (no refit):

        tighten = prior_std / posterior_std

      Big ratio (say >=10x) => the posterior is far narrower than the prior => the likelihood
      is a sharp spike sitting inside a nearly-flat region of the prior, so the prior's exact
      value barely shapes the posterior => INERT, the round guess is safe. Ratio ~1x (or <1)
      => the data barely moved the prior => the prior is INFORMATIVE (carrying the answer).

    WHAT IT DOES NOT PROVE (be honest about the proxy's blind spots):
      1. It measures WIDTH, not LOCATION. A narrow prior centred in the wrong place could
         still bias the posterior. We PARTLY close this by also reporting whether the
         posterior mean lands INSIDE the prior's central mass (`post_in_prior`); if it sits
         out in the prior's tail, the "inert" verdict is NOT trustworthy -- investigate.
      2. It never actually VARIES the prior. The gold-standard proof is a refit under a
         deliberately different prior, checking the P(bear) OUTPUT is unchanged. This screen
         TRIAGES (flags which priors even need that refit); it does not replace it.

    So: use this to sort priors into "inert -> guess freely" vs "informative -> justify /
    ground / refit-test". It is a screen, not a certificate.

    fit_pkl: path to an outputs/*.pkl saved by save_run (must contain posterior 'samples').
             Defaults to the shipped 3-state global fit. tighten_inert: ratio at/above which
             we label a prior inert (default 10x -- a soft, disclosed threshold).
    Returns a list of per-prior result dicts (also printed as a table).
    """
    import pickle

    _OUT = _pl.Path(__file__).resolve().parents[2] / "outputs"
    if fit_pkl is None:
        # prefer the shipped-config global fit; fall back to any 3-state fit with samples.
        for cand in ("regime_3state_r_t_v_t_dd_global.pkl",
                     "regime___main___r_t_v_t_dd_global.pkl"):
            if (_OUT / cand).exists():
                fit_pkl = str(_OUT / cand)
                break
    if fit_pkl is None:
        raise FileNotFoundError(
            "No cached 3-state global fit with posterior samples found in outputs/. "
            "Run: python src/models/regime_model_3state.py  (global mode saves samples)."
        )

    payload = pickle.load(open(fit_pkl, "rb"))
    samples = payload.get("samples", {})
    if not samples:
        raise ValueError(f"{fit_pkl} has no posterior 'samples' (curve-only cache?). "
                         "Use a save_run pkl (model global run), not a strategy cache.")

    specs = _shipped_prior_specs()
    rows = []
    for name, (human, prior_std, prior_mean, family) in specs.items():
        if name not in samples:
            continue
        post = np.asarray(samples[name]).ravel()
        post_std = float(post.std())
        post_mean = float(post.mean())
        tighten = prior_std / post_std if post_std > 0 else float("inf")

        # LOCATION check (blind-spot #1): does the posterior mean sit inside the prior's
        # central mass? Use a family-appropriate central interval.
        if family == "normal":
            lo, hi = prior_mean - 2 * prior_std, prior_mean + 2 * prior_std
        elif family == "halfnormal":
            sigma = prior_std / np.sqrt(1 - 2 / np.pi)  # recover the declared sigma
            lo, hi = 0.0, 2.5 * sigma  # ~99% of a HalfNormal's mass
        else:  # beta / gamma: use mean +- 2 std as a rough central band
            lo, hi = prior_mean - 2 * prior_std, prior_mean + 2 * prior_std
        post_in_prior = bool(lo <= post_mean <= hi)

        if tighten < 1.5:
            verdict = "INFORMATIVE (prior carries the answer -- by design?)"
        elif not post_in_prior:
            verdict = "CHECK (tightened, but posterior in prior TAIL -- location suspect)"
        elif tighten >= tighten_inert:
            verdict = "INERT (guess is safe)"
        else:
            verdict = "PARTLY (prior still shapes it -- worth grounding)"

        rows.append(dict(name=name, prior=human, prior_std=prior_std, post_std=post_std,
                         post_mean=post_mean, tighten=tighten,
                         post_in_prior=post_in_prior, verdict=verdict))

    # ---- print ----
    print("\n" + "=" * 100)
    print(f"PRIOR INERTNESS SCREEN  (fit: {_pl.Path(fit_pkl).name})")
    print("  tighten = prior_std / posterior_std.  >=%.0fx => data overrules the prior => INERT."
          % tighten_inert)
    print("  post_in_prior = does the posterior mean land inside the prior's central mass?")
    print("  (SCREEN not proof: it never varies the prior -- for the doubtful ones, refit "
          "under a\n   different prior and check P(bear) is unchanged. See this fn's docstring.)")
    print("=" * 100)
    print(f"{'param':22s} {'prior':18s} {'tighten':>8s} {'post_mean':>10s} {'in_prior':>9s}  verdict")
    for r in sorted(rows, key=lambda r: r["tighten"]):
        tx = "inf" if r["tighten"] == float("inf") else f"{r['tighten']:.0f}x"
        print(f"{r['name']:22s} {r['prior']:18s} {tx:>8s} {r['post_mean']:10.4f} "
              f"{str(r['post_in_prior']):>9s}  {r['verdict']}")
    print("=" * 100)
    print("READ: INERT priors may be arbitrary round guesses safely. INFORMATIVE priors must "
          "be\n  justified (grounded via tool 1, or intentional like p_self). CHECK/PARTLY "
          "-> look closer.")
    return rows


def main() -> None:
    ds = load_regime_dataset(include_vix=False)
    train_obs, _test_obs = ds.split()
    r_t = train_obs["r_t"]
    v_t = train_obs["v_t"]

    print(f"Train span: {r_t.index[0].date()} -> {r_t.index[-1].date()} ({len(r_t)} weeks)")
    print()

    print("--- Unconditional (whole-sample) stats ---")
    for k, v in unconditional_stats(r_t, v_t).items():
        print(f"  {k}: {v:.5f}")
    print()

    print("--- v_t (log realized vol) distribution shape ---")
    shape = check_v_t_distribution_shape(v_t)
    print(f"  skew: {shape['v_t_skew']:.3f}, kurtosis: {shape['v_t_kurtosis']:.3f}")
    print("  percentiles (v_t, log scale):")
    for p, val in shape["percentiles"].items():
        print(f"    p{p}: {val:.4f}  (realized_vol = {np.exp(val):.5f})")
    # TAKEAWAY: near-zero skew + mild kurtosis => v_t is ~UNIMODAL (one smooth hump, no
    # natural calm/turbulent break). So the 70/30 split line is a PROXY through a continuum,
    # not a real boundary -- which is exactly why we keep the vol priors WIDE (weakly-
    # informative) and lean on the inertness screen rather than trusting the split's digits.
    unimodal = abs(shape["v_t_skew"]) < 0.5 and shape["v_t_kurtosis"] < 1.0
    print(f"  => {'~UNIMODAL' if unimodal else 'possible break'}: the split point is "
          f"{'arbitrary -> keep priors wide, rely on inertness' if unimodal else 'somewhat data-backed'}.")
    print()

    print("--- Percentile split by v_t (70/30, calm-proxy vs turbulent-proxy) ---")
    split = percentile_split_stats(r_t, v_t, split_pct=70.0)
    print(f"  v_t threshold: {split['v_t_threshold']:.5f}")
    for group_name in ["calm_proxy", "turbulent_proxy"]:
        stats = split[group_name]
        print(f"  {group_name} (n={stats['n_weeks']}):")
        print(f"    r_t: mean={stats['r_t_mean']:.5f}  std={stats['r_t_std']:.5f}")
        print(f"    v_t: mean={stats['v_t_mean']:.4f}  std={stats['v_t_std']:.4f}  (realized_vol mean={stats['realized_vol_mean']:.5f})")
    print()

    # DERIVED PRIOR RECOMMENDATIONS: map the split moments -> named priors, side-by-side with
    # what the model currently ships (so the grounding is a VISIBLE derivation, not a manual
    # eyeball-and-round). Pulls current values from the model module if importable.
    print("--- Recommended prior centers (grounded)  vs  what the model ships ---")
    try:
        import regime_model_3state as _m
        shipped = {
            "_PRIOR_CALM_LOG_VOL_MEAN": getattr(_m, "_PRIOR_CALM_LOG_VOL_MEAN", None),
            "_PRIOR_LOG_VOL_GAP": getattr(_m, "_PRIOR_LOG_VOL_GAP", None),
            "_PRIOR_LOG_VOL_SPREAD": getattr(_m, "_PRIOR_LOG_VOL_SPREAD", None),
            "_PRIOR_BEAR_DRIFT_MEAN": getattr(_m, "_PRIOR_BEAR_DRIFT_MEAN", None),
            "_PRIOR_DRIFT_GAP": getattr(_m, "_PRIOR_DRIFT_GAP", None),
        }
    except Exception:
        shipped = {}
    print(f"  {'prior':28s} {'raw':>9s} {'->round':>9s} {'shipped':>9s}   grounding")
    for row in recommend_priors(r_t, v_t, split_pct=70.0):
        sv = shipped.get(row["prior"])
        sv_s = f"{sv:>9.4f}" if isinstance(sv, (int, float)) else f"{'?':>9s}"
        print(f"  {row['prior']:28s} {row['raw']:9.4f} {row['rounded']:9.4f} {sv_s}   {row['grounding']}")
    print("  (round grid keeps priors as ballparks; the inertness screen below shows the")
    print("   exact center is overruled 5-50x anyway -- grounding is for defensibility, not fit.)")
    print()

    # TOOL 2: the inertness screen (reads a cached fit; skips gracefully if none present).
    try:
        prior_sensitivity()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[prior_sensitivity skipped] {e}")

    plot_v_t_histogram(v_t, split_pct=70.0)


if __name__ == "__main__":
    # `--sensitivity` runs ONLY the inertness screen (no data download / no plot) -- cheap.
    if "--sensitivity" in _sys.argv:
        prior_sensitivity()
    else:
        main()
