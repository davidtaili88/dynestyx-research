"""Historical r_t / v_t analysis to inform the regime model's priors (section 5).

Deliberately unconditional on the section 4 drawdown-rally label -- that label
is the answer key for evaluation only (section 4: "never as a model input"),
and priors are not inert, so conditioning on it here would let the ground
truth leak into what the model is allowed to believe before it sees data.

Instead, weeks are split by v_t (log realized vol) percentile -- an observed
input the model already consumes -- as a proxy for "calm" vs "turbulent"
to get empirically grounded prior locations/scales, replacing the generic
HalfNormal(0.02)/HalfNormal(0.03) guesses currently in regime_model.py.

Note: v_t from data.py's RegimeDataset is already log(realized_vol) (see
weekly_log_realized_vol) -- NOT raw vol. Do not np.log() it again.

Scoped to the training split only, matching what the model itself is fit on.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import _syspath  # noqa: E402,F401  (puts sibling src/ subfolders on sys.path)

import numpy as np
import pandas as pd

from data import load_regime_dataset


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

    v_t and r_t share the same index by construction (data.py's observations()
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

    plot_v_t_histogram(v_t, split_pct=70.0)


if __name__ == "__main__":
    main()
