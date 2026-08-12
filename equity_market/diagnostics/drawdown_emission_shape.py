"""DESIGN-TIME diagnostic: what distribution should the DRAWDOWN emission channel use?

WHAT THIS IS (and the discipline behind it)
-------------------------------------------
The drawdown channel emits  dd_t = price_t / max(price, trailing 52wk) - 1  (<= 0, causal).
The shipped model emits it as a single-scale Normal per state -- which is misspecified:
dd is HARD-BOUNDED at 0, so a Normal fit to bull-state dd puts ~27% of its mass ABOVE 0,
an IMPOSSIBLE region. This script visualises the ACTUAL per-state shape of dd so the
emission family can be chosen from the data's STRUCTURE.

This is SPECIFICATION, not inference -- and there's a bright line we stay on the safe side
of. Legitimate use of these plots: REJECT families the variable structurally can't be
(unbounded -> no; smooth density over an atom -> no). NOT-legitimate: SELECTING/tuning a
family to best match this particular sample's histogram wiggles (that overfits the sample,
and the marginal isn't even the emission's target -- the emission is per-state/conditional).
So: the eye PROPOSES and REJECTS on structure; out-of-sample likelihood (the emission
sweep, later) DISPOSES between the structurally-admissible survivors. See the on-plot notes.

We reparametrise to  u = -dd >= 0  ("depth underwater") so every standard non-negative
family (Half-Normal, Half-t, log-Normal, gamma) applies cleanly and the 0-boundary is at
the ORIGIN where it's easy to see the atom.

USES THE P&S LABELS -- on purpose, and safely. The split into bull/bear weeks uses the
Pagan-Sossounov dating because the emission is a PER-STATE likelihood, so the per-state
shape is exactly what we need to see. That's a design-time choice; it would only be
"cheating" if we then SCORED the model on those same labels as a clean test (we don't --
family choice is a documented design-time decision, and scoring stays in the sweeps).

WHAT IT PRODUCES
----------------
outputs/drawdown_emission_shape.png -- a grid:
  row 1: dd itself (signed), bull vs bear, showing the 0-ceiling + the bull atom at 0.
  row 2: u = -dd, per state, with candidate NON-NEGATIVE family overlays (Half-Normal,
         Half-t(df=4), Half-Cauchy, log-Normal) fit to the u>0 part, plus the atom mass
         P(u=0) drawn as a bar at the origin -- so you can SEE which families can't
         represent the spike (all the continuous ones) and which tail matches the bear bulk.
  row 3: log(u) for the u>0 weeks, per state -- the "is the underwater DEPTH roughly
         log-symmetric?" view that separates a log-Normal-ish body from a heavy left tail.
Plus a printed REJECT/KEEP summary keyed to structure (bound, atom, tail), NOT to which
family had the lowest in-sample error.

RUN (no NUTS, cheap):
    python diagnostics/drawdown_emission_shape.py
    python diagnostics/drawdown_emission_shape.py --window 26   # see how a shorter window grows the bull atom
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

# diagnostics/ is a SIBLING of src/. Put src/ subfolders on sys.path like every script does.
_SRC = _pl.Path(__file__).resolve().parents[1] / "src"
_sys.path.insert(0, str(_SRC))
import _syspath  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from data import load_regime_dataset  # noqa: E402
from labels import pagan_sossounov_label  # noqa: E402


DATA_START = "1957-03-01"
ATOM_TOL = 1e-6  # u <= this counts as "at the peak" (the boundary atom)


# ----------------------------------------------------------------------------
# candidate NON-NEGATIVE families for u = -dd (fit to the u>0 part only; the atom
# at u=0 is handled SEPARATELY as a point mass, since no continuous family can hold it)
# ----------------------------------------------------------------------------
def _halft_pdf(x, df, scale):
    """Half-Student-t: a Student-t folded at 0 (density = 2 * t.pdf for x>=0), scaled.

    scipy has no 'halft', so we build it. df -> inf recovers Half-Normal; df=1 is
    Half-Cauchy. df=4 is the classic heavy-but-finite-variance middle choice, and is
    what we overlay as the 'moderately heavy tail' candidate for the bear bulk.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 2.0 * stats.t.pdf(x / scale, df) / scale, 0.0)


def _fit_families(u_pos: np.ndarray) -> dict:
    """Fit each candidate family to the STRICTLY-POSITIVE underwater depths u>0.

    Returns {name: (pdf_callable, label)}. All fixed at loc=0 (the boundary) so they're
    genuine NON-NEGATIVE densities, not shifted ones. Half-t df is fixed at 4 (a shape
    choice, not fit -- we're illustrating tail weight, not optimising it).
    """
    fams = {}

    # Half-Normal: the thin-tailed, drop-in candidate (family B in the write-up).
    _, hn_scale = stats.halfnorm.fit(u_pos, floc=0)
    fams["half_normal"] = (lambda x, s=hn_scale: stats.halfnorm.pdf(x, 0, s),
                           f"Half-Normal (scale={hn_scale:.3f})")

    # Half-t (df=4): bounded + moderately heavy tail. scale via a simple MoM-ish match
    # to the positive data's scale (median/0.674-ish); we fit scale by max-likelihood grid.
    scales = np.linspace(u_pos.std() * 0.3, u_pos.std() * 1.5, 60)
    lls = [np.log(_halft_pdf(u_pos, 4, s) + 1e-300).sum() for s in scales]
    ht_scale = scales[int(np.argmax(lls))]
    fams["half_t_df4"] = (lambda x, s=ht_scale: _halft_pdf(x, 4, s),
                          f"Half-t (df=4, scale={ht_scale:.3f})")

    # Half-Cauchy: the very-heavy-tail extreme (df=1). Shows the upper bound on tail weight.
    _, hc_scale = stats.halfcauchy.fit(u_pos, floc=0)
    fams["half_cauchy"] = (lambda x, s=hc_scale: stats.halfcauchy.pdf(x, 0, s),
                           f"Half-Cauchy (scale={hc_scale:.3f})")

    # log-Normal on u>0: the "is depth log-symmetric" candidate (family C).
    ln_s, _, ln_scale = stats.lognorm.fit(u_pos, floc=0)
    fams["lognormal"] = (lambda x, s=ln_s, sc=ln_scale: stats.lognorm.pdf(x, s, 0, sc),
                         f"log-Normal (s={ln_s:.2f})")

    return fams


# ----------------------------------------------------------------------------
# structure summary (the REJECT/KEEP logic -- keyed to structure, NOT in-sample fit)
# ----------------------------------------------------------------------------
def _structure_summary(dd: pd.Series, bull: np.ndarray, bear: np.ndarray) -> str:
    """Text readout of the STRUCTURAL facts that license rejecting/keeping families.

    Deliberately reports only structure (bound, atom size, skew, tail), because those are
    the out-of-sample-STABLE properties we're allowed to decide on -- not which family had
    the smallest error on this sample.
    """
    def atom(x):  # fraction sitting AT the boundary (u ~ 0)
        return float(np.mean(-x <= ATOM_TOL))

    lines = []
    lines.append("STRUCTURAL FACTS (what we may REJECT a family on -- these are OOS-stable):")
    lines.append(f"  dd is hard-bounded at 0 (max observed = {dd.max():+.4f}).")
    lines.append(f"  BULL: atom at peak P(dd~0) = {atom(bull)*100:4.1f}%   skew={stats.skew(bull):+.2f}")
    lines.append(f"  BEAR: atom at peak P(dd~0) = {atom(bear)*100:4.1f}%   skew={stats.skew(bear):+.2f}")
    lines.append("")
    lines.append("REJECT / KEEP (from STRUCTURE only; tail-weight ties go to the OOS sweep):")
    lines.append("  x Normal (unbounded)        REJECT: puts mass above 0, an impossible region.")
    lines.append(f"  x any smooth density, BULL  REJECT for bull: can't represent a {atom(bull)*100:.0f}% atom at 0")
    lines.append("                              -> bull wants a HURDLE (atom P(u=0) + slab on u>0).")
    lines.append("  ? Half-Normal vs Half-t     KEEP BOTH as candidates for the BEAR bulk; the eye")
    lines.append("                              can't settle thin-vs-heavy tail -> let OOS likelihood pick.")
    lines.append("  ~ log-Normal on u>0         admissible for the bull SLAB / bear body; check row 3.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------------
def make_figure(window_weeks: int, save_path: _pl.Path) -> None:
    import matplotlib.pyplot as plt

    ds = load_regime_dataset(start=DATA_START, include_vix=False, include_macro=False)
    dd = ds.drawdown(window_weeks).dropna()
    lab = pagan_sossounov_label(ds.weekly_price).reindex(dd.index).ffill()

    bull = dd[lab == 0].to_numpy()
    bear = dd[lab == 1].to_numpy()
    print(_structure_summary(dd, bull, bear))

    states = [("BULL", bull, "steelblue"), ("BEAR", bear, "firebrick")]

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(
        f"Drawdown emission shape  [window={window_weeks}wk]  "
        "-- design-time: reject on STRUCTURE, let OOS pick the tail",
        fontsize=12, y=0.995,
    )

    # ---- ROW 1: dd itself (signed), per state -- shows the 0-ceiling + bull atom ----
    for j, (nm, x, c) in enumerate(states):
        ax = axes[0, j]
        ax.hist(x, bins=60, density=True, color=c, alpha=0.55)
        ax.axvline(0, color="k", lw=1.2, ls="--", label="dd=0 (hard ceiling)")
        atom = np.mean(-x <= ATOM_TOL)
        ax.set_title(f"{nm}: dd (signed)   atom at 0 = {atom*100:.0f}%   n={len(x)}", fontsize=10)
        ax.set_xlabel("dd = price/peak - 1"); ax.set_ylabel("density")
        ax.legend(fontsize=7)

    # ---- ROW 2: u = -dd, per state, with candidate NON-NEGATIVE family overlays ----
    for j, (nm, x, c) in enumerate(states):
        ax = axes[1, j]
        u = -x
        atom_mass = float(np.mean(u <= ATOM_TOL))
        u_pos = u[u > ATOM_TOL]
        # histogram of the CONTINUOUS part (u>0)
        ax.hist(u_pos, bins=50, density=True, color=c, alpha=0.35,
                label=f"u=-dd, u>0 (mass {1-atom_mass:.0%})")
        # the atom drawn as a bar at the origin, scaled to be visible
        ax.bar([0], [atom_mass / max(np.diff(np.histogram(u_pos, bins=50)[1])[0], 1e-6)],
               width=0.004, color="k", alpha=0.8,
               label=f"ATOM P(u=0)={atom_mass:.0%} (no continuous density holds this)")
        # family overlays (fit to u>0; each scaled by the u>0 mass so it sits under the hist)
        fams = _fit_families(u_pos)
        grid = np.linspace(0, u_pos.max() * 1.05, 400)
        styles = {"half_normal": "-", "half_t_df4": "--", "half_cauchy": ":", "lognormal": "-."}
        for name, (pdf, lbl) in fams.items():
            ax.plot(grid, pdf(grid) * (1 - atom_mass), styles.get(name, "-"),
                    lw=1.6, label=lbl)
        ax.set_title(f"{nm}: u=-dd with candidate families (atom handled separately)", fontsize=10)
        ax.set_xlabel("u = -dd  (depth underwater, >=0)"); ax.set_ylabel("density")
        ax.set_xlim(-0.01, np.percentile(u_pos, 99) * 1.1 if len(u_pos) else 0.5)
        ax.legend(fontsize=6.5)

    # ---- ROW 3: log(u) for u>0, per state -- the log-symmetry / heavy-tail view ----
    for j, (nm, x, c) in enumerate(states):
        ax = axes[2, j]
        u = -x
        lu = np.log(u[u > ATOM_TOL])
        ax.hist(lu, bins=50, density=True, color=c, alpha=0.55)
        # a Normal overlay in log space = a log-Normal in u space; deviation shows heavy tail
        mu, sd = lu.mean(), lu.std()
        g = np.linspace(lu.min(), lu.max(), 300)
        ax.plot(g, stats.norm.pdf(g, mu, sd), "k-", lw=1.4,
                label=f"Normal in log-space (= log-Normal)\nskew={stats.skew(lu):+.2f} kurt={stats.kurtosis(lu):+.2f}")
        ax.set_title(f"{nm}: log(u) for u>0  (left-skew/heavy = not log-Normal)", fontsize=10)
        ax.set_xlabel("log(-dd)"); ax.set_ylabel("density")
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved figure -> {save_path}")
    try:
        plt.show()
    except Exception:
        pass


def main(argv):
    window = 52
    if "--window" in argv:
        window = int(argv[argv.index("--window") + 1])
    out = _pl.Path(__file__).resolve().parents[1] / "outputs" / (
        "drawdown_emission_shape.png" if window == 52
        else f"drawdown_emission_shape_{window}wk.png")
    make_figure(window, out)


if __name__ == "__main__":
    main(_sys.argv[1:])
