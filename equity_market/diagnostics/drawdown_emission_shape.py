"""DESIGN-TIME diagnostic: what distribution should the DRAWDOWN emission channel use?

RESOLUTION (2026-08) -- READ THIS FIRST; the DISPOSE step has since fired.
------------------------------------------------------------------------
This script PROPOSES/REJECTS on structure and hands the tie to an OOS sweep (see the
discipline note below). That sweep has now run: a 2x2 ablation (emission {Normal, hurdle}
x reference-peak {trailing-window, event-reset}) DISPOSED in favour of the *misspecified*
single-scale NORMAL, and the shipped model keeps it on purpose -- see
regime_model_4state.INCLUDE_DRAWDOWN / _DRAWDOWN_RESET_PCT and its 82-85 comment. The
reason is counter-intuitive and is exactly why this file is kept as an evidence trail: the
correctly-specified hurdle is FAITHFUL to a laggy channel, which makes the nowcast laggier;
the Normal's mis-specification quietly DOWN-WEIGHTS dd, and that down-weighting helps.
So the "bull wants a HURDLE" recommendation in the structure summary below is SUPERSEDED --
structurally right, empirically rejected. Keep this script to show the misspecification was
seen and chosen through, not overlooked.

WHAT THIS IS (and the discipline behind it)
-------------------------------------------
The drawdown channel emits  dd_t = price_t / (event-reset peak) - 1  (<= 0, causal).
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
    python diagnostics/drawdown_emission_shape.py --reset-pct 0.15   # a different bull-confirm threshold
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

from data_acquisition import load_regime_dataset  # noqa: E402
from pagan_sossounov import pagan_sossounov_label  # noqa: E402


DATA_START = "1957-03-01"
ATOM_TOL = 1e-6  # u <= this counts as "at the peak" (the boundary atom)


# ----------------------------------------------------------------------------
# candidate NON-NEGATIVE families for u = -dd (fit to the u>0 part only; the atom
# at u=0 is handled SEPARATELY as a point mass, since no continuous family can hold it)
# ----------------------------------------------------------------------------
def _halft_pdf(x, df, scale):
    """Half-Student-t: a Student-t folded at 0 (density = 2 * t.pdf for x>=0), scaled.

    Exists ONLY to draw the half-t OVERLAY CURVE on the plot -- scipy ships halfnorm and
    halfcauchy but no 'halft', so we build it. df -> inf recovers Half-Normal; df=1 is
    Half-Cauchy; df=4 is the classic heavy-but-finite-variance middle. We plot it so the
    eye can REJECT it, not adopt it: a half-family pins its MODE at the 0 boundary, but the
    bear underwater-depth has an INTERIOR mode (~5% under), so all half-families are wrong
    for bear. The model went with log-t instead (regime_model_3state ~line 257). So this is
    a rejected-candidate renderer -- kept for the evidence trail, not a proposed emission.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 2.0 * stats.t.pdf(x / scale, df) / scale, 0.0)


def _logt_pdf(x, df, log_mode, log_sigma):
    """log-Student-t on x>0: log(x) ~ StudentT(df, log_mode, log_sigma), density in X-space.

    This is the family the SHIPPED model uses for the dd slab (regime_model_3state ~line 256:
    log(u) ~ StudentT), so it belongs in the overlay lineup as the SURVIVOR -- not to be
    ruled out. Unlike the half-families it has an INTERIOR mode (at exp(log_mode)), matching
    the bear underwater-depth. scipy has no log-t, so we build it: transform to y=log(x) and
    apply the log-Jacobian d y / d x = 1/x, i.e. pdf_X(x) = t.pdf((log x - mode)/sig)/(sig*x).
    df=inf here would recover log-Normal (the other survivor); df=4 gives it a heavier tail.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    ly = (np.log(x[pos]) - log_mode) / log_sigma
    out[pos] = stats.t.pdf(ly, df) / (log_sigma * x[pos])
    return out


def _fit_families(u_pos: np.ndarray) -> dict:
    """Fit each candidate family to the STRICTLY-POSITIVE underwater depths u>0.

    Returns {name: (pdf_callable, label)}. All fixed at loc=0 (the boundary) so they're
    genuine NON-NEGATIVE densities, not shifted ones. Half-t df is fixed at 4 (a shape
    choice, not fit -- we're illustrating tail weight, not optimising it). These are
    DISPLAY candidates for the eye to accept/reject on structure; the half-families were
    subsequently rejected (mode pinned at 0 vs the bear's interior mode) in favour of log-t.
    """
    fams = {}

    # The three HALF-families below (mode pinned AT the 0 boundary) are the ones the plot
    # exists to RULE OUT -- bear's depth has an interior mode, so all three are wrong for
    # bear. They span the tail-weight axis (thin -> very-heavy) so the rejection is visibly
    # not a tail-weight artefact. log-Normal (last) is the survivor the model's log-t built on.

    # Half-Normal: the thin-tailed extreme. scale by scipy MLE.
    _, hn_scale = stats.halfnorm.fit(u_pos, floc=0)
    fams["half_normal"] = (lambda x, s=hn_scale: stats.halfnorm.pdf(x, 0, s),
                           f"Half-Normal (scale={hn_scale:.3f})")

    # Half-t (df=4): the moderate-tail middle. df fixed at 4 (illustrating tail weight, not
    # optimising it); scale picked by a 60-point max-likelihood GRID over u_pos (scipy has
    # no 'halft' to .fit(), so we grid it ourselves -- df stays fixed, only scale is chosen).
    scales = np.linspace(u_pos.std() * 0.3, u_pos.std() * 1.5, 60)
    lls = [np.log(_halft_pdf(u_pos, 4, s) + 1e-300).sum() for s in scales]
    ht_scale = scales[int(np.argmax(lls))]
    fams["half_t_df4"] = (lambda x, s=ht_scale: _halft_pdf(x, 4, s),
                          f"Half-t (df=4, scale={ht_scale:.3f})")

    # Half-Cauchy: the very-heavy-tail extreme (df=1). scale by scipy MLE. Upper bound on tail.
    _, hc_scale = stats.halfcauchy.fit(u_pos, floc=0)
    fams["half_cauchy"] = (lambda x, s=hc_scale: stats.halfcauchy.pdf(x, 0, s),
                           f"Half-Cauchy (scale={hc_scale:.3f})")

    # log-Normal on u>0: an INTERIOR mode (not pinned at 0), matching bear.
    # scipy MLE. 
    ln_s, _, ln_scale = stats.lognorm.fit(u_pos, floc=0)
    fams["lognormal"] = (lambda x, s=ln_s, sc=ln_scale: stats.lognorm.pdf(x, s, 0, sc),
                         f"log-Normal (s={ln_s:.2f})")

    # log-t (df=4): the SHIPPED family (regime_model_3state log(u) ~ StudentT). df fixed at 4
    # to match half-t's convention; mode & sigma from the log(u) sample (interior mode, heavier
    # tail than log-Normal). scipy has no log-t -> built by hand in _logt_pdf (log-Jacobian).
    lt_mode, lt_sigma = np.log(u_pos).mean(), np.log(u_pos).std()
    fams["logt_df4"] = (lambda x, m=lt_mode, s=lt_sigma: _logt_pdf(x, 4, m, s),
                        f"log-t (df=4, mode={np.exp(lt_mode):.3f})")

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

    def slab_mode(x):  # histogram mode of the u>0 slab -- is the depth mode AT 0 or INTERIOR?
        u = -np.asarray(x, dtype=float)
        up = u[u > ATOM_TOL]
        if up.size == 0:
            return float("nan")
        h, edges = np.histogram(up, bins=40, density=True)
        return float(0.5 * (edges[np.argmax(h)] + edges[np.argmax(h) + 1]))

    bull_mode, bear_mode = slab_mode(bull), slab_mode(bear)

    lines = []
    lines.append("STRUCTURAL FACTS (bound / atom / skew / where the u>0 mode sits -- OOS-stable):")
    lines.append(f"  dd is hard-bounded at 0 (max observed = {dd.max():+.4f}).")
    lines.append(f"  BULL: atom P(dd~0) = {atom(bull)*100:4.1f}%   skew={stats.skew(bull):+.2f}   "
                 f"u>0 slab mode ~= {bull_mode*100:.1f}% under (at the boundary)")
    lines.append(f"  BEAR: atom P(dd~0) = {atom(bear)*100:4.1f}%   skew={stats.skew(bear):+.2f}   "
                 f"u>0 slab mode ~= {bear_mode*100:.1f}% under (INTERIOR)")
    lines.append("")
    lines.append("REJECT / KEEP -- the OOS sweep has ALREADY fired; this is the POST-resolution readout:")
    lines.append("  x Normal (unbounded)        REJECT structurally: puts mass above 0, an impossible")
    lines.append("                              region -- YET it is what SHIPS (see below; its very")
    lines.append("                              misspecification helps by down-weighting a laggy channel).")
    lines.append(f"  BULL, u>0 slab mode AT 0    a boundary-mode family (Half-*) FITS bull's slab shape;")
    lines.append(f"                              bull's real problem is the {atom(bull)*100:.0f}% ATOM at 0, which NO")
    lines.append("                              continuous density holds -> bull STRUCTURALLY wants a HURDLE")
    lines.append("                              (atom + slab). BUT the 2x2 OOS sweep REJECTED the hurdle:")
    lines.append("                              faithful to a laggy channel -> laggier nowcast; the")
    lines.append("                              misspecified Normal ships. See RESOLUTION at top of file.")
    lines.append(f"  x Half-* for BEAR           REJECT for bear: Half-Normal/Half-t/Half-Cauchy pin the mode")
    lines.append(f"                              AT 0, but bear's slab mode is INTERIOR (~{bear_mode*100:.1f}% under).")
    lines.append("                              (Overlaid in row 2 only so they can be seen to fail.)")
    lines.append("  ~ log-Normal / log-t on u>0 the survivors: interior mode, matching bear. The model")
    lines.append("                              uses log-t (per-state df learns the tail); log-Normal is the")
    lines.append("                              df->inf special case. Check row 3 for the tail weight.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# the figure
# ----------------------------------------------------------------------------
def make_figure(reset_pct: float, save_path: _pl.Path) -> None:
    import matplotlib.pyplot as plt

    ds = load_regime_dataset(start=DATA_START, include_vix=False, include_macro=False)
    dd = ds.drawdown(reset_pct=reset_pct).dropna()  # shipped EVENT-RESET dd (no window)
    lab = pagan_sossounov_label(ds.weekly_price).reindex(dd.index).ffill()

    bull = dd[lab == 0].to_numpy()
    bear = dd[lab == 1].to_numpy()
    print(_structure_summary(dd, bull, bear))

    states = [("BULL", bull, "steelblue"), ("BEAR", bear, "firebrick")]

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(
        f"Drawdown emission shape  [event-reset {reset_pct:.0%}]  "
        "-- design-time evidence trail (OOS RESOLVED: misspecified Normal ships; see docstring)",
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
        # rejected half-families: thin greyish lines; the two SURVIVORS (log-Normal, log-t):
        # bold coloured so the "what we chose" curves stand out from the "ruled out" ones.
        styles = {"half_normal": "-", "half_t_df4": "--", "half_cauchy": ":",
                  "lognormal": "-.", "logt_df4": (0, (5, 1))}
        emphasis = {"lognormal": dict(lw=2.2, color="darkgreen"),
                    "logt_df4": dict(lw=2.4, color="black")}
        for name, (pdf, lbl) in fams.items():
            ax.plot(grid, pdf(grid) * (1 - atom_mass), linestyle=styles.get(name, "-"),
                    label=lbl, **{"lw": 1.4, **emphasis.get(name, {})})
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
    reset_pct = 0.20  # the shipped event-reset threshold
    if "--reset-pct" in argv:
        reset_pct = float(argv[argv.index("--reset-pct") + 1])
    out = _pl.Path(__file__).resolve().parents[1] / "outputs" / (
        "drawdown_emission_shape.png" if reset_pct == 0.20
        else f"drawdown_emission_shape_reset{int(reset_pct*100)}.png")
    make_figure(reset_pct, out)


if __name__ == "__main__":
    main(_sys.argv[1:])
