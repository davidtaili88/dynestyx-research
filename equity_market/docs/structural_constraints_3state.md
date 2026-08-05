# Structural constraints of the 3-state regime nowcast

*The shipped model: `regime_model_3state.py`. This inventories every STRUCTURAL
choice — the things baked into the model's shape that the data works THROUGH and
cannot undo (as opposed to PRIORS, which only seed learnable values the likelihood
then overrules). Reference framing: notebooks/07_hidden_markov_model.ipynb, where the
loaded-die emission probs are structural constants and the transition matrix is a
learned prior.*

---

## The one-sentence framing (use as the opening slide)

A **prior** says *"here's my guess for a value; data, go refine it."* A **structural
constraint** says *"here's a fact about the world's SHAPE; build it in."* With ~3,600
weeks of data the likelihood overrules priors, so **every real modeling decision in
this model is structural** — structure is what changes what the model can express.

---

## The 7 structural constraints

> **A note on "two volatilities" (constraints #4 and #5):** the model has TWO distinct
> quantities both called "vol," measured from different things:
> - **`v_t` / the vol ladder (#4)** = the MEAN of an OBSERVED channel: the std of the
>   ~5 DAILY returns *inside* one week ("how jumpy were the days this week"). It is
>   DATA the model sees, and it is the strong bear-vs-tbull separator.
> - **`return_vols` (#5)** = an INFERRED PARAMETER: the width (`scale`) of the WEEKLY
>   return `r_t` distribution ("across weeks in this state, how spread are the weekly
>   totals"). Not observed; not a separator.
> Same word, different observables — intra-week daily scatter (#4) vs weekly-return
> width (#5).

### 1. Three states (BEAR / TURBULENT_BULL / BULL)
- **What:** `K = 3`. A hidden regime each week is exactly one of three types.
- **Why:** A 2-state model has nowhere to put a transient vol spike except BEAR, so
  P(bear) whipsawed on every ~1-2 week burst (~40% of all turbulent weeks). The 3rd
  state, TURBULENT_BULL, is a *home for turbulence that is not bear*.
- **Effect:** This is THE structural fix for the whipsaw. Reduced 0.5-crossings from
  193 (2-state) to ~16.

### 2. First-order Markov transitions (memoryless dwell)
- **What:** Next week's regime depends only on this week's regime, via a fixed 3×3
  transition matrix `A`. Row = [stay, split the rest over the other two].
- **Why:** Standard HMM structure; keeps the model tractable and the filter exact.
- **Consequence (important, and a known limitation):** dwell time is GEOMETRIC —
  P(leave) is the same on week 1 and week 100 of a regime. Attempts to add
  duration/inertia on top (min-dwell floor, negative-binomial tail) were built and
  **disproven** — belief drains to the terminal phase and collapses back to geometric.
  So mature-regime inertia is *not* available on the transition side.

### 3. Drift ladder: mu_bear < mu_tbull < mu_bull (ordering by construction)
- **What:** `mu_tbull = mu_bear + gap1`, `mu_bull = mu_tbull + gap2`, with each gap a
  non-negative HalfNormal. The magnitudes are learned; the **ordering** is structural.
- **Why:** Bull-vs-bear is *definitionally* a drift distinction. Building the means as
  a rising ladder PINS THE LABELS — NUTS never sees the mislabeled (swapped) modes, so
  no label-switching. (A free per-state mean would let the sampler relabel states.)
- **Role:** Drift does IDENTIFICATION (which state is which), not separation.

### 4. v_t vol ladder: calm(BULL) < turbulent(TBULL) < violent(BEAR)  ← THE KEY FIX
- **What:** a THREE-rung realized-vol ladder, same construction: each higher rung =
  lower rung + a non-negative HalfNormal gap. BEAR gets its OWN vol mean, ABOVE TBULL.
- **Why / the story:** Originally BEAR and TBULL were FORCED to share one vol mean
  (structural constraint: `v_loc = [turb, turb, calm]`). Then the only thing
  separating bear from turbulent-bull was the tiny weekly drift gap. When a real bear
  hit, the STRONG channel (vol) said "turbulent" for BOTH equally and couldn't break
  the tie → the filter split mass ~50/50 → **P(bear) capped at ~0.71** (under-confident).
- **The fix:** ADD a parameter (`log_vol_bear_extra`) giving BEAR a higher vol mean.
  Now a deep, violent drawdown reads unambiguously as BEAR through the strong channel.
- **Effect:** lifted max P(bear) in real bears from **0.71 → 1.0**.
- **Teaching point (great slide):** this LOOKS like "just a prior" (`HalfNormal(...)`),
  but what changed behavior is that a PARAMETER NOW EXISTS. The data always contained
  "bears are more violent than transient spikes"; the model previously had no dial to
  absorb it. Structure = adding the dial, not tuning it.

### 5. Return-vol structure: a THREE-rung ordered ladder (bull < tbull < bear)
- **What:** each state has its OWN weekly-return spread, built as an ordered ladder
  `bull < tbull < bear` (non-negative HalfNormal gaps), same construction as the drift
  and v_t ladders. `return_vols = [bear, tbull, bull]` with bear widest.
- **Why 3-way, not shared:** originally bear & tbull SHARED one high value (`[high,
  high, low]`), on the theory that return-vol was redundant with the v_t ladder — an
  8wk-return-std vs 8wk-mean-v_t correlation of ~0.76 suggested so. **Testing it proved
  the opposite:** fitting a 3-way ladder, BEAR learned a return-vol ~2x TBULL's
  (bear ~0.038, tbull ~0.019, bull ~0.011). The 0.76 was on SMOOTHED 8-week windows,
  which averages away the per-week crash tails (-8%, -12% weeks) that make bear's
  *weekly* return spread much wider. So return-vol carries REAL bear-vs-tbull signal
  that v_t (intra-week choppiness) doesn't — both channels earn their place.
- **Lesson for a slide:** a plausible redundancy argument (0.76 correlation) was WRONG
  because it was measured at the wrong timescale; testing beat intuition. (Note: making
  it 3-way did NOT reduce whipsaw — like #4, more separation = a bit more single-week
  jumpiness. Whipsaw is a separate, still-open, emission-side problem.)

### 6. Emission families: StudentT on returns, Normal on log-vol (+ conditional independence)
- **What:** `r_t ~ StudentT`, `v_t ~ Normal`, and given the state the two are treated
  as independent (the `_JointRV` sums their per-dimension log-probs). Hard family
  choices — only the loc/scale inside them are learned.
- **Why StudentT on r_t:** weekly equity returns have fat tails; a crash is a routine
  in-regime OUTLIER, not evidence of a regime change. Fat tails stop a single -10%
  week from forcing a state flip.
- **Why Normal on v_t:** v_t is already log(realized vol), which is ~symmetric, so
  Normal-on-log = LogNormal on raw vol (right shape). A StudentT-on-v_t experiment
  collapsed to ~Gaussian, so no benefit.
- **Why conditional independence:** standard, tractable simplification; the state
  absorbs most of the r_t–v_t correlation (leverage effect).

### (NOT a separate constraint) Bear = low-drift AND high-vol corner
- This is a PROPERTY that FALLS OUT of #3 and #4, not an additional mechanism. There
  is no extra code, parameter, or sample for it. #3 already makes BEAR the lowest drift
  rung; #4 already makes it the highest vol rung. The only substantive point is that
  the two ladders are ALIGNED (both put BEAR at the extreme), and that alignment is
  just how the arrays are indexed: `mean_return=[bear,tbull,bull]` and
  `v_loc=[bear,tbull,calm]` both place BEAR at index 0. Listed here only to note the
  reinforcement; it is NOT counted among the constraints. (Earlier drafts double-counted
  it as "#7" — it is a rehash of #3 + #4.)

### 7. Fixed uniform initial belief
- **What:** `initial_condition = Categorical(uniform over 3 states)` — a hard constant,
  not learned.
- **Why:** We have no prior reason to favor a start state; it only sets the filter's
  t=0 belief and washes out within a few weeks.

---

## What is NOT structural (priors — for contrast on a slide)

These are `numpyro.sample(...)` on learnable values; the data refines them and can
overrule the seed:
- `p_self` (per-state persistence), `off_split` (leave-direction)
- the drift MAGNITUDES (`mean_return_bear`, `drift_gap1`, `drift_gap2`)
- the vol MAGNITUDES (`log_vol_calm`, `log_vol_gap`, `log_vol_bear_extra`, `v_scale`)
- return-vol magnitudes, `tail_dof_raw`
- All `_PRIOR_*` module constants are prior CENTERS (data-informed seeds), not values
  the model is forced to use.

**The recurring pattern:** most parameters are [PRIOR+STRUCTURAL] — a learned
MAGNITUDE wrapped in a structural ORDERING (`x = lower + HalfNormal_gap`). We commit
to what we KNOW (bull>bear drift, bear>tbull vol) and stay humble about how much.

---

## Evaluation context (numbers for a results slide)

3-state (shipped), OOS filtered P(bear), P&S ground truth, 1957-2026:
- max P(bear) in real bears: **1.0** (with the vol-gap fix) — was 0.71 without it
- whipsaw-resistant: rises are gradual multi-week ramps into the 3 real bears
  (1974 / 2002 / 2008-09), and it does NOT fire on bull-market dips
- the vol-gap version trades a little smoothness for that confidence; the exact
  smoothness/confidence balance is the remaining tuning knob (the size of gap #4)
