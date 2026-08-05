# Time-dependent regime switching — investigation & recommendation

**Question (from you):** P(bear) fluctuates too much and reacts to large drawdowns
even when they're just dips. Idea 1 was "something time-dependent — if we just
entered bull we shouldn't be able to switch the next day."

**Short answer:** The principled version of idea 1 is real and worth doing, but it
**cannot** be a prior/knob tweak on the current model — a plain HMM is structurally
incapable of duration-dependence. The fix is **state augmentation** (an HSMM
expressed as an HMM over `(regime, dwell)` pairs), which runs inside the existing
dynestyx filter with **zero framework changes**.

---

## Why the current model can't do it (the structural finding)

The dynestyx HMM filter (`inference/hmm_filters.py`) carries a belief of shape
`(K,)` — one probability per regime — and nothing else. The forward recursion is:

```
log_pred = logsumexp(log_filt_prev[:, None] + log_A_t, axis=0)   # (K,)
log_filt = log_softmax(log_emit_t + log_pred)                    # (K,)
```

The transition matrix `A_t` is built per timestep from wall-clock `(t_now, t_next)`,
**not** from how long the chain has been in its current state. This is the Markov
property: the future depends only on the *current state*, never on time-since-entry.

A direct consequence: **regime dwell time is geometric.** P(leave) is identical on
week 1 and week 100 of a regime. "Just entered → can't flip next day" is precisely a
*non-geometric* dwell requirement, so no persistence prior (`Beta(500,3)` or
anything else) can produce it. We already saw this empirically — `Beta(500,3)` still
gets dragged to ~0.986 by the likelihood (see `regime_model_2state.py` docstring).
That's not under-tuning; it's the model class.

**So: idea 1, done as a prior, is a dead end. Done as state augmentation, it works.**

---

## The fix: state augmentation (HSMM-as-HMM)

Give each regime a **dwell counter** and expand the state space from `K` regimes to
`K'` = `(regime, dwell-phase)` pairs. A "just-entered bull" is a *different augmented
state* than a "20-week-old bull," so they can carry *different exit probabilities*.
The filter code is untouched — it just runs over a larger `K'`. Confirmed feasible:
`state_dim` is inferred from `initial_condition`'s support, and `state_evolution`
returns an arbitrary `Categorical(probs=A_aug[x])`, so any `K'×K'` matrix is legal.

Two variants, in increasing fidelity:

### Variant A — min-dwell "staircase" (simplest, most literal to your ask)
Split each regime into `d` chained sub-phases `s1 → s2 → ... → sd`, where **exit is
only permitted from the later phases**. Entering a regime lands you in `s1`; you're
*forced* to advance through the early phases before any switch is allowed. This
**mechanically forbids** a next-week flip — exactly "we just entered, can't switch."

- Cost: `K' = K × d`. For the 3-state model with `d = 8`: 24 states. Filter is
  O(K'²) per step → 576 vs 9 flops/step, still trivially fast (< a second delta).
- `d` sets the *hard minimum* regime length (weeks). Pick from the P&S dwell
  distribution — e.g. no P&S bear shorter than ~10 weeks ⇒ `d ≈ 8-10` for bear.
- Downside: the floor is a hard hyperparameter, not learned; and a hard floor can
  hurt genuinely-fast transitions (2020 COVID was real and fast — though note P&S
  doesn't even date COVID as a bear, so a floor that suppresses it is arguably
  *correct* against this ground truth).

### Variant B — explicit-duration HSMM, negative-binomial dwell (more faithful)
Same augmentation, but instead of a hard staircase, the dwell phases implement a
**negative-binomial** duration distribution whose parameters NUTS *learns*. This
lets the data tell you the typical regime length rather than you hard-coding it, and
gives a soft (probabilistic) reluctance to switch early instead of a hard wall.

- Cost: same `K × d` order; `d` becomes the NB's max-phase truncation.
- Upside: learned durations, smoother behaviour, still causal and Bayesian.
- Downside: more moving parts; the per-phase→regime bookkeeping for emissions and
  for reading P(bear) needs care (P(bear_t) = sum of filtered mass over all bear
  phases).

---

## How this interacts with the 3-state model (important)

The **3-state TURBULENT_BULL** state and **duration structure** attack the whipsaw
from *different* angles and are **complementary, not redundant**:

- 3rd state fixes *"a transient vol spike has nowhere to go but bear"* — it re-homes
  the ~40% of turbulent weeks that come in 1-2 week bursts into a non-bear state.
  This kills the *emission-driven* whipsaw (vol pops → P(bear) jumps).
- Duration structure fixes *"the chain is allowed to leave a fresh regime instantly"*
  — the *transition-driven* whipsaw. It makes any regime (including the real bear)
  sticky in time.

**Recommended order:** evaluate the now-updated 3-state model *first*. If it already
damps the whipsaw to an acceptable level (plausible — it targets the dominant cause),
you may not need duration structure at all, and you'd avoid the `K×d` complexity. If
residual fast-flipping remains, add Variant A on top of whichever state count wins.

---

## Recommendation

1. **Evaluate the updated 3-state model** (now at parity with the 2-state: OOS
   filtering, walk-forward, P&S ground truth). Measure whipsaw quantitatively —
   switch count, run-length distribution vs P&S, false-alarm rate on dips — so the
   next decision is evidence-based, not eyeballed.
2. **If whipsaw persists**, implement **Variant A (min-dwell staircase)** first: it's
   the smallest change, directly encodes your literal ask, needs no framework work,
   and the `d` floor is readable straight off the P&S dwell histogram. Promote to
   **Variant B** only if the hard floor proves too blunt.
3. **Idea 2 (leading indicators)** is a separate, larger project that changes *what*
   P(bear) reacts to rather than its dynamics. Keep it decoupled from the whipsaw
   fix; revisit after the dynamics are right.

## Notes for whoever implements the augmentation
- Build `A_aug` (K'×K') once from the regime-level params; the emission for augmented
  state `(regime, phase)` is just the regime's emission (phases share it).
- `initial_condition = Categorical(ones(K')/K')`; `state_dim` infers K' automatically.
- P(bear_t) = filtered mass summed over all bear *phases*, not a single index — the
  `filtered_p_bear*` helpers need a phase→regime reduction added.
- Everything else (`fit`, `filtered_p_bear_over`, walk-forward, plotting) is unchanged.
