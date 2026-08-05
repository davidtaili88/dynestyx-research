# HSMM (duration-augmented 3-state) — implementation plan

**Status: IMPLEMENTED + EVALUATED (2026-07-21). RESULT: partial success / design flaw
found — see "EVALUATION RESULT" below before reading the rest.** Both HSMM models exist
(`regime_model_2state_hsmm.py`, `regime_model_3state_hsmm.py`).

---

## EVALUATION RESULT (2026-07-21) — the min-dwell staircase does NOT fix whipsaw

Four-model comparison, filtered P(bear) over 1957-2026, P&S ground truth:

| model         | whipsaw (0.5-crossings) | max Pb in bears | mean Pb in bulls | %bull>0.5 |
|---------------|-------------------------|-----------------|------------------|-----------|
| plain 2-state | 193                     | 1.000           | 0.357            | 34.5%     |
| plain 3-state | **16**                  | 0.710           | **0.106**        | **2.7%**  |
| HSMM 2-state  | 161                     | 1.000           | 0.364            | 35.3%     |
| HSMM 3-state  | 170                     | 1.000           | 0.310            | 30.0%     |

- HSMM FIXED under-confidence: max P(bear) in real bears 0.71 → 1.00. Good.
- HSMM DID NOT fix whipsaw: 170 crossings / 30% false alarms, ~same as plain 2-state
  and vastly WORSE than plain 3-state (16 / 2.7%).

ROOT CAUSE (diagnosed, not a bug): min-dwell via forced phase-advance only floors the
dwell of a FRESHLY-ENTERED regime. Because all bull phases emit IDENTICALLY (phase is
latent, emissions phase-invariant), the filter drains phase-belief straight to the
TERMINAL phase and parks it there (observed: 96.6% mass on bull_phase_17 before a
whipsaw week). At the terminal, bear is one step away, so a crash flips ~0.5 mass to
bear in a SINGLE week. The 17-wk floor protected nothing because a mature bull's belief
long since reached the exit gate. The floor only bites for the first 17 weeks after a
bear→bull exit — a tiny fraction of history.

So: the staircase delivers EXACTLY the user's literal ask ("just-entered bull can't
switch") and NOTHING MORE. It does not make MATURE regimes sticky, which is what the
whipsaw actually needs.

STRIKING SECONDARY FINDING: the PLAIN 3-STATE is the clear whipsaw winner (16 crossings,
2.7% false alarms) — its only weakness was under-confidence (0.71 cap). That reframes
the whole problem: we may not need duration structure at all; we need to fix the plain
3-state's UNDER-CONFIDENCE while keeping its excellent whipsaw resistance.

### Two real fixes (supersede the staircase)
1. **Make the geometric TAIL stickier, not the floor** (fixes mature-regime whipsaw):
   the plain 3-state already nearly nails whipsaw via its Beta(500,3) persistence. Its
   under-confidence comes from BEAR↔TURBULENT_BULL emission overlap, NOT from dwell. So
   fix that directly: widen the bear-vs-tbull separation (drift and/or vol), OR let the
   filter accumulate bear evidence faster once turbulence PERSISTS (a self-loop-heavy
   bear that, once entered, is hard to leave). This keeps the winning plain-3-state
   whipsaw and lifts the 0.71 cap.
2. **Duration on the TAIL (negative-binomial / high self-loop), not a hard floor:** if
   we still want duration, put it on the persistence hazard of MATURE regimes (raise
   p_self toward 1 with a longer effective dwell) rather than a fresh-entry staircase.
   This is Variant B from the investigation doc; the staircase (Variant A) is now known
   to be the wrong tool for THIS problem.

RECOMMENDATION: park the HSMM staircase (keep files as the documented negative result),
and pivot to fixing the plain 3-state's under-confidence — it is 90% of the way there.

---

## Original plan (kept for provenance; superseded by the result above)

## What changed the goal (why this plan supersedes the doc's framing)

The investigation doc framed duration structure as a **whipsaw** fix (make regimes
sticky in time). The new observation from the P(bear) charts reframes it:

- 2-state: P(bear) reaches 1.0 but flaps to 1.0 on every dip (decisive, wrong).
- 3-state: P(bear) never exceeds ~0.65 even in 2008 (calm on dips, but under-confident
  in *real* bears).

**Both are the same root defect:** BEAR and TURBULENT_BULL are separated ONLY by the
drift ladder, and weekly drift is a weak signal (~0.6% gap vs ~2-3% weekly noise).
So in any high-vol stretch the filter can't decide *which* high-vol state it's in;
mass splits, and P(bear) caps out.

**The reframing:** duration is not just a stickiness knob — it is the STRONG
DISCRIMINATOR that drift fails to be. In the world, BEAR and TURBULENT_BULL differ
mainly by PERSISTENCE:
- TURBULENT_BULL = transient (1-2 week vol bursts, ~40% of turbulent weeks).
- BEAR = sustained (P&S bears run months).

If we encode that asymmetry in the dwell structure, a high-vol stretch that PERSISTS
past a couple weeks can only be explained as BEAR — forcing the filter to commit.
That fixes under-confidence AND whipsaw at once. This is the design driver.

## Design decisions (this plan's departures from the generic doc)

### D1. Asymmetric min-dwell per regime — the crux
NOT a uniform `d`. The design is a MINIMUM-dwell floor per regime, asymmetric by
intent, driven by the user's rule "bulls ≥ 4 months; bears however long if the
drawdown is large enough."

IMPORTANT distinction that shapes the whole design: a staircase floor is CAUSAL — it
can only enforce "stay ≥ N weeks," it CANNOT enforce "stay while the drawdown is
large" (the filter doesn't know future drawdown depth at time t). So:
- "bulls ≥ 4 months" → a clean MIN-DWELL floor. ✓ mechanism-native.
- "bears however long if drawdown large" → NOT a floor. "However long" = little/no
  floor; "if large enough" is an EMISSION condition already handled by the return
  channel. So the bear side leans on the likelihood (deep negative returns keep it in
  bear), not on the dwell clock.

**Validated P&S phase-length histograms (1957-2026, `pagan_sossounov_label`):**
- BULL: min 27 wk, p10 43, median 114, max 321.  → a 17-wk (4-mo) floor never clips
  a single real bull (shortest is 27). Safe & conservative. Could go to 26 wk (6 mo)
  and still clip none, but 17 is the user's spec and the guaranteed-safe choice.
- BEAR: min 15 wk, p10 20, median 42, max 131.  → P&S bears have a NATURAL ~15-wk
  minimum. So a small bear floor is data-justified, not just tolerated.

Chosen floors:
- **BULL: 17 wk (4 mo).** User rule; safe vs the 27-wk shortest real bull.
- **TURBULENT_BULL (3-state only): 2-3 wk cap** — transient by construction; forces
  persistent turbulence out into BEAR.
- **BEAR: SEE OPEN QUESTION Q1'** — either (a) small floor ~4-6 wk (lean on emissions,
  matches "however long" intent) or (b) data-driven ~15 wk (matches P&S's natural
  bear minimum). Leaning (a) per user's stated instinct; (b) is the more P&S-faithful
  alternative. Resolve before coding.

### D2. Variant A (min-dwell staircase) first, not the NB-HSMM
The doc offered a negative-binomial (learned duration) variant. Start with the hard
staircase because: (a) the asymmetry above is the effect we want and the staircase
encodes it directly; (b) fewer parameters while we validate the mechanism; (c) the
`d` values come straight off P&S, so they're not free knobs. Promote to NB only if
the hard floors prove too blunt (e.g. suppress a genuinely fast bear onset).

### D3. State layout — SHARED bull-complex clock (revised per user)
CRITICAL REVISION: the min-dwell is on the BULL COMPLEX (BULL + TURBULENT_BULL
together), NOT on each state separately. The 17-wk clock counts time since entering
the bull complex and spans BULL↔TBULL freely.

Rule (3-state):
- A single dwell phase `i = 1..17` tracks weeks-since-bull-complex-entry.
- During phases 1..16 (protected window): free movement BULL↔TURBULENT_BULL at every
  phase, and NO transition to BEAR is allowed. The clock KEEPS COUNTING across a
  BULL→TBULL→BULL excursion — it does NOT reset on a turbulence spike (user decision).
  Both BULL and TBULL at phase i advance to phase i+1.
- At phase 17 (terminal): BEAR becomes reachable, from EITHER BULL or TURBULENT_BULL
  (user decision — no forced turbulence-first hop). Hard exit.
- Entering the bull complex (from a bear exit) lands at phase 1, in BULL.

Augmented states:
- Bull complex: `(sub, i)` for sub ∈ {BULL, TBULL}, i ∈ 1..17  → 2×17 = 34 states.
  (Two coordinates in the protected window: the phase clock AND which sub-state, so
  the emission can differ BULL vs TBULL while they share one clock.)
- Bear: `bear_1..bear_5` (5-wk floor, see floor table) → 5 states.
- K' (3-state) = 34 + 5 = 39. Still trivial (O(K'²) = 1521 flops/step).

2-state HSMM (no TBULL): bull complex is just `bull_1..bull_17` (17 states, single
sub-state), bear `bear_1..bear_5`. K' (2-state) = 22. The 2-vs-3 contrast is now
EVEN sharper: 2-state has a plain 17-wk bull floor; 3-state additionally lets
turbulence come and go INSIDE that protected window without escaping to bear.

- Emission for `(sub, i)` = sub's existing regime emission (BULL vs TBULL vs BEAR
  emission params, unchanged). Phase i is a clock, not a new emission.

### D4. Reading P(bear) — must sum over phases
P(bear_t) = filtered mass summed over ALL bear phases `bear_1..bear_{d_bear}`, not a
single index. Same for any per-regime readout. The `filtered_p_bear*` helpers need a
phase→regime reduction (a (K', K_regime) 0/1 grouping matrix applied to the filtered
vector). This is the ONLY change to the readout/eval code; `fit`,
`filtered_p_bear_over`, walk-forward, plotting are otherwise unchanged.

### D5. Label-switching / identification
Unchanged in spirit: the drift ladder + vol ordering still pin regime labels. The
phase expansion is deterministic bookkeeping on top of the identified regimes, so it
introduces no new label-switching axis. Build A_aug from the SAME regime-level
sampled params (p_self, off_split, drift ladder, vols) — do not add per-phase free
params.

## Feasibility (already confirmed)
- `state_dim` is inferred from `initial_condition`'s support → set
  `initial_condition = Categorical(ones(K')/K')` and K' infers automatically.
- `state_evolution` may return any `Categorical(probs=A_aug[x])` → arbitrary K'×K'
  legal. NO dynestyx framework changes. (Verified in hmm_filters.py: filter is
  agnostic to K, carries a (K',) belief, O(K'²) scan.)

## Build order
1. **Diagnostics first (prerequisite for D1):** P&S bear dwell-length histogram +
   turbulent-burst length distribution → pick d_bear, d_tbull, d_bull. Do NOT guess
   these; they are the whole design.
2. New module `regime_model_3state_hsmm.py` (keep the plain 3-state as baseline, same
   as 2-state was kept). Reuse `_JointRV` and the regime-level priors verbatim.
3. Build `A_aug` (K'×K') from regime params + the phase staircase; expand
   `mean_return`/`v_loc`/`return_vols` to length K' via a phase→regime index map.
4. Add phase→regime grouping to `filtered_p_bear` / `filtered_p_bear_over`.
5. Smoke-test (small window, tiny NUTS) → then full `main()` with P&S ground truth.
6. Compare P(bear) against BOTH current models: does it (a) reach ~1 in 2008/2020-P&S
   bears [fixes under-confidence] AND (b) stay calm on bull-market dips [keeps whipsaw
   fix]? That two-sided check is the acceptance criterion.

## Decisions locked with user
- **Exit rule (was Q2): HARD.** Exit only from the terminal phase = strict min-dwell.
  A regime physically cannot switch before its floor. Soften to NB only if too blunt.
- **Scope (was Q3): BUILD BOTH a 2-state and a 3-state HSMM and compare/contrast**,
  mirroring the existing plain 2state-vs-3state split. The contrast is meaningful:
  the 2-state HSMM has no TURBULENT_BULL, so its ONLY whipsaw lever is the 17-wk bull
  floor. The comparison answers: does a 4-month bull floor ALONE tame the whipsaw
  (2-state), or is the transient-turbulence state ALSO needed (3-state)?
- **Bull floor: 17 wk (4 mo).** User rule; safe vs 27-wk shortest real bull.

## Bear floor: DECIDED — small (~5 wk)
Lean on the emission likelihood, not the dwell clock, for "bears however long if the
drawdown is large enough." The ~5-wk floor only prevents 1-week bear blips; deep
negative returns (the emission) are what hold a real bear and let P(bear) reach ~1,
and a recovery in returns lets bear exit promptly. This is the user's stated intent.

## FINAL FLOOR TABLE (all decided)
| Regime          | Min dwell | Source                                             |
|-----------------|-----------|----------------------------------------------------|
| BULL            | 17 wk     | user rule (4 mo); safe vs 27-wk shortest real bull |
| TURBULENT_BULL  | 2-3 wk    | transient by construction (3-state only)           |
| BEAR            | ~5 wk     | small floor; emissions carry "however long"        |

K' (3-state) = 39 states (34 bull-complex + 5 bear), K' (2-state) = 22 states.
See D3 for the SHARED bull-complex clock (revised). Both trivial for O(K'²).
