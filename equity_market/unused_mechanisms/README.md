# Unused mechanisms (archive)

Model variants and their consumers that were **explored but are not part of the shipped
pipeline**. Kept as reference implementations, moved here 2026-08-16 for a clean codebase.
The one live model is `src/models/regime_model_3state.py`; the one live strategy is
`trading_strategies/regime_pbear_strategy.py`.

This folder is an ARCHIVE: it is **not** on `sys.path` (unlike `src/` subfolders via
`_syspath`), so these files are not importable in place and are not expected to run as-is.
To resurrect one, move it back to its original location (below) so the flat imports resolve.

## Contents

| File | What it is | Original location |
|---|---|---|
| `regime_model_2state.py` | The 2-state predecessor of the 3-state model (BEAR vs BULL). Superseded once the third TURBULENT_BULL state fixed the bull-market P(bear) whipsaw. | `src/models/` |
| `regime_model_4state.py` | 4-state variant adding a CALM_BEAR state (negative drift + bull-like low vol) to catch calm/grinding bears the 3-state misses. A real, working model — not shipped in favour of the simpler 3-state. | `src/models/` |
| `regime_model_2state_hsmm.py` | Duration-augmented (HSMM) 2-state variant — explicit min-dwell floors to damp whipsaw. | `src/models/` |
| `regime_model_3state_hsmm.py` | Duration-augmented (HSMM) 3-state variant. | `src/models/` |
| `regime_pbear_strategy_4state.py` | The trading strategy hardwired to the 4-state model (persistence-short variant). Consumer of `regime_model_4state`. | `trading_strategies/` |
| `stationarity_vol.py` | Param-analysis script (`from regime_model_2state import fit`). Consumer of the 2-state model. | `src/param_analysis/` |

## Dependencies to know before resurrecting

- The HSMM models import `regime_model_2state` (and `2state` imports nothing here) — move
  them back together.
- `regime_pbear_strategy_4state.py` imports `regime_model_4state` (move both back). Its fit
  cache uses the model-agnostic `save_fit`/`load_fit` from `src/models/model_utils/persistence.py`
  (no longer coupled to any specific model); that dir is on `sys.path` in the live tree.
- The live 3-state strategy's `--model 4state` option was removed when the 4-state model was
  archived (`_load_model` now raises for any key other than "3state").
