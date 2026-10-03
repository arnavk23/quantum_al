# Legacy results (pre-rebuild)

These files were produced by secondary-experiment scripts from before the
0.1.0 rebuild (`scripts/discrete_classification.py`,
`scripts/transfer_learning.py`, `scripts/multi_property_optimization.py`,
`scripts/spurious_correlation_analysis.py` and
`scripts/observable_sensitivity_analysis.py`). Those scripts were removed in commit `ed33f9c` and are
recoverable from git history (`git show ed33f9c^:scripts/<name>.py`).

They are kept for provenance only. **No claim in the JOSS paper, the
documentation or the current manuscript depends on them**, and several have
known problems documented in `../SUMMARY.md` ("Secondary experiments"):

| File | Status |
|---|---|
| `discrete_classification.json`, `classification_results.png` | synthetic 6-class data, not real crystal-system labels |
| `transfer_learning*` | transfer and from-scratch outputs are identical: a bug, not a finding |
| `multi_property*` | R² near zero throughout; inconclusive |
| `spurious_correlation*` | deliberately synthetic noise-injection study; ran correctly |
| `observable_sensitivity_legacy*`, `observable_sensitivity_analysis.png` | superseded by `../observable_sensitivity.json` from `benchmarks/run_primary_benchmark.py --stage sensitivity` |

Every file directly under `results/` (outside this folder) is produced by a
script in `benchmarks/`; see `benchmarks/README.md`.
