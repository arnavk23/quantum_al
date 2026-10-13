# Changelog

Notable changes to `quantum_al`, by release. Dates are UTC, from git history.

## [Unreleased]

## [0.2.0] - 2026-10-14

Archived on Zenodo: [10.5281/zenodo.23351262](https://doi.org/10.5281/zenodo.23351262).

### Added
- `quantum_al.acquisition`: multi-property criteria from Bayesian optimal
  experimental design (joint and marginal EIG, total correlation, whitened
  trace and maximum eigenvalue), `EnsembleCriterionSelector`, and greedy batch
  EIG with a (1 - 1/e) guarantee (`GreedyBatchEIGSelector`).
- `quantum_al.diagnostics`: measures whether the joint and correlation-blind
  criteria would select different batches, and why.
- `quantum_al.synthetic`: multi-property problems with tunable cross-property
  correlation and noise.
- `quantum_al.loop` and `quantum_al.stats`: the shared pool-based loop,
  learning curves and AULC, paired tests, bootstrap confidence intervals,
  effect sizes and Holm-Bonferroni correction.
- `quantum_al.data_utils.load_multi_task`, which joins separately fetched
  Materials Project properties into a multi-property dataset.
- Sparse-by-construction observables (`QuantumObservableBank(sparse=True)`),
  the measurement-grouping tool in `quantum_al.circuit`, and
  `benchmarks/run_measurement_grouping.py`.
- `benchmarks/run_correlation_sweep.py`: a controlled study of when joint
  acquisition can beat correlation-blind acquisition, with results.
- Sphinx documentation with full derivations, Read the Docs configuration,
  runnable `examples/`, community files, a ruff lint job in CI, and a release
  workflow that publishes to PyPI through trusted publishing.
- Baselines take an optional `seed` for reproducible use outside the
  benchmark scripts (`get_all_baselines(seed=...)`).

### Changed
- `quantum_al.joint_eig` merged into `quantum_al.acquisition`:
  `JointEIGSelector` and `MarginalSumSelector` are
  `EnsembleCriterionSelector("eig")` and `("marginal_eig")`, and
  `per_tree_predictions`, `predictive_covariance` and `oob_residual_variance`
  now live in `acquisition`.
- `quantum_al.operator`, `operator_v2`, `operator_v3` and `operator_sparse`
  merged into `quantum_al.quantum_inspired`, together with `QuantumSelector`
  (previously defined in a benchmark script). Selections are bit-identical
  to the previous modules.
- `quantum_al.baselines`: the nested-class `BaselineFactory` replaced by
  nine top-level classes listed in `BASELINES`; CoreSet is vectorized. With
  the default `seed=None`, selections are identical to before.
- `paper.md` and `paper.bib` moved to `paper/`; `README.md` shortened;
  `results/SUMMARY.md` replaced by a `results/README.md` index, with its
  narrative kept in `docs/findings.rst`.
- Real-data joint-EIG results rerun from a fresh Materials Project download,
  now with per-round selection diagnostics; reported p-values are
  consistently Holm-corrected.

### Removed
- `self_test()` functions, which duplicated the test suite.
- `benchmarks/run_v3_test.py` and `run_v3_ablation.py` and their outputs,
  which reproduced subsets of `v3_all_tasks.json` and `primary_benchmark.json`
  exactly.
- `results/legacy/` and `papers/legacy/` (outputs of scripts removed in
  0.1.0, and a superseded manuscript; both remain in git history) and
  `requirements.txt` (out of sync with `pyproject.toml`).
- The IEEE-format manuscript on the quantum-inspired formalism
  (`papers/quantum_paper_ieee.*`) and the bibliography entries only it cited;
  `papers/` now holds the npj manuscript only.

### Fixed
- Baseline fallbacks raised `ValueError` instead of falling back when
  `n_select` exceeded the number of candidates.
- `oob_residual_variance` raised `AttributeError` rather than its intended
  `ValueError` for a forest fitted without `oob_score=True`.
- `run_joint_eig_experiment.py` printed "band_gap & formation_energy" in its
  summary header for every property pair.

## [0.1.0] - 2026-08-19

- Full honest rebuild of the package after an earlier, less rigorous
  evaluation was found not to reproduce from any code in the repository (see
  the History section of `docs/findings.rst`). Reorganized into an installable package
  (`src/quantum_al/`) with a real Materials Project data pipeline, 9 classical
  active-learning baselines behind a common interface, a real-data benchmark
  harness with paired significance testing (Holm-Bonferroni correction), a
  pytest suite run in CI, and a verified Qiskit circuit realization of the
  covariance-aware formalism with NISQ resource characterization.
- Added `operator_v3.py`, a residual-coupled variant that encodes state from a
  downstream model's per-tree predictions instead of raw features, and its own
  ablation isolating what the fix actually contributes.
- Added `operator_sparse.py`'s dense-vs-sparse Pauli-overhead comparison.

## 2025-09-27 - 2026-06-24: pre-rebuild history

- Initial ML surrogate DFT pipeline, first draft of the quantum-inspired
  covariance-aware active-learning formalism and accompanying paper, and a
  conference submission (NQComp 2026). Reviewer feedback and a subsequent
  direct re-run of the submitted code surfaced that its reported numbers did
  not reproduce; this triggered the 0.1.0 rebuild above. Kept as git history
  for provenance rather than squashed, per the project's policy of reporting
  what actually happened rather than presenting a cleaned-up narrative.
