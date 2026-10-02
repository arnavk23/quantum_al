# Changelog

Notable changes to `quantum_al`, by release. Dates are UTC, from git history.

## [Unreleased]

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
- `quantum_al.circuit` measurement-grouping tool and
  `benchmarks/run_measurement_grouping.py`, which regenerates
  `results/measurement_grouping.json`.
- `benchmarks/run_correlation_sweep.py`: a controlled study of when joint
  acquisition can beat correlation-blind acquisition.
- Sphinx documentation (`docs/`) with full derivations of every property the
  tests check, Read the Docs configuration, runnable `examples/`, a
  `benchmarks/README.md` mapping every result file to its command, and
  community files (code of conduct, issue and pull request templates).

### Changed
- `paper.md` and `README.md` rewritten around the information-theoretic
  criteria and their diagnostics, and every reference in `paper.bib` checked
  against Crossref or arXiv.
- `fetch_data` sends an honest `quantum_al` User-Agent; `data_utils` takes a
  configurable data directory and raises a clear error when data is missing.
- Baseline fallbacks now emit a `RuntimeWarning` instead of printing, with
  the same behavior.
- Outputs of pre-rebuild scripts moved to `results/legacy/`, with a README.
### Earlier unreleased changes
- Added `src/quantum_al/joint_eig.py`: a joint expected-information-gain
  acquisition score derived from Bayesian experimental design, with a proved
  and numerically-verified total-correlation decomposition theorem, tested
  against its own classical-limit ablation on four real, independently-measured
  Materials Project property pairs (`benchmarks/run_joint_eig_experiment.py`).
- Added `src/quantum_al/operator_sparse.py`: sparse-by-construction observables
  for the quantum-inspired formalism, a further NISQ measurement-cost reduction
  beyond qubit-wise-commuting grouping.
- Added `quantum_al.data_utils.load_multi_task`, an inner-join utility for
  constructing genuinely correlated real multi-property datasets from
  separately-fetched Materials Project property files.
- Restructured the accompanying npj-style manuscript around the joint-EIG
  result, with the quantum-inspired formalism retained as an independently
  evaluated earlier attempt at the same goal.

## [0.1.0] - 2026-08-19

- Full honest rebuild of the package after an earlier, less rigorous
  evaluation was found not to reproduce from any code in the repository (see
  `results/SUMMARY.md`). Reorganized into an installable package
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
