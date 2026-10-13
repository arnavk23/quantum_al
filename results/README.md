# Results

Every file here is written by a script in [`../benchmarks/`](../benchmarks/);
none is edited by hand. The findings they support are described, with every
number, in [`docs/findings.rst`](../docs/findings.rst), and
[`../benchmarks/README.md`](../benchmarks/README.md) gives the exact command
for each file. Figures in [`../figures/`](../figures/) are drawn from these
files by `benchmarks/make_paper_figures.py`.

| File | Script | Content |
|---|---|---|
| `joint_eig_experiment.json`, `_replication.json`, `_pair_bg_mm.json`, `_pair_bm_dc.json` | `run_joint_eig_experiment.py` | Joint EIG vs. its correlation-blind ablation and random selection on four real property pairs, with per-round selection diagnostics |
| `correlation_sweep.json`, `correlation_sweep_run_log.txt` | `run_correlation_sweep.py` | Controlled synthetic study over signal correlation and noise |
| `primary_benchmark.json`, `primary_benchmark_summary.json` | `run_primary_benchmark.py --stage primary` | Quantum-inspired selector vs. nine baselines on five real tasks |
| `statistical_tests.json` | `--stage stats` | Paired tests with Holm-Bonferroni correction for the primary benchmark |
| `ablation.json` | `--stage ablation` | Ablations of the quantum-inspired selector's components |
| `observable_sensitivity.json` | `--stage sensitivity` | Sensitivity to the number and grouping of observables |
| `runtime_memory.json` | `--stage runtime` | Selection time and peak memory per method |
| `improvement_attempt.json` | `run_improvement_attempt.py` | Domain-informed groups and importance-weighted encoding |
| `v3_all_tasks.json` | `run_v3_all_tasks.py` | Tree-ensemble (residual-coupled) variant and its ablation on all five tasks |
| `sparse_observable_experiment.json` | `run_sparse_observable_experiment.py` | Measurement cost and accuracy of sparse observables |
| `quantum_circuit_experiment.json` | `run_quantum_circuit_experiment.py` | Qiskit resource counts and shot/noise sensitivity |
| `measurement_grouping.json` | `run_measurement_grouping.py` | Pauli terms before and after commuting-group measurement |

Real-data results reproduce exactly only with scikit-learn 1.7.2 (see
`benchmarks/README.md`); newer releases change random-forest outputs slightly.
