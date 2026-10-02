# Benchmarks

Each script here writes JSON to `../results/`, and `make_paper_figures.py`
draws `../figures/*.pdf` from those files. No number in the paper, the
manuscripts or the documentation is entered by hand. The shared protocol and
statistics are described in the documentation (`docs/benchmarks.rst`).

Run everything from the repository root after `pip install -e ".[all]"`.

**Exact reproduction needs scikit-learn 1.7.2.** Random-forest outputs change
between scikit-learn releases, so a newer version gives slightly different
numbers even with identical data and seeds; on the small, noisy property pairs
the differences are as large as the effects being tested. The published
real-data `results/` reproduce with scikit-learn 1.7.2 and NumPy 2.3.5:

```bash
pip install "scikit-learn==1.7.2" "numpy==2.3.5"
```

## No data needed

```bash
python benchmarks/run_correlation_sweep.py          # about 1 hour on 8 cores; --quick for a smoke test
python benchmarks/run_measurement_grouping.py       # seconds; needs the [circuit] extra
```

| Script | Output | Question |
|---|---|---|
| `run_correlation_sweep.py` | `correlation_sweep.json` | On controlled synthetic problems, when (signal correlation, noise level) can joint acquisition beat correlation-blind acquisition, and do the two criteria even select different batches? |
| `run_measurement_grouping.py` | `measurement_grouping.json` | How many measurement settings does the quantum-inspired score need on hardware, before and after grouping? |

## Real Materials Project data

Fetch the data once (needs the `[data]` extra and a free API key):

```bash
export MP_API_KEY=your_key
python -m quantum_al.fetch_data
```

### Multi-property acquisition (main result)

```bash
python benchmarks/run_joint_eig_experiment.py --tasks band_gap formation_energy --n-trials 5 --out joint_eig_experiment.json
python benchmarks/run_joint_eig_experiment.py --tasks formation_energy magnetic_moment --n-trials 5 --out joint_eig_experiment_replication.json
python benchmarks/run_joint_eig_experiment.py --tasks band_gap magnetic_moment --n-trials 5 --out joint_eig_experiment_pair_bg_mm.json
python benchmarks/run_joint_eig_experiment.py --tasks bulk_modulus dielectric_constant --n-trials 5 --n0 15 --t-iters 5 --batch-size 3 --out joint_eig_experiment_pair_bm_dc.json
```

Add `--extended` to also run the greedy batch EIG and the whitened trace and
max-eigenvalue criteria (all compared with Joint-EIG under one
Holm-Bonferroni family); write to a new `--out` file so the published
results are not overwritten. Every run also records, per round, whether the
joint and correlation-blind criteria would have selected different batches
(`selection_diagnostics`).

### Quantum-inspired formalism (earlier attempt)

| Script | Output |
|---|---|
| `run_primary_benchmark.py --stage all` | `primary_benchmark.json`, `primary_benchmark_summary.json`, `statistical_tests.json`, `ablation.json`, `observable_sensitivity.json`, `runtime_memory.json` |
| `run_improvement_attempt.py` | `improvement_attempt.json` (two narrow fixes that did not help) |
| `run_v3_test.py`, `run_v3_ablation.py`, `run_v3_all_tasks.py` | `v3_test.json`, `v3_ablation.json`, `v3_all_tasks.json` (the residual-coupled variant and its ablation) |
| `run_sparse_observable_experiment.py` | `sparse_observable_experiment.json` (needs `[circuit]`) |
| `run_quantum_circuit_experiment.py` | `quantum_circuit_experiment.json` (needs `[circuit]`) |

`run_primary_benchmark.py` must run before `run_v3_all_tasks.py` and
`run_sparse_observable_experiment.py`, which read its summary.

## Figures

```bash
python benchmarks/make_paper_figures.py   # needs the [plot] extra
```

Files in `results/legacy/` come from pre-rebuild scripts that are no longer
in the tree; see the README there.
