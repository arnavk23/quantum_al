# quantum_al

[![Tests](https://github.com/arnavk23/quantum_al/actions/workflows/tests.yml/badge.svg)](https://github.com/arnavk23/quantum_al/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/quantum-al/badge/?version=latest)](https://quantum-al.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/quantum-al)](https://pypi.org/project/quantum-al/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23351261.svg)](https://doi.org/10.5281/zenodo.23351261)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Correlation-aware active learning for materials discovery.**

A single DFT calculation returns a band gap, a formation energy and a magnetic
moment computed from the same electronic structure, so materials properties
are correlated. Active learning chooses which material to compute next.
`quantum_al` is a Python package for building and rigorously testing
acquisition criteria that try to exploit that correlation:

- **Information-theoretic criteria** (`quantum_al.acquisition`): the joint
  expected information gain (EIG), its correlation-blind counterpart, the
  exact total-correlation term that separates them, unit-invariant A- and
  E-optimal analogues, and a greedy batch EIG with a (1 − 1/e) guarantee.
- **Diagnostics** (`quantum_al.diagnostics`) that explain *why* two criteria
  do or do not lead to different outcomes.
- **A synthetic testbed** (`quantum_al.synthetic`) with tunable
  cross-property correlation and noise.
- **A benchmark harness**: a shared pool-based loop (`quantum_al.loop`), nine
  classical baselines (`quantum_al.baselines`), paired statistics with
  Holm-Bonferroni correction (`quantum_al.stats`) and real Materials Project
  data (`quantum_al.data_utils`).
- **A quantum-inspired covariance formalism** (`quantum_al.quantum_inspired`)
  with a verified Qiskit circuit realization and measurement-cost analysis
  (`quantum_al.circuit`).

Every criterion is tested against the mathematical properties it claims, and
every reported number is written to [`results/`](results/) by a script in
[`benchmarks/`](benchmarks/).

## Installation

```bash
pip install quantum-al
```

Python 3.10 or newer is required. Optional extras: `data` (pymatgen, for
fetching Materials Project data), `circuit` (Qiskit and Qiskit Aer), `plot`
(matplotlib), `test`, `docs`, and `all`; for example
`pip install "quantum-al[circuit]"`. To run the examples and benchmarks, or to
develop, install from a clone:

```bash
git clone https://github.com/arnavk23/quantum_al.git
cd quantum_al
pip install -e ".[test]"
```

## Example

Compare the joint EIG with its correlation-blind ablation on a two-property
problem, with no data download:

```python
from sklearn.model_selection import train_test_split

from quantum_al import EnsembleCriterionSelector, make_correlated_tasks, run_active_learning
from quantum_al.data_utils import standardize

X, Y, info = make_correlated_tasks(n_samples=300, correlation=0.9, noise=0.3, seed=0)
X_pool, X_test, Y_pool, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_pool, X_test = standardize(X_pool, X_test)

for criterion in ["eig", "marginal_eig"]:
    selector = EnsembleCriterionSelector(criterion, n_estimators=50, seed=0)
    curve = run_active_learning(selector, X_pool, Y_pool, X_test, Y_test,
                                n_initial=20, n_rounds=5, batch_size=8, seed=0)
    print(criterion, curve.aulc())
```

Any object with a `select_next_experiments(X_candidates, X_train, Y_train,
n_select)` method can be benchmarked the same way. [`examples/`](examples/)
goes further: the information decomposition, batch selection, plugging in
your own criterion, and real Materials Project data.

## What it has found

On four real Materials Project property pairs, the joint EIG never
significantly beat its correlation-blind ablation. The diagnostics explain
why: the total-correlation term is second order in the epistemic-to-aleatoric
variance ratio, and on real candidate pools it varies only 3–9% as much as the
marginal scores, so the two criteria select nearly the same batches. The
quantum-inspired formalism did not beat nine classical baselines, and
ablations attribute its later improvement to ensemble disagreement rather than
to its quantum-specific components. See
[`docs/findings.rst`](docs/findings.rst) for every number, and its History
section for claims from an earlier version of this repository that did not
reproduce and were retracted.

## Documentation

The documentation at **[quantum-al.readthedocs.io](https://quantum-al.readthedocs.io)**
covers installation, a quickstart, the theory with full proofs, the benchmark
protocol, the findings, and the API reference. Build it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Reproducing the results

[`benchmarks/README.md`](benchmarks/README.md) lists the exact command behind
every file in `results/`. In short:

```bash
pip install -e ".[data,circuit,plot]"
python benchmarks/run_correlation_sweep.py   # synthetic, no data needed
export MP_API_KEY=your_key                   # https://next-gen.materialsproject.org/api
python -m quantum_al.fetch_data              # real-data experiments
python benchmarks/run_joint_eig_experiment.py --tasks band_gap formation_energy
python benchmarks/make_paper_figures.py
```

[`paper/`](paper/) holds the JOSS paper and [`papers/`](papers/) the research
manuscript.

## Citation

If you use `quantum_al`, please cite it using [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button).

## Contributing

Bug reports, questions and contributions are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).
