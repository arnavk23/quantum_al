# quantum_al: Correlation-Aware Active Learning for Materials Discovery

[![Tests](https://github.com/arnavk23/quantum_al/actions/workflows/tests.yml/badge.svg)](https://github.com/arnavk23/quantum_al/actions/workflows/tests.yml)

A single DFT calculation returns a band gap, a formation energy and a magnetic
moment computed from the same electronic structure, so materials properties
are correlated. Active learning chooses which material to compute next.
Should that choice take the correlation into account, and when does it help?
`quantum_al` is a research toolkit for answering that question rigorously:

- **Information-theoretic acquisition criteria** from Bayesian optimal
  experimental design (`quantum_al.acquisition`): the joint expected
  information gain (EIG), its correlation-blind counterpart, the exact
  total-correlation term that separates them, unit-invariant A- and
  E-optimal analogues, and a greedy batch EIG with a (1 − 1/e) guarantee.
- **Diagnostics** (`quantum_al.diagnostics`) that explain *why* two criteria
  do or do not lead to different outcomes.
- **A controlled testbed** (`quantum_al.synthetic`) of multi-property
  problems with tunable cross-property correlation and noise.
- **A real-data benchmark harness** on Materials Project properties with
  nine classical baselines, a shared pool-based loop (`quantum_al.loop`) and
  paired statistics with Holm-Bonferroni correction (`quantum_al.stats`).
- **A quantum-inspired covariance formalism** (`quantum_al.operator*`) with a
  verified Qiskit circuit realization and NISQ cost analysis, kept as an
  independently evaluated alternative construction.

Every criterion is tested against its own mathematical properties, and every
reported number is written to `results/` by a script in `benchmarks/`.
[`paper.md`](paper.md) is the JOSS software paper; [`papers/`](papers/) holds
the research manuscripts.

## Quickstart

```bash
pip install -e .
python examples/01_quickstart.py
```

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
n_select)` method can be benchmarked the same way; see
[`examples/04_custom_selector.py`](examples/04_custom_selector.py).

## Theory in one paragraph

For an ensemble with predictive covariance Σ(x) over K properties and noise
variances R, the joint EIG is ½ log det(I + R^{-1/2} Σ R^{-1/2}). It equals
the sum of the per-property EIGs minus the total correlation
TC = −½ log det Corr(Σ + R) ≥ 0 (Hadamard's inequality). TC is **second order**
in the epistemic-to-aleatoric ratio s = Σ_kk / R_k, while each marginal EIG is
first order. For two properties the bound
TC ≤ ½ log(1 + s₁s₂ / (1 + s₁ + s₂)) is exact and tight. The joint and
correlation-blind criteria can therefore rank two candidates differently only
when their marginal scores are nearly tied. Full derivations are in
[`docs/theory.rst`](docs/theory.rst), and every property is checked in
[`tests/test_acquisition.py`](tests/test_acquisition.py).

## Findings

**Joint EIG on real data.** On four real Materials Project property pairs
(label correlation −0.37 to 0.33, n = 49 to 498), the joint EIG **never
significantly beats its correlation-blind ablation**, and it beats random
sampling significantly on only one pair. The total-correlation term is
measurably non-zero on every pair but small (a mean of 0.004 to 0.021 nats
per candidate), which is the regime in which the theory says the two criteria
can only reorder near-ties.

**Quantum-inspired formalism.** Against nine classical baselines on five real
regression tasks, the formalism as originally specified loses on four, and
its covariance term has no effect distinguishable from noise. Coupling it to a
random forest's per-tree disagreement brings it to parity with the best
baseline, but ablations attribute the gain entirely to the disagreement
signal, not to the quantum-specific machinery. The Qiskit realization matches
the classical simulation exactly. Each measured quantity has 528 Pauli terms,
reduced 4× by qubit-wise-commuting grouping and 12× by general commuting
grouping; sparse observables cut the total a further 3×, at a small but
significant cost in accuracy.

Every number, with the details: [`results/SUMMARY.md`](results/SUMMARY.md)
and [`docs/findings.rst`](docs/findings.rst).

An earlier version of this repository claimed a 35% sample-efficiency
improvement and p < 0.01 over nine baselines. Those numbers did not reproduce
from any code in the repository and have been retracted. Everything above
comes from rerun experiments.

## Installation

```bash
git clone https://github.com/arnavk23/quantum_al.git
cd quantum_al
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e ".[test]"
pytest
```

Optional extras: `data` (pymatgen, for fetching Materials Project data),
`circuit` (Qiskit and Qiskit Aer), `plot` (matplotlib, for the paper
figures), `docs` (Sphinx) and `all`. Python 3.10 or newer is required.

## Documentation

The documentation in [`docs/`](docs/) covers installation, a quickstart, the
full theory with proofs, the benchmark protocol and statistics, the findings
and the API reference. Build it locally with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

## Repository structure

```
src/quantum_al/
  acquisition.py       multi-property EIG criteria, total correlation, greedy batch EIG
  diagnostics.py       why two criteria do or do not select different batches
  synthetic.py         multi-property problems with tunable correlation and noise
  loop.py              shared pool-based active-learning loop, learning curves
  stats.py             paired tests, bootstrap CIs, effect sizes, Holm-Bonferroni
  joint_eig.py         the original joint-EIG selector and its correlation-blind ablation
  baselines.py         nine classical single-property acquisition strategies
  operator*.py         the quantum-inspired formalism and its variants
  circuit.py           Qiskit circuit realization, measurement grouping, NISQ costs
  data_utils.py        loading real Materials Project data, multi-property joins
  fetch_data.py        (re)fetching data from the Materials Project API
examples/              runnable examples, from quickstart to real data
benchmarks/            the scripts behind every result (see benchmarks/README.md)
results/               JSON output of every benchmark; legacy/ holds pre-rebuild outputs
figures/               paper figures, drawn from results/ by benchmarks/make_paper_figures.py
docs/                  Sphinx documentation
tests/                 pytest suite, run in CI on every push
papers/                research manuscripts and their provenance
data/                  Materials Project data (gitignored; see below)
```

## Reproducing the results

[`benchmarks/README.md`](benchmarks/README.md) lists the exact command behind
every file in `results/`. In short:

```bash
export MP_API_KEY=your_key            # https://next-gen.materialsproject.org/api
pip install -e ".[data,circuit,plot]"
python -m quantum_al.fetch_data       # real-data experiments only
python benchmarks/run_correlation_sweep.py   # synthetic, no data needed
python benchmarks/make_paper_figures.py
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contributing

Contributions, bug reports and questions are welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md).

## Contact

Arnav Kapoor — arnavkapoor23@iiserb.ac.in
