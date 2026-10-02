# Contributing

Thanks for your interest in `quantum_al`. Contributions of all sizes are
welcome: bug reports, documentation fixes, new acquisition criteria,
baselines, datasets, or independent reproductions of the results. The project
is maintained by one person, so responses may take a few days.

## Getting help and reporting issues

- **Questions** ("how do I…", "what does this result mean"): open an issue
  with the *Question* template.
- **Bugs and results that do not reproduce**: open an issue with the *Bug
  report* template. Please include the exact command, what you expected, what
  happened, your Python version and the relevant `pip freeze` lines, and
  whether `pytest` passes on your machine.
- **Ideas**: open an issue with the *Feature request* template before
  starting large changes, so we can agree on the design first.

## Development setup

```bash
git clone https://github.com/arnavk23/quantum_al.git
cd quantum_al
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
pytest                           # ~100 tests, ~15 s
sphinx-build -W docs docs/_build/html
```

## Where things go

- `src/quantum_al/`: the installable library. Every addition needs tests in
  `tests/`, and NumPy-style docstrings (they become the API reference).
- `benchmarks/`: scripts that produce the files in `results/`. See
  `benchmarks/README.md`.
- `examples/`: short, runnable scripts that need no data download; CI runs
  them on every push.
- `docs/`: Sphinx documentation, including the derivations in
  `docs/theory.rst`.

## Adding an acquisition criterion

A single-candidate multi-property criterion is a function
`f(Sigma, R) -> scores` of the predictive covariance (shape `(n, K, K)`) and
per-target noise variances (shape `(K,)`). Add it to
`quantum_al.acquisition.CRITERIA` and it works with
`EnsembleCriterionSelector`, the loop and the benchmark scripts. Any other
strategy only needs a `select_next_experiments(X_candidates, X_train,
Y_train, n_select) -> (selected_idx, scores, info)` method; see
`examples/04_custom_selector.py`.

If you claim a mathematical property for a criterion (an invariance, a
bound, a special case it reduces to), add a test that checks it numerically,
as `tests/test_acquisition.py` does for the existing ones.

## The one hard rule: results come from code

Never report a result by editing numbers into `results/`, the papers or the
documentation by hand. If a script fails, fix it or report the failure; never
substitute an illustrative or synthetic number for a real-data one. This
project was rebuilt after an earlier version broke this rule (see
`results/SUMMARY.md`), and the benchmark loop raises on invalid selections
rather than silently falling back for the same reason.

If you change `src/quantum_al/operator.py`, `joint_eig.py` or
`acquisition.py`, the self-tests and `tests/` are what guarantee the code
still matches the mathematics in the papers and `docs/theory.rst`; keep them
passing.

## Pull requests

- One logical change per PR where practical.
- Fill in the PR template checklist; CI runs the tests on Linux and macOS for
  Python 3.10-3.13, runs the examples and builds the docs.
- Add a line to `CHANGELOG.md` under "Unreleased".

## Code of conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
Disagreement about results or methodology is expected and welcome; bad-faith
conduct is not.
