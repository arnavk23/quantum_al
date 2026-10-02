Contributing and support
========================

Contributions, bug reports and questions are welcome.

* **Questions and bug reports:** open an issue at
  https://github.com/arnavk23/quantum_al/issues (templates are provided).
* **Contributing code:** see ``CONTRIBUTING.md`` in the repository for the
  development setup, testing requirements and the project's rule that no
  result is ever reported without being produced by a script.
* **Conduct:** participation is governed by ``CODE_OF_CONDUCT.md``.

Adding a new acquisition criterion
----------------------------------

A single-candidate criterion is a function ``f(Sigma, R) -> scores`` of the
predictive covariance (shape ``(n, K, K)``) and noise variances (shape
``(K,)``). Register it in :data:`quantum_al.acquisition.CRITERIA` and it is
immediately usable through
:class:`~quantum_al.acquisition.EnsembleCriterionSelector` and the
benchmark scripts. Please add tests for the mathematical properties you
claim for it (e.g. invariances, bounds or special cases) alongside the
existing ones in ``tests/test_acquisition.py``.

Changelog
---------

See ``CHANGELOG.md`` in the repository.
