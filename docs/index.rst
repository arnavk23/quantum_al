quantum_al
==========

**Correlation-aware active learning for materials discovery.**

Materials properties are correlated: a single DFT calculation returns a band
gap, a formation energy and a magnetic moment computed from the same
electronic structure. Active learning chooses which material to compute
next. Should that choice take the correlation between properties into
account, and if so, when does it help? ``quantum_al`` is a research toolkit
for answering that question rigorously. It provides:

* **Information-theoretic acquisition criteria** derived from Bayesian
  optimal experimental design: the joint expected information gain (EIG),
  its correlation-blind counterpart, the exact total-correlation term that
  separates them, unit-invariant A- and E-optimal analogues, and a greedy
  batch EIG with a :math:`(1 - 1/e)` optimality guarantee
  (:doc:`theory`).
* **Diagnostics** that explain *why* two criteria do or do not lead to
  different outcomes, e.g. whether they would even pick different batches.
* **A controlled testbed** of multi-property problems with a tunable
  cross-property correlation and noise level, for mechanistic studies.
* **A real-data benchmark harness** on Materials Project properties with
  nine classical baselines, a shared pool-based loop, and paired statistics
  with multiple-comparison correction (:doc:`benchmarks`).
* **A quantum-inspired covariance formalism** with a verified Qiskit circuit
  realization and near-term hardware cost analysis, kept as an
  independently evaluated alternative construction.

Every criterion is tested against its own mathematical properties, and every
reported number is produced by a script in ``benchmarks/``. Negative results
are reported as such (:doc:`findings`).

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   theory
   benchmarks
   findings

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   contributing
