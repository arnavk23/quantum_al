Findings
========

This page summarizes what the benchmarks in :doc:`benchmarks` found. Every
number comes from a JSON file in ``results/``, named next to each result.
Results that are not significant after correction are reported as
inconclusive, not as evidence that two methods are equal. With five paired
trials, only large effects are detectable.

Joint EIG on real Materials Project data
----------------------------------------

The joint EIG (``Joint-EIG``) is compared with its correlation-blind ablation
(``Marginal-Sum``, the sum of per-property EIGs) and with random selection, on
every Materials Project property pair with enough shared materials to run the
protocol. Each query reveals both properties. The metric is the final-round
test :math:`R^2`, averaged over the two properties, and comparisons are paired
over 5 trials with Holm-Bonferroni correction.

.. list-table::
   :header-rows: 1
   :widths: 30 8 10 22 22 10

   * - Pair
     - *n*
     - label *r*
     - Joint − Marginal-Sum
     - Joint − Random
     - mean TC
   * - band gap + formation energy
     - 498
     - −0.365
     - +0.0076, *p* = 0.27
     - **+0.0623, p = 0.034** (corrected)
     - 0.0118
   * - formation energy + magnetic moment
     - 220
     - 0.205
     - +0.0006, *p* = 0.99
     - −0.0164, *p* = 0.66
     - 0.0041
   * - band gap + magnetic moment
     - 193
     - −0.024
     - +0.0086, *p* = 0.25
     - +0.0186, *p* = 0.17
     - 0.0042
   * - bulk modulus + dielectric constant
     - 49
     - 0.334
     - −0.0314, *p* = 0.61
     - +0.0388, *p* = 0.61
     - 0.0211

Sources: ``results/joint_eig_experiment.json``,
``joint_eig_experiment_replication.json``, ``joint_eig_experiment_pair_bg_mm.json``
and ``joint_eig_experiment_pair_bm_dc.json``. The last pair has only 49 shared
materials and uses a reduced protocol (:math:`N_0 = 15`, :math:`T = 5`,
:math:`b = 3`), so it is low-powered and inconclusive.

**Reading.** The joint EIG never significantly beats its correlation-blind
ablation. It beats random selection significantly on one pair, the largest,
where the correlation-blind ablation also does nominally better than random,
so the gain is attributable to ensemble uncertainty in general rather than to
the joint term. The total-correlation term (TC, in nats per candidate) is
non-zero on every pair, as Proposition 1 in :doc:`theory` requires when the
ensemble's errors are correlated. It is also small, which is the regime in
which Proposition 4 and its corollary say the two criteria can only reorder
near-ties. The real-data candidate pools have not yet been analyzed with
:mod:`quantum_al.diagnostics`; ``examples/05_materials_project.py`` does this
for one pair.

Controlled study: when can the joint criterion help?
----------------------------------------------------

``benchmarks/run_correlation_sweep.py`` varies the signal correlation
between two synthetic properties (0, 0.5, 0.9, 0.99) and the noise level
(0.1, 0.5) independently, with 10 paired trials per setting. In each setting
it compares the joint EIG, the marginal-EIG sum, greedy batch EIG, the
whitened trace and random selection, and records the diagnostics of
:mod:`quantum_al.diagnostics` for the joint and marginal criteria. Results
will be added here once the full run completes
(``results/correlation_sweep.json``).

The quantum-inspired covariance formalism
-----------------------------------------

The formalism in :mod:`quantum_al.operator` was compared with nine classical
baselines on five real regression tasks (5 trials, 8 rounds, a 100-tree
random forest; ``results/primary_benchmark_summary.json``,
``statistical_tests.json``).

* **As specified, it does not beat the baselines.** It ranks last of ten on
  band gap (:math:`R^2` = 0.513 against 0.597 for uncertainty sampling), sixth
  on formation energy, eighth on bulk modulus and last on magnetic moment.
  It is nominally first on dielectric constant, where every method has
  :math:`R^2 \approx 0` and the ranking is meaningless. On the two primary
  tasks (band gap and formation energy), no paired comparison with a
  baseline survives Holm-Bonferroni correction in either direction.
* **Its distinctive components are inert.** Removing the covariance term,
  restricting to commuting observables, using real coefficients or a single
  observable each changes band-gap :math:`R^2` by less than 0.011, well inside
  the trial-to-trial standard deviation of 0.046 to 0.076
  (``results/ablation.json``).
* **Coupling it to ensemble disagreement reaches parity, for a different
  reason.** Encoding states from a random forest's per-tree predictions
  (:mod:`quantum_al.operator_v3`) brings it to statistical parity with the
  best baseline on all five tasks (``results/v3_all_tasks.json``). Its
  ablation shows that dropping the covariance term, or collapsing to one
  observable, matches or beats the full construction on four of five tasks
  (``results/v3_ablation.json``): the gain comes from the disagreement signal.
* **The circuit realization is exact, and expensive.** The Qiskit circuit in
  :mod:`quantum_al.circuit` reproduces the classical simulation to
  floating-point precision. Each of the nine measured quantities has 528
  Pauli terms, reduced to 122 measurement settings by qubit-wise-commuting
  grouping and to 43 by general commuting grouping
  (``results/measurement_grouping.json``). Sparse observables cut the total
  further, from 1098 to 366 grouped settings, at a cost in accuracy: band-gap
  :math:`R^2` falls from 0.513 to 0.489 (*p* = 0.031, paired) for the original
  formalism (``results/sparse_observable_experiment.json``).

Summary
-------

Two unrelated constructions of correlation-aware acquisition, one an analogy
to quantum measurement and one derived from Bayesian experimental design,
did not reliably outperform acquisition based on ensemble disagreement alone
on real materials data. For the information-theoretic construction, the
theory says why such a result is expected: the correlation term is second
order in the epistemic-to-aleatoric ratio. The full record, including
retracted earlier claims and pre-rebuild experiments, is in
``results/SUMMARY.md``.
