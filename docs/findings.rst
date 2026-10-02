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
over 5 trials. The *p*-values are from paired *t*-tests with Holm-Bonferroni
correction over the two comparisons for each pair.

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
     - **+0.0623, p = 0.034**
     - 0.0118
   * - formation energy + magnetic moment
     - 220
     - 0.205
     - +0.0006, *p* = 1.00
     - −0.0164, *p* = 1.00
     - 0.0041
   * - band gap + magnetic moment
     - 193
     - −0.024
     - +0.0077, *p* = 0.28
     - +0.0149, *p* = 0.18
     - 0.0042
   * - bulk modulus + dielectric constant
     - 49
     - 0.334
     - −0.0317, *p* = 0.60
     - +0.0384, *p* = 0.60
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
near-ties.

**Why: the two criteria select almost the same batches.**
:func:`quantum_al.diagnostics.criterion_disagreement`, run on every round's
real candidate pool (``selection_diagnostics`` in the same JSON files),
measures this directly:

.. list-table::
   :header-rows: 1
   :widths: 34 16 14 12 14 10

   * - Pair
     - top-batch overlap (Jaccard)
     - shared picks
     - rank corr.
     - :math:`\mathrm{std(TC)} / \mathrm{std}(\sum_k \mathrm{EIG}_k)`
     - median :math:`s_k`
   * - band gap + formation energy
     - 0.763
     - 12.9 of 15
     - 0.998
     - 0.085
     - 0.67
   * - formation energy + magnetic moment
     - 0.955
     - 14.6 of 15
     - 0.999
     - 0.033
     - 0.49
   * - band gap + magnetic moment
     - 0.971
     - 14.8 of 15
     - 0.998
     - 0.035
     - 0.47
   * - bulk modulus + dielectric constant
     - 0.900
     - 2.8 of 3
     - 0.983
     - 0.092
     - 0.79

Across candidates, the correlation term varies only 3 to 9% as much as the
marginal scores, so the two criteria rank candidates almost identically
(Spearman 0.98 to 0.999) and share most of every batch. A correlation-aware
criterion can only beat a correlation-blind one through the decisions it
makes differently, and here it makes very few.

**Reproducibility.** All four experiments were rerun from a fresh Materials
Project download. Two pairs reproduce every learning curve to floating-point
precision; in each of the other two, one trial differs slightly, most likely
because a few database entries changed after the original download. Exact
reproduction needs scikit-learn 1.7.2 (see ``benchmarks/README.md``).

Controlled study: when can the joint criterion help?
----------------------------------------------------

``benchmarks/run_correlation_sweep.py`` varies the signal correlation
between two synthetic properties (0, 0.5, 0.9, 0.99) and the noise level
(0.1, 0.5) independently, with 10 paired trials per setting. In each setting
it compares the joint EIG, the marginal-EIG sum, greedy batch EIG, the
whitened trace and random selection, and records the diagnostics of
:mod:`quantum_al.diagnostics` for the joint and marginal criteria
(``results/correlation_sweep.json``; 600 samples, 8 features, 30 initial
labels, 8 rounds of 10). The metric is the normalized area under the
learning curve (AULC), and *p*-values are Holm-corrected over the 8 settings.

.. list-table::
   :header-rows: 1
   :widths: 10 10 16 14 26 24

   * - :math:`\rho`
     - :math:`\sigma`
     - batch overlap
     - TC spread ratio
     - Joint − Marginal (AULC)
     - Joint − Random (AULC)
   * - 0
     - 0.1
     - 0.786
     - 0.079
     - −0.0014, *p* = 1.00
     - −0.0098, *p* = 1.00
   * - 0.5
     - 0.1
     - 0.646
     - 0.177
     - −0.0142, *p* = 1.00
     - −0.0234, *p* = 1.00
   * - 0.9
     - 0.1
     - 0.794
     - 0.281
     - −0.0227, *p* = 0.13
     - −0.0372, *p* = 0.40
   * - 0.99
     - 0.1
     - 0.957
     - 0.307
     - +0.0170, *p* = 0.09
     - −0.0139, *p* = 1.00
   * - 0
     - 0.5
     - 0.777
     - 0.085
     - −0.0027, *p* = 1.00
     - −0.0125, *p* = 1.00
   * - 0.5
     - 0.5
     - 0.679
     - 0.144
     - +0.0081, *p* = 1.00
     - −0.0114, *p* = 1.00
   * - 0.9
     - 0.5
     - 0.742
     - 0.225
     - +0.0063, *p* = 1.00
     - −0.0212, *p* = 0.51
   * - 0.99
     - 0.5
     - 0.726
     - 0.250
     - +0.0032, *p* = 1.00
     - −0.0217, *p* = 1.00

"Batch overlap" is the Jaccard overlap of the two criteria's top-10 batches
and "TC spread ratio" is :math:`\mathrm{std(TC)} / \mathrm{std}(\sum_k
\mathrm{EIG}_k)` across candidates, both averaged over rounds and trials.
``figures/fig8_correlation_sweep.pdf`` plots the effect with 95% bootstrap
confidence intervals.

**Reading.** The diagnostics respond to correlation as the theory predicts:
as :math:`\rho` rises from 0 to 0.99, the mean total correlation grows from
about 0.007 to 0.11 nats and its spread relative to the marginal scores from
0.08 to 0.31. Here the two criteria disagree more than on real data, sharing
65 to 96% of each batch. Even so, no difference between them is significant
after correction in any setting. The two raw-significant results (at
:math:`\rho` = 0.9 and 0.99 with low noise) point in opposite directions,
which is what noise looks like with 10 trials. So within this design, larger
and more variable correlation terms change which points are picked without
changing how fast the model learns.

Random selection has the highest mean AULC in all eight settings, although
no difference from the joint EIG is significant after correction. The likely
explanation is a known failure mode of uncertainty-driven acquisition, not a
property of the joint criterion: both EIG criteria favor candidates where the
ensemble disagrees most, which on Gaussian inputs tend to lie in the
low-density tails, while the test set follows the pool's density. The stored
results do not test this explanation directly. The study therefore probes the
joint-versus-marginal question in a regime where neither criterion beats
random; whether a correlation-aware criterion helps when uncertainty sampling
does work well is not answered here.

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
