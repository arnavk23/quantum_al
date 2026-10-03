Benchmarks and reproducibility
==============================

Every number in the paper, the manuscripts and :doc:`findings` comes from a
script in ``benchmarks/`` writing JSON to ``results/``; figures are drawn from
those JSON files by ``benchmarks/make_paper_figures.py``. Nothing is entered
by hand, and a failed computation is recorded as a failure, never replaced by
an illustrative number.

Protocol
--------

All experiments use :func:`quantum_al.loop.run_active_learning`:

1. Split the data 70/30 into pool and held-out test set (``random_state =
   trial``) and standardize features with pool statistics only.
2. Label :math:`N_0` random pool points (50 on real data).
3. For :math:`T` rounds (8), fit a 100-tree random forest on the labeled set,
   record test :math:`R^2` (mean over targets for multi-property tasks), and
   acquire a batch of :math:`b` points (15). Multi-property queries reveal
   all properties of the chosen material.
4. Repeat for 5 (real data) or 10 (synthetic) trials. Trial :math:`t` uses the
   same split, initial set and evaluation model for every method, so
   comparisons are **paired**.

Statistics
----------

:func:`quantum_al.stats.paired_comparison` reports, for per-trial scores of
two methods, the mean difference with a bootstrap 95% confidence interval,
the paired t-test, the Wilcoxon signed-rank test, Shapiro-Wilk normality of
the differences and the standardized effect size :math:`d_z`. Families of
comparisons are corrected with Holm's step-down procedure
(:func:`quantum_al.stats.holm_bonferroni`). Besides the final-round
:math:`R^2`, the normalized area under the learning curve
(:func:`quantum_al.stats.area_under_learning_curve`) summarizes the whole
trajectory. With 5 trials, only large effects are detectable; the documents
report non-significant results as inconclusive, not as evidence of equality.

Experiments
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Script
     - Output in ``results/``
     - Needs
   * - ``run_joint_eig_experiment.py``
     - ``joint_eig_experiment*.json``
     - MP data
   * - ``run_primary_benchmark.py --stage all``
     - ``primary_benchmark*.json``, ``statistical_tests.json``, ``ablation.json``, ``observable_sensitivity.json``, ``runtime_memory.json``
     - MP data
   * - ``run_improvement_attempt.py``
     - ``improvement_attempt.json``
     - MP data
   * - ``run_v3_test.py``, ``run_v3_ablation.py``, ``run_v3_all_tasks.py``
     - ``v3_test.json``, ``v3_ablation.json``, ``v3_all_tasks.json``
     - MP data
   * - ``run_sparse_observable_experiment.py``
     - ``sparse_observable_experiment.json``
     - MP data, [circuit]
   * - ``run_quantum_circuit_experiment.py``
     - ``quantum_circuit_experiment.json``
     - MP data, [circuit]
   * - ``run_measurement_grouping.py``
     - ``measurement_grouping.json``
     - [circuit]
   * - ``run_correlation_sweep.py``
     - ``correlation_sweep.json``
     - nothing
   * - ``make_paper_figures.py``
     - ``figures/*.pdf``
     - [plot]

``benchmarks/README.md`` lists the exact commands, including the arguments
used for each of the four real property pairs.

Testing
-------

``pytest`` runs over 100 tests in about 15 seconds. Besides interface and
regression tests they check, numerically, every mathematical claim in
:doc:`theory`: the decomposition identity, the sandwich and two-property
bounds (including tightness), unit invariance, the second-order behavior of
the correlation term, batch-EIG identities, monotonicity and submodularity on
every subset of a small pool, the :math:`1 - 1/e` greedy guarantee against
brute force, the quantum formalism's classical limit, and exact agreement
between the Qiskit circuit and the classical formula. GitHub Actions runs the
suite on Linux and macOS for Python 3.10-3.13, runs the examples, and builds
this documentation with warnings treated as errors.
