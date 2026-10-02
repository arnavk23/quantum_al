Quickstart
==========

The selector interface
----------------------

Every acquisition strategy in the package, and any strategy you write, is an
object with one method::

    selected_idx, scores, info = selector.select_next_experiments(
        X_candidates, X_train, Y_train, n_select)

``selected_idx`` indexes ``X_candidates``; ``Y_train`` may hold one target
(shape ``(n,)``) or several (shape ``(n, K)``). This is the only contract the
benchmark loop relies on.

A first comparison
------------------

.. code-block:: python

   from sklearn.model_selection import train_test_split
   from quantum_al import EnsembleCriterionSelector, make_correlated_tasks, run_active_learning
   from quantum_al.data_utils import standardize

   # Two properties whose noiseless signals are 90% correlated.
   X, Y, info = make_correlated_tasks(n_samples=300, correlation=0.9, noise=0.3, seed=0)
   X_pool, X_test, Y_pool, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
   X_pool, X_test = standardize(X_pool, X_test)

   for criterion in ["eig", "marginal_eig", "trace"]:
       selector = EnsembleCriterionSelector(criterion, n_estimators=100, seed=0)
       curve = run_active_learning(selector, X_pool, Y_pool, X_test, Y_test,
                                   n_initial=20, n_rounds=5, batch_size=8, seed=0)
       print(criterion, [round(r, 3) for r in curve.r2], "AULC", round(curve.aulc(), 3))

Trials with the same ``seed`` share their initial labeled set and evaluation
model, so results of different selectors are paired; compare them with
:func:`quantum_al.stats.paired_comparison` and correct for multiple
comparisons with :func:`quantum_al.stats.holm_bonferroni`.

Inspecting a criterion
----------------------

The criteria are plain functions of the ensemble's predictive covariance
``Sigma`` (shape ``(n, K, K)``) and noise variances ``R``:

.. code-block:: python

   from quantum_al.acquisition import (fit_ensemble, expected_information_gain,
                                       marginal_information_gain_sum, total_correlation)
   from quantum_al.joint_eig import predictive_covariance
   from quantum_al.diagnostics import criterion_disagreement

   forest, R = fit_ensemble(X_pool[:40], Y_pool[:40], n_estimators=200)
   Sigma = predictive_covariance(forest, X_pool[40:])
   joint = expected_information_gain(Sigma, R)            # nats, >= 0
   gap = marginal_information_gain_sum(Sigma, R) - joint  # == total_correlation(Sigma, R)
   print(criterion_disagreement(Sigma, R, batch_size=10)) # would they pick different batches?

Batches
-------

Top-k selection can fill a batch with near-duplicates. The greedy batch EIG
accounts for redundancy through the ensemble's cross-candidate covariance:

.. code-block:: python

   from quantum_al import GreedyBatchEIGSelector
   selector = GreedyBatchEIGSelector(n_estimators=200, seed=0)

More
----

The ``examples/`` directory contains runnable scripts covering the
decomposition identities, batch selection, writing and benchmarking your own
selector against the nine classical baselines, and real Materials Project
data.
