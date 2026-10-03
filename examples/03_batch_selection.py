"""Greedy batch EIG vs. naive top-k: when candidates' ensemble
disagreement is redundant, picking the k individually best points wastes
labels. The greedy batch is provably within (1 - 1/e) of the optimal batch.
"""
import numpy as np

from quantum_al.acquisition import (
    batch_information_gain, expected_information_gain, fit_ensemble, greedy_batch_eig, top_k,
)
from quantum_al.joint_eig import per_tree_predictions, predictive_covariance
from quantum_al.synthetic import make_correlated_tasks

X, Y, _ = make_correlated_tasks(n_samples=500, correlation=0.5, noise=0.2, seed=1)
# Make the candidate pool redundant: every candidate appears with 4 near-copies.
X_cand = np.repeat(X[100:200], 5, axis=0) + 1e-3 * np.random.default_rng(0).normal(size=(500, X.shape[1]))
forest, R = fit_ensemble(X[:40], Y[:40], n_estimators=200, seed=0)
F = per_tree_predictions(forest, X_cand)

b = 10
naive = top_k(expected_information_gain(predictive_covariance(forest, X_cand), R), b)
greedy, gains = greedy_batch_eig(F, R, b)

print(f"distinct source points in top-{b} batch:  {len(set(naive // 5))}")
print(f"distinct source points in greedy batch: {len(set(greedy // 5))}")
print(f"batch EIG, top-{b}:  {batch_information_gain(F[:, naive], R):.3f} nats")
print(f"batch EIG, greedy:  {batch_information_gain(F[:, greedy], R):.3f} nats")
print("greedy marginal gains (non-increasing by submodularity):", np.round(gains, 3))
