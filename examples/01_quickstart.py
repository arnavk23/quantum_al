"""Quickstart: compare joint and correlation-blind acquisition on a
two-property problem, with no data download or API key.

Runs in well under a minute. See 05_materials_project.py for real data.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from quantum_al import EnsembleCriterionSelector, make_correlated_tasks, run_active_learning
from quantum_al.data_utils import standardize
from quantum_al.stats import paired_comparison

selectors = {
    "joint EIG": lambda seed: EnsembleCriterionSelector("eig", n_estimators=50, seed=seed),
    "marginal EIG": lambda seed: EnsembleCriterionSelector("marginal_eig", n_estimators=50, seed=seed),
}

aulc = {name: [] for name in selectors}
for seed in range(3):
    # Two targets whose noiseless signals are 90% correlated.
    X, Y, info = make_correlated_tasks(n_samples=300, n_features=6, correlation=0.9,
                                       noise=0.3, seed=seed)
    X_pool, X_test, Y_pool, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)
    X_pool, X_test = standardize(X_pool, X_test)

    for name, make in selectors.items():
        curve = run_active_learning(make(seed), X_pool, Y_pool, X_test, Y_test,
                                    n_initial=20, n_rounds=5, batch_size=8, seed=seed)
        aulc[name].append(curve.aulc())
        print(f"seed {seed} {name:13s} R^2 by round: "
              + " ".join(f"{r:.2f}" for r in curve.r2))

result = paired_comparison(aulc["joint EIG"], aulc["marginal EIG"])
print(f"\nmean AULC difference (joint - marginal): {result['mean_diff']:+.4f}, "
      f"paired t-test p = {result['p_value_t']:.3f} over {result['n_paired']} paired seeds")
