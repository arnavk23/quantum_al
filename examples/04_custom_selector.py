"""Plug your own acquisition function into the harness and compare it to
the nine classical baselines with paired, multiple-comparison-corrected
statistics. Any object with select_next_experiments(...) works.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from quantum_al import get_all_baselines, make_correlated_tasks, run_active_learning
from quantum_al.acquisition import top_k
from quantum_al.data_utils import standardize
from quantum_al.stats import holm_bonferroni, paired_comparison


class DistanceSelector:
    """Toy example: query the candidates farthest from the labeled set."""

    name = "Farthest-first"

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        d = np.linalg.norm(X_candidates[:, None, :] - X_train[None, :, :], axis=2).min(axis=1)
        return top_k(d, n_select), d, {}


warnings.filterwarnings("ignore", category=ConvergenceWarning)  # tiny toy datasets

finals = {}
seeds = range(4)
for seed in seeds:
    X, Y, _ = make_correlated_tasks(n_samples=250, n_features=5, n_tasks=1, seed=seed)
    y = Y[:, 0]                                  # baselines are single-target
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    X_pool, X_test = standardize(X_pool, X_test)
    methods = {"Farthest-first": DistanceSelector(), **get_all_baselines()}
    for name, selector in methods.items():
        np.random.seed(seed)                     # some baselines use the global RNG
        curve = run_active_learning(selector, X_pool, y_pool, X_test, y_test,
                                    n_initial=20, n_rounds=4, batch_size=8, seed=seed)
        finals.setdefault(name, []).append(curve.final_r2)

ours = finals.pop("Farthest-first")
names = list(finals)
results = [paired_comparison(ours, finals[n]) for n in names]
adjusted, reject = holm_bonferroni([r["p_value_t"] or 1.0 for r in results])
print(f"{'baseline':22s} {'mean diff':>9s} {'p (Holm)':>9s}")
for n, r, p, rej in zip(names, results, adjusted, reject):
    print(f"{n:22s} {r['mean_diff']:+9.3f} {p:9.3f}{'  *' if rej else ''}")
