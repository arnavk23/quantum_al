"""Real data: joint vs marginal acquisition on two Materials Project
properties. Requires the data files; fetch them once with

    pip install -e ".[data]"
    MP_API_KEY=... python -m quantum_al.fetch_data

The full, published protocol is benchmarks/run_joint_eig_experiment.py.
"""
import sys

from sklearn.model_selection import train_test_split

from quantum_al import EnsembleCriterionSelector, run_active_learning
from quantum_al.data_utils import load_multi_task, standardize
from quantum_al.diagnostics import criterion_disagreement
from quantum_al.joint_eig import predictive_covariance

try:
    X, Y, meta = load_multi_task(["band_gap", "formation_energy"])
except FileNotFoundError as e:
    sys.exit(f"Skipping: {e}")

print(f"{len(X)} materials with both properties computed")
X_pool, X_test, Y_pool, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_pool, X_test = standardize(X_pool, X_test)


def diagnose(it, info, X_candidates):
    return criterion_disagreement(predictive_covariance(info["forest"], X_candidates), info["R"], 15)


curve = run_active_learning(EnsembleCriterionSelector("eig", seed=0), X_pool, Y_pool, X_test, Y_test,
                            n_initial=50, n_rounds=8, batch_size=15, seed=0, on_round=diagnose)
print("mean R^2 by round:", " ".join(f"{r:.3f}" for r in curve.r2))
print("joint-vs-marginal batch overlap by round:",
      " ".join(f"{d['topk_overlap']:.2f}" for d in curve.diagnostics))
