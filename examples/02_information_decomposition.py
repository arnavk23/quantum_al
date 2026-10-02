"""The exact identities behind the joint-EIG criterion, on one fitted
ensemble: EIG = sum_k EIG_k - TC, the sandwich bound, unit invariance,
and why the total-correlation term is second order in low-signal regimes.
"""
import numpy as np

from quantum_al.acquisition import (
    epistemic_aleatoric_ratio, expected_information_gain, fit_ensemble,
    marginal_information_gains, total_correlation, whitened_trace,
)
from quantum_al.joint_eig import predictive_covariance
from quantum_al.synthetic import make_correlated_tasks

X, Y, _ = make_correlated_tasks(n_samples=400, correlation=0.8, noise=0.3, seed=0)
forest, R = fit_ensemble(X[:60], Y[:60], n_estimators=100, seed=0)
Sigma = predictive_covariance(forest, X[60:])          # (n, 2, 2) per candidate

joint = expected_information_gain(Sigma, R)
marg = marginal_information_gains(Sigma, R)
tc = total_correlation(Sigma, R)

print("1. Decomposition  max |sum_k EIG_k - TC - EIG| =",
      f"{np.max(np.abs(marg.sum(1) - tc - joint)):.2e}")
print("2. Sandwich bound max_k EIG_k <= EIG <= sum_k EIG_k holds for all candidates:",
      bool(np.all(marg.max(1) <= joint + 1e-12) and np.all(joint <= marg.sum(1) + 1e-12)))

# 3. Express target 2 in different units (x1000): information is unit-free.
D = np.array([1.0, 1000.0])
Sigma_u, R_u = Sigma * np.outer(D, D)[None], R * D ** 2
print("3. Unit invariance: max |EIG change| =",
      f"{np.max(np.abs(expected_information_gain(Sigma_u, R_u) - joint)):.2e};",
      "raw-trace ranking unchanged?",
      np.array_equal(np.argsort(np.trace(Sigma, axis1=1, axis2=2)),
                     np.argsort(np.trace(Sigma_u, axis1=1, axis2=2))))

# 4. Shrink the epistemic uncertainty: TC vanishes faster than the EIGs.
print("4. Low-signal regime (scale Sigma by c):")
for c in [1.0, 0.1, 0.01]:
    s = np.median(epistemic_aleatoric_ratio(c * Sigma, R))
    ratio = np.median(total_correlation(c * Sigma, R)
                      / marginal_information_gains(c * Sigma, R).sum(1))
    print(f"   c={c:<5} median s_k={s:.4f}  median TC / sum_k EIG_k = {ratio:.5f}")
print("   (first-order criterion for comparison, whitened trace / 2, first candidate:",
      f"{whitened_trace(Sigma, R)[0] / 2:.4f} vs EIG {joint[0]:.4f})")
