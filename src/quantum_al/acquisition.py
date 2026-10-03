"""Multi-target acquisition criteria from Bayesian optimal experimental design.

Every criterion here is a function of two quantities that an ensemble
predictive model provides for each candidate ``x``:

* ``Sigma(x)``, the ``(K, K)`` epistemic covariance of the ensemble's
  ``K``-dimensional prediction (disagreement between ensemble members), and
* ``R``, the length-``K`` vector of per-task aleatoric (noise) variances.

Under a Gaussian approximation ``f(x) ~ N(mu(x), Sigma(x))`` with
independent observation noise ``y = f(x) + eps``, ``eps ~ N(0, diag(R))``,
the expected information gain (mutual information between the observed
labels and the latent function values) is

.. math::

    \\mathrm{EIG}(x) = I(y; f) = \\tfrac12 \\log\\det\\left(I_K +
    R^{-1/2} \\Sigma(x) R^{-1/2}\\right) \\ge 0.

Properties proved in the documentation (``docs/theory.rst``) and checked
numerically in ``tests/test_acquisition.py``:

1. **Total-correlation decomposition.** ``sum_k EIG_k(x) - EIG(x) =
   TC(x) = -1/2 log det Corr(Sigma(x) + R) >= 0`` (Hadamard's inequality),
   with equality iff the predictive covariance is diagonal.
2. **Sandwich bound.** ``max_k EIG_k(x) <= EIG(x) <= sum_k EIG_k(x)``.
3. **Unit invariance.** ``EIG``, the marginal EIGs and ``TC`` are exactly
   invariant to rescaling any target by a positive constant (changing
   units). The unwhitened trace ``tr Sigma(x)`` is not.
4. **Second-order gap.** If ``s_k = Sigma_kk / R_k`` is the per-task
   epistemic-to-aleatoric ratio, the correlation entering ``TC`` obeys
   ``|c_jk| <= sqrt(s_j s_k / ((1 + s_j)(1 + s_k)))``, so ``TC = O(s^2)``
   while each ``EIG_k = O(s)``. For two targets this is exact:
   ``TC <= 1/2 log(1 + s_1 s_2 / (1 + s_1 + s_2))``, with equality iff the
   ensemble's disagreement is perfectly correlated (rank-one ``Sigma``).
   In the low-signal regime typical of later active-learning rounds, the
   joint and correlation-blind criteria agree to first order.
5. **Batch submodularity.** For a set ``S`` of candidates, ``EIG(S)``
   computed from the ensemble's cross-candidate covariance is monotone
   submodular, so :func:`greedy_batch_eig` is within ``1 - 1/e`` of the
   optimal batch (Nemhauser et al., 1978).

The scores in :mod:`quantum_al.joint_eig` (``1/2 log det(Sigma + R)``) are
the same criteria up to a candidate-independent additive constant
(``1/2 log det R``), so they induce identical rankings.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quantum_al.joint_eig import (
    oob_residual_variance,
    per_tree_predictions,
    predictive_covariance,
)

__all__ = [
    "expected_information_gain",
    "marginal_information_gains",
    "marginal_information_gain_sum",
    "total_correlation",
    "whitened_trace",
    "whitened_max_eigenvalue",
    "epistemic_aleatoric_ratio",
    "CRITERIA",
    "batch_information_gain",
    "greedy_batch_eig",
    "top_k",
    "fit_ensemble",
    "EnsembleCriterionSelector",
    "GreedyBatchEIGSelector",
]


def _check(Sigma, R):
    Sigma = np.asarray(Sigma, dtype=float)
    R = np.asarray(R, dtype=float)
    if Sigma.ndim == 2:
        Sigma = Sigma[None]
    if Sigma.ndim != 3 or Sigma.shape[1] != Sigma.shape[2]:
        raise ValueError(f"Sigma must have shape (n, K, K), got {Sigma.shape}")
    if R.shape != (Sigma.shape[1],):
        raise ValueError(f"R must have shape ({Sigma.shape[1]},), got {R.shape}")
    if np.any(R <= 0):
        raise ValueError("noise variances R must be strictly positive")
    return Sigma, R


def _whiten(Sigma, R):
    """R^{-1/2} Sigma R^{-1/2}, per candidate."""
    w = 1.0 / np.sqrt(R)
    return Sigma * w[None, :, None] * w[None, None, :]


def expected_information_gain(Sigma, R):
    """Joint expected information gain ``I(y; f)`` in nats, per candidate.

    Parameters
    ----------
    Sigma : ndarray, shape (n, K, K) or (K, K)
        Epistemic predictive covariance for each candidate.
    R : ndarray, shape (K,)
        Strictly positive per-task noise variances.

    Returns
    -------
    ndarray, shape (n,)
        ``0.5 * log det(I + R^{-1/2} Sigma R^{-1/2})``. Non-negative, zero
        iff ``Sigma`` is zero, and invariant to per-task changes of units.
    """
    Sigma, R = _check(Sigma, R)
    K = Sigma.shape[1]
    sign, logdet = np.linalg.slogdet(np.eye(K)[None] + _whiten(Sigma, R))
    if np.any(sign <= 0):
        raise ValueError("Sigma must be positive semi-definite")
    return 0.5 * logdet


def marginal_information_gains(Sigma, R):
    """Per-task expected information gains ``EIG_k``, shape (n, K)."""
    Sigma, R = _check(Sigma, R)
    diag = np.diagonal(Sigma, axis1=1, axis2=2)
    return 0.5 * np.log1p(np.maximum(diag, 0.0) / R[None, :])


def marginal_information_gain_sum(Sigma, R):
    """Correlation-blind criterion ``sum_k EIG_k``, shape (n,).

    Ranks candidates identically to
    :func:`quantum_al.joint_eig.marginal_sum_score`.
    """
    return marginal_information_gains(Sigma, R).sum(axis=1)


def total_correlation(Sigma, R):
    """Total correlation ``sum_k EIG_k - EIG >= 0`` among the K predicted
    labels, in nats, shape (n,). Equal to ``-0.5 log det`` of the
    correlation matrix of ``Sigma + diag(R)``."""
    return marginal_information_gain_sum(Sigma, R) - expected_information_gain(Sigma, R)


def whitened_trace(Sigma, R):
    """``tr(R^{-1} Sigma) = sum_k Sigma_kk / R_k``, shape (n,).

    Twice the first-order (small-signal) expansion of both the joint and
    the marginal EIG; a unit-invariant analogue of A-optimal design."""
    Sigma, R = _check(Sigma, R)
    return (np.diagonal(Sigma, axis1=1, axis2=2) / R[None, :]).sum(axis=1)


def whitened_max_eigenvalue(Sigma, R):
    """Largest eigenvalue of ``R^{-1/2} Sigma R^{-1/2}``, shape (n,).

    A unit-invariant analogue of E-optimal design: scores the single most
    uncertain linear combination of the K targets."""
    Sigma, R = _check(Sigma, R)
    return np.linalg.eigvalsh(_whiten(Sigma, R))[:, -1]


def epistemic_aleatoric_ratio(Sigma, R):
    """Per-task ratio ``s_k = Sigma_kk / R_k``, shape (n, K).

    Controls the size of the total-correlation term (property 4 in the
    module docstring)."""
    Sigma, R = _check(Sigma, R)
    return np.diagonal(Sigma, axis1=1, axis2=2) / R[None, :]


CRITERIA = {
    "eig": expected_information_gain,
    "marginal_eig": marginal_information_gain_sum,
    "trace": whitened_trace,
    "max_eigenvalue": whitened_max_eigenvalue,
    "total_correlation": total_correlation,
}
"""Name -> criterion function ``f(Sigma, R) -> scores`` used by
:class:`EnsembleCriterionSelector`."""


def _whitened_members(member_preds, R):
    """Centered, noise-whitened ensemble deviations, shape (M, n, K),
    scaled so that ``G[:, i, :].T @ G[:, i, :]`` is the whitened
    predictive covariance of candidate ``i``."""
    F = np.asarray(member_preds, dtype=float)
    if F.ndim == 2:
        F = F[:, :, None]
    M = F.shape[0]
    if M < 2:
        raise ValueError("need at least two ensemble members")
    R = np.asarray(R, dtype=float)
    if R.shape != (F.shape[2],) or np.any(R <= 0):
        raise ValueError("R must be a strictly positive vector of length K")
    G = (F - F.mean(axis=0, keepdims=True)) / np.sqrt(M - 1)
    return G / np.sqrt(R)[None, None, :]


def batch_information_gain(member_preds, R):
    """Exact joint EIG ``I(y_S; f)`` of observing all K labels at every
    candidate in a batch, in nats.

    Parameters
    ----------
    member_preds : ndarray, shape (M, b, K) or (M, b)
        Predictions of each of ``M`` ensemble members at the ``b`` batch
        candidates. Cross-candidate covariance between members captures
        redundancy between batch elements.
    R : ndarray, shape (K,)
        Per-task noise variances, assumed independent across candidates.

    Returns
    -------
    float
        ``0.5 * log det(I_{bK} + G G^T)``, computed in ``M``-dimensional
        form via Sylvester's determinant identity.
    """
    G = _whitened_members(member_preds, R)
    M = G.shape[0]
    Gm = G.reshape(M, -1)
    sign, logdet = np.linalg.slogdet(np.eye(M) + Gm @ Gm.T)
    return 0.5 * logdet


def greedy_batch_eig(member_preds, R, batch_size):
    """Greedy maximization of the batch information gain.

    Because the batch EIG is monotone submodular in the selected set, the
    greedy batch achieves at least ``(1 - 1/e)`` of the optimal batch EIG.
    Each step costs ``O(M^3 + n M^2 K)`` using the ``M``-dimensional
    (Sylvester) form of the determinant.

    Parameters
    ----------
    member_preds : ndarray, shape (M, n, K) or (M, n)
        Ensemble-member predictions for all ``n`` candidates.
    R : ndarray, shape (K,)
        Per-task noise variances.
    batch_size : int
        Number of candidates to select; clamped to ``n``.

    Returns
    -------
    selected : ndarray of int, shape (batch_size,)
        Candidate indices in the order chosen.
    gains : ndarray, shape (batch_size,)
        Marginal information gain of each pick given the earlier picks;
        non-increasing by submodularity and summing to the batch EIG.
    """
    G = _whitened_members(member_preds, R)  # (M, n, K)
    M, n, K = G.shape
    batch_size = int(min(batch_size, n))
    A = np.eye(M)
    available = np.ones(n, dtype=bool)
    selected, gains = [], []
    for _ in range(batch_size):
        L = np.linalg.cholesky(A)
        # V[:, i, :] = L^{-1} G_i^T; gain_i = 0.5 logdet(I_K + V_i^T V_i)
        V = np.linalg.solve(L, G.reshape(M, n * K)).reshape(M, n, K)
        gram = np.einsum("mik,mil->ikl", V, V) + np.eye(K)[None]
        gain = 0.5 * np.linalg.slogdet(gram)[1]
        gain[~available] = -np.inf
        best = int(np.argmax(gain))
        selected.append(best)
        gains.append(float(gain[best]))
        available[best] = False
        A = A + G[:, best, :] @ G[:, best, :].T
    return np.asarray(selected, dtype=int), np.asarray(gains)


def top_k(scores, k):
    """Indices of the ``k`` highest scores (ascending score order), the
    selection rule shared by every selector in this package."""
    k = int(min(k, len(scores)))
    return np.argsort(scores)[-k:] if k > 0 else np.array([], dtype=int)


def fit_ensemble(X_train, Y_train, n_estimators=200, seed=0):
    """Fit the random-forest ensemble used by every selector here.

    Returns
    -------
    forest : RandomForestRegressor
        Fitted with ``oob_score=True``; use
        :func:`quantum_al.joint_eig.predictive_covariance` or
        :func:`quantum_al.joint_eig.per_tree_predictions` on it.
    R : ndarray, shape (K,)
        Out-of-bag residual variance per target (the noise floor).
    """
    forest = RandomForestRegressor(
        n_estimators=n_estimators, random_state=seed, oob_score=True, n_jobs=-1,
    )
    forest.fit(X_train, Y_train)
    return forest, oob_residual_variance(forest, X_train, Y_train)


class EnsembleCriterionSelector:
    """Scores candidates with any criterion in :data:`CRITERIA`, using a
    (multi-output) random-forest ensemble's per-tree disagreement as
    ``Sigma(x)`` and its out-of-bag residual variance as ``R``.

    Parameters
    ----------
    criterion : str
        Key of :data:`CRITERIA`.
    n_estimators : int
        Number of trees in the acquisition ensemble.
    seed : int
        ``random_state`` of the forest.
    name : str, optional
        Display name; defaults to the criterion name.

    Notes
    -----
    Implements the package-wide selector interface
    ``select_next_experiments(X_candidates, X_train, Y_train, n_select)
    -> (selected_idx, scores, info)``. ``Y_train`` may be 1-D (single
    target) or ``(n, K)``. ``criterion="eig"`` reproduces the ranking of
    :class:`quantum_al.joint_eig.JointEIGSelector` and
    ``criterion="marginal_eig"`` that of
    :class:`quantum_al.joint_eig.MarginalSumSelector` for the same seed.
    """

    def __init__(self, criterion="eig", n_estimators=200, seed=0, name=None):
        if criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}; choose from {sorted(CRITERIA)}")
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.seed = seed
        self.name = name or criterion

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        forest, R = fit_ensemble(X_train, Y_train, self.n_estimators, self.seed)
        Sigma = predictive_covariance(forest, X_candidates)
        scores = CRITERIA[self.criterion](Sigma, R)
        return top_k(scores, n_select), scores, {"forest": forest, "R": R, "Sigma": Sigma}


class GreedyBatchEIGSelector:
    """Selects a whole batch by greedily maximizing the joint batch EIG
    (:func:`greedy_batch_eig`), penalizing candidates whose ensemble
    disagreement is redundant with candidates already in the batch.

    Parameters
    ----------
    n_estimators : int
        Number of trees (ensemble members ``M``). Batch EIG is computed in
        ``M`` dimensions, so its capacity saturates for batches with
        ``b * K`` much larger than ``M``.
    seed : int
        ``random_state`` of the forest.
    """

    def __init__(self, n_estimators=200, seed=0, name="Greedy-Batch-EIG"):
        self.n_estimators = n_estimators
        self.seed = seed
        self.name = name

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        forest, R = fit_ensemble(X_train, Y_train, self.n_estimators, self.seed)
        F = per_tree_predictions(forest, X_candidates)
        selected, gains = greedy_batch_eig(F, R, n_select)
        # per-candidate scores: stand-alone EIG (the greedy step-1 gains)
        scores = expected_information_gain(predictive_covariance(forest, X_candidates), R)
        return selected, scores, {"forest": forest, "R": R, "marginal_gains": gains}
