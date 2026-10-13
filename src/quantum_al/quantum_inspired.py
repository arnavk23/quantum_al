"""Covariance-aware quantum-inspired uncertainty formalism.

Each candidate ``x`` is encoded as a normalized complex amplitude vector
``|psi(x)>`` (Definition 1 of the manuscript), and scored by

.. math::

    U_{\\mathrm{total}}(x) = \\Big(\\sum_k |\\alpha_k|^2 \\mathrm{Var}(O_k)
        + \\sum_{k \\ne l} \\mathrm{Re}(\\bar\\alpha_k \\alpha_l)
          \\mathrm{Cov}(O_k, O_l)\\Big)^{1/2},

where the ``O_k`` are Hermitian observables concentrated on feature groups,
``Var`` and ``Cov`` are quantum (symmetrized) moments in ``|psi(x)>``, and
the ``alpha_k`` are complex coupling coefficients. With commuting
observables, real coefficients and no covariance term, ``U_total`` reduces
exactly to a weighted sum of classical variances (the classical limit,
Proposition 2).

The module also provides the variants evaluated in ``benchmarks/``: sparse
observables (cheaper to measure on hardware), domain-informed feature
groups with importance-weighted encoding, and an encoding of per-tree
random-forest predictions instead of raw features. :mod:`quantum_al.circuit`
realizes the same quantities as a Qiskit circuit.
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quantum_al.acquisition import top_k
from quantum_al.data_utils import FEATURE_COLUMNS

__all__ = [
    "softmax",
    "feature_phase_weights",
    "encode_states",
    "make_observable",
    "make_sparse_observable",
    "default_feature_groups",
    "domain_feature_groups",
    "tree_group_split",
    "QuantumObservableBank",
    "QuantumSelector",
    "ImportanceWeightedQuantumSelector",
    "TreeEnsembleQuantumSelector",
]


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def feature_phase_weights(X_labeled):
    """Phase ``phi_j`` of each feature: ``pi`` times its mean absolute
    correlation with the other features over the labeled pool. Zero when
    fewer than three points are labeled."""
    d = X_labeled.shape[1]
    if X_labeled.shape[0] < 3:
        return np.zeros(d)
    C = np.nan_to_num(np.corrcoef(X_labeled, rowvar=False), nan=0.0)
    np.fill_diagonal(C, 0.0)
    return np.pi * np.mean(np.abs(C), axis=1)


def encode_states(X, phase_weights):
    """Amplitude vectors ``alpha_ij = sqrt(softmax(x_i^2)_j) exp(i phi_j)``,
    shape (n, d), one unit-norm state per row of standardized ``X``."""
    p = softmax(X ** 2, axis=1)
    return np.sqrt(p) * np.exp(1j * phase_weights[None, :])


def _unit_spectral_norm(H):
    eigmax = np.max(np.abs(np.linalg.eigvalsh(H)))
    return H / eigmax if eigmax > 1e-12 else H


def make_observable(d, feature_group, seed, cross_weight=0.15, block_boost=2.5):
    """Dense random Hermitian ``d x d`` matrix, weighted toward the
    ``feature_group`` block and scaled to unit spectral norm."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    S = (A + A.T) / 2.0
    mask = np.zeros((d, d))
    mask[np.ix_(feature_group, feature_group)] = 1.0
    return _unit_spectral_norm(cross_weight * S + block_boost * mask * S)


def make_sparse_observable(d, feature_group, seed):
    """Random Hermitian matrix that is zero outside the
    ``feature_group`` block, scaled to unit spectral norm."""
    rng = np.random.default_rng(seed)
    k = len(feature_group)
    A = rng.normal(size=(k, k))
    H = np.zeros((d, d))
    H[np.ix_(feature_group, feature_group)] = (A + A.T) / 2.0
    return _unit_spectral_norm(H)


def default_feature_groups(d):
    """Split feature indices into three contiguous, ``structural``,
    ``electronic`` and ``thermodynamic`` groups that overlap by one index."""
    third = max(1, d // 3)
    return {
        "structural": list(range(0, third + 1)),
        "electronic": list(range(third, 2 * third + 1)),
        "thermodynamic": list(range(2 * third, d)),
    }


DOMAIN_GROUPS_BY_NAME = {
    "structural": ["density", "volume_per_atom", "nsites", "space_group_number",
                   "atomic_radius_mean", "atomic_radius_std", "atomic_radius_range"],
    "electronic": ["X_mean", "X_std", "X_range", "group_mean", "group_std", "group_range"],
    "thermodynamic": ["nelements", "energy_above_hull",
                      "atomic_mass_mean", "atomic_mass_std", "atomic_mass_range",
                      "row_mean", "row_std", "row_range"],
}


def domain_feature_groups(feature_columns=FEATURE_COLUMNS):
    """Feature groups chosen by physical meaning rather than by index, for
    the Materials Project feature set of :mod:`quantum_al.data_utils`."""
    idx = {name: i for i, name in enumerate(feature_columns)}
    return {group: [idx[c] for c in cols if c in idx]
            for group, cols in DOMAIN_GROUPS_BY_NAME.items()}


def tree_group_split(n_trees, k=3):
    """Split ``n_trees`` indices into ``k`` contiguous, near-equal groups."""
    block = n_trees // k
    return {f"tree_group_{i}": list(range(i * block, n_trees if i == k - 1 else (i + 1) * block))
            for i in range(k)}


class QuantumObservableBank:
    """``K`` Hermitian observables with complex coupling coefficients.

    Parameters
    ----------
    d : int
        Dimension of the encoded states (number of features).
    feature_groups : dict
        ``name -> feature indices`` for each observable.
    coefficients : dict, optional
        ``name -> complex alpha_k``. Defaults to ``1``, ``1.2 e^{i pi/4}``
        and ``0.8`` for ``structural``, ``electronic`` and
        ``thermodynamic``, and ``1`` for any other name.
    seed : int
        Observable ``k`` is drawn with seed ``seed + k``.
    commuting_only : bool
        Use diagonal (commuting) observables, for the classical-limit
        ablation.
    real_only_coeff : bool
        Drop the imaginary part of the coefficients, for the same ablation.
    sparse : bool
        Use observables that are zero outside their own feature block
        (:func:`make_sparse_observable`).
    """

    def __init__(self, d, feature_groups, coefficients=None, seed=0,
                 commuting_only=False, real_only_coeff=False, sparse=False):
        self.d = d
        self.names = list(feature_groups)
        self.K = len(self.names)

        if commuting_only:
            self.O = {}
            for i, name in enumerate(self.names):
                diag = np.zeros(d)
                diag[feature_groups[name]] = 1.0
                diag += 0.05 * (i + 1)  # break degeneracy so eigenbases align
                self.O[name] = np.diag(diag / np.max(np.abs(diag)))
        else:
            make = make_sparse_observable if sparse else make_observable
            self.O = {name: make(d, feature_groups[name], seed=seed + i)
                      for i, name in enumerate(self.names)}

        if coefficients is None:
            default = {
                "structural": 1.0 + 0j,
                "electronic": 1.2 * np.exp(1j * np.pi / 4),
                "thermodynamic": 0.8 + 0j,
            }
            coefficients = {n: default.get(n, 1.0 + 0j) for n in self.names}
        if real_only_coeff:
            coefficients = {k: complex(v.real, 0.0) for k, v in coefficients.items()}
        self.alpha = coefficients

    def commutator_norm(self):
        """Sum of ``||[O_k, O_l]||_F`` over pairs; zero iff all commute."""
        total = 0.0
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                total += np.linalg.norm(self.O[a] @ self.O[b] - self.O[b] @ self.O[a])
        return total

    def variance(self, psi, name):
        """``<psi|O^2|psi> - <psi|O|psi>^2``, clipped at zero."""
        O = self.O[name]
        exp_O = np.real(np.conj(psi) @ (O @ psi))
        exp_O2 = np.real(np.conj(psi) @ (O @ (O @ psi)))
        return max(0.0, exp_O2 - exp_O ** 2)

    def covariance(self, psi, name_k, name_l):
        """Symmetrized covariance ``<{O_k, O_l}>/2 - <O_k><O_l>``."""
        Ok, Ol = self.O[name_k], self.O[name_l]
        exp_sym = np.real(np.conj(psi) @ (((Ok @ Ol + Ol @ Ok) / 2.0) @ psi))
        exp_k = np.real(np.conj(psi) @ (Ok @ psi))
        exp_l = np.real(np.conj(psi) @ (Ol @ psi))
        return exp_sym - exp_k * exp_l

    def total_uncertainty(self, psi, use_covariance=True):
        """``U_total`` of one state."""
        var_term = sum(abs(self.alpha[k]) ** 2 * self.variance(psi, k) for k in self.names)
        cov_term = 0.0
        if use_covariance:
            for k in self.names:
                for l in self.names:  # noqa: E741
                    if k != l:
                        cov_term += (np.real(np.conj(self.alpha[k]) * self.alpha[l])
                                     * self.covariance(psi, k, l))
        return np.sqrt(max(0.0, var_term + cov_term))

    def scores(self, psi, use_covariance=True):
        """``U_total`` for each row of a state matrix ``psi``."""
        return np.array([self.total_uncertainty(p, use_covariance) for p in psi])

    def batch_scores(self, X_labeled, X_pool, use_covariance=True):
        """``U_total`` of every candidate in ``X_pool``, with phases taken
        from the labeled pool ``X_labeled``."""
        psi = encode_states(X_pool, feature_phase_weights(X_labeled))
        return self.scores(psi, use_covariance)


class QuantumSelector:
    """Selects the candidates with the largest ``U_total``.

    Parameters are those of :class:`QuantumObservableBank`;
    ``feature_groups`` defaults to :func:`default_feature_groups`.
    """

    def __init__(self, d, feature_groups=None, seed=0, use_covariance=True,
                 commuting_only=False, real_only_coeff=False, sparse=False,
                 name="Quantum-Enhanced"):
        self.name = name
        groups = feature_groups if feature_groups is not None else default_feature_groups(d)
        self.bank = QuantumObservableBank(d, groups, seed=seed, commuting_only=commuting_only,
                                          real_only_coeff=real_only_coeff, sparse=sparse)
        self.use_covariance = use_covariance

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        scores = self.bank.batch_scores(X_train, X_candidates, self.use_covariance)
        return top_k(scores, n_select), scores, {"quantum_scores": scores}


class ImportanceWeightedQuantumSelector:
    """:class:`QuantumSelector` with domain feature groups, and features
    rescaled by ``sqrt`` of random-forest feature importances before
    encoding."""

    def __init__(self, d, feature_groups=None, seed=0, use_covariance=True,
                 n_estimators=50, name="Quantum-ImportanceWeighted"):
        self.name = name
        groups = feature_groups if feature_groups is not None else domain_feature_groups()
        self.bank = QuantumObservableBank(d, groups, seed=seed)
        self.use_covariance = use_covariance
        self.n_estimators = n_estimators

    def _importance_weights(self, X_train, y_train):
        try:
            rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            rf.fit(X_train, y_train)
        except ValueError as e:
            warnings.warn(f"RandomForest fit failed ({e}); using uniform importance weights "
                          "for this round.", RuntimeWarning, stacklevel=3)
            return np.ones(X_train.shape[1])
        imp = np.clip(rf.feature_importances_, 1e-6, None)
        return np.sqrt(imp / imp.mean())

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        w = self._importance_weights(X_train, y_train)
        scores = self.bank.batch_scores(X_train * w, X_candidates * w, self.use_covariance)
        return top_k(scores, n_select), scores, {
            "quantum_scores": scores, "importance_weights": w.tolist(),
        }


class TreeEnsembleQuantumSelector:
    """``U_total`` computed on per-tree random-forest predictions instead of
    raw features, which couples the score to ensemble disagreement.

    The trees are split into ``n_groups`` exchangeable groups, one
    observable each; the groups have no physical meaning.
    """

    def __init__(self, seed=0, n_estimators=99, n_groups=3, use_covariance=True,
                 sparse=False, name="Quantum-TreeEnsemble"):
        self.name = name
        self.seed = seed
        self.n_estimators = n_estimators
        self.n_groups = n_groups
        self.use_covariance = use_covariance
        self.sparse = sparse

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.seed)
        rf.fit(X_train, y_train)
        tree_cand = np.stack([t.predict(X_candidates) for t in rf.estimators_], axis=1)
        tree_train = np.stack([t.predict(X_train) for t in rf.estimators_], axis=1)

        mu = tree_train.mean(axis=0)
        sigma = tree_train.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z_train = (tree_train - mu) / sigma
        Z_cand = (tree_cand - mu) / sigma

        n_trees = Z_cand.shape[1]
        bank = QuantumObservableBank(n_trees, tree_group_split(n_trees, self.n_groups),
                                     seed=self.seed, sparse=self.sparse)
        scores = bank.batch_scores(Z_train, Z_cand, self.use_covariance)
        return top_k(scores, n_select), scores, {"quantum_scores": scores}
