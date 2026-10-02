"""Two variants of QuantumObservableBank tested against the baseline gap
(results/improvement_attempt.json): domain-informed observable groups
(vs. arbitrary contiguous thirds) and predictor-importance-weighted
feature rescaling before state encoding. Core formalism unchanged.
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from quantum_al.operator import (
    QuantumObservableBank, encode_states, feature_phase_weights, softmax,
)
from quantum_al.data_utils import FEATURE_COLUMNS

DOMAIN_GROUPS_BY_NAME = {
    "structural": ["density", "volume_per_atom", "nsites", "space_group_number",
                   "atomic_radius_mean", "atomic_radius_std", "atomic_radius_range"],
    "electronic": ["X_mean", "X_std", "X_range",  # electronegativity + periodic group
                   "group_mean", "group_std", "group_range"],
    "thermodynamic": ["nelements", "energy_above_hull",
                       "atomic_mass_mean", "atomic_mass_std", "atomic_mass_range",
                       "row_mean", "row_std", "row_range"],
}


def domain_feature_groups(feature_columns=FEATURE_COLUMNS):
    idx = {name: i for i, name in enumerate(feature_columns)}
    return {
        group: [idx[c] for c in cols if c in idx]
        for group, cols in DOMAIN_GROUPS_BY_NAME.items()
    }


class ImportanceWeightedQuantumSelector:
    """U_total ranking, but features rescaled by ``sqrt(RF feature_importances_)``
    before state encoding."""

    def __init__(self, d, feature_groups=None, seed=0, use_covariance=True,
                 n_estimators=50, name="Quantum-ImportanceWeighted"):
        self.name = name
        groups = feature_groups if feature_groups is not None else domain_feature_groups()
        self.bank = QuantumObservableBank(d, groups, seed=seed)
        self.use_covariance = use_covariance
        self.n_estimators = n_estimators

    def _importance_weights(self, X_train, y_train, d):
        """Falls back to uniform weights if RF fitting fails on too few samples."""
        try:
            rf = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
            rf.fit(X_train, y_train)
        except ValueError as e:
            warnings.warn(
                f"RandomForest fit failed ({e}); falling back to uniform "
                "importance weights for this iteration."
            )
            return np.ones(d)
        imp = rf.feature_importances_
        imp = np.clip(imp, 1e-6, None)
        return np.sqrt(imp / imp.mean())

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        d = X_train.shape[1]
        w = self._importance_weights(X_train, y_train, d)
        X_train_w = X_train * w[None, :]
        X_cand_w = X_candidates * w[None, :]

        phase_weights = feature_phase_weights(X_train_w)
        psi_pool = encode_states(X_cand_w, phase_weights)
        scores = np.array([
            self.bank.total_uncertainty(psi_pool[i], use_covariance=self.use_covariance)
            for i in range(psi_pool.shape[0])
        ])
        n_select = min(n_select, len(X_candidates))
        selected_idx = np.argsort(scores)[-n_select:]
        return selected_idx, scores, {"quantum_scores": scores, "importance_weights": w.tolist()}
