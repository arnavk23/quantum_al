"""Nine classical single-target active-learning baselines.

Every class implements the package-wide selector interface
``select_next_experiments(X_candidates, X_train, y_train, n_select) ->
(selected_idx, scores, info)``; :func:`get_all_baselines` returns one fresh
instance of each.

Selectors that draw random numbers (``RandomSampling``, ``CoreSet`` and the
fallbacks) take a ``seed``. With the default ``seed=None`` they use NumPy's
global random state, which is how the results in ``results/`` were
produced (the benchmark scripts seed it once per trial).

If a baseline's model fails to fit (e.g. a Gaussian process on a degenerate
labeled set), it falls back to random selection for that round and emits a
``RuntimeWarning``. Use ``warnings.simplefilter("error", RuntimeWarning)``
to make the fallback fatal instead.
"""
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor

from quantum_al.acquisition import top_k

__all__ = [
    "UncertaintySampling",
    "QueryByCommittee",
    "ExpectedImprovement",
    "MaximumEntropy",
    "DiversitySampling",
    "BADGE",
    "CoreSet",
    "RandomForestUncertainty",
    "RandomSampling",
    "get_all_baselines",
]


class _Baseline:
    name = None

    def __init__(self, seed=None):
        self.seed = seed
        self._rng = np.random if seed is None else np.random.RandomState(seed)

    def select_next_experiments(self, X_candidates, X_train, y_train, n_select=10):
        try:
            return self._select(X_candidates, X_train, y_train, n_select)
        except Exception as e:
            warnings.warn(
                f"{self.name} failed ({type(e).__name__}: {e}); falling back to random "
                "selection for this round.",
                RuntimeWarning,
                stacklevel=2,
            )
            n = len(X_candidates)
            return self._rng.choice(n, min(n_select, n), replace=False), None, {}

    def _select(self, X_candidates, X_train, y_train, n_select):
        raise NotImplementedError


def _tree_predictions(forest, X):
    return np.array([t.predict(X) for t in forest.estimators_])


def _closest_to_centroids(points, n_clusters):
    """Index of the point nearest each k-means centroid."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(points)
    selected = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if members.size:
            dist = np.linalg.norm(points[members] - kmeans.cluster_centers_[c], axis=1)
            selected.append(members[np.argmin(dist)])
    return np.array(selected)


class UncertaintySampling(_Baseline):
    """Largest Gaussian-process posterior standard deviation (RBF kernel)."""

    name = "Uncertainty Sampling"

    def _select(self, X_candidates, X_train, y_train, n_select):
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6).fit(X_train, y_train)
        _, std = gp.predict(X_candidates, return_std=True)
        return top_k(std, n_select), std, {"gp_uncertainty": std}


class QueryByCommittee(_Baseline):
    """Largest prediction variance across a committee of three random
    forests and two neural networks."""

    name = "Query by Committee"

    def _select(self, X_candidates, X_train, y_train, n_select):
        committee = [
            RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0),
            RandomForestRegressor(n_estimators=50, max_depth=10, random_state=1),
            RandomForestRegressor(n_estimators=50, max_depth=15, random_state=2),
            MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=3),
            MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=4),
        ]
        preds = np.array([m.fit(X_train, y_train).predict(X_candidates) for m in committee])
        disagreement = np.var(preds, axis=0)
        return top_k(disagreement, n_select), disagreement, {"committee_variance": disagreement}


class ExpectedImprovement(_Baseline):
    """Expected improvement over the best label so far, from a Gaussian
    process with a Matern-5/2 kernel."""

    name = "Expected Improvement"

    def _select(self, X_candidates, X_train, y_train, n_select):
        kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6).fit(X_train, y_train)
        mean, std = gp.predict(X_candidates, return_std=True)
        improvement = mean - np.max(y_train)
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        return top_k(ei, n_select), ei, {"expected_improvement": ei}


class MaximumEntropy(_Baseline):
    """Largest Gaussian entropy of the random-forest per-tree predictions."""

    name = "Maximum Entropy"

    def _select(self, X_candidates, X_train, y_train, n_select):
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        variance = np.var(_tree_predictions(rf, X_candidates), axis=0)
        entropy = 0.5 * np.log(2 * np.pi * np.e * variance + 1e-9)
        return top_k(entropy, n_select), entropy, {"entropy": entropy}


class DiversitySampling(_Baseline):
    """The candidate nearest each of ``n_select`` k-means centroids."""

    name = "Diversity Sampling"

    def _select(self, X_candidates, X_train, y_train, n_select):
        if len(X_candidates) <= n_select:
            return np.arange(len(X_candidates)), None, {}
        selected = _closest_to_centroids(X_candidates, n_select)
        scores = np.zeros(len(X_candidates))
        scores[selected] = 1.0
        return selected, scores, {"diversity_selected": True}


class BADGE(_Baseline):
    """Regression analogue of BADGE: k-means++ diversity over candidates
    embedded by importance-weighted features plus random-forest
    uncertainty."""

    name = "BADGE"

    def _select(self, X_candidates, X_train, y_train, n_select):
        if len(X_candidates) <= n_select:
            return np.arange(len(X_candidates)), None, {}
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        uncertainty = np.var(_tree_predictions(rf, X_candidates), axis=0)
        embeddings = np.column_stack([X_candidates * rf.feature_importances_, uncertainty])
        selected = _closest_to_centroids(embeddings, n_select)
        scores = np.zeros(len(X_candidates))
        scores[selected] = uncertainty[selected]
        return selected, scores, {"badge_embeddings": True}


class CoreSet(_Baseline):
    """Greedy k-center selection: repeatedly pick the candidate farthest
    from the labeled set and the candidates already picked, starting from a
    random candidate."""

    name = "CoreSet"

    def _select(self, X_candidates, X_train, y_train, n_select):
        n = len(X_candidates)
        n_select = min(n_select, n)
        first = self._rng.randint(n)
        selected = [first]
        min_dist = np.linalg.norm(X_candidates[:, None, :] - X_train[None, :, :], axis=2).min(axis=1)
        available = np.ones(n, dtype=bool)
        available[first] = False
        for _ in range(n_select - 1):
            min_dist = np.minimum(min_dist, np.linalg.norm(X_candidates - X_candidates[selected[-1]], axis=1))
            best = int(np.argmax(np.where(available, min_dist, -np.inf)))
            selected.append(best)
            available[best] = False
        selected = np.array(selected)
        scores = np.zeros(n)
        scores[selected] = 1.0
        return selected, scores, {"coreset_selected": True}


class RandomForestUncertainty(_Baseline):
    """Largest variance of the random-forest per-tree predictions."""

    name = "RF Uncertainty"

    def _select(self, X_candidates, X_train, y_train, n_select):
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        uncertainty = np.var(_tree_predictions(rf, X_candidates), axis=0)
        return top_k(uncertainty, n_select), uncertainty, {"rf_uncertainty": uncertainty}


class RandomSampling(_Baseline):
    """Uniformly random selection."""

    name = "Random Sampling"

    def _select(self, X_candidates, X_train, y_train, n_select):
        n = len(X_candidates)
        selected = self._rng.choice(n, min(n_select, n), replace=False)
        # Unused by the selection; drawn so the random stream matches the
        # one that produced the published results.
        scores = self._rng.random_sample(n)
        return selected, scores, {"random_selection": True}


BASELINES = [
    UncertaintySampling, QueryByCommittee, ExpectedImprovement, MaximumEntropy,
    DiversitySampling, BADGE, CoreSet, RandomForestUncertainty, RandomSampling,
]
"""The nine baseline classes, in the order used throughout ``results/``."""


def get_all_baselines(seed=None):
    """``name -> fresh selector`` for all nine baselines."""
    return {cls.name: cls(seed=seed) for cls in BASELINES}
