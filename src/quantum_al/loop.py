"""Pool-based batch active-learning loop shared by every benchmark.

Protocol for one trial: shuffle the pool with ``seed``, label the first
``n_initial`` points, then for ``n_rounds`` rounds (i) fit the evaluation
model on the labeled set and score it on the held-out test set, (ii) ask
the selector for ``batch_size`` of the remaining pool points, (iii) reveal
their labels (every target at once, as a single DFT calculation returns
all computed properties). The model is scored once more after the last
round, so a trial produces ``n_rounds + 1`` points of a learning curve.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from quantum_al.stats import area_under_learning_curve

__all__ = ["LearningCurve", "default_model_factory", "run_active_learning"]


def default_model_factory(seed):
    """The evaluation model used throughout the benchmarks: a 100-tree
    random forest seeded by the trial seed."""
    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)


@dataclass
class LearningCurve:
    """Per-round record of one active-learning trial.

    Attributes
    ----------
    n_labeled : list of int
        Labeled-set size at each evaluation.
    r2 : list of float
        Test R^2 averaged over targets at each evaluation.
    r2_per_task, mae_per_task : list of list of float
        Per-target test R^2 / mean absolute error at each evaluation.
    selected : list of list of int
        Pool indices acquired in each round.
    diagnostics : list
        Whatever ``on_round`` returned in each round (if given).
    """

    n_labeled: list = field(default_factory=list)
    r2: list = field(default_factory=list)
    r2_per_task: list = field(default_factory=list)
    mae_per_task: list = field(default_factory=list)
    selected: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)

    @property
    def final_r2(self):
        """Mean test R^2 after the last round."""
        return self.r2[-1]

    def aulc(self):
        """Normalized area under the mean-R^2 learning curve."""
        return area_under_learning_curve(self.n_labeled, self.r2)


def _validate_selection(sel_idx, n_candidates, n_requested):
    sel = np.asarray(sel_idx)
    if sel.ndim != 1 or (sel.size and not np.issubdtype(sel.dtype, np.integer)):
        raise ValueError(f"selector must return a 1-D integer index array, got {sel!r}")
    if sel.size == 0 or sel.size > n_requested:
        raise ValueError(f"selector returned {sel.size} indices, expected 1..{n_requested}")
    if sel.min() < 0 or sel.max() >= n_candidates:
        raise ValueError("selector returned an index outside the candidate set")
    if len(np.unique(sel)) != sel.size:
        raise ValueError("selector returned duplicate indices")
    return sel.astype(int)


def run_active_learning(selector, X_pool, Y_pool, X_test, Y_test, *, n_initial=50,
                        n_rounds=8, batch_size=15, seed=0,
                        model_factory=default_model_factory, on_round=None):
    """Run one pool-based batch active-learning trial.

    Parameters
    ----------
    selector : object
        Any object with ``select_next_experiments(X_candidates, X_train,
        Y_train, n_select) -> (selected_idx, scores, info)`` where
        ``selected_idx`` indexes ``X_candidates``.
    X_pool, Y_pool : ndarray
        Unlabeled pool features, shape (n, d), and the labels revealed on
        query, shape (n,) or (n, K).
    X_test, Y_test : ndarray
        Held-out evaluation set.
    n_initial : int
        Size of the random initial labeled set.
    n_rounds : int
        Number of acquisition rounds.
    batch_size : int
        Points acquired per round.
    seed : int
        Seeds the initial labeled set and the evaluation model, so trials
        of different selectors with equal ``seed`` are paired.
    model_factory : callable
        ``seed -> unfitted regressor`` used for evaluation.
    on_round : callable, optional
        ``on_round(round_index, info, X_candidates)`` is called after each
        selection with the selector's ``info`` dict; its return value is
        stored in :attr:`LearningCurve.diagnostics`.

    Returns
    -------
    LearningCurve

    Raises
    ------
    ValueError
        If the selector returns an invalid selection. Failures are never
        replaced by a fallback selection.
    """
    n_pool = X_pool.shape[0]
    if not 0 < n_initial < n_pool:
        raise ValueError("n_initial must be between 1 and the pool size - 1")
    perm = np.random.RandomState(seed).permutation(n_pool)
    labeled = perm[:n_initial].tolist()
    remaining = perm[n_initial:].tolist()
    Y_test_2d = Y_test.reshape(len(Y_test), -1)
    curve = LearningCurve()

    for it in range(n_rounds + 1):
        X_train, Y_train = X_pool[labeled], Y_pool[labeled]
        model = model_factory(seed)
        model.fit(X_train, Y_train)
        pred = model.predict(X_test).reshape(Y_test_2d.shape)
        per_task = r2_score(Y_test_2d, pred, multioutput="raw_values")
        curve.n_labeled.append(len(labeled))
        curve.r2.append(float(np.mean(per_task)))
        curve.r2_per_task.append([float(v) for v in per_task])
        curve.mae_per_task.append(
            [float(v) for v in mean_absolute_error(Y_test_2d, pred, multioutput="raw_values")]
        )
        if it == n_rounds or not remaining:
            break

        X_candidates = X_pool[remaining]
        n_select = min(batch_size, len(remaining))
        sel_idx, _scores, info = selector.select_next_experiments(
            X_candidates, X_train, Y_train, n_select=n_select
        )
        sel = _validate_selection(sel_idx, len(remaining), n_select)
        if on_round is not None:
            curve.diagnostics.append(on_round(it, info, X_candidates))
        chosen = [remaining[i] for i in sel]
        curve.selected.append(chosen)
        labeled.extend(chosen)
        chosen_set = set(chosen)
        remaining = [i for i in remaining if i not in chosen_set]
    return curve
