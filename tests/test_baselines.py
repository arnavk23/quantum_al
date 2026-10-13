"""Smoke and reproducibility tests for quantum_al.baselines on synthetic
data (no API key needed)."""
import numpy as np
import pytest

from quantum_al.baselines import BASELINES, CoreSet, get_all_baselines


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(20, 6))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + rng.normal(scale=0.1, size=20)
    X_candidates = rng.normal(size=(15, 6))
    return X_candidates, X_train, y_train


def test_get_all_baselines_returns_nine():
    baselines = get_all_baselines()
    assert len(baselines) == 9
    assert list(baselines) == [cls.name for cls in BASELINES]


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.name)
@pytest.mark.parametrize("n_select", [5, 25])
def test_baseline_selects_valid_batch(cls, n_select, toy_data):
    """n_select larger than the pool clamps instead of raising."""
    X_candidates, X_train, y_train = toy_data
    sel, _, _ = cls(seed=0).select_next_experiments(X_candidates, X_train, y_train, n_select=n_select)
    sel = np.asarray(sel)
    expected = min(n_select, len(X_candidates))
    assert len(set(sel.tolist())) == len(sel) == expected
    assert sel.min() >= 0 and sel.max() < len(X_candidates)


@pytest.mark.parametrize("cls", BASELINES, ids=lambda c: c.name)
def test_seeded_baselines_are_reproducible(cls, toy_data):
    picks = [cls(seed=3).select_next_experiments(*toy_data, n_select=5)[0] for _ in range(2)]
    assert np.array_equal(picks[0], picks[1])


def test_coreset_matches_brute_force_k_center(toy_data):
    X_candidates, X_train, _ = toy_data
    sel, _, _ = CoreSet(seed=1).select_next_experiments(X_candidates, X_train, None, n_select=6)
    chosen = [sel[0]]
    for _ in range(5):
        ref = np.vstack([X_train, X_candidates[chosen]])
        d = [np.inf if i in chosen else min(np.linalg.norm(x - r) for r in ref)
             for i, x in enumerate(X_candidates)]
        d = [-np.inf if v == np.inf else v for v in d]
        chosen.append(int(np.argmax(d)))
    assert sel.tolist() == chosen


def test_fallback_warns_and_selects_randomly(toy_data):
    X_candidates, X_train, _ = toy_data
    bad_y = np.full(len(X_train), np.nan)  # every model fit fails
    with pytest.warns(RuntimeWarning, match="falling back"):
        sel, scores, _ = get_all_baselines(seed=0)["RF Uncertainty"].select_next_experiments(
            X_candidates, X_train, bad_y, n_select=20)
    assert scores is None and len(set(np.asarray(sel).tolist())) == len(X_candidates)
