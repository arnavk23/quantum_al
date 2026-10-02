"""Tests for the shared active-learning loop, the statistics helpers, the
synthetic correlated-task generator and the selection diagnostics."""
import numpy as np
import pytest

from quantum_al.acquisition import EnsembleCriterionSelector
from quantum_al.diagnostics import criterion_disagreement, rank_agreement, topk_overlap
from quantum_al.loop import run_active_learning
from quantum_al.stats import (
    area_under_learning_curve,
    bootstrap_ci,
    holm_bonferroni,
    paired_comparison,
)
from quantum_al.synthetic import make_correlated_tasks


class FirstK:
    name = "first-k"

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        return np.arange(n_select), np.zeros(len(X_candidates)), {"n": len(X_candidates)}


class Duplicates(FirstK):
    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        return np.zeros(n_select, dtype=int), None, {}


@pytest.fixture
def split():
    X, Y, _ = make_correlated_tasks(n_samples=200, n_features=5, correlation=0.6, seed=0)
    return X[:140], Y[:140], X[140:], Y[140:]


# ---- synthetic ---------------------------------------------------------------

@pytest.mark.parametrize("rho", [0.0, 0.3, 0.9, 1.0])
def test_signal_correlation_is_exact(rho):
    _, _, info = make_correlated_tasks(n_samples=300, n_tasks=3, correlation=rho, noise=0.5, seed=1)
    C = np.corrcoef(info["signal"], rowvar=False)
    assert np.allclose(C[~np.eye(3, dtype=bool)], rho, atol=1e-10)


def test_label_correlation_tracks_expectation():
    _, Y, info = make_correlated_tasks(n_samples=5000, correlation=0.8, noise=0.5, seed=2)
    assert abs(np.corrcoef(Y, rowvar=False)[0, 1] - info["expected_label_correlation"]) < 0.05


def test_task_scales_and_determinism():
    X1, Y1, _ = make_correlated_tasks(n_samples=50, task_scales=[1.0, 100.0], seed=3)
    X2, Y2, _ = make_correlated_tasks(n_samples=50, task_scales=[1.0, 100.0], seed=3)
    assert np.array_equal(X1, X2) and np.array_equal(Y1, Y2)
    assert Y1[:, 1].std() > 30 * Y1[:, 0].std()


@pytest.mark.parametrize("kwargs", [{"correlation": 1.5}, {"noise": -1}, {"task_scales": [1.0]}])
def test_synthetic_validation(kwargs):
    with pytest.raises(ValueError):
        make_correlated_tasks(**kwargs)


# ---- loop --------------------------------------------------------------------

def test_loop_shapes_and_bookkeeping(split):
    curve = run_active_learning(FirstK(), *split, n_initial=20, n_rounds=3, batch_size=7,
                                on_round=lambda it, info, Xc: info["n"])
    assert curve.n_labeled == [20, 27, 34, 41]
    assert len(curve.r2) == 4 and len(curve.r2_per_task[0]) == 2
    assert curve.diagnostics == [120, 113, 106]
    flat = [i for batch in curve.selected for i in batch]
    assert len(set(flat)) == len(flat) == 21
    assert np.isfinite(curve.aulc())


def test_loop_is_deterministic_given_seed(split):
    sel = EnsembleCriterionSelector("eig", n_estimators=20, seed=0)
    a = run_active_learning(sel, *split, n_initial=20, n_rounds=2, batch_size=5, seed=4)
    b = run_active_learning(sel, *split, n_initial=20, n_rounds=2, batch_size=5, seed=4)
    assert np.allclose(a.r2, b.r2, rtol=1e-12) and a.selected == b.selected


def test_loop_single_target(split):
    X, Y, Xt, Yt = split
    curve = run_active_learning(FirstK(), X, Y[:, 0], Xt, Yt[:, 0], n_initial=20, n_rounds=1, batch_size=5)
    assert len(curve.r2_per_task[0]) == 1


def test_loop_rejects_invalid_selection(split):
    with pytest.raises(ValueError, match="duplicate"):
        run_active_learning(Duplicates(), *split, n_initial=20, n_rounds=1, batch_size=5)


def test_loop_stops_when_pool_exhausted(split):
    X, Y, Xt, Yt = split
    curve = run_active_learning(FirstK(), X[:30], Y[:30], Xt, Yt, n_initial=20, n_rounds=5, batch_size=7)
    assert curve.n_labeled == [20, 27, 30]


# ---- stats -------------------------------------------------------------------

def test_holm_bonferroni_known_values():
    adj, reject = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    assert reject == [True, False, False]


def test_paired_comparison_detects_shift():
    rng = np.random.default_rng(0)
    b = rng.normal(size=20)
    out = paired_comparison(b + 0.5 + 0.05 * rng.normal(size=20), b)
    assert out["p_value_t"] < 1e-6 and out["p_value_wilcoxon"] < 1e-3
    lo, hi = out["ci95_diff"]
    assert lo < 0.5 < hi and out["effect_size_dz"] > 1


def test_paired_comparison_identical_inputs():
    out = paired_comparison([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert out["mean_diff"] == 0 and out["p_value_t"] is None


def test_bootstrap_ci_contains_mean():
    v = np.arange(10.0)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi


def test_aulc():
    assert area_under_learning_curve([0, 10], [0.0, 1.0]) == pytest.approx(0.5)
    assert area_under_learning_curve([0, 10, 20], [1.0, 1.0, 1.0], normalize=False) == pytest.approx(20)


# ---- diagnostics -------------------------------------------------------------

def test_overlap_and_rank_agreement():
    s = np.arange(10.0)
    assert topk_overlap(s, s, 3) == 1.0
    assert topk_overlap(s, -s, 3) == 0.0
    assert rank_agreement(s, s) == pytest.approx(1.0)


def test_criterion_disagreement_uncorrelated_is_identical():
    rng = np.random.default_rng(0)
    Sigma = np.stack([np.diag(rng.uniform(0.1, 1, size=2)) for _ in range(30)])
    d = criterion_disagreement(Sigma, np.ones(2), 5)
    assert d["topk_overlap"] == 1.0 and d["mean_tc"] == pytest.approx(0.0, abs=1e-12)
