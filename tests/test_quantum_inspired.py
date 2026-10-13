"""Correctness tests for quantum_al.quantum_inspired: state normalization,
non-commutativity, the exact classical-limit reduction, sparse observables,
and valid selections from every selector."""
import numpy as np
import pytest

from quantum_al.data_utils import FEATURE_COLUMNS
from quantum_al.quantum_inspired import (
    ImportanceWeightedQuantumSelector,
    QuantumObservableBank,
    QuantumSelector,
    TreeEnsembleQuantumSelector,
    default_feature_groups,
    domain_feature_groups,
    encode_states,
    feature_phase_weights,
    make_sparse_observable,
    softmax,
    tree_group_split,
)


def test_softmax_sums_to_one_and_is_stable():
    x = np.array([
        [1e3, -1e3, 0.0, 500.0],
        [-1e3, -1e3, -1e3, -999.0],
        [0.1, 0.2, 0.3, 0.4],
    ])
    p = softmax(x, axis=1)
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(p, axis=1), np.argmax(x, axis=1))


def test_encode_states_normalized():
    X = np.random.default_rng(1).normal(size=(4, 12))
    psi = encode_states(X, feature_phase_weights(X))
    assert np.allclose(np.sum(np.abs(psi) ** 2, axis=1), 1.0, atol=1e-10)


@pytest.mark.parametrize("sparse", [False, True])
def test_default_bank_is_noncommuting(sparse):
    bank = QuantumObservableBank(12, default_feature_groups(12), seed=1, sparse=sparse)
    assert bank.commutator_norm() > 1e-6


def test_commuting_only_bank_commutes():
    bank = QuantumObservableBank(12, default_feature_groups(12), seed=1,
                                 commuting_only=True, real_only_coeff=True)
    assert bank.commutator_norm() < 1e-9


def test_classical_limit_reduction_exact():
    """Proposition 2: with commuting observables, real coefficients and no
    covariance term, U_total^2 = sum_k alpha_k^2 Var_k exactly."""
    d = 12
    bank = QuantumObservableBank(d, default_feature_groups(d), seed=1,
                                 commuting_only=True, real_only_coeff=True)
    x = np.random.default_rng(0).normal(size=(1, d))
    psi = encode_states((x - x.mean()) / x.std(), np.zeros(d))[0]
    manual = np.sqrt(sum(bank.alpha[k].real ** 2 * bank.variance(psi, k) for k in bank.names))
    assert abs(bank.total_uncertainty(psi, use_covariance=False) - manual) < 1e-9


@pytest.mark.parametrize("real_only", [True, False])
@pytest.mark.parametrize("commuting_only", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_batch_scores_finite_nonnegative(real_only, commuting_only, sparse):
    d = 10
    bank = QuantumObservableBank(d, default_feature_groups(d), seed=6, sparse=sparse,
                                 commuting_only=commuting_only, real_only_coeff=real_only)
    rng = np.random.default_rng(7)
    scores = bank.batch_scores(rng.normal(size=(10, d)), rng.normal(size=(25, d)))
    assert scores.shape == (25,)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0)
    psi = encode_states(rng.normal(size=(5, d)), np.zeros(d))
    assert all(bank.variance(p, n) >= 0 for p in psi for n in bank.names)


def test_sparse_observable_is_zero_outside_block():
    group = [2, 3, 4, 5]
    O = make_sparse_observable(12, group, seed=0)
    mask = np.ones((12, 12), dtype=bool)
    mask[np.ix_(group, group)] = False
    assert np.all(O[mask] == 0.0)
    assert np.allclose(O, O.T)


def test_disjoint_sparse_observables_have_exactly_zero_covariance():
    O1 = make_sparse_observable(12, [0, 1, 2], seed=1)
    O2 = make_sparse_observable(12, [6, 7, 8], seed=2)
    assert np.max(np.abs(O1 @ O2 + O2 @ O1)) == 0.0


def test_feature_groups():
    groups = domain_feature_groups()
    assert sorted(i for idx in groups.values() for i in idx) == list(range(len(FEATURE_COLUMNS)))
    trees = tree_group_split(10, 3)
    assert sorted(i for idx in trees.values() for i in idx) == list(range(10))


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    d = len(FEATURE_COLUMNS)
    X_train = rng.normal(size=(20, d))
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + rng.normal(scale=0.1, size=20)
    return rng.normal(size=(15, d)), X_train, y_train


SELECTORS = {
    "dense": lambda d: QuantumSelector(d, seed=0),
    "sparse": lambda d: QuantumSelector(d, seed=0, sparse=True),
    "no-covariance": lambda d: QuantumSelector(d, seed=0, use_covariance=False),
    "importance-weighted": lambda d: ImportanceWeightedQuantumSelector(d, seed=0),
    "tree-ensemble": lambda d: TreeEnsembleQuantumSelector(seed=0, n_estimators=9),
    "tree-ensemble-sparse": lambda d: TreeEnsembleQuantumSelector(seed=0, n_estimators=9, sparse=True),
    "tree-ensemble-no-cov": lambda d: TreeEnsembleQuantumSelector(
        seed=0, n_estimators=9, use_covariance=False),
}


@pytest.mark.parametrize("name", SELECTORS)
def test_selector_selects_valid_batch(name, toy_data):
    X_candidates, X_train, y_train = toy_data
    selector = SELECTORS[name](X_train.shape[1])
    sel, scores, info = selector.select_next_experiments(X_candidates, X_train, y_train, n_select=5)
    sel = np.asarray(sel)
    assert len(set(sel.tolist())) == 5
    assert sel.min() >= 0 and sel.max() < len(X_candidates)
    assert np.all(np.isfinite(scores))
