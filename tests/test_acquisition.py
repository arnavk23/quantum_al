"""Checks every mathematical property claimed in quantum_al.acquisition:
decomposition, sandwich bound, unit invariance, second-order gap, batch
EIG identities, monotone submodularity, and ranking equivalence with the
original quantum_al.joint_eig selectors."""
import itertools

import numpy as np
import pytest

from quantum_al.acquisition import (
    CRITERIA,
    EnsembleCriterionSelector,
    GreedyBatchEIGSelector,
    batch_information_gain,
    epistemic_aleatoric_ratio,
    expected_information_gain,
    greedy_batch_eig,
    marginal_information_gain_sum,
    marginal_information_gains,
    top_k,
    total_correlation,
    whitened_max_eigenvalue,
    whitened_trace,
)
from quantum_al.joint_eig import (
    JointEIGSelector,
    MarginalSumSelector,
    joint_eig_score,
    marginal_sum_score,
)


def random_psd(rng, n, K, rank=None):
    rank = rank or K
    A = rng.normal(size=(n, K, rank))
    return np.einsum("nij,nkj->nik", A, A)


@pytest.fixture
def cov():
    rng = np.random.default_rng(0)
    return random_psd(rng, 40, 3), rng.uniform(0.1, 2.0, size=3)


def test_eig_nonnegative_and_zero_without_epistemic_uncertainty(cov):
    Sigma, R = cov
    assert np.all(expected_information_gain(Sigma, R) >= 0)
    assert np.allclose(expected_information_gain(np.zeros_like(Sigma), R), 0.0)


def test_total_correlation_decomposition_identity(cov):
    Sigma, R = cov
    tc = total_correlation(Sigma, R)
    total = Sigma + np.diag(R)[None]
    d = np.diagonal(total, axis1=1, axis2=2)
    corr = total / np.sqrt(d[:, :, None] * d[:, None, :])
    assert np.all(tc >= -1e-12)
    assert np.allclose(tc, -0.5 * np.linalg.slogdet(corr)[1], atol=1e-10)


def test_total_correlation_zero_iff_diagonal():
    rng = np.random.default_rng(1)
    Sigma = np.stack([np.diag(rng.uniform(0.1, 3, size=4)) for _ in range(10)])
    assert np.allclose(total_correlation(Sigma, np.ones(4)), 0.0, atol=1e-12)


def test_sandwich_bound(cov):
    Sigma, R = cov
    joint = expected_information_gain(Sigma, R)
    marg = marginal_information_gains(Sigma, R)
    assert np.all(marg.max(axis=1) <= joint + 1e-12)
    assert np.all(joint <= marg.sum(axis=1) + 1e-12)


def test_unit_invariance(cov):
    """Changing a target's units (Sigma -> D Sigma D, R -> D^2 R) leaves the
    information-theoretic criteria unchanged but not the raw trace."""
    Sigma, R = cov
    D = np.array([1e-3, 1.0, 250.0])
    Sigma_s = Sigma * D[None, :, None] * D[None, None, :]
    R_s = R * D ** 2
    for name in ["eig", "marginal_eig", "total_correlation", "trace", "max_eigenvalue"]:
        assert np.allclose(CRITERIA[name](Sigma, R), CRITERIA[name](Sigma_s, R_s), rtol=1e-8)
    raw_trace = np.trace(Sigma, axis1=1, axis2=2)
    raw_trace_s = np.trace(Sigma_s, axis1=1, axis2=2)
    assert not np.array_equal(np.argsort(raw_trace), np.argsort(raw_trace_s))


def test_rankings_match_original_joint_eig_scores(cov):
    """The normalized criteria differ from quantum_al.joint_eig's scores only
    by the candidate-independent constant 0.5 * sum(log R)."""
    Sigma, R = cov
    shift = 0.5 * np.log(R).sum()
    assert np.allclose(expected_information_gain(Sigma, R), joint_eig_score(Sigma, R) - shift)
    assert np.allclose(marginal_information_gain_sum(Sigma, R), marginal_sum_score(Sigma, R) - shift)


def test_gap_is_second_order_in_snr():
    """TC / sum_k EIG_k -> 0 as the epistemic-to-aleatoric ratio -> 0, and
    the correlation bound |c_jk| <= sqrt(s_j s_k / ((1+s_j)(1+s_k))) holds."""
    rng = np.random.default_rng(2)
    base = random_psd(rng, 50, 3)
    R = np.ones(3)
    ratios = []
    for scale in [1.0, 1e-1, 1e-2, 1e-3]:
        Sigma = scale * base
        ratios.append(np.median(total_correlation(Sigma, R) / marginal_information_gain_sum(Sigma, R)))
        s = epistemic_aleatoric_ratio(Sigma, R)
        total = Sigma + np.diag(R)[None]
        d = np.diagonal(total, axis1=1, axis2=2)
        c = np.abs(total / np.sqrt(d[:, :, None] * d[:, None, :]))
        bound = np.sqrt((s / (1 + s))[:, :, None] * (s / (1 + s))[:, None, :])
        off = ~np.eye(3, dtype=bool)
        assert np.all(c[:, off] <= bound[:, off] + 1e-12)
    assert all(b < a for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] < 1e-2


def test_trace_and_max_eigenvalue_agree_with_definitions(cov):
    Sigma, R = cov
    W = Sigma / np.sqrt(R)[None, :, None] / np.sqrt(R)[None, None, :]
    assert np.allclose(whitened_trace(Sigma, R), np.trace(W, axis1=1, axis2=2))
    assert np.allclose(whitened_max_eigenvalue(Sigma, R), np.linalg.eigvalsh(W)[:, -1])


def test_input_validation():
    with pytest.raises(ValueError):
        expected_information_gain(np.eye(2), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        expected_information_gain(np.eye(2), np.ones(3))


# ---- batch EIG -------------------------------------------------------------

@pytest.fixture
def members():
    rng = np.random.default_rng(3)
    M, n, K = 12, 9, 2
    shared = rng.normal(size=(M, 1, K))
    return shared + 0.7 * rng.normal(size=(M, n, K)), np.array([0.5, 2.0])


def test_batch_eig_single_candidate_equals_pointwise_eig(members):
    F, R = members
    M = F.shape[0]
    for i in range(F.shape[1]):
        Fi = F[:, i, :]
        Sigma = np.cov(Fi, rowvar=False, ddof=1)[None]
        assert np.isclose(batch_information_gain(Fi[:, None, :], R),
                          expected_information_gain(Sigma, R)[0])
    assert M > 1


def test_batch_eig_matches_direct_bk_dimensional_logdet(members):
    F, R = members
    S = [0, 3, 5]
    Fs = F[:, S, :].reshape(F.shape[0], -1)
    cov_S = np.cov(Fs, rowvar=False, ddof=1)
    noise = np.tile(R, len(S))
    direct = 0.5 * np.linalg.slogdet(np.eye(len(noise)) + cov_S / np.sqrt(noise)[:, None] / np.sqrt(noise)[None, :])[1]
    assert np.isclose(batch_information_gain(F[:, S, :], R), direct)


def test_batch_eig_monotone_and_submodular(members):
    F, R = members
    n = F.shape[1]
    f = lambda S: batch_information_gain(F[:, list(S), :], R) if S else 0.0  # noqa: E731
    for A in itertools.combinations(range(n), 2):
        for extra in range(n):
            if extra in A:
                continue
            B = A + (extra,)
            for x in range(n):
                if x in B:
                    continue
                gain_A = f(A + (x,)) - f(A)
                gain_B = f(B + (x,)) - f(B)
                assert gain_A >= -1e-12  # monotone
                assert gain_A >= gain_B - 1e-10  # diminishing returns


def test_greedy_gains_sum_to_batch_eig_and_decrease(members):
    F, R = members
    sel, gains = greedy_batch_eig(F, R, 5)
    assert len(set(sel.tolist())) == 5
    assert np.isclose(gains.sum(), batch_information_gain(F[:, sel, :], R))
    assert np.all(np.diff(gains) <= 1e-10)


def test_greedy_first_pick_is_pointwise_eig_argmax(members):
    F, R = members
    Sigma = np.stack([np.cov(F[:, i, :], rowvar=False, ddof=1) for i in range(F.shape[1])])
    sel, _ = greedy_batch_eig(F, R, 1)
    assert sel[0] == int(np.argmax(expected_information_gain(Sigma, R)))


def test_greedy_within_one_minus_one_over_e_of_optimum(members):
    F, R = members
    b = 3
    sel, _ = greedy_batch_eig(F, R, b)
    best = max(batch_information_gain(F[:, list(S), :], R)
               for S in itertools.combinations(range(F.shape[1]), b))
    assert batch_information_gain(F[:, sel, :], R) >= (1 - 1 / np.e) * best


# ---- selectors -------------------------------------------------------------

@pytest.fixture
def multitask_data():
    rng = np.random.default_rng(4)
    X_train = rng.normal(size=(60, 6))
    Y_train = np.stack([X_train[:, 0], X_train[:, 0] + 0.3 * X_train[:, 1]], axis=1)
    Y_train += 0.1 * rng.normal(size=Y_train.shape)
    return rng.normal(size=(30, 6)), X_train, Y_train


@pytest.mark.parametrize("criterion", sorted(CRITERIA))
def test_criterion_selector_valid_batch(criterion, multitask_data):
    Xc, Xt, Yt = multitask_data
    sel, scores, info = EnsembleCriterionSelector(criterion, n_estimators=30).select_next_experiments(Xc, Xt, Yt, 5)
    assert len(set(np.asarray(sel).tolist())) == 5 and scores.shape == (30,)
    assert np.all(np.isfinite(scores))


def test_criterion_selector_reproduces_original_selectors(multitask_data):
    Xc, Xt, Yt = multitask_data
    a, _, _ = EnsembleCriterionSelector("eig", n_estimators=40, seed=1).select_next_experiments(Xc, Xt, Yt, 6)
    b, _, _ = JointEIGSelector(n_estimators=40, seed=1).select_next_experiments(Xc, Xt, Yt, 6)
    assert set(a.tolist()) == set(b.tolist())
    a, _, _ = EnsembleCriterionSelector("marginal_eig", n_estimators=40, seed=1).select_next_experiments(Xc, Xt, Yt, 6)
    b, _, _ = MarginalSumSelector(n_estimators=40, seed=1).select_next_experiments(Xc, Xt, Yt, 6)
    assert set(a.tolist()) == set(b.tolist())


def test_criterion_selector_single_target(multitask_data):
    Xc, Xt, Yt = multitask_data
    sel, scores, _ = EnsembleCriterionSelector("eig", n_estimators=30).select_next_experiments(Xc, Xt, Yt[:, 0], 4)
    assert len(sel) == 4 and np.all(scores >= 0)


def test_greedy_batch_selector(multitask_data):
    Xc, Xt, Yt = multitask_data
    sel, scores, info = GreedyBatchEIGSelector(n_estimators=30).select_next_experiments(Xc, Xt, Yt, 5)
    assert len(set(sel.tolist())) == 5 and scores.shape == (30,)
    assert np.all(np.diff(info["marginal_gains"]) <= 1e-10)


def test_unknown_criterion():
    with pytest.raises(ValueError):
        EnsembleCriterionSelector("nope")


def test_top_k():
    assert set(top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2).tolist()) == {1, 3}
    assert len(top_k(np.arange(3.0), 10)) == 3


def test_two_task_exact_total_correlation_bound():
    """K=2: TC <= 0.5 log(1 + s1 s2 / (1 + s1 + s2)), tight for rank-1 Sigma."""
    rng = np.random.default_rng(5)
    R = np.array([0.3, 1.7])
    for rank, tight in [(2, False), (1, True)]:
        Sigma = random_psd(rng, 200, 2, rank=rank)
        s = epistemic_aleatoric_ratio(Sigma, R)
        bound = 0.5 * np.log1p(s[:, 0] * s[:, 1] / (1 + s[:, 0] + s[:, 1]))
        tc = total_correlation(Sigma, R)
        assert np.all(tc <= bound + 1e-12)
        if tight:
            assert np.allclose(tc, bound, atol=1e-10)
