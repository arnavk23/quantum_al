"""Checks quantum_al.circuit's statevector-simulated variance/covariance
match quantum_al.operator's classical formula exactly. Skipped if qiskit
isn't installed (pip install -e ".[circuit]").
"""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from quantum_al.operator import make_observable, default_feature_groups, encode_states
from quantum_al.operator_sparse import make_sparse_observable
from quantum_al.circuit import circuit_variance, circuit_covariance


@pytest.fixture
def sample_state():
    rng = np.random.default_rng(0)
    d = 21
    x = rng.normal(size=(1, d))
    x = (x - x.mean()) / (x.std() + 1e-8)
    phase_weights = np.zeros(d)
    alpha = encode_states(x, phase_weights)[0]
    groups = default_feature_groups(d)
    O_struct = make_observable(d, groups["structural"], seed=1)
    O_elec = make_observable(d, groups["electronic"], seed=2)
    return alpha, O_struct, O_elec


def test_circuit_variance_matches_classical(sample_state):
    alpha, O_struct, _ = sample_state
    exp_O = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_O2 = np.real(np.conj(alpha) @ (O_struct @ (O_struct @ alpha)))
    classical_var = max(0.0, exp_O2 - exp_O ** 2)

    circuit_var, n_terms_O, n_terms_O2 = circuit_variance(alpha, O_struct, exact=True)

    assert abs(classical_var - circuit_var) < 1e-6
    assert n_terms_O > 0 and n_terms_O2 > 0


def test_circuit_covariance_matches_classical(sample_state):
    alpha, O_struct, O_elec = sample_state
    sym = (O_struct @ O_elec + O_elec @ O_struct) / 2.0
    exp_sym = np.real(np.conj(alpha) @ (sym @ alpha))
    exp_k = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_l = np.real(np.conj(alpha) @ (O_elec @ alpha))
    classical_cov = exp_sym - exp_k * exp_l

    circuit_cov, n_terms = circuit_covariance(alpha, O_struct, O_elec, exact=True)

    assert abs(classical_cov - circuit_cov) < 1e-6
    assert n_terms > 0


def test_sparse_circuit_variance_matches_classical(sample_state):
    alpha, _, _ = sample_state
    groups = default_feature_groups(21)
    O_sparse = make_sparse_observable(21, groups["structural"], seed=1)

    exp_O = np.real(np.conj(alpha) @ (O_sparse @ alpha))
    exp_O2 = np.real(np.conj(alpha) @ (O_sparse @ (O_sparse @ alpha)))
    classical_var = max(0.0, exp_O2 - exp_O ** 2)

    circuit_var, n_terms_O, n_terms_O2 = circuit_variance(alpha, O_sparse, exact=True)

    assert abs(classical_var - circuit_var) < 1e-6


def test_measurement_grouping_reduces_settings():
    from quantum_al.circuit import grouping_counts, measurement_grouping_report, pauli_decompose
    from quantum_al.operator import QuantumObservableBank

    bank = QuantumObservableBank(21, default_feature_groups(21), seed=0)
    report = measurement_grouping_report(bank.O)
    assert report["n_qubits"] == 5
    assert len(report["per_quantity"]) == 3 * 2 + 3  # O, O^2 per observable + pairwise products
    t = report["totals"]
    assert t["raw"] >= t["qwc_groups"] >= t["full_commuting_groups"] > 0
    # a diagonal observable is a sum of Z-strings: one setting suffices
    diag = pauli_decompose(np.diag(np.arange(4.0)).astype(complex))
    assert grouping_counts(diag)["qwc_groups"] == 1
