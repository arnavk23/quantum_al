"""Qiskit circuit realization of the operator.py formalism: amplitude
encoding on ceil(log2(d)) qubits, Pauli-decomposed measurement of <O>,
<O^2>, and covariance cross-terms. Also NISQ resource tools: gate/qubit
counts, shot-based estimation, depolarizing noise, measurement grouping.
"""
import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.primitives import StatevectorEstimator
    _HAVE_QISKIT = True
except ImportError:
    _HAVE_QISKIT = False

try:
    from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    _HAVE_AER = True
except ImportError:
    _HAVE_AER = False


def _require_qiskit():
    if not _HAVE_QISKIT:
        raise ImportError(
            "quantum_al.circuit requires qiskit, which is not installed. "
            "Install it with: pip install -e '.[circuit]'"
        )


def n_qubits_for_dim(d):
    return int(np.ceil(np.log2(max(d, 2))))


def pad_amplitudes(alpha, n_qubits):
    """Zero-pad a length-d amplitude vector to length 2^n_qubits, renormalize."""
    dim = 2 ** n_qubits
    padded = np.zeros(dim, dtype=complex)
    padded[: len(alpha)] = alpha
    norm = np.linalg.norm(padded)
    if norm > 1e-12:
        padded = padded / norm
    return padded


def pad_observable(O, n_qubits):
    """Zero-pad a d x d Hermitian matrix to 2^n_qubits x 2^n_qubits."""
    dim = 2 ** n_qubits
    d = O.shape[0]
    padded = np.zeros((dim, dim), dtype=complex)
    padded[:d, :d] = O
    return padded


def amplitude_encoding_circuit(alpha, n_qubits=None):
    """Real state-preparation circuit for ``|psi> = sum_j alpha_j |j>``."""
    _require_qiskit()
    if n_qubits is None:
        n_qubits = n_qubits_for_dim(len(alpha))
    padded = pad_amplitudes(alpha, n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(padded), range(n_qubits))
    return qc, padded


def pauli_decompose(O_padded):
    """SparsePauliOp decomposition of a Hermitian matrix.

    atol=0, rtol=0: default truncation in from_operator introduces ~1e-5
    error on operators like O^2, breaking exact agreement with the
    classical formula. We truncate genuinely-zero terms (<1e-12) ourselves
    instead.
    """
    _require_qiskit()
    op = SparsePauliOp.from_operator(O_padded, atol=0, rtol=0)
    mask = np.abs(op.coeffs) > 1e-12
    op = SparsePauliOp(op.paulis[mask], op.coeffs[mask])
    return op


def exact_circuit_expectation(qc, pauli_op):
    """Noiseless expectation value via exact statevector simulation."""
    _require_qiskit()
    estimator = StatevectorEstimator()
    job = estimator.run([(qc, pauli_op)])
    result = job.result()[0]
    return float(np.real(result.data.evs))


def shot_based_expectation(qc, pauli_op, shots, noise_model=None, seed=0):
    """Finite-shot expectation via AerEstimatorV2; shots maps to a target
    precision of 1/sqrt(shots), not a literal execution count. Decomposes
    the circuit first since Aer doesn't accept StatePreparation directly."""
    if not _HAVE_AER:
        raise RuntimeError("qiskit-aer is required for shot-based simulation")
    backend_options = {"seed_simulator": seed}
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    estimator = AerEstimatorV2(options={"backend_options": backend_options})
    qc_decomposed = qc.decompose(reps=5)
    job = estimator.run([(qc_decomposed, pauli_op)], precision=1.0 / np.sqrt(shots))
    result = job.result()[0]
    return float(np.real(result.data.evs))


def circuit_variance(alpha, O, n_qubits=None, exact=True, shots=None, noise_model=None, seed=0):
    """Var(O; psi) = <O^2> - <O>^2 via circuit measurement of O and O^2."""
    if n_qubits is None:
        n_qubits = n_qubits_for_dim(len(alpha))
    qc, padded_alpha = amplitude_encoding_circuit(alpha, n_qubits)
    O_pad = pad_observable(O, n_qubits)
    O2_pad = O_pad @ O_pad

    op_O = pauli_decompose(O_pad)
    op_O2 = pauli_decompose(O2_pad)

    if exact:
        exp_O = exact_circuit_expectation(qc, op_O)
        exp_O2 = exact_circuit_expectation(qc, op_O2)
    else:
        exp_O = shot_based_expectation(qc, op_O, shots, noise_model, seed)
        exp_O2 = shot_based_expectation(qc, op_O2, shots, noise_model, seed + 1)

    var = exp_O2 - exp_O ** 2
    return max(0.0, var), len(op_O.paulis), len(op_O2.paulis)


def circuit_covariance(alpha, Ok, Ol, n_qubits=None, exact=True, shots=None, noise_model=None, seed=0):
    if n_qubits is None:
        n_qubits = n_qubits_for_dim(len(alpha))
    qc, padded_alpha = amplitude_encoding_circuit(alpha, n_qubits)
    Ok_pad = pad_observable(Ok, n_qubits)
    Ol_pad = pad_observable(Ol, n_qubits)
    sym = (Ok_pad @ Ol_pad + Ol_pad @ Ok_pad) / 2.0

    op_k = pauli_decompose(Ok_pad)
    op_l = pauli_decompose(Ol_pad)
    op_sym = pauli_decompose(sym)

    if exact:
        exp_sym = exact_circuit_expectation(qc, op_sym)
        exp_k = exact_circuit_expectation(qc, op_k)
        exp_l = exact_circuit_expectation(qc, op_l)
    else:
        exp_sym = shot_based_expectation(qc, op_sym, shots, noise_model, seed)
        exp_k = shot_based_expectation(qc, op_k, shots, noise_model, seed + 1)
        exp_l = shot_based_expectation(qc, op_l, shots, noise_model, seed + 2)

    cov = exp_sym - exp_k * exp_l
    return cov, len(op_sym.paulis)


def transpiled_resource_report(alpha, basis_gates=("cx", "rz", "sx", "x")):
    """Qubit count, gate count, and depth for the state-prep circuit alone."""
    from qiskit import transpile
    n_qubits = n_qubits_for_dim(len(alpha))
    qc, _ = amplitude_encoding_circuit(alpha, n_qubits)
    tqc = transpile(qc, basis_gates=list(basis_gates), optimization_level=1)
    gate_counts = dict(tqc.count_ops())
    return {
        "n_qubits": n_qubits,
        "depth": tqc.depth(),
        "gate_counts": gate_counts,
        "total_gates": sum(gate_counts.values()),
        "cx_count": gate_counts.get("cx", 0),
    }


def grouping_counts(pauli_op):
    """Measurement settings needed for one Pauli-decomposed observable.

    Returns a dict with ``raw`` (number of Pauli terms, i.e. settings with
    no grouping), ``qwc_groups`` (qubit-wise commuting groups, measurable
    with single-qubit basis changes) and ``full_commuting_groups`` (general
    commuting groups, which need entangling basis changes). Groups come
    from Qiskit's greedy graph colouring, so counts are upper bounds on
    the true minimum."""
    _require_qiskit()
    if len(pauli_op.paulis) == 0:
        return {"raw": 0, "qwc_groups": 0, "full_commuting_groups": 0}
    return {
        "raw": len(pauli_op.paulis),
        "qwc_groups": len(pauli_op.group_commuting(qubit_wise=True)),
        "full_commuting_groups": len(pauli_op.group_commuting(qubit_wise=False)),
    }


def measurement_grouping_report(observables, n_qubits=None):
    """Measurement cost of every quantity ``U_total`` needs on hardware.

    Parameters
    ----------
    observables : dict
        ``name -> (d, d)`` Hermitian matrix (e.g. ``QuantumObservableBank.O``).
    n_qubits : int, optional
        Register size; defaults to ``ceil(log2 d)``.

    Returns
    -------
    dict
        ``per_quantity`` grouping counts (see :func:`grouping_counts`) for
        each ``<name>_O``, ``<name>_O2`` and symmetrized product
        ``<a>_<b>_sym``, and their ``totals`` with reduction factors.
    """
    names = list(observables)
    d = observables[names[0]].shape[0]
    n_qubits = n_qubits or n_qubits_for_dim(d)
    padded = {n: pad_observable(observables[n], n_qubits) for n in names}
    per_quantity = {}
    for n in names:
        per_quantity[f"{n}_O"] = grouping_counts(pauli_decompose(padded[n]))
        per_quantity[f"{n}_O2"] = grouping_counts(pauli_decompose(padded[n] @ padded[n]))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sym = (padded[a] @ padded[b] + padded[b] @ padded[a]) / 2.0
            per_quantity[f"{a}_{b}_sym"] = grouping_counts(pauli_decompose(sym))
    totals = {key: sum(q[key] for q in per_quantity.values())
              for key in ("raw", "qwc_groups", "full_commuting_groups")}
    totals["qwc_reduction_factor"] = totals["raw"] / max(totals["qwc_groups"], 1)
    totals["full_reduction_factor"] = totals["raw"] / max(totals["full_commuting_groups"], 1)
    return {"n_qubits": n_qubits, "per_quantity": per_quantity, "totals": totals}


def make_depolarizing_noise_model(p1=0.001, p2=0.01):
    """1-/2-qubit depolarizing noise at representative near-term error rates."""
    if not _HAVE_AER:
        return None
    noise_model = NoiseModel()
    err1 = depolarizing_error(p1, 1)
    err2 = depolarizing_error(p2, 2)
    noise_model.add_all_qubit_quantum_error(err1, ["sx", "x", "rz"])
    noise_model.add_all_qubit_quantum_error(err2, ["cx"])
    return noise_model


def self_test():
    """Checks circuit-computed Var/Cov match operator.py's classical
    closed-form exactly in the noiseless limit."""
    from quantum_al.operator import make_observable, default_feature_groups, encode_states

    rng = np.random.default_rng(0)
    d = 21
    x = rng.normal(size=(1, d))
    x = (x - x.mean()) / (x.std() + 1e-8)
    phase_weights = np.zeros(d)
    alpha = encode_states(x, phase_weights)[0]

    groups = default_feature_groups(d)
    O_struct = make_observable(d, groups["structural"], seed=1)
    O_elec = make_observable(d, groups["electronic"], seed=2)

    # Classical closed-form reference.
    exp_O = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_O2 = np.real(np.conj(alpha) @ (O_struct @ (O_struct @ alpha)))
    classical_var = max(0.0, exp_O2 - exp_O ** 2)

    circuit_var, n_terms_O, n_terms_O2 = circuit_variance(alpha, O_struct, exact=True)
    print(f"[self_test] classical Var = {classical_var:.8f}")
    print(f"[self_test] circuit   Var = {circuit_var:.8f}  "
          f"({n_terms_O} Pauli terms for O, {n_terms_O2} for O^2)")
    assert abs(classical_var - circuit_var) < 1e-6, "circuit/classical variance mismatch"

    sym = (O_struct @ O_elec + O_elec @ O_struct) / 2.0
    exp_sym = np.real(np.conj(alpha) @ (sym @ alpha))
    exp_k = np.real(np.conj(alpha) @ (O_struct @ alpha))
    exp_l = np.real(np.conj(alpha) @ (O_elec @ alpha))
    classical_cov = exp_sym - exp_k * exp_l

    circuit_cov, n_terms_cov = circuit_covariance(alpha, O_struct, O_elec, exact=True)
    print(f"[self_test] classical Cov = {classical_cov:.8f}")
    print(f"[self_test] circuit   Cov = {circuit_cov:.8f}  ({n_terms_cov} Pauli terms)")
    assert abs(classical_cov - circuit_cov) < 1e-6, "circuit/classical covariance mismatch"

    resources = transpiled_resource_report(alpha)
    print(f"[self_test] state-prep resources: {resources}")

    print("[self_test] circuit realization matches classical formalism exactly. All checks passed.")


if __name__ == "__main__":
    self_test()
