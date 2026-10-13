"""NISQ resource characterization on real band-gap candidates: qubit/gate
counts, Pauli measurement overhead vs K, shot/noise sensitivity of the
acquisition ranking vs the exact statevector computation.
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)  # only for sibling benchmarks/*.py scripts below,
# which are standalone (not part of the installed quantum_al package);
# quantum_al.* imports themselves work without this once pip-installed.

from quantum_al.circuit import (  # noqa: E402
    amplitude_encoding_circuit,
    exact_circuit_expectation,
    make_depolarizing_noise_model,
    n_qubits_for_dim,
    pad_observable,
    pauli_decompose,
    shot_based_expectation,
    transpiled_resource_report,
)
from quantum_al.data_utils import load_task, standardize  # noqa: E402
from quantum_al.quantum_inspired import (  # noqa: E402
    QuantumObservableBank,
    default_feature_groups,
    encode_states,
    feature_phase_weights,
    make_observable,
)

N_CANDIDATES = 30
SHOT_BUDGETS = [100, 1000, 10000, 100000]
BATCH_SIZE = 15


def default_feature_groups_k(d, k):
    if k <= 1:
        return {"group0": list(range(d))}
    block = max(1, d // k)
    groups = {}
    for i in range(k):
        start = i * block
        end = d if i == k - 1 else min(d, (i + 1) * block + 1)
        start = max(0, start - (1 if i > 0 else 0))
        groups[f"group{i}"] = list(range(start, end))
    return groups


def build_bank(d, groups, seed=0):
    names = list(groups.keys())
    n_qubits = n_qubits_for_dim(d)
    O = {name: make_observable(d, groups[name], seed=seed + i) for i, name in enumerate(names)}
    O_pad = {name: pad_observable(O[name], n_qubits) for name in names}
    return names, O, O_pad


def u_total_from_expectations(names, alpha_dict, coeff, exp_O, exp_O2, exp_sym):
    var_term = 0.0
    for k in names:
        var_term += (abs(coeff[k]) ** 2) * max(0.0, exp_O2[k] - exp_O[k] ** 2)
    cov_term = 0.0
    for k in names:
        for l in names:
            if k == l:
                continue
            cov_term += np.real(np.conj(coeff[k]) * coeff[l]) * (exp_sym[(k, l)] - exp_O[k] * exp_O[l])
    return float(np.sqrt(max(0.0, var_term + cov_term)))


def compute_u_total_circuit(alpha, names, O_pad, coeff, qc_cache, mode="exact", shots=None,
                             noise_model=None, seed=0):
    qc, _ = amplitude_encoding_circuit(alpha)
    exp_O, exp_O2, exp_sym = {}, {}, {}
    s = seed
    for k in names:
        op_k = qc_cache[("op", k)]
        op_k2 = qc_cache[("op2", k)]
        if mode == "exact":
            exp_O[k] = exact_circuit_expectation(qc, op_k)
            exp_O2[k] = exact_circuit_expectation(qc, op_k2)
        else:
            exp_O[k] = shot_based_expectation(qc, op_k, shots, noise_model, s)
            exp_O2[k] = shot_based_expectation(qc, op_k2, shots, noise_model, s + 1)
            s += 2
    for i, k in enumerate(names):
        for l in names[i + 1:]:
            op_sym = qc_cache[("sym", k, l)]
            if mode == "exact":
                v = exact_circuit_expectation(qc, op_sym)
            else:
                v = shot_based_expectation(qc, op_sym, shots, noise_model, s)
                s += 1
            exp_sym[(k, l)] = v
            exp_sym[(l, k)] = v
    return u_total_from_expectations(names, alpha, coeff, exp_O, exp_O2, exp_sym)


def main():
    print("Loading real band_gap candidates...")
    X, y, meta = load_task("band_gap")
    X = standardize(X)
    d = X.shape[1]
    n_qubits = n_qubits_for_dim(d)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=N_CANDIDATES, replace=False)
    X_cand = X[idx]
    phase_weights = feature_phase_weights(X[:100])
    alphas = encode_states(X_cand, phase_weights)

    groups = default_feature_groups(d)
    names, O, O_pad = build_bank(d, groups, seed=1)
    coeff = {"structural": 1.0 + 0j, "electronic": 1.2 * np.exp(1j * np.pi / 4), "thermodynamic": 0.8 + 0j}

    print("Decomposing observables into Pauli operators (K=3, default)...")
    qc_cache = {}
    pauli_term_counts = {}
    for k in names:
        op_k = pauli_decompose(O_pad[k])
        op_k2 = pauli_decompose(O_pad[k] @ O_pad[k])
        qc_cache[("op", k)] = op_k
        qc_cache[("op2", k)] = op_k2
        pauli_term_counts[f"{k}_O"] = len(op_k.paulis)
        pauli_term_counts[f"{k}_O2"] = len(op_k2.paulis)
    for i, k in enumerate(names):
        for l in names[i + 1:]:
            sym = (O_pad[k] @ O_pad[l] + O_pad[l] @ O_pad[k]) / 2.0
            op_sym = pauli_decompose(sym)
            qc_cache[("sym", k, l)] = op_sym
            pauli_term_counts[f"{k}_{l}_sym"] = len(op_sym.paulis)
    print("Pauli term counts (K=3):", pauli_term_counts)

    print("\nComputing exact U_total for all candidates (ground truth)...")
    exact_scores = []
    for i in range(N_CANDIDATES):
        u = compute_u_total_circuit(alphas[i], names, O_pad, coeff, qc_cache, mode="exact")
        exact_scores.append(u)
    exact_scores = np.array(exact_scores)
    classical = QuantumObservableBank(d, groups, coefficients=coeff, seed=1).scores(alphas)
    max_dev = float(np.max(np.abs(exact_scores - classical)))
    if max_dev > 1e-6:
        raise RuntimeError(f"circuit U_total deviates from the classical formula by {max_dev:.2e}")
    print(f"Circuit matches the classical formula (max deviation {max_dev:.1e})")
    exact_ranking = np.argsort(exact_scores)[::-1]
    exact_top_b = set(exact_ranking[:BATCH_SIZE].tolist())
    print("Exact scores range:", exact_scores.min(), exact_scores.max())

    results = {
        "n_candidates": N_CANDIDATES, "n_qubits": n_qubits, "batch_size": BATCH_SIZE,
        "pauli_term_counts_K3": pauli_term_counts,
        "exact_scores": exact_scores.tolist(),
        "shot_sweep": {},
    }

    for shots in SHOT_BUDGETS:
        print(f"\n--- shots={shots} (noiseless) ---")
        t0 = time.time()
        shot_scores = []
        for i in range(N_CANDIDATES):
            u = compute_u_total_circuit(alphas[i], names, O_pad, coeff, qc_cache,
                                         mode="shot", shots=shots, seed=i * 100)
            shot_scores.append(u)
        shot_scores = np.array(shot_scores)
        mae = float(np.mean(np.abs(shot_scores - exact_scores)))
        tau, _ = scipy_stats.kendalltau(exact_scores, shot_scores)
        shot_ranking = np.argsort(shot_scores)[::-1]
        shot_top_b = set(shot_ranking[:BATCH_SIZE].tolist())
        overlap = len(exact_top_b & shot_top_b) / BATCH_SIZE
        dt = time.time() - t0
        print(f"  MAE={mae:.4f}  Kendall-tau={tau:.4f}  top-{BATCH_SIZE} overlap={overlap:.2f}  ({dt:.1f}s)")
        results["shot_sweep"][str(shots)] = {
            "mae_vs_exact": mae, "kendall_tau": float(tau), "top_b_overlap_frac": overlap,
            "wall_time_sec": dt,
        }

    print("\n--- shots=10000 WITH depolarizing noise (p1=0.001, p2=0.01) ---")
    noise_model = make_depolarizing_noise_model(p1=0.001, p2=0.01)
    t0 = time.time()
    noisy_scores = []
    for i in range(N_CANDIDATES):
        u = compute_u_total_circuit(alphas[i], names, O_pad, coeff, qc_cache,
                                     mode="shot", shots=10000, noise_model=noise_model, seed=i * 100)
        noisy_scores.append(u)
    noisy_scores = np.array(noisy_scores)
    mae_noisy = float(np.mean(np.abs(noisy_scores - exact_scores)))
    tau_noisy, _ = scipy_stats.kendalltau(exact_scores, noisy_scores)
    noisy_ranking = np.argsort(noisy_scores)[::-1]
    noisy_top_b = set(noisy_ranking[:BATCH_SIZE].tolist())
    overlap_noisy = len(exact_top_b & noisy_top_b) / BATCH_SIZE
    dt = time.time() - t0
    print(f"  MAE={mae_noisy:.4f}  Kendall-tau={tau_noisy:.4f}  "
          f"top-{BATCH_SIZE} overlap={overlap_noisy:.2f}  ({dt:.1f}s)")
    results["noisy_10000shots_p1_0.001_p2_0.01"] = {
        "mae_vs_exact": mae_noisy, "kendall_tau": float(tau_noisy), "top_b_overlap_frac": overlap_noisy,
        "wall_time_sec": dt,
    }

    print("\nComputing state-prep resource scaling for K=1,3,6 (Pauli term counts)...")
    resource_by_k = {}
    example_alpha = alphas[0]
    resource_by_k["state_prep"] = transpiled_resource_report(example_alpha)
    for k in [1, 3, 6]:
        g = default_feature_groups_k(d, k) if k != 3 else groups
        gnames, gO, gO_pad = build_bank(d, g, seed=1)
        total_terms = 0
        for name in gnames:
            op = pauli_decompose(gO_pad[name])
            total_terms += len(op.paulis)
        n_pairs = k * (k - 1) // 2
        resource_by_k[f"K={k}"] = {
            "n_observables": k, "n_pairs_for_covariance": n_pairs,
            "avg_pauli_terms_per_observable": total_terms / max(k, 1),
        }
    results["resource_scaling"] = resource_by_k
    print(json.dumps(resource_by_k, indent=2))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "quantum_circuit_experiment.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/quantum_circuit_experiment.json")


if __name__ == "__main__":
    main()
