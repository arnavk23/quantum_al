"""Pauli measurement cost of the quantum-inspired score U_total on hardware,
before and after measurement grouping (K=3 default observable bank on the
d=21 Materials Project feature space, 5 qubits). No data or API key needed.

Saves results/measurement_grouping.json (Figure 5 of the manuscript).

Usage: python benchmarks/run_measurement_grouping.py
"""
import json
import os

from quantum_al.circuit import measurement_grouping_report
from quantum_al.data_utils import FEATURE_COLUMNS
from quantum_al.operator import QuantumObservableBank, default_feature_groups

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    d = len(FEATURE_COLUMNS)
    bank = QuantumObservableBank(d, default_feature_groups(d), seed=0)
    report = measurement_grouping_report(bank.O)
    out = {"per_quantity": report["per_quantity"], "totals_K3": report["totals"]}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "measurement_grouping.json"), "w") as f:
        json.dump(out, f, indent=2)
    t = report["totals"]
    print(f"{report['n_qubits']} qubits: {t['raw']} Pauli terms -> {t['qwc_groups']} qubit-wise "
          f"commuting groups ({t['qwc_reduction_factor']:.1f}x) -> {t['full_commuting_groups']} "
          f"commuting groups ({t['full_reduction_factor']:.1f}x)")
    print("Saved results/measurement_grouping.json")


if __name__ == "__main__":
    main()
