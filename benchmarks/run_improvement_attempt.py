"""Tests two fixes to the quantum-inspired selector against the original, same protocol
as the primary benchmark."""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)

from run_primary_benchmark import TRIAL_SEEDS, prepare_task_split, run_al_trial  # noqa: E402

from quantum_al.data_utils import FEATURE_COLUMNS  # noqa: E402
from quantum_al.quantum_inspired import (  # noqa: E402
    ImportanceWeightedQuantumSelector,
    QuantumSelector,
    domain_feature_groups,
)

VARIANTS = {
    "original_default_thirds": lambda d: QuantumSelector(d, seed=0, name="Quantum-Original"),
    "domain_groups_only": lambda d: QuantumSelector(
        d, feature_groups=domain_feature_groups(), seed=0, name="Quantum-DomainGroups"),
    "domain_groups_plus_importance_weighting": lambda d: ImportanceWeightedQuantumSelector(d, seed=0),
}


def run_on_task(task):
    print(f"\n{'='*70}\ntask: {task}\n{'='*70}")
    d = len(FEATURE_COLUMNS)
    out = {}
    for vname, ctor in VARIANTS.items():
        print(f"\n--- variant: {vname} ---")
        trials_out = []
        for trial in TRIAL_SEEDS:
            X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)

            def factory(ctor=ctor, d_=d):
                return ctor(d_)

            result = run_al_trial(factory, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                                   method_seed_offset=0)
            trials_out.append(result)
            final_r2 = result["r2"][-1] if result["r2"] else float("nan")
            print(f"  trial={trial} final R2={final_r2:.4f} error={result['error']}")
        finals = [t["r2"][-1] for t in trials_out if not t["error"]]
        out[vname] = {
            "trials": trials_out,
            "final_r2_mean": float(np.mean(finals)) if finals else None,
            "final_r2_std": float(np.std(finals)) if finals else None,
        }
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Stage 1: quick feedback on band_gap only.
    results = {"band_gap": run_on_task("band_gap")}
    with open(os.path.join(RESULTS_DIR, "improvement_attempt.json"), "w") as f:
        json.dump(results, f)

    print("\n\n=== SUMMARY (band_gap) ===")
    for vname, v in results["band_gap"].items():
        print(f"  {vname:40s} R2={v['final_r2_mean']:.4f}+-{v['final_r2_std']:.4f}")


if __name__ == "__main__":
    main()
