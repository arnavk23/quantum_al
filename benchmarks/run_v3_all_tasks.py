"""The residual-coupled variant (v3, TreeEnsembleQuantumSelector) and its
ablation on all 5 tasks, same protocol as the primary benchmark, plus
paired t-tests against each task's best baseline."""
import json
import os
import sys

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)  # only for sibling benchmarks/*.py scripts below,
# which are standalone (not part of the installed quantum_al package);
# quantum_al.* imports themselves work without this once pip-installed.

from run_primary_benchmark import TRIAL_SEEDS, prepare_task_split, run_al_trial  # noqa: E402

from quantum_al.data_utils import REGRESSION_TASKS  # noqa: E402
from quantum_al.quantum_inspired import TreeEnsembleQuantumSelector  # noqa: E402

VARIANTS = {
    "v3_full": lambda: TreeEnsembleQuantumSelector(seed=0, use_covariance=True, n_groups=3),
    "v3_no_covariance": lambda: TreeEnsembleQuantumSelector(seed=0, use_covariance=False, n_groups=3),
    "v3_single_group": lambda: TreeEnsembleQuantumSelector(seed=0, use_covariance=True, n_groups=1),
}


def run_variant(task, name, factory_fn):
    trials_out = []
    for trial in TRIAL_SEEDS:
        X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)
        result = run_al_trial(factory_fn, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                               method_seed_offset=0)
        trials_out.append(result)
        final_r2 = result["r2"][-1] if result["r2"] else float("nan")
        print(f"  [{task}/{name}] trial={trial} final R2={final_r2:.4f} error={result['error']}", flush=True)
    finals = [t["r2"][-1] for t in trials_out if not t["error"]]
    mean, std = (float(np.mean(finals)), float(np.std(finals))) if finals else (None, None)
    print(f"  [{task}/{name}] MEAN R2 = {mean} +- {std}", flush=True)
    return {"trials": trials_out, "final_r2_mean": mean, "final_r2_std": std,
            "final_r2_per_trial": finals}


def main():
    with open(os.path.join(RESULTS_DIR, "primary_benchmark_summary.json")) as f:
        primary_summary = json.load(f)

    out = {}
    for task in REGRESSION_TASKS:
        print(f"\n{'='*70}\ntask: {task}\n{'='*70}", flush=True)
        out[task] = {}
        for vname, ctor in VARIANTS.items():
            out[task][vname] = run_variant(task, vname, ctor)

        # Best baseline for this task (excluding our own quantum method)
        task_methods = primary_summary[task]
        baseline_only = {k: v for k, v in task_methods.items() if k != "Quantum-Enhanced"}
        best_name = max(baseline_only, key=lambda k: baseline_only[k]["final_r2_mean"])
        best_r2 = baseline_only[best_name]["final_r2_mean"]
        best_per_trial = baseline_only[best_name]["final_r2_per_trial"]

        v3_full_per_trial = out[task]["v3_full"]["final_r2_per_trial"]
        n = min(len(v3_full_per_trial), len(best_per_trial))
        if n >= 3:
            t_stat, p_val = stats.ttest_rel(v3_full_per_trial[:n], best_per_trial[:n])
        else:
            t_stat, p_val = (None, None)

        out[task]["comparison_vs_best_baseline"] = {
            "best_baseline_name": best_name,
            "best_baseline_r2": best_r2,
            "v3_full_r2": out[task]["v3_full"]["final_r2_mean"],
            "t_statistic": t_stat,
            "p_value_raw": p_val,
        }
        print(f"\n  [{task}] v3_full={out[task]['v3_full']['final_r2_mean']:.4f} vs "
              f"best baseline {best_name}={best_r2:.4f}  (t={t_stat}, p={p_val})", flush=True)

    with open(os.path.join(RESULTS_DIR, "v3_all_tasks.json"), "w") as f:
        json.dump(out, f)

    print("\n\n=== FINAL SUMMARY (all 5 tasks) ===")
    for task in REGRESSION_TASKS:
        c = out[task]["comparison_vs_best_baseline"]
        print(f"{task:22s} v3_full={c['v3_full_r2']:.4f}  "
              f"v3_no_cov={out[task]['v3_no_covariance']['final_r2_mean']:.4f}  "
              f"v3_single={out[task]['v3_single_group']['final_r2_mean']:.4f}  "
              f"best_baseline({c['best_baseline_name']})={c['best_baseline_r2']:.4f}  p={c['p_value_raw']}")


if __name__ == "__main__":
    main()
