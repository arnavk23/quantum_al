"""Tests the joint-EIG acquisition function (quantum_al.joint_eig) against
its own classical-limit ablation (marginal-sum) and random sampling, on
two real, jointly-labeled MP properties with a genuine, non-trivial
correlation (band_gap, formation_energy: r=-0.365, n=498 shared materials).

Unlike the original single-property comparisons in run_primary_benchmark.py,
this experiment's setting is: one AL query returns BOTH labels for the
chosen material (as a real DFT run would), so a genuinely joint acquisition
score is a meaningful, distinct question from per-property scalar
uncertainty. Protocol constants match run_primary_benchmark.py for
consistency (N0=50, T=8, batch=15, 5 trials, 30% held-out test set).

The default methods (Joint-EIG, Marginal-Sum, Random) reproduce the
published results/joint_eig_experiment*.json. ``--extended`` adds the
greedy batch-EIG selector and the unit-invariant trace (A-optimal-like)
and max-eigenvalue (E-optimal-like) criteria from quantum_al.acquisition;
every method is compared against Joint-EIG with Holm-Bonferroni
correction across the whole family.

Usage: python benchmarks/run_joint_eig_experiment.py [--extended]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)

from quantum_al.acquisition import EnsembleCriterionSelector, GreedyBatchEIGSelector  # noqa: E402
from quantum_al.data_utils import load_multi_task, standardize  # noqa: E402
from quantum_al.diagnostics import criterion_disagreement  # noqa: E402
from quantum_al.joint_eig import (  # noqa: E402
    JointEIGSelector, MarginalSumSelector, predictive_covariance, total_correlation_gap,
)
from quantum_al.loop import LearningCurve, run_active_learning  # noqa: E402
from quantum_al.stats import holm_bonferroni, paired_comparison  # noqa: E402

os.makedirs(RESULTS_DIR, exist_ok=True)

N0_DEFAULT = 50
T_ITERS_DEFAULT = 8
BATCH_SIZE_DEFAULT = 15
TEST_FRACTION = 0.3
N_ESTIMATORS = 200


class RandomSelector:
    name = "Random"

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        n_select = min(n_select, len(X_candidates))
        idx = np.random.permutation(len(X_candidates))[:n_select]
        return idx, np.zeros(len(X_candidates)), {}


def _track_joint_diagnostics(batch_size):
    """on_round hook: total-correlation gap and joint-vs-marginal selection
    disagreement, from the acquisition forest the selector already fit."""
    def hook(it, info, X_candidates):
        Sigma = predictive_covariance(info["forest"], X_candidates)
        out = criterion_disagreement(Sigma, info["R"], batch_size)
        out["mean_gap"] = float(np.mean(total_correlation_gap(Sigma, info["R"])))
        return out
    return hook


def run_al_trial(method_factory, X_pool, Y_pool, X_test, Y_test, trial_seed,
                  n0=N0_DEFAULT, t_iters=T_ITERS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT,
                  method_seed_offset=0, track_gap=False):
    # Random selection uses the global NumPy RNG; seed it per (trial, method).
    np.random.seed(trial_seed * 10007 + method_seed_offset)
    method = method_factory()
    error, curve = None, LearningCurve()
    try:
        curve = run_active_learning(
            method, X_pool, Y_pool, X_test, Y_test, n_initial=n0, n_rounds=t_iters,
            batch_size=batch_size, seed=trial_seed,
            on_round=_track_joint_diagnostics(batch_size) if track_gap else None,
        )
    except Exception as e:  # recorded and excluded from the summary, never imputed
        error = f"{type(e).__name__}: {e}"

    return {
        "trial_seed": trial_seed,
        "iteration": list(range(len(curve.r2))),
        "r2_joint": curve.r2,
        "r2_per_task": curve.r2_per_task,
        "n_labeled": curve.n_labeled,
        "aulc": curve.aulc() if len(curve.r2) > 1 else None,
        "mean_total_correlation_gap": [d["mean_gap"] for d in curve.diagnostics],
        "selection_diagnostics": curve.diagnostics,
        "error": error,
    }


EXTENDED_FACTORIES = {
    "Greedy-Batch-EIG": lambda: GreedyBatchEIGSelector(n_estimators=N_ESTIMATORS, seed=0),
    "Trace": lambda: EnsembleCriterionSelector("trace", n_estimators=N_ESTIMATORS, seed=0),
    "Max-Eigenvalue": lambda: EnsembleCriterionSelector("max_eigenvalue", n_estimators=N_ESTIMATORS, seed=0),
}


def main(tasks, n_trials, out_name, n0=N0_DEFAULT, t_iters=T_ITERS_DEFAULT, batch_size=BATCH_SIZE_DEFAULT,
         extended=False):
    trial_seeds = list(range(n_trials))
    print("=" * 70)
    print(f"Joint-EIG experiment: {tasks[0]} + {tasks[1]} (n_trials={n_trials}, "
          f"n0={n0}, T={t_iters}, batch={batch_size})")
    print("=" * 70)
    X, Y, meta = load_multi_task(tasks)
    print(f"n_materials={X.shape[0]} n_features={X.shape[1]} n_tasks={Y.shape[1]}")
    pair_corr = float(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1])
    print(f"raw label correlation r={pair_corr:.3f}")

    factories = {
        "Joint-EIG": lambda: JointEIGSelector(n_estimators=N_ESTIMATORS, seed=0),
        "Marginal-Sum": lambda: MarginalSumSelector(n_estimators=N_ESTIMATORS, seed=0),
        "Random": lambda: RandomSelector(),
    }
    seed_offsets = {"Joint-EIG": 0, "Marginal-Sum": 1, "Random": 2}
    if extended:
        factories.update(EXTENDED_FACTORIES)
        seed_offsets.update({name: 3 + i for i, name in enumerate(EXTENDED_FACTORIES)})

    out = {
        "config": {
            "tasks": tasks, "n0": n0, "T": t_iters, "batch_size": batch_size,
            "n_trials": n_trials, "test_fraction": TEST_FRACTION,
            "n_estimators": N_ESTIMATORS, "raw_label_correlation": pair_corr,
            "note": "one query returns labels for BOTH tasks (joint DFT-run setting)",
        },
        "methods": {},
    }

    for name, factory in factories.items():
        out["methods"][name] = {"trials": []}
        for trial in trial_seeds:
            X_pool_raw, X_test_raw, Y_pool, Y_test = train_test_split(
                X, Y, test_size=TEST_FRACTION, random_state=trial
            )
            X_pool, X_test = standardize(X_pool_raw, X_test_raw)
            t0 = time.time()
            result = run_al_trial(
                factory, X_pool, Y_pool, X_test, Y_test, trial_seed=trial,
                n0=n0, t_iters=t_iters, batch_size=batch_size,
                method_seed_offset=seed_offsets[name],
                track_gap=(name == "Joint-EIG"),
            )
            dt = time.time() - t0
            status = "ERROR" if result["error"] else "ok"
            final = result["r2_joint"][-1] if result["r2_joint"] else float("nan")
            print(f"  {name:14s} trial={trial} final joint-R2={final:.4f} ({dt:.1f}s) {status}")
            out["methods"][name]["trials"].append(result)

    summary = {}
    for name in out["methods"]:
        trials = out["methods"][name]["trials"]
        finals = [t["r2_joint"][-1] for t in trials if not t["error"]]
        finals_per_task = [t["r2_per_task"][-1] for t in trials if not t["error"]]
        finals_per_task = np.array(finals_per_task) if finals_per_task else np.zeros((0, len(tasks)))
        summary[name] = {
            "final_joint_r2_mean": float(np.mean(finals)) if finals else None,
            "final_joint_r2_std": float(np.std(finals)) if finals else None,
            "final_joint_r2_per_trial": finals,
            "final_per_task_r2_mean": finals_per_task.mean(axis=0).tolist() if len(finals_per_task) else None,
        }
    out["summary"] = summary

    # paired stats: Joint-EIG vs every other method, one Holm-Bonferroni family
    jeig = np.array(summary["Joint-EIG"]["final_joint_r2_per_trial"])
    comparisons = {}
    raw_pvals, names_in_order = [], []
    for name in [m for m in factories if m != "Joint-EIG"]:
        base = np.array(summary[name]["final_joint_r2_per_trial"])
        n = min(len(jeig), len(base))
        pc = paired_comparison(jeig[:n], base[:n])
        comparisons[name] = {
            "n_paired": n,
            "joint_eig_mean": pc["mean_a"],
            "baseline_mean": pc["mean_b"],
            "mean_diff": pc["mean_diff"],
            "ci95_mean_diff": pc["ci95_diff"],
            "effect_size_dz": pc["effect_size_dz"],
            "t_statistic": pc["t_statistic"],
            "p_value_raw": pc["p_value_t"],
            "p_value_wilcoxon": pc["p_value_wilcoxon"],
            "shapiro_p": pc["shapiro_p"],
            "joint_eig_wins": bool(pc["mean_diff"] > 0),
        }
        raw_pvals.append(pc["p_value_t"] if pc["p_value_t"] is not None else 1.0)
        names_in_order.append(name)

    adj, reject = holm_bonferroni(raw_pvals, alpha=0.05)
    for name, p_adj, rej in zip(names_in_order, adj, reject):
        comparisons[name]["p_value_holm_bonferroni"] = float(p_adj)
        comparisons[name]["significant_at_0.05_after_correction"] = bool(rej)
    out["comparisons"] = comparisons

    # evidence the theorem's gap term is real and non-trivial on this data
    jeig_trials = out["methods"]["Joint-EIG"]["trials"]
    all_gaps = [g for t in jeig_trials for g in t["mean_total_correlation_gap"]]
    out["total_correlation_gap_stats"] = {
        "mean": float(np.mean(all_gaps)) if all_gaps else None,
        "min": float(np.min(all_gaps)) if all_gaps else None,
        "max": float(np.max(all_gaps)) if all_gaps else None,
        "note": "mean over candidates per AL round; always >=0 (Hadamard); "
                ">0 confirms the two tasks' epistemic uncertainties are "
                "genuinely correlated across the ensemble, not just their labels",
    }
    diags = [d for t in jeig_trials for d in t["selection_diagnostics"]]
    out["selection_diagnostics_summary"] = {
        key: float(np.mean([d[key] for d in diags])) if diags else None
        for key in ("topk_overlap", "spearman", "tc_spread_ratio", "median_snr")
    }
    out["selection_diagnostics_summary"]["note"] = (
        "Joint-EIG vs Marginal-Sum on the same candidates and forest, averaged over "
        "rounds and trials: topk_overlap=1 means both criteria would pick the same batch"
    )

    with open(os.path.join(RESULTS_DIR, out_name), "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY (final joint R^2, mean of band_gap & formation_energy R^2)")
    print("=" * 70)
    for name, s in summary.items():
        print(f"  {name:14s} {s['final_joint_r2_mean']:.4f} +/- {s['final_joint_r2_std']:.4f}")
    print("\nComparisons vs Joint-EIG:")
    for name, c in comparisons.items():
        print(f"  vs {name:14s} mean_diff={c['mean_diff']:+.4f} p_raw={c['p_value_raw']:.4f} "
              f"p_holm={c['p_value_holm_bonferroni']:.4f} wins={c['joint_eig_wins']}")
    print(f"\nTotal-correlation gap (mean/min/max): "
          f"{out['total_correlation_gap_stats']['mean']:.4f} / "
          f"{out['total_correlation_gap_stats']['min']:.4f} / "
          f"{out['total_correlation_gap_stats']['max']:.4f}")
    sd = out["selection_diagnostics_summary"]
    if sd["topk_overlap"] is not None:
        print(f"Joint vs marginal batch overlap (Jaccard): {sd['topk_overlap']:.3f}, "
              f"rank correlation: {sd['spearman']:.3f}, median epistemic/aleatoric ratio: "
              f"{sd['median_snr']:.3f}")
    print(f"\nSaved results/{out_name}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs=2, default=["band_gap", "formation_energy"])
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--out", default="joint_eig_experiment.json")
    parser.add_argument("--n0", type=int, default=N0_DEFAULT)
    parser.add_argument("--t-iters", type=int, default=T_ITERS_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--extended", action="store_true",
                        help="also run greedy batch EIG and the trace / max-eigenvalue criteria")
    args = parser.parse_args()
    main(args.tasks, args.n_trials, args.out, n0=args.n0, t_iters=args.t_iters,
         batch_size=args.batch_size, extended=args.extended)
