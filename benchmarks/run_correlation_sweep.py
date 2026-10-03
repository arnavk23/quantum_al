"""Controlled study: when can correlation-aware (joint) acquisition beat
correlation-blind acquisition?

On real Materials Project pairs the joint EIG never significantly beat the
marginal-EIG sum (results/joint_eig_experiment*.json), but real data fixes
cross-property correlation, noise and sample size together. This script
varies them independently with quantum_al.synthetic.make_correlated_tasks:

* signal correlation rho between the two targets, from 0 (independent) to
  0.99 (almost the same function);
* noise level sigma, which sets the epistemic-to-aleatoric ratio that
  controls the size of the total-correlation term (quantum_al.acquisition,
  property 4).

For every setting it runs paired trials (same seed = same problem draw,
split and initial labeled set) of the joint EIG, the marginal-EIG sum, the
greedy batch EIG, the whitened trace and random selection, and records the
selection diagnostics of quantum_al.diagnostics (do the joint and marginal
criteria even pick different batches?). All numbers are synthetic by
design and are reported separately from the real-data results.

Saves results/correlation_sweep.json.

Usage: python benchmarks/run_correlation_sweep.py [--quick]
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

from quantum_al.acquisition import EnsembleCriterionSelector, GreedyBatchEIGSelector
from quantum_al.data_utils import standardize
from quantum_al.diagnostics import criterion_disagreement
from quantum_al.joint_eig import predictive_covariance
from quantum_al.loop import run_active_learning
from quantum_al.stats import holm_bonferroni, paired_comparison
from quantum_al.synthetic import make_correlated_tasks

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

FULL = dict(correlations=[0.0, 0.5, 0.9, 0.99], noises=[0.1, 0.5], n_trials=10,
            n_samples=600, n_features=8, n0=30, n_rounds=8, batch_size=10, n_estimators=100)
QUICK = dict(correlations=[0.0, 0.9], noises=[0.1], n_trials=2,
             n_samples=200, n_features=5, n0=20, n_rounds=2, batch_size=5, n_estimators=20)


class RandomSelector:
    name = "Random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def select_next_experiments(self, X_candidates, X_train, Y_train, n_select=10):
        n_select = min(n_select, len(X_candidates))
        return self.rng.permutation(len(X_candidates))[:n_select], np.zeros(len(X_candidates)), {}


def method_factories(n_estimators):
    return {
        "Joint-EIG": lambda s: EnsembleCriterionSelector("eig", n_estimators=n_estimators, seed=s),
        "Marginal-EIG": lambda s: EnsembleCriterionSelector("marginal_eig", n_estimators=n_estimators, seed=s),
        "Greedy-Batch-EIG": lambda s: GreedyBatchEIGSelector(n_estimators=n_estimators, seed=s),
        "Trace": lambda s: EnsembleCriterionSelector("trace", n_estimators=n_estimators, seed=s),
        "Random": lambda s: RandomSelector(s),
    }


def diagnostics_hook(batch_size):
    def hook(it, info, X_candidates):
        Sigma = predictive_covariance(info["forest"], X_candidates)
        return criterion_disagreement(Sigma, info["R"], batch_size)
    return hook


def run_setting(rho, noise, cfg):
    factories = method_factories(cfg["n_estimators"])
    finals = {m: [] for m in factories}
    aulcs = {m: [] for m in factories}
    diags = []
    for trial in range(cfg["n_trials"]):
        X, Y, info = make_correlated_tasks(
            n_samples=cfg["n_samples"], n_features=cfg["n_features"], n_tasks=2,
            correlation=rho, noise=noise, seed=1000 + trial,
        )
        X_pool, X_test, Y_pool, Y_test = train_test_split(X, Y, test_size=0.3, random_state=trial)
        X_pool, X_test = standardize(X_pool, X_test)
        for name, make in factories.items():
            curve = run_active_learning(
                make(trial), X_pool, Y_pool, X_test, Y_test, n_initial=cfg["n0"],
                n_rounds=cfg["n_rounds"], batch_size=cfg["batch_size"], seed=trial,
                on_round=diagnostics_hook(cfg["batch_size"]) if name == "Joint-EIG" else None,
            )
            finals[name].append(curve.final_r2)
            aulcs[name].append(curve.aulc())
            diags.extend(curve.diagnostics)
    return finals, aulcs, diags


def main(cfg, out_name):
    out = {"config": cfg, "settings": []}
    t_start = time.time()
    for noise in cfg["noises"]:
        for rho in cfg["correlations"]:
            t0 = time.time()
            finals, aulcs, diags = run_setting(rho, noise, cfg)
            setting = {
                "correlation": rho,
                "noise": noise,
                "expected_label_correlation": rho / (1 + noise ** 2),
                "final_r2": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_trial": v}
                             for m, v in finals.items()},
                "aulc": {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_trial": v}
                         for m, v in aulcs.items()},
                "joint_vs_marginal_diagnostics": {
                    key: float(np.mean([d[key] for d in diags]))
                    for key in ("topk_overlap", "spearman", "mean_tc", "tc_spread_ratio", "median_snr")
                },
                "comparisons": {
                    "Joint-EIG vs Marginal-EIG": paired_comparison(aulcs["Joint-EIG"], aulcs["Marginal-EIG"]),
                    "Greedy-Batch-EIG vs Joint-EIG": paired_comparison(aulcs["Greedy-Batch-EIG"], aulcs["Joint-EIG"]),
                    "Joint-EIG vs Random": paired_comparison(aulcs["Joint-EIG"], aulcs["Random"]),
                },
            }
            out["settings"].append(setting)
            d = setting["joint_vs_marginal_diagnostics"]
            c = setting["comparisons"]["Joint-EIG vs Marginal-EIG"]
            print(f"rho={rho:<5} sigma={noise:<4} overlap={d['topk_overlap']:.3f} "
                  f"spearman={d['spearman']:.4f} snr={d['median_snr']:.3f} | AULC joint-marginal "
                  f"{c['mean_diff']:+.4f} (p={c['p_value_t']}) | "
                  + " ".join(f"{m}={np.mean(aulcs[m]):.3f}" for m in aulcs)
                  + f" [{time.time() - t0:.0f}s]", flush=True)

    # Holm-Bonferroni within each comparison family, across all settings
    for family in out["settings"][0]["comparisons"]:
        pvals = [s["comparisons"][family]["p_value_t"] for s in out["settings"]]
        adj, reject = holm_bonferroni([1.0 if p is None else p for p in pvals])
        for s, p_adj, rej in zip(out["settings"], adj, reject):
            s["comparisons"][family]["p_value_holm_bonferroni"] = p_adj
            s["comparisons"][family]["significant_at_0.05_after_correction"] = rej
    out["metric_note"] = ("comparisons use the normalized area under the mean-R^2 learning "
                          "curve (AULC), paired by trial; Holm-Bonferroni within each family")
    out["runtime_sec"] = time.time() - t_start
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, out_name), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/{out_name} ({out['runtime_sec']:.0f}s)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    main(cfg, args.out or ("correlation_sweep_quick.json" if args.quick else "correlation_sweep.json"))
