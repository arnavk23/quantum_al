"""Benchmark suite on data/*.json (see quantum_al.fetch_data):

  primary      quantum vs 9 baselines, 5 tasks, 5 trials, 8 iterations
  stats        paired t-tests + Holm-Bonferroni + Shapiro-Wilk, band_gap/formation_energy
  ablation     full model vs 4 ablated variants, band_gap
  sensitivity  K=2,3,6 + best/worst-of-N K=3 combinations
  runtime      wall-clock time + peak memory, all 10 methods

Usage: python benchmarks/run_primary_benchmark.py --stage {primary,stats,ablation,sensitivity,runtime,all}
"""
import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
sys.path.insert(0, SCRIPT_DIR)

from quantum_al.baselines import BASELINES  # noqa: E402
from quantum_al.data_utils import FEATURE_COLUMNS, REGRESSION_TASKS, load_task, standardize  # noqa: E402
from quantum_al.quantum_inspired import QuantumSelector, default_feature_groups  # noqa: E402
from quantum_al.stats import holm_bonferroni  # noqa: E402

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Protocol constants (paper Table III protocol)
# ----------------------------------------------------------------------
N0 = 50            # initial labeled samples
T_ITERS = 8         # number of AL query rounds
BATCH_SIZE = 15      # b
N_TRIALS = 5         # trial_seeds 0..4
TEST_FRACTION = 0.3
TRIAL_SEEDS = list(range(N_TRIALS))
PRIMARY_TASKS_FOR_STATS = ["band_gap", "formation_energy"]


def make_quantum_selector(d, **kwargs):
    return QuantumSelector(d, **kwargs)


# ----------------------------------------------------------------------
# Core AL loop
# ----------------------------------------------------------------------
def run_al_trial(method_factory, X_pool, y_pool, X_test, y_test, trial_seed,
                  method_seed_offset=0, n0=N0, T=T_ITERS, b=BATCH_SIZE,
                  track_selection_time=False):
    """Run one active-learning trial for one method on one task.

    method_factory: zero-arg callable returning a fresh selector object
        with .select_next_experiments(X_candidates, X_train, y_train, n_select)
    Returns a dict with per-iteration r2/mae/n_labeled lists (iteration 0
    is the initial n0-sample model, before any AL queries).
    """
    rng = np.random.RandomState(trial_seed)
    n_pool = X_pool.shape[0]
    perm = rng.permutation(n_pool)
    labeled_idx = perm[:n0].tolist()
    remaining = perm[n0:].tolist()

    # CoreSet/Random Sampling use global np.random state, not a RandomState arg.
    np.random.seed(trial_seed * 10007 + method_seed_offset)

    method = method_factory()

    iterations, r2s, maes, n_labeled_list, sel_times = [], [], [], [], []
    error = None
    try:
        for it in range(T + 1):
            X_train = X_pool[labeled_idx]
            y_train = y_pool[labeled_idx]
            rf = RandomForestRegressor(n_estimators=100, random_state=trial_seed)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            r2 = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)

            iterations.append(it)
            r2s.append(float(r2))
            maes.append(float(mae))
            n_labeled_list.append(len(labeled_idx))

            if it == T or len(remaining) == 0:
                break

            X_candidates = X_pool[remaining]
            t0 = time.time()
            sel_idx, scores, info = method.select_next_experiments(
                X_candidates, X_train, y_train, n_select=min(b, len(remaining))
            )
            sel_times.append(time.time() - t0)

            sel_idx = np.asarray(sel_idx).astype(int)
            chosen_global = [remaining[i] for i in sel_idx]
            labeled_idx.extend(chosen_global)
            remaining = [i for i in remaining if i not in set(chosen_global)]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    out = {
        "trial_seed": trial_seed,
        "iteration": iterations,
        "r2": r2s,
        "mae": maes,
        "n_labeled": n_labeled_list,
        "error": error,
    }
    if track_selection_time:
        out["selection_times_sec"] = sel_times
    return out


def prepare_task_split(task_name, trial_seed):
    X, y, meta = load_task(task_name)
    X_pool_raw, X_test_raw, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=trial_seed
    )
    X_pool, X_test = standardize(X_pool_raw, X_test_raw)
    return X_pool, y_pool, X_test, y_test


def all_method_factories(d, quantum_seed=0):
    """Returns dict name -> zero-arg factory for quantum + 9 baselines."""
    factories = {"Quantum-Enhanced": lambda: make_quantum_selector(d, seed=quantum_seed)}
    factories.update({cls.name: cls for cls in BASELINES})
    return factories


METHOD_SEED_OFFSETS = {cls.name: i + 1 for i, cls in enumerate(BASELINES)}
METHOD_SEED_OFFSETS["Quantum-Enhanced"] = 0


# ----------------------------------------------------------------------
# Stage 1: Primary benchmark (Table III)
# ----------------------------------------------------------------------
def run_primary_benchmark():
    print("=" * 70)
    print("STAGE: primary benchmark (Table III)")
    print("=" * 70)
    full = {
        "config": {
            "n0": N0, "T": T_ITERS, "batch_size": BATCH_SIZE,
            "n_trials": N_TRIALS, "trial_seeds": TRIAL_SEEDS,
            "test_fraction": TEST_FRACTION,
            "predictor": "RandomForestRegressor(n_estimators=100, random_state=trial_seed)",
            "n_features": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
        },
        "tasks": {},
    }
    summary = {}

    for task in REGRESSION_TASKS:
        print(f"\n--- task: {task} ---")
        task_out = {"methods": {}}
        summary[task] = {}
        d = len(FEATURE_COLUMNS)

        for trial in TRIAL_SEEDS:
            X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)
            factories = all_method_factories(d)
            if trial == 0:
                task_out["n_pool"] = int(X_pool.shape[0])
                task_out["n_test"] = int(X_test.shape[0])
                task_out["n_features"] = int(X_pool.shape[1])

            for name, factory in factories.items():
                t0 = time.time()
                result = run_al_trial(
                    factory, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                    method_seed_offset=METHOD_SEED_OFFSETS[name],
                )
                dt = time.time() - t0
                status = "ERROR" if result["error"] else "ok"
                final_r2 = result["r2"][-1] if result["r2"] else float("nan")
                print(f"  [{task}] trial={trial} {name:22s} final R2={final_r2:.4f} "
                      f"({dt:.1f}s) {status}")
                task_out["methods"].setdefault(name, {"trials": []})
                task_out["methods"][name]["trials"].append(result)

        for name in task_out["methods"]:
            trials = task_out["methods"][name]["trials"]
            finals_r2 = [t["r2"][-1] for t in trials if not t["error"] and t["r2"]]
            finals_mae = [t["mae"][-1] for t in trials if not t["error"] and t["mae"]]
            n_errors = sum(1 for t in trials if t["error"])
            summary[task][name] = {
                "n_trials_ok": len(finals_r2),
                "n_trials_error": n_errors,
                "final_r2_mean": float(np.mean(finals_r2)) if finals_r2 else None,
                "final_r2_std": float(np.std(finals_r2)) if finals_r2 else None,
                "final_mae_mean": float(np.mean(finals_mae)) if finals_mae else None,
                "final_mae_std": float(np.std(finals_mae)) if finals_mae else None,
                "final_r2_per_trial": finals_r2,
                "final_mae_per_trial": finals_mae,
            }

        full["tasks"][task] = task_out

    with open(os.path.join(RESULTS_DIR, "primary_benchmark.json"), "w") as f:
        json.dump(full, f)
    with open(os.path.join(RESULTS_DIR, "primary_benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/primary_benchmark.json and results/primary_benchmark_summary.json")
    return full, summary


# ----------------------------------------------------------------------
# Stage 2: Statistical testing (Table IV)
# ----------------------------------------------------------------------
def run_statistical_tests(primary_summary_full=None):
    print("=" * 70)
    print("STAGE: statistical tests (Table IV)")
    print("=" * 70)
    primary_path = os.path.join(RESULTS_DIR, "primary_benchmark.json")
    if primary_summary_full is None:
        with open(primary_path) as f:
            primary_summary_full = json.load(f)

    out = {}
    for task in PRIMARY_TASKS_FOR_STATS:
        task_data = primary_summary_full["tasks"][task]
        quantum_trials = task_data["methods"]["Quantum-Enhanced"]["trials"]
        quantum_r2 = [t["r2"][-1] for t in quantum_trials if not t["error"]]

        task_out = {"quantum_final_r2": quantum_r2, "comparisons": {}}
        baseline_names = [cls.name for cls in BASELINES]
        raw_pvals = []
        names_in_order = []

        for name in baseline_names:
            base_trials = task_data["methods"][name]["trials"]
            base_r2 = [t["r2"][-1] for t in base_trials if not t["error"]]
            n = min(len(quantum_r2), len(base_r2))
            if n < 3:
                task_out["comparisons"][name] = {
                    "error": f"insufficient paired trials (n={n}) for a t-test"
                }
                continue
            q = np.array(quantum_r2[:n])
            b = np.array(base_r2[:n])
            diff = q - b
            t_stat, p_val = stats.ttest_rel(q, b)
            try:
                if n >= 3:
                    shapiro_stat, shapiro_p = stats.shapiro(diff)
                else:
                    shapiro_stat, shapiro_p = (None, None)
            except Exception:
                shapiro_stat, shapiro_p = (None, None)

            task_out["comparisons"][name] = {
                "n_paired": n,
                "quantum_mean_r2": float(np.mean(q)),
                "baseline_mean_r2": float(np.mean(b)),
                "mean_diff": float(np.mean(diff)),
                "t_statistic": float(t_stat),
                "p_value_raw": float(p_val),
                "shapiro_stat": float(shapiro_stat) if shapiro_stat is not None else None,
                "shapiro_p": float(shapiro_p) if shapiro_p is not None else None,
                "quantum_wins": bool(np.mean(diff) > 0),
            }
            raw_pvals.append(p_val)
            names_in_order.append(name)

        adj, reject = holm_bonferroni(raw_pvals, alpha=0.05)
        for name, p_adj, rej in zip(names_in_order, adj, reject):
            task_out["comparisons"][name]["p_value_holm_bonferroni"] = float(p_adj)
            task_out["comparisons"][name]["significant_at_0.05_after_correction"] = bool(rej)

        out[task] = task_out

    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/statistical_tests.json")
    return out


# ----------------------------------------------------------------------
# Stage 3: Ablation study (Table V) -- band_gap only
# ----------------------------------------------------------------------
def run_ablation():
    print("=" * 70)
    print("STAGE: ablation study (Table V) -- band_gap")
    print("=" * 70)
    task = "band_gap"
    d = len(FEATURE_COLUMNS)
    groups_full = default_feature_groups(d)
    single_group = {"structural": groups_full["structural"]}

    variants = {
        "full_model": dict(feature_groups=None, use_covariance=True,
                            commuting_only=False, real_only_coeff=False),
        "no_covariance": dict(feature_groups=None, use_covariance=False,
                               commuting_only=False, real_only_coeff=False),
        "commuting_only": dict(feature_groups=None, use_covariance=True,
                                commuting_only=True, real_only_coeff=False),
        "real_only_coeff": dict(feature_groups=None, use_covariance=True,
                                 commuting_only=False, real_only_coeff=True),
        "single_observable_K1": dict(feature_groups=single_group, use_covariance=True,
                                      commuting_only=False, real_only_coeff=False),
    }

    out = {"task": task, "n_trials": N_TRIALS, "variants": {}}
    for vname, kwargs in variants.items():
        print(f"\n--- variant: {vname} ---")
        trials_out = []
        for trial in TRIAL_SEEDS:
            X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)

            def factory(kw=kwargs, d_=X_pool.shape[1]):
                return make_quantum_selector(d_, seed=0, **kw)

            result = run_al_trial(factory, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                                   method_seed_offset=0)
            trials_out.append(result)
            final_r2 = result["r2"][-1] if result["r2"] else float("nan")
            print(f"  trial={trial} final R2={final_r2:.4f}")

        finals = [t["r2"][-1] for t in trials_out if not t["error"]]
        out["variants"][vname] = {
            "trials": trials_out,
            "final_r2_mean": float(np.mean(finals)) if finals else None,
            "final_r2_std": float(np.std(finals)) if finals else None,
        }

    full_mean = out["variants"]["full_model"]["final_r2_mean"]
    for vdata in out["variants"].values():
        if vdata["final_r2_mean"] is not None and full_mean is not None:
            vdata["delta_vs_full"] = vdata["final_r2_mean"] - full_mean
        else:
            vdata["delta_vs_full"] = None

    with open(os.path.join(RESULTS_DIR, "ablation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/ablation.json")
    return out


# ----------------------------------------------------------------------
# Stage 4: Observable sensitivity -- band_gap only
# ----------------------------------------------------------------------
def default_feature_groups_k(d, k):
    """Generalization of quantum_inspired.default_feature_groups to
    arbitrary K: split [0, d) into K overlapping contiguous blocks (each
    block overlaps its neighbor by 1 index at each boundary, same
    construction principle as the manuscript's K=3 default)."""
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


def named_k3_combinations(d):
    """4 different 3-observable-group index splits, for best-vs-worst comparison."""
    combos = {}
    combos["default_thirds"] = default_feature_groups(d)
    g = default_feature_groups(d)
    names = list(g.keys())
    vals = list(g.values())
    combos["reversed_assignment"] = {names[0]: vals[2], names[1]: vals[1], names[2]: vals[0]}
    combos["narrow_first_wide_rest"] = {
        "structural": list(range(0, 3)),
        "electronic": list(range(2, d - 2)),
        "thermodynamic": list(range(d - 3, d)),
    }
    third = d // 3
    combos["disjoint_thirds"] = {
        "structural": list(range(0, third)),
        "electronic": list(range(third, 2 * third)),
        "thermodynamic": list(range(2 * third, d)),
    }
    return combos


def run_observable_sensitivity():
    print("=" * 70)
    print("STAGE: observable sensitivity -- band_gap")
    print("=" * 70)
    task = "band_gap"
    out = {"task": task, "n_trials": N_TRIALS, "by_K": {}, "by_combination": {}}

    for K in [2, 3, 6]:
        print(f"\n--- K={K} ---")
        trials_out = []
        for trial in TRIAL_SEEDS:
            X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)
            d = X_pool.shape[1]
            groups = default_feature_groups(d) if K == 3 else default_feature_groups_k(d, K)

            def factory(g=groups, d_=d):
                return make_quantum_selector(d_, seed=0, feature_groups=g)

            result = run_al_trial(factory, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                                   method_seed_offset=0)
            trials_out.append(result)
            final_r2 = result["r2"][-1] if result["r2"] else float("nan")
            print(f"  trial={trial} final R2={final_r2:.4f}")
        finals = [t["r2"][-1] for t in trials_out if not t["error"]]
        out["by_K"][str(K)] = {
            "feature_groups": groups,
            "trials": trials_out,
            "final_r2_mean": float(np.mean(finals)) if finals else None,
            "final_r2_std": float(np.std(finals)) if finals else None,
        }

    combos = named_k3_combinations(len(FEATURE_COLUMNS))
    for cname, groups in combos.items():
        print(f"\n--- combination: {cname} ---")
        trials_out = []
        for trial in TRIAL_SEEDS:
            X_pool, y_pool, X_test, y_test = prepare_task_split(task, trial)
            d = X_pool.shape[1]

            def factory(g=groups, d_=d):
                return make_quantum_selector(d_, seed=0, feature_groups=g)

            result = run_al_trial(factory, X_pool, y_pool, X_test, y_test, trial_seed=trial,
                                   method_seed_offset=0)
            trials_out.append(result)
        finals = [t["r2"][-1] for t in trials_out if not t["error"]]
        out["by_combination"][cname] = {
            "feature_groups": groups,
            "trials": trials_out,
            "final_r2_mean": float(np.mean(finals)) if finals else None,
            "final_r2_std": float(np.std(finals)) if finals else None,
        }

    combo_means = {k: v["final_r2_mean"] for k, v in out["by_combination"].items()
                   if v["final_r2_mean"] is not None}
    if combo_means:
        best = max(combo_means, key=combo_means.get)
        worst = min(combo_means, key=combo_means.get)
        out["best_combination"] = {"name": best, "final_r2_mean": combo_means[best]}
        out["worst_combination"] = {"name": worst, "final_r2_mean": combo_means[worst]}
        out["combination_spread"] = combo_means[best] - combo_means[worst]

    with open(os.path.join(RESULTS_DIR, "observable_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/observable_sensitivity.json")
    return out


# ----------------------------------------------------------------------
# Stage 5: Runtime & memory benchmark (Table VI)
# ----------------------------------------------------------------------
def run_runtime_memory():
    print("=" * 70)
    print("STAGE: runtime & memory benchmark (Table VI)")
    print("=" * 70)
    task = "band_gap"
    X, y, meta = load_task(task)
    X_pool_raw, X_test_raw, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=0
    )
    X_pool, X_test = standardize(X_pool_raw, X_test_raw)
    d = X_pool.shape[1]

    rng = np.random.RandomState(0)
    n_pool = X_pool.shape[0]
    perm = rng.permutation(n_pool)
    labeled_idx = perm[:N0]
    pool_idx = perm[N0:]
    X_train = X_pool[labeled_idx]
    y_train = y_pool[labeled_idx]
    X_candidates = X_pool[pool_idx]
    n_pool_candidates = X_candidates.shape[0]

    factories = all_method_factories(d)
    n_repeats = 3
    out = {
        "task": task,
        "n_repeats": n_repeats,
        "n_train_labeled": int(X_train.shape[0]),
        "n_pool_candidates_real": int(n_pool_candidates),
        "note": (
            "n_pool_candidates_real is the actual achievable pool size from the "
            "real Materials Project data after the 70/30 split and removing the "
            "n0=50 initial labeled set (no padding/duplication of rows). An "
            "additional bootstrap-resampled extrapolation point is reported "
            "separately below and is explicitly labeled as such -- it is NOT a "
            "second real measurement."
        ),
        "methods": {},
    }

    def time_one_call(factory):
        method = factory()
        tracemalloc.start()
        t0 = time.time()
        sel_idx, scores, info = method.select_next_experiments(
            X_candidates, X_train, y_train, n_select=BATCH_SIZE
        )
        elapsed = time.time() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return elapsed, peak / (1024 ** 2)  # MB

    for name, factory in factories.items():
        print(f"  timing {name} ...")
        np.random.seed(0)
        times, peaks = [], []
        errs = []
        for _ in range(n_repeats):
            try:
                t, m = time_one_call(factory)
                times.append(t)
                peaks.append(m)
            except Exception as e:
                errs.append(f"{type(e).__name__}: {e}")
        entry = {
            "runs_ok": len(times),
            "runs_error": len(errs),
            "errors": errs,
            "wall_time_sec_mean": float(np.mean(times)) if times else None,
            "wall_time_sec_std": float(np.std(times)) if times else None,
            "peak_memory_MB_mean": float(np.mean(peaks)) if peaks else None,
            "peak_memory_MB_std": float(np.std(peaks)) if peaks else None,
            "raw_times_sec": times,
            "raw_peak_memory_MB": peaks,
        }
        out["methods"][name] = entry
        print(f"    {name}: {entry['wall_time_sec_mean']} s, {entry['peak_memory_MB_mean']} MB")

    # Bootstrap-resampled extrapolation point at a larger N, explicitly labeled.
    boot_rng = np.random.RandomState(1)
    N_BOOT = 2000
    boot_idx = boot_rng.choice(n_pool_candidates, size=N_BOOT, replace=True)
    X_boot = X_candidates[boot_idx]
    out["bootstrap_extrapolation"] = {
        "warning": "EXTRAPOLATION: candidates resampled with replacement from the "
                   "real pool to reach N=2000; not a second independent real "
                   "measurement, reported only to show scaling trend.",
        "n_pool_candidates": N_BOOT,
        "methods": {},
    }
    for name, factory in factories.items():
        np.random.seed(0)
        try:
            method = factory()
            tracemalloc.start()
            t0 = time.time()
            method.select_next_experiments(X_boot, X_train, y_train, n_select=BATCH_SIZE)
            elapsed = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            out["bootstrap_extrapolation"]["methods"][name] = {
                "wall_time_sec": elapsed, "peak_memory_MB": peak / (1024 ** 2),
            }
        except Exception as e:
            out["bootstrap_extrapolation"]["methods"][name] = {"error": f"{type(e).__name__}: {e}"}

    with open(os.path.join(RESULTS_DIR, "runtime_memory.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results/runtime_memory.json")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                         choices=["primary", "stats", "ablation", "sensitivity", "runtime", "all"])
    args = parser.parse_args()

    if args.stage in ("primary", "all"):
        run_primary_benchmark()
    if args.stage in ("stats", "all"):
        run_statistical_tests()
    if args.stage in ("ablation", "all"):
        run_ablation()
    if args.stage in ("sensitivity", "all"):
        run_observable_sensitivity()
    if args.stage in ("runtime", "all"):
        run_runtime_memory()


if __name__ == "__main__":
    main()
