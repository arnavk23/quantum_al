"""Statistics for comparing active-learning strategies across paired trials.

Trials of two strategies that share a seed share the same train/test split
and initial labeled set, so comparisons are paired. The functions here are
the ones used by every benchmark script in ``benchmarks/``.
"""
import numpy as np
from scipy import stats as _stats

__all__ = [
    "holm_bonferroni",
    "paired_comparison",
    "bootstrap_ci",
    "area_under_learning_curve",
]


def holm_bonferroni(pvals, alpha=0.05):
    """Holm's step-down correction for multiple comparisons (Holm, 1979).

    Parameters
    ----------
    pvals : sequence of float
        Raw p-values of the family of comparisons.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    adjusted : list of float
        Holm-adjusted p-values (monotone, capped at 1), in input order.
    reject : list of bool
        Whether each null hypothesis is rejected at ``alpha``.
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, pvals[idx] * (m - rank))
        adj[idx] = min(running_max, 1.0)
    return adj.tolist(), (adj < alpha).tolist()


def bootstrap_ci(values, n_boot=10000, level=0.95, seed=0):
    """Percentile bootstrap confidence interval for the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def paired_comparison(a, b, n_boot=10000, seed=0):
    """Paired comparison of per-trial scores ``a`` (method) vs ``b`` (reference).

    Returns a dict with the mean difference, a bootstrap 95% CI for it, the
    paired t-test, the Wilcoxon signed-rank test (a distribution-free
    check), Shapiro-Wilk normality of the differences, and the
    standardized effect size ``d_z = mean(diff) / sd(diff)``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length (paired trials)")
    diff = a - b
    n = len(diff)
    out = {
        "n_paired": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_diff": bootstrap_ci(diff, n_boot=n_boot, seed=seed) if n > 1 else (None, None),
    }
    sd = diff.std(ddof=1) if n > 1 else 0.0
    out["effect_size_dz"] = float(diff.mean() / sd) if sd > 0 else None
    if n > 1 and sd > 0:
        t_stat, p_t = _stats.ttest_rel(a, b)
        out["t_statistic"], out["p_value_t"] = float(t_stat), float(p_t)
        out["p_value_wilcoxon"] = float(_stats.wilcoxon(diff).pvalue)
    else:
        out["t_statistic"] = out["p_value_t"] = out["p_value_wilcoxon"] = None
    out["shapiro_p"] = float(_stats.shapiro(diff).pvalue) if n >= 3 and sd > 0 else None
    return out


def area_under_learning_curve(n_labeled, scores, normalize=True):
    """Trapezoidal area under a learning curve (score vs. labels used).

    With ``normalize=True`` the area is divided by the label range, giving
    the curve's average score, which is comparable across budgets. It
    summarizes the whole trajectory, not just the final round.
    """
    x = np.asarray(n_labeled, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.shape != y.shape or len(x) < 2:
        raise ValueError("need at least two (n_labeled, score) points of equal length")
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return area / float(x[-1] - x[0]) if normalize else area
