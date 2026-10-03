"""Diagnostics that explain *why* two acquisition criteria do or do not
lead to different outcomes.

Two criteria can only produce different learning curves if they select
different points. Since ``EIG = sum_k EIG_k - TC`` exactly, the joint and
correlation-blind criteria rank candidates differently only to the extent
that the total-correlation term ``TC`` varies *across candidates*
relative to the spread of the marginal term; a large but nearly constant
``TC`` changes nothing.
"""
import numpy as np
from scipy import stats as _stats

from quantum_al.acquisition import (
    epistemic_aleatoric_ratio,
    expected_information_gain,
    marginal_information_gain_sum,
    top_k,
    total_correlation,
)

__all__ = ["topk_overlap", "rank_agreement", "criterion_disagreement"]


def topk_overlap(scores_a, scores_b, k):
    """Jaccard overlap of the top-``k`` sets under two score vectors
    (1 = identical batches, 0 = disjoint)."""
    a = set(top_k(np.asarray(scores_a), k).tolist())
    b = set(top_k(np.asarray(scores_b), k).tolist())
    return len(a & b) / len(a | b) if a | b else 1.0


def rank_agreement(scores_a, scores_b):
    """Spearman rank correlation between two score vectors."""
    return float(_stats.spearmanr(scores_a, scores_b).statistic)


def criterion_disagreement(Sigma, R, batch_size):
    """Summarize how differently the joint EIG and the correlation-blind
    marginal-EIG sum would select a batch from the same candidates.

    Parameters
    ----------
    Sigma : ndarray, shape (n, K, K)
        Epistemic predictive covariance per candidate.
    R : ndarray, shape (K,)
        Noise variances.
    batch_size : int
        Batch size used for the overlap.

    Returns
    -------
    dict
        ``topk_overlap`` and ``spearman`` between the two criteria;
        ``mean_tc`` and ``tc_spread_ratio = std(TC) / std(sum_k EIG_k)``
        (the quantity that controls whether rankings can differ); and
        ``median_snr``, the median epistemic-to-aleatoric ratio ``s_k``
        (small ratios make ``TC`` second-order, see
        :mod:`quantum_al.acquisition`).
    """
    joint = expected_information_gain(Sigma, R)
    marginal = marginal_information_gain_sum(Sigma, R)
    tc = total_correlation(Sigma, R)
    spread = marginal.std()
    return {
        "topk_overlap": topk_overlap(joint, marginal, batch_size),
        "spearman": rank_agreement(joint, marginal),
        "mean_tc": float(tc.mean()),
        "tc_spread_ratio": float(tc.std() / spread) if spread > 0 else float("nan"),
        "median_snr": float(np.median(epistemic_aleatoric_ratio(Sigma, R))),
    }
