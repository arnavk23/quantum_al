"""quantum_al: correlation-aware active learning for materials discovery.

A toolkit for building, analysing and rigorously benchmarking acquisition
functions that try to exploit correlation between several materials
properties when choosing which candidate to label next.

Modules
-------
acquisition
    Multi-target criteria from Bayesian optimal experimental design (joint
    and marginal expected information gain, total correlation, whitened
    trace / max-eigenvalue) and greedy batch EIG with a (1 - 1/e)
    guarantee.
joint_eig
    The original joint-EIG selector and its correlation-blind ablation.
diagnostics
    Why two criteria do or do not select different batches.
loop, stats
    The shared pool-based active-learning loop and paired statistics.
synthetic
    Multi-target problems with a tunable cross-task correlation.
baselines
    Nine classical single-target acquisition strategies.
operator, operator_v2, operator_v3, operator_sparse, circuit
    The covariance-aware quantum-inspired formalism, its variants, and its
    Qiskit circuit realization (``circuit`` needs the ``[circuit]`` extra).
data_utils, fetch_data
    Real Materials Project data (``fetch_data`` needs the ``[data]`` extra).
"""

__version__ = "0.2.0.dev0"

from quantum_al.acquisition import (  # noqa: E402
    CRITERIA,
    EnsembleCriterionSelector,
    GreedyBatchEIGSelector,
    batch_information_gain,
    expected_information_gain,
    greedy_batch_eig,
    marginal_information_gain_sum,
    total_correlation,
)
from quantum_al.baselines import get_all_baselines  # noqa: E402
from quantum_al.joint_eig import JointEIGSelector, MarginalSumSelector  # noqa: E402
from quantum_al.loop import LearningCurve, run_active_learning  # noqa: E402
from quantum_al.synthetic import make_correlated_tasks  # noqa: E402

__all__ = [
    "__version__",
    "CRITERIA",
    "EnsembleCriterionSelector",
    "GreedyBatchEIGSelector",
    "JointEIGSelector",
    "MarginalSumSelector",
    "LearningCurve",
    "batch_information_gain",
    "expected_information_gain",
    "get_all_baselines",
    "greedy_batch_eig",
    "make_correlated_tasks",
    "marginal_information_gain_sum",
    "run_active_learning",
    "total_correlation",
]
