---
title: 'quantum_al: correlation-aware active learning for materials discovery'
tags:
  - Python
  - active learning
  - Bayesian experimental design
  - materials discovery
  - materials informatics
  - uncertainty quantification
  - quantum-inspired algorithms
authors:
  - name: Arnav Kapoor
    orcid: 0009-0007-9818-7908
    affiliation: '1'
affiliations:
  - name: Indian Institute of Science Education and Research Bhopal, India
    index: 1
date: 2 October 2026
bibliography: paper.bib
---

# Summary

A single density-functional-theory (DFT) calculation returns many properties
of a material at once (band gap, formation energy, magnetic moment), all
derived from the same electronic structure and therefore correlated. Active
learning chooses which material to calculate or synthesize next under a fixed
budget [@lookman2019active]. Whether that choice should account for the
correlation between properties, and when doing so can help, is an open
question. `quantum_al` is a Python package for answering it rigorously. It
provides (1) multi-property acquisition criteria derived from Bayesian optimal
experimental design [@lindley1956measure; @chaloner1995bayesian]: the joint
expected information gain (EIG), its correlation-blind counterpart, the exact
total-correlation term [@watanabe1960information] that separates them,
unit-invariant A- and E-optimal analogues, and a greedy batch EIG with a
$(1 - 1/e)$ optimality guarantee [@nemhauser1978analysis; @krause2005near];
(2) diagnostics that explain why two criteria do or do not lead to different
outcomes; (3) a synthetic testbed with tunable cross-property correlation and
noise; (4) a benchmark harness on real Materials Project data
[@jain2013materials] with nine classical baselines and paired statistics with
multiple-comparison correction [@holm1979simple]; and (5) a quantum-inspired
covariance formalism with a Qiskit [@javadi2024qiskit] circuit realization,
kept as an independently evaluated alternative construction.

# Statement of need

Active learning is now a standard way to reduce the cost of materials
discovery [@lookman2019active; @vandermause2020fly;
@pyzerknapp2022accelerating], and multi-task models that share information
across properties are well established [@zhang2022survey]. Proposals to make
the *acquisition* step itself correlation-aware, including quantum and
quantum-inspired constructions [@biamonte2017quantum; @schuld2021machine],
are harder to assess. Evaluating one honestly requires the criterion and its
correlation-blind ablation behind the same interface, real multi-property data
in which each query reveals all properties, strong classical baselines,
paired trials with corrected statistics, and a way to tell whether a null
result means "the idea does not work" or "the experiment could not detect
it". `quantum_al` packages all of these.

Its central contribution is the link between theory and diagnostics. The
documentation proves that the joint EIG equals the sum of per-property EIGs
minus a total-correlation term, that this term is non-negative (Hadamard's
inequality), and that it is *second order* in the ratio of epistemic to
aleatoric variance while the marginal terms are first order. The joint and
correlation-blind criteria can therefore rank two candidates differently only
when their marginal scores are nearly tied. `quantum_al.diagnostics` measures
exactly these quantities on a candidate pool (the overlap of the two criteria's
top batches, the spread of the correlation term relative to the marginal
scores, the median signal-to-noise ratio), which turns a null benchmark result
into a mechanistic explanation. The package serves materials-informatics
researchers comparing acquisition strategies, including their own, on real
data; researchers in Bayesian experimental design who want a tested,
documented multi-output EIG implementation; and quantum-computing researchers
who want a worked example of taking a quantum-inspired construction to a
verified circuit.

# State of the field

General-purpose active-learning libraries such as modAL [@danka2018modal] and
ALiPy [@tang2019alipy] provide many classical single-target strategies behind
a scikit-learn-compatible interface [@pedregosa2011scikit], but they include
no multi-property acquisition criteria and no materials data. BoTorch
[@balandat2020botorch] implements Monte-Carlo acquisition functions for
Bayesian optimization, including multi-output models, but it targets
optimization rather than learning a predictive model over a pool, and it is
built around Gaussian-process surrogates rather than the tree ensembles common
in materials informatics. Materials benchmarks such as Matbench
[@dunn2020matbench], the sequential-learning benchmark of
@rohr2020benchmarking and Olympus [@hase2021olympus] standardize datasets and
protocols but do not provide or analyze multi-property acquisition criteria.
BatchBALD [@kirsch2019batchbald] and BALD [@houlsby2011bayesian] are the
closest information-theoretic methods, for classification. `quantum_al`
contributes the Gaussian, multi-output, ensemble-based analogue for
regression, with proofs of its properties, the diagnostics that follow from
them, and a statistically careful real-data benchmark, rather than
reimplementing general active-learning infrastructure that these packages
already provide.

# Software design

Every acquisition strategy implements one method,
`select_next_experiments(X_candidates, X_train, Y_train, n_select)`, and is
run by the shared pool-based loop `quantum_al.loop.run_active_learning`, so a
new criterion or baseline never requires changes to the benchmark harness.
`quantum_al.acquisition` computes all criteria from two quantities that a
random-forest ensemble [@breiman2001random] supplies for each candidate: the
$K \times K$ covariance of per-tree predictions and the per-property
out-of-bag residual variance. Batch selection uses the cross-candidate
covariance through Sylvester's determinant identity, at a cost independent of
the batch size. The test suite checks each criterion against its own
mathematical properties: the decomposition identity, the sandwich bound
$\max_k \mathrm{EIG}_k \le \mathrm{EIG} \le \sum_k \mathrm{EIG}_k$, exact
invariance to changing units, the tight two-property bound on the correlation
term, and monotonicity, submodularity and the $(1-1/e)$ bound of the greedy
batch by brute-force enumeration on small pools. `quantum_al.stats` reports
paired $t$ and Wilcoxon tests, bootstrap confidence intervals, effect sizes
and Holm correction. `quantum_al.circuit` asserts agreement between the Qiskit
circuit and its classical closed form before reporting any resource estimate.
Core dependencies are NumPy [@harris2020array], SciPy [@virtanen2020scipy]
and scikit-learn; Qiskit and pymatgen [@ong2013pymatgen] are optional extras.

One design rule applies throughout: the benchmark scripts raise an error
instead of substituting illustrative numbers when a computation fails, and
every number in the documentation is written to `results/` by a script in
`benchmarks/`. The main trade-off is the narrow selector interface, a scored
ranking over a candidate pool given a labeled set. Multi-fidelity and
cost-aware protocols are out of scope for now.

A minimal example, which needs no data download:

```python
from sklearn.model_selection import train_test_split
from quantum_al import EnsembleCriterionSelector, make_correlated_tasks, run_active_learning
from quantum_al.data_utils import standardize

X, Y, info = make_correlated_tasks(n_samples=300, correlation=0.9, noise=0.3, seed=0)
X_pool, X_test, Y_pool, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
X_pool, X_test = standardize(X_pool, X_test)

for criterion in ["eig", "marginal_eig"]:
    selector = EnsembleCriterionSelector(criterion, n_estimators=50, seed=0)
    curve = run_active_learning(selector, X_pool, Y_pool, X_test, Y_test,
                                n_initial=20, n_rounds=5, batch_size=8, seed=0)
    print(criterion, curve.aulc())
```

# Research impact statement

The author has used `quantum_al` for two empirical studies, reported in a
manuscript in preparation [@kapoor2026]. On four real Materials Project
property pairs, the joint EIG never significantly outperformed its
correlation-blind ablation, and it beat random selection significantly on one
pair. The quantum-inspired formalism did not outperform nine classical
baselines on five real regression tasks, and ablations attributed a later
improvement entirely to its use of ensemble disagreement rather than to the
quantum-specific components. The theory in the package predicts this kind
of null result: the correlation term is second order and was small on every
real pair (a mean of 0.004 to 0.021 nats per candidate), so it can reorder
only candidates whose marginal scores are already nearly tied.
These results are negative, and the package is built so that
they are reproducible and checkable: every number is regenerated by a script
from stored inputs.

The software has not yet been adopted outside the author's own work. Its
near-term value is reuse. A researcher with a new correlation-aware criterion
can implement one method and test it against the same baselines, data,
statistics and diagnostics, and can use the synthetic testbed to find out in
advance whether their benchmark has the power to detect the effect they
expect.

# AI usage disclosure

Generative AI (Claude, Anthropic, several model versions across development
sessions in 2025 and 2026) was used extensively: to write and refactor most
of the Python code in `src/quantum_al/` and `benchmarks/`, to draft the test
suite and documentation, including the derivations in the theory page, and to
draft this paper and the accompanying manuscripts. AI assistance also
fact-checked an earlier version of this project by executing its code, which
showed that an earlier evaluation had reported numbers that did not
reproduce. The author directed the work throughout: framing the research
questions, choosing which experiments and fixes to pursue, requiring that
every reported result come from an actual run of the code, rejecting drafts
and results that did not meet this standard, and reviewing the final code,
tests, derivations, statistical methodology and reported findings. The author
takes full responsibility for the correctness, originality and reporting of
everything in this repository and the accompanying manuscripts.

# Acknowledgements

This work uses data from the Materials Project [@jain2013materials].

# References
