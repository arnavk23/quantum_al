# Real-Data Results Summary

Every number below comes from an actual, reproducible script run against real
Materials Project data (`data/*.json`, via `quantum_al.fetch_data`)
using the operator formalism in `quantum_al/operator.py` (Eq. 1-6 of
the manuscript, self-tested against the classical-limit reduction). This
replaces the previous manuscript's tables, which did not reproduce from any
code in the repository.

Source files: `results/primary_benchmark_summary.json`, `results/statistical_tests.json`,
`results/ablation.json`, `results/observable_sensitivity.json`,
`results/runtime_memory.json`, `results/improvement_attempt.json`, and, for the
pre-rebuild secondary experiments, `results/legacy/` (see its README).

## Headline finding (updated after the v3 residual-coupled experiment)

**As originally specified, the quantum-inspired covariance formalism does not
outperform standard active-learning baselines on real materials data.** It
loses on 4 of 5 real regression tasks, and no comparison survives
Holm-Bonferroni correction. Two narrow improvement attempts (domain-informed
observable grouping; predictor-importance-weighted state encoding) did not
close the gap.

**A third, more direct fix — coupling the state encoding to the downstream
random forest's per-tree predictions (i.e. genuinely injecting ensemble
disagreement) — reaches statistical parity with the best baseline on all 5
real tasks** (`results/v3_all_tasks.json`; p>0.15 throughout, no significant
loss anywhere). This is a real, non-cherry-picked improvement over the
original formalism (p=0.049 vs. the original method on band gap).

**However, this is not evidence for the quantum-specific machinery.** An
ablation of the v3 variant (`results/v3_ablation.json`, `results/v3_all_tasks.json`)
shows the improvement is attributable entirely to exposure to ensemble
disagreement: dropping the covariance term or collapsing to a single
observable matches or beats the full covariance-aware, multi-observable,
complex-coupled construction on 4 of 5 tasks, and the full version is
markedly unstable on the noisiest task (dielectric constant: full
R²=−0.450±0.724, individual trials as low as −1.86, vs. single-observable
ablation R²=−0.160). The honest reading: what helps is becoming more like
RF-Uncertainty/QBC (measuring model disagreement), not the covariance/
non-commutativity/complex-coefficient apparatus the paper is built around,
which remains inert-to-harmful in every configuration tested.

## Table III equivalent — primary benchmark (5 real tasks, 5 trials, 8 AL iterations, RF-100 predictor)

Ranked by final-iteration mean R² (full table in `primary_benchmark_summary.json`):

**band_gap** (N=1000 real materials, d=21 features)
| Rank | Method | R² | Std |
|---|---|---|---|
| 1 | Uncertainty Sampling | 0.5966 | 0.0432 |
| 2 | Query by Committee | 0.5958 | 0.0419 |
| 3 | Maximum Entropy / RF Uncertainty (tied) | 0.5827 | 0.0459 |
| 5 | BADGE | 0.5814 | 0.0627 |
| 6 | Random Sampling | 0.5806 | 0.0667 |
| 7 | CoreSet | 0.5558 | 0.0272 |
| 8 | Diversity Sampling | 0.5531 | 0.0599 |
| 9 | Expected Improvement | 0.5508 | 0.1046 |
| **10 (last)** | **Quantum-Enhanced (ours)** | **0.5131** | **0.0458** |

**formation_energy**: quantum 0.8247 (6th of 10); best = Maximum Entropy/RF Uncertainty 0.8526.

**bulk_modulus**: quantum 0.7931 (8th of 10); best = Diversity Sampling/BADGE 0.8257.

**magnetic_moment**: quantum **−0.0874** (worst — worse than predicting the mean); best = BADGE 0.4181.

**dielectric_constant**: quantum 0.0341 (nominally best, but every method is near R²≈0 on this task — it is effectively unpredictable from these features; not a meaningful win).

## Table IV equivalent — paired significance (band_gap, formation_energy; 5 trials)

Quantum vs. every baseline: `quantum_wins = false` in every single comparison
on both primary tasks. Several raw p-values are <0.05 (i.e. nominally
"significant" — but in the baseline's favor, not the quantum method's), none
survive Holm-Bonferroni correction across the 9 comparisons. Shapiro-Wilk
confirms paired differences are approximately normal (p>0.05 throughout), so
the paired t-test is the right tool. Full numbers in `statistical_tests.json`.

## Table V equivalent — ablation (band_gap, 5 trials)

| Variant | R² | Δ vs. full model |
|---|---|---|
| Full model | 0.5131 ± 0.0458 | — |
| No covariance | 0.5137 ± 0.0763 | +0.0007 (noise) |
| Commuting-only observables | 0.5192 ± 0.0588 | +0.0062 (noise) |
| Real-only coefficients | 0.5050 ± 0.0742 | −0.0081 (noise) |
| Single observable (K=1) | 0.5023 ± 0.0652 | −0.0107 (noise) |

**None of the paper's claimed mechanisms (covariance coupling, non-commutativity,
complex coefficients) produce an effect distinguishable from trial-to-trial
noise.** The original manuscript's claim that covariance coupling is "the
primary driver of improvement" (a claimed −2.8% R² drop from removing it) does
not hold on real data — the actual effect is +0.07%, i.e. nothing.

## Observable sensitivity (band_gap, 5 trials)

K=2: 0.4759 ± 0.0682. K=3: 0.5131 ± 0.0458. K=6: 0.5122 ± 0.0689. K=3 is
marginally best among the three tested (consistent in direction with the
original claim, though the underlying method still trails every baseline
regardless of K).

## Table VI equivalent — runtime & memory (band_gap, real pool N=650, 3 repeats, single thread)

| Method | Time (s) | Memory (MB) |
|---|---|---|
| Quantum-Enhanced | 0.369 ± 0.002 | 0.669 ± 0.00003 |
| Uncertainty Sampling | 0.081 ± 0.004 | 0.793 ± 0.012 |
| Query by Committee | 1.539 ± 0.019 | 0.958 ± 0.070 |
| Expected Improvement | 0.083 ± 0.001 | 1.279 ± 0.002 |
| Maximum Entropy | 0.880 ± 0.040 | 1.173 ± 0.001 |

(Full table incl. remaining baselines and a clearly-labeled N=2000
bootstrap-resampled extrapolation point in `runtime_memory.json`.) These
numbers bear no resemblance to the original manuscript's Table VI
(8.3s/127MB) — that table did not come from measuring anything.

## Improvement attempt (band_gap, 5 trials)

Tested two principled, pre-specified fixes on top of the unmodified core
formalism, isolated from each other:

| Variant | R² |
|---|---|
| Original (arbitrary index-thirds observable groups) | 0.5131 ± 0.0458 |
| Domain-informed observable groups only | 0.4915 ± 0.0903 (worse) |
| Domain groups + predictor-importance-weighted encoding | 0.5124 ± 0.0773 (no change) |

Neither closes the gap to the best baseline (0.5966). The likely root cause:
`U_total` is computed purely from a candidate's position in the standardized
feature Hilbert space and the labeled pool's correlation structure — it never
sees the downstream model's actual residuals/disagreement, unlike every
baseline that wins (GP posterior std, committee variance, RF tree variance).
Reweighting by feature importance did not fix this because importance is
static per iteration and still doesn't track per-candidate predictive
uncertainty the way an ensemble or GP does.

## Secondary experiments (pre-rebuild scripts; outputs now in `results/legacy/`)

- **discrete_classification** (synthetic 6-class data, not yet wired to real
  `data/crystal_system.json`): final accuracy — quantum-margin 89.3%, entropy
  89.7%, random 79.3%. Quantum ≈ entropy (entropy slightly ahead), both far
  above random. This is a much more modest, but directionally real, finding
  than the original "91% vs 85%" claim — and quantum does *not* clearly beat
  entropy here as originally claimed.
- **transfer_learning**: `r2_transfer` and `r2_scratch` are byte-for-byte
  identical for both sulfides and nitrides in the saved output. This is not
  a real result — it indicates a bug (the transfer and from-scratch code
  paths are not actually diverging) rather than a genuine "no benefit from
  transfer" finding. Needs a real fix, not reported as a finding.
- **multi_property_optimization**: R² scores are all near zero or slightly
  negative across every property and iteration — the active-learning loop is
  not learning effectively on this synthetic setup. Inconclusive; flagged for
  follow-up rather than reported as a result.
- **spurious_correlation_analysis**: ran successfully as a controlled
  synthetic-noise study (this one is legitimately synthetic by design, not
  meant to model real materials). Numbers in `legacy/spurious_correlation.json` are
  real outputs of the noise-injection experiment.

## Joint expected-information-gain (JEIG): a mathematically derived
## alternative to the quantum-covariance ansatz

Source: `src/quantum_al/joint_eig.py`, `benchmarks/run_joint_eig_experiment.py`,
`results/joint_eig_experiment.json`, `results/joint_eig_experiment_replication.json`.

The original formalism's covariance term was an analogy (borrowed quantum
operator math), not a derivation, and measured as inert in every ablation
above. This experiment replaces it with a term derived directly from
Bayesian experimental design: for K real, jointly-labeled targets and an
ensemble predictive model, the joint expected information gain is

    JEIG(x) = 1/2 * log det(Sigma_pred(x) + R)

where `Sigma_pred(x)` is the (K,K) covariance of per-tree predictions
across a random-forest ensemble at candidate x, and `R` is the diagonal
out-of-bag residual variance per task.

**A real, checked theorem** (`self_test()` in `joint_eig.py`, plus
`tests/test_joint_eig.py`): summing the per-task marginal EIG terms and
subtracting the joint term always equals `-1/2 * log det(Corr(x))`, the
total correlation among the K predictive uncertainties (Hadamard's
determinant inequality guarantees this gap is >= 0). It is exactly zero
when the tasks' epistemic uncertainties are uncorrelated, in which case
JEIG collapses exactly to ordinary per-task ensemble-variance scoring
(the classical baseline is an exact special case, not an approximation) —
this is the honest version of the "classical-limit reduction" the original
Prop. 2 claimed but never actually used for anything.

**Empirical test, real MP data, one AL query returns all K labels (a
real DFT run gives every computed property at once):**

*Primary pair* — band_gap + formation_energy, n=498 shared materials,
real label correlation r=-0.365, 5 trials, same N0/T/batch protocol as
the primary benchmark:

| Method | Final joint R² (mean of both tasks) |
|---|---|
| Joint-EIG | 0.7000 ± 0.0492 |
| Marginal-Sum (classical-limit ablation) | 0.6924 ± 0.0407 |
| Random | 0.6378 ± 0.0466 |

Joint-EIG beats Random significantly (mean diff +0.0623, p=0.017 raw,
p=0.034 after Holm-Bonferroni, survives correction). Joint-EIG beats the
correlation-blind Marginal-Sum ablation only nominally (+0.0076, p=0.27,
not significant). Mean total-correlation gap on this pair: 0.0118
(max 0.0493) — confirms the two tasks' epistemic uncertainties are
genuinely, non-trivially correlated across the ensemble on real data.

**Four independent real property pairs were tested** (all pairs with
enough shared-material overlap to run a meaningful AL protocol):

| Pair | n | label r | Joint-EIG vs Marginal-Sum | Joint-EIG vs Random | mean gap |
|---|---|---|---|---|---|
| band_gap + formation_energy | 498 | -0.365 | +0.0076, p=0.27 (ns) | +0.0623, **p=0.034 (survives correction)** | 0.0118 |
| formation_energy + magnetic_moment | 220 | 0.205 | +0.0006, p=0.99 (ns) | -0.0164, p=0.66 (ns) | 0.0041 |
| band_gap + magnetic_moment | 193 | -0.024 | +0.0086, p=0.25 (ns) | +0.0186, p=0.17 (ns, corrected) | 0.0042 |
| bulk_modulus + dielectric_constant | 49 | 0.334 | -0.0314, p=0.61 (ns) | +0.0388, p=0.61 (ns) | 0.0211 |

(`results/joint_eig_experiment.json`, `_replication.json`,
`_pair_bg_mm.json`, `_pair_bm_dc.json`. The last pair used a scaled-down
protocol, n0=15/T=5/batch=3, because n=49 is too small for the default
N0=50 seed; treat it as low-power/inconclusive, not a fifth vote either
way.)

**Honest reading:** across 4 real, independently-measured pairs spanning
weak to strong label correlation, Joint-EIG **never** significantly beats
the correlation-blind Marginal-Sum ablation — the joint/covariance term
adds no measurable value over plain per-task ensemble-disagreement
scoring in any of the 4 tests. It significantly beats random sampling in
exactly 1 of 4 (the largest-N, most strongly-correlated pair), which is
attributable to ensemble uncertainty in general (Marginal-Sum wins there
too, nominally) rather than to the joint term specifically. The
total-correlation gap is real and non-zero in every pair (confirming the
theorem is measuring something genuine, not numerical noise), but it
does not translate into a reliable accuracy advantage, and its magnitude
does not even track label correlation strength cleanly (the near-zero-
label-correlation pair, r=-0.024, has a similar gap to the r=0.205 pair).

This is not evidence that covariance-aware acquisition "far supersedes"
existing active-learning methods — across every formulation tried in
this repository (the original quantum operator, three narrower fixes,
and this information-theoretic redesign), no version of "make the
acquisition score aware of cross-property/cross-observable correlation"
has produced a reliable, replicated accuracy gain over the simpler
alternative that ignores correlation. What is different this time: the
mathematics is a real, checkable derivation (Hadamard's inequality, an
exact classical-limit reduction), not an analogy, and it correctly
predicts its own null result: the gap term is provably zero when
correlation is absent and provably non-negative in general, so a small
or absent empirical benefit is exactly what the theorem allows, not a
contradiction of it. This is a stronger theoretical contribution than
the original manuscript, but it does not change the empirical
conclusion: on this class of real materials data, in this batch-AL
setting, correlation-aware acquisition is not distinguishable from
ensemble uncertainty alone.

## Known limitations of this rebuild (be upfront about these too)

- Thermal conductivity was dropped (not available at scale in MP) — 5 real
  regression tasks, not 6.
- Feature vectors are 21-dimensional, not the previously-claimed 100.
- `transfer_learning.py` and `multi_property_optimization.py` still need a
  real fix, not just a crash fix — their current output should not be quoted.
- The "9 baselines" here are Tier-1 acquisition strategies only (QBC, EI,
  Uncertainty Sampling, Max Entropy, Diversity, BADGE, CoreSet, RF
  Uncertainty, Random); Tier-2/3 (full GP posterior, deep ensembles,
  multi-task predictors) from the original manuscript were not rebuilt.
