Theory
======

This page derives every acquisition criterion in :mod:`quantum_al.acquisition`
and proves the properties that ``tests/test_acquisition.py`` checks
numerically. The aim is to make precise *what* a correlation-aware criterion
can gain over a correlation-blind one, and *when* that gain must be small.

.. contents::
   :local:
   :depth: 1

Setting
-------

A candidate material :math:`x` has :math:`K` real properties
:math:`y = (y_1, \dots, y_K)`. Querying :math:`x` reveals all :math:`K`
labels at once, as a single DFT calculation does. We model

.. math::

   y = f(x) + \varepsilon, \qquad
   f(x) \sim \mathcal N\big(\mu(x), \Sigma(x)\big), \qquad
   \varepsilon \sim \mathcal N\big(0, R\big), \quad R = \operatorname{diag}(r_1, \dots, r_K),

where :math:`\Sigma(x)` is the *epistemic* uncertainty (what more data could
remove) and :math:`R` the *aleatoric* noise (what it could not). In the
package, :math:`\Sigma(x)` is the covariance of the :math:`M` per-tree
predictions of a multi-output random forest at :math:`x`
(:func:`quantum_al.joint_eig.predictive_covariance`) and :math:`r_k` the
out-of-bag residual variance of target :math:`k`
(:func:`quantum_al.joint_eig.oob_residual_variance`). Ensemble disagreement as
a stand-in for the epistemic posterior is the same approximation behind
query-by-committee and deep-ensemble active learning; see *Limitations*
below.

Write :math:`s_k(x) = \Sigma_{kk}(x) / r_k` for the per-target
epistemic-to-aleatoric ratio, and :math:`\tilde\Sigma = R^{-1/2} \Sigma R^{-1/2}`
for the noise-whitened covariance.

Expected information gain
-------------------------

The value of querying :math:`x` is the mutual information between the labels
it would reveal and the unknown function values,

.. math::

   \mathrm{EIG}(x) = I(y; f) = H(y) - H(y \mid f)
   = \tfrac12 \log\det\big(\Sigma + R\big) - \tfrac12 \log\det R
   = \tfrac12 \log\det\big(I_K + \tilde\Sigma\big),

Lindley's (1956) measure of the information in an experiment and the
Gaussian, multi-output form of BALD (Houlsby et al., 2011). It is
non-negative, zero iff :math:`\Sigma = 0`, and maximizing it is the
Bayesian D-optimal design criterion (Chaloner & Verdinelli, 1995). The
per-target terms are :math:`\mathrm{EIG}_k = \tfrac12 \log(1 + s_k)`, and
:math:`\sum_k \mathrm{EIG}_k` is the natural *correlation-blind* criterion:
each property's uncertainty is scored independently and summed.

:func:`quantum_al.joint_eig.joint_eig_score` returns
:math:`\tfrac12\log\det(\Sigma + R)`, which differs from
:math:`\mathrm{EIG}` by the candidate-independent constant
:math:`\tfrac12 \log\det R`; both induce the same ranking.

Proposition 1: total-correlation decomposition
----------------------------------------------

.. math::

   \sum_{k=1}^K \mathrm{EIG}_k(x) - \mathrm{EIG}(x)
   = \mathrm{TC}(x) := -\tfrac12 \log\det C(x) \;\ge\; 0,

where :math:`C` is the correlation matrix of :math:`T = \Sigma + R`, with
equality iff :math:`\Sigma` is diagonal.

*Proof.* Let :math:`D = \operatorname{diag}(T)`, so :math:`T = D^{1/2} C D^{1/2}`
and :math:`\log\det T = \sum_k \log(\Sigma_{kk} + r_k) + \log\det C`. Since
:math:`R` is diagonal, :math:`\log\det R = \sum_k \log r_k`, and
subtracting gives the identity. :math:`T` is positive definite because
:math:`R \succ 0`, so Hadamard's inequality gives :math:`\det C \le \prod_k C_{kk} = 1`,
with equality iff :math:`C` (equivalently :math:`\Sigma`, whose off-diagonal
equals :math:`T`'s) is diagonal. :math:`\square`

:math:`\mathrm{TC}` is Watanabe's (1960) total correlation of the predicted
labels: the information the :math:`K` labels share. A correlation-blind
criterion counts that shared information once per property; the joint
criterion counts it once. With no epistemic correlation the two criteria are
*identical*, so the correlation-blind score is an exact special case, not an
approximation.

Proposition 2: sandwich bound
-----------------------------

.. math::

   \max_k \mathrm{EIG}_k(x) \;\le\; \mathrm{EIG}(x) \;\le\; \sum_k \mathrm{EIG}_k(x).

*Proof.* The upper bound is Proposition 1. For the lower bound, the chain
rule gives :math:`I(y; f) = I(y_k; f) + I(y_{-k}; f \mid y_k) \ge I(y_k; f)`,
and :math:`I(y_k; f) = I(y_k; f_k) = \mathrm{EIG}_k` because :math:`y_k`
depends on :math:`f` only through :math:`f_k`. :math:`\square`

Measuring :math:`K` properties is therefore worth at least as much as the
most informative one and at most as much as all of them measured separately.

Proposition 3: unit invariance
------------------------------

Changing the units of the targets, :math:`y \mapsto Dy` with
:math:`D = \operatorname{diag}(d_k)`, :math:`d_k > 0`, maps
:math:`\Sigma \mapsto D\Sigma D` and :math:`R \mapsto D^2 R`, leaving
:math:`\tilde\Sigma` unchanged. Every criterion that depends on
:math:`(\Sigma, R)` only through :math:`\tilde\Sigma` is therefore exactly
unit-invariant: :math:`\mathrm{EIG}`, :math:`\mathrm{EIG}_k`,
:math:`\mathrm{TC}`, the whitened trace :math:`\operatorname{tr}\tilde\Sigma`
(an A-optimal analogue) and the largest eigenvalue
:math:`\lambda_{\max}(\tilde\Sigma)` (an E-optimal analogue). The raw trace
:math:`\operatorname{tr}\Sigma`, i.e. summing ensemble variances across
properties, is not: a property reported in GPa dominates one in eV.
``examples/02_information_decomposition.py`` shows the raw-trace ranking
changing under a unit change.

Proposition 4: the correlation term is second order
---------------------------------------------------

The off-diagonal entries of :math:`C` satisfy

.. math::

   |c_{jk}| \;\le\; \sqrt{\frac{s_j}{1 + s_j}\,\frac{s_k}{1 + s_k}} .

*Proof.* Off the diagonal, :math:`T_{jk} = \Sigma_{jk}`, and
:math:`|\Sigma_{jk}| \le \sqrt{\Sigma_{jj}\Sigma_{kk}}` because every
:math:`2\times2` principal minor of a positive semi-definite matrix is
non-negative. Divide by :math:`\sqrt{T_{jj}T_{kk}}`. :math:`\square`

For two properties, :math:`\det C = 1 - c_{12}^2`, which gives the exact bound

.. math::

   \mathrm{TC} \;\le\; \tfrac12 \log\!\Big(1 + \frac{s_1 s_2}{1 + s_1 + s_2}\Big),

with equality iff the ensemble's disagreement is perfectly correlated
(:math:`\Sigma` of rank one). For general :math:`K`, write :math:`C = I + E`
with :math:`\operatorname{tr} E = 0`; expanding :math:`\log\det` gives

.. math::

   \mathrm{TC} = \tfrac12 \sum_{j<k} c_{jk}^2 + O(\lVert E \rVert^3)
   \;\le\; \tfrac12 \sum_{j<k} \frac{s_j s_k}{(1+s_j)(1+s_k)} + O(s^3).

Since :math:`\mathrm{EIG}_k = \tfrac12 s_k + O(s_k^2)`, the correlation term is
**second order** in the epistemic-to-aleatoric ratio while the marginal terms
are first order. To first order, the joint EIG, the marginal-EIG sum and half
the whitened trace are the same criterion.

Corollary: when the joint criterion can change a decision
---------------------------------------------------------

Let :math:`M(x) = \sum_k \mathrm{EIG}_k(x)`, so
:math:`\mathrm{EIG} = M - \mathrm{TC}`. Two candidates are ranked the same
way by both criteria whenever

.. math::

   |M(x) - M(x')| \;>\; |\mathrm{TC}(x) - \mathrm{TC}(x')| .

A large total correlation that is *the same for all candidates* changes
nothing; only its *variation across candidates* can. By Proposition 4 that
variation is bounded by quantities of order :math:`s^2`, so in the regime
where the model's disagreement is small compared with the noise (most rounds
of a typical materials campaign, and every round once the model is decent),
the two criteria can disagree only between candidates whose marginal scores
are already nearly tied, and such swaps have little effect on what is
learned. :func:`quantum_al.diagnostics.criterion_disagreement` measures the
quantities in this corollary on real candidate pools: the ratio
:math:`\operatorname{std}(\mathrm{TC}) / \operatorname{std}(M)`, the Jaccard
overlap of the two top-:math:`b` batches, and the median :math:`s_k`.

This offers an explanation, not just a report, of the null result on
Materials Project data (:doc:`findings`): a correlation-aware criterion can
only beat a correlation-blind one through decisions it makes differently, and
the mathematics confines those to a second-order regime. :doc:`findings`
tests this prediction on real candidate pools and in a controlled study.

Batch selection
---------------

A batch :math:`S` of candidates, each revealing :math:`K` labels with
independent noise, has information gain

.. math::

   \mathrm{EIG}(S) = I(y_S; f) = \tfrac12 \log\det\big(I_{|S|K} + G_S G_S^\top\big)
   = \tfrac12 \log\det\big(I_M + G_S^\top G_S\big),

where :math:`G_S \in \mathbb R^{|S|K \times M}` stacks the centered, whitened
per-member predictions at the batch candidates, scaled by
:math:`(M-1)^{-1/2}`, so that :math:`G_S G_S^\top` is the ensemble's
whitened joint covariance over the batch, *including cross-candidate
covariance*. The second form (Sylvester's determinant identity) costs
:math:`O(M^3)` regardless of the batch size
(:func:`quantum_al.acquisition.batch_information_gain`). Two candidates on
which the ensemble disagrees *in the same way* are redundant, and
:math:`\mathrm{EIG}(S)` counts their information once.

Because the observations at different candidates are conditionally
independent given :math:`f`, :math:`S \mapsto \mathrm{EIG}(S)` is monotone
and submodular (Krause & Guestrin, 2005; Krause et al., 2008). The greedy
algorithm, which adds the candidate with the largest marginal gain

.. math::

   \Delta(x \mid S) = \tfrac12 \log\det\big(I_K + G_x A_S^{-1} G_x^\top\big),
   \qquad A_S = I_M + G_S^\top G_S,

therefore returns a batch within a factor :math:`1 - 1/e` of the optimal one
(Nemhauser et al., 1978). This is the Gaussian, multi-output analogue of
BatchBALD (Kirsch et al., 2019). :func:`quantum_al.acquisition.greedy_batch_eig`
implements it with one Cholesky factorization per step; the tests check the
marginal gains against direct :math:`|S|K`-dimensional determinants,
monotonicity and diminishing returns on every subset of a small pool, and the
:math:`1 - 1/e` bound against brute-force enumeration.

The ensemble covariance has rank at most :math:`M - 1`, so
:math:`\mathrm{EIG}(S) \le \tfrac12 \sum_{i<M} \log(1 + \lambda_i)` saturates
for large batches: the number of ensemble members bounds how much batch
diversity the criterion can see.

The quantum-inspired formalism
------------------------------

:mod:`quantum_al.operator` implements an earlier, independently evaluated
construction with the same goal of aggregating several uncertainty sources
in a correlation-aware way, by analogy with quantum measurement rather than
by derivation. A standardized feature vector :math:`x \in \mathbb R^d` is
encoded as a normalized state

.. math::

   |\psi(x)\rangle = \sum_{j=1}^d \sqrt{p_j(x)}\, e^{i\varphi_j} |j\rangle,
   \qquad p(x) = \operatorname{softmax}(x \odot x),

with phases :math:`\varphi_j` set by feature correlations in the labeled
pool. :math:`K` Hermitian observables :math:`O_k`, each concentrated on one
feature group and non-commuting by construction, give variances
:math:`\operatorname{Var}_\psi(O_k) = \langle O_k^2\rangle - \langle O_k\rangle^2`
and symmetrized covariances
:math:`\operatorname{Cov}_\psi(O_k, O_l) = \tfrac12\langle\{O_k, O_l\}\rangle - \langle O_k\rangle\langle O_l\rangle`,
aggregated with complex weights :math:`\alpha_k` into

.. math::

   U(x) = \Big(\sum_k |\alpha_k|^2 \operatorname{Var}_\psi(O_k)
   + \sum_{k \ne l} \operatorname{Re}(\alpha_k^* \alpha_l) \operatorname{Cov}_\psi(O_k, O_l)\Big)^{1/2}.

With commuting observables, real weights and no covariance term this
reduces exactly to a weighted sum of classical variances (checked by
:func:`quantum_al.operator.self_test`). :mod:`quantum_al.circuit` realizes
:math:`|\psi\rangle` by amplitude encoding on :math:`\lceil \log_2 d\rceil`
qubits and estimates each expectation from a Pauli decomposition; it matches
the classical formula to floating-point precision in the noiseless limit.
:func:`quantum_al.circuit.measurement_grouping_report` counts the measurement
settings this needs (4752 Pauli terms for :math:`K=3`, :math:`d=21`, reduced
4.3x by qubit-wise commuting grouping and 12.3x by general commuting
grouping).

The contrast with the information-theoretic criteria is instructive. Here
the aggregation weights, the observables and the encoding are posited; there
they follow from a probability model, which is what makes Propositions 1-4
possible. Empirically, neither construction beats direct ensemble
disagreement (:doc:`findings`).

Limitations of the approximations
---------------------------------

* **Gaussianity.** EIG is computed under a Gaussian approximation of the
  ensemble's predictive distribution. Random-forest predictions are
  piecewise constant and can be strongly non-Gaussian.
* **Ensemble as posterior.** Tree disagreement is a heuristic proxy for
  epistemic uncertainty, not a Bayesian posterior; it can be overconfident
  away from the data.
* **Homoscedastic noise.** :math:`R` is one out-of-bag residual variance per
  target, shared by all candidates, and assumed diagonal (independent noise
  across properties). Out-of-bag residuals also contain some epistemic error,
  so :math:`R` is an overestimate, which makes :math:`s_k`, and hence the
  correlation term, smaller than under a true noise model.
* **Myopia.** All criteria are one-step (or one-batch) lookahead.

References
----------

* Chaloner, K. & Verdinelli, I. (1995). Bayesian experimental design: a review. *Statistical Science* 10(3), 273-304.
* Houlsby, N., Huszár, F., Ghahramani, Z. & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. arXiv:1112.5745.
* Kirsch, A., van Amersfoort, J. & Gal, Y. (2019). BatchBALD: efficient and diverse batch acquisition for deep Bayesian active learning. *NeurIPS*.
* Krause, A. & Guestrin, C. (2005). Near-optimal nonmyopic value of information in graphical models. *UAI*.
* Krause, A., Singh, A. & Guestrin, C. (2008). Near-optimal sensor placements in Gaussian processes. *JMLR* 9, 235-284.
* Lindley, D. V. (1956). On a measure of the information provided by an experiment. *Annals of Mathematical Statistics* 27(4), 986-1005.
* Nemhauser, G. L., Wolsey, L. A. & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions I. *Mathematical Programming* 14, 265-294.
* Watanabe, S. (1960). Information theoretical analysis of multivariate correlation. *IBM Journal of Research and Development* 4(1), 66-82.
