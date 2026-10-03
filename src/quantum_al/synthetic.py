"""Controlled multi-target regression problems with a tunable cross-task
correlation, for studying *when* correlation-aware acquisition can help.

Real materials data fixes the correlation structure, noise level and
sample size together, so a null result on it cannot say which of them is
responsible. This module generates problems where each is an explicit
knob. It is a testbed for mechanism, clearly separate from the real
Materials Project benchmarks: no synthetic number is ever substituted for
a real-data result anywhere in this package.

Construction: a shared latent function ``g`` and independent task-specific
functions ``h_1..h_K`` are drawn from the span of random Fourier features
(an approximate draw from a Gaussian process with an RBF kernel, Rahimi &
Recht, 2007), orthonormalized on the sample, and mixed as

.. math::

    f_k(x) = \\sqrt{\\rho}\\, g(x) + \\sqrt{1 - \\rho}\\, h_k(x),
    \\qquad y_k = a_k\\,(f_k(x) + \\sigma\\,\\varepsilon_k),

so the noiseless signals have unit variance and pairwise sample
correlation exactly ``rho``, the labels have expected pairwise correlation
``rho / (1 + sigma^2)``, and ``a_k`` sets each task's units.
"""
import numpy as np

__all__ = ["make_correlated_tasks"]


def make_correlated_tasks(n_samples=500, n_features=8, n_tasks=2, correlation=0.5,
                          noise=0.3, task_scales=None, n_components=200,
                          length_scale=None, seed=0):
    """Draw a multi-target regression problem with known task correlation.

    Parameters
    ----------
    n_samples : int
        Number of materials-like samples.
    n_features : int
        Input dimension; inputs are i.i.d. standard normal.
    n_tasks : int
        Number of targets ``K``.
    correlation : float in [0, 1]
        Pairwise correlation ``rho`` between the noiseless target functions.
    noise : float >= 0
        Noise standard deviation ``sigma`` relative to unit signal variance.
    task_scales : sequence of float, optional
        Per-task unit scale ``a_k`` (default all ones). Use very different
        scales to test unit invariance of an acquisition criterion.
    n_components : int
        Number of random Fourier features spanning the function space.
    length_scale : float, optional
        RBF length scale; defaults to ``sqrt(n_features)``.
    seed : int
        Seed for every random draw.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    Y : ndarray, shape (n_samples, n_tasks)
    info : dict
        ``signal`` (noiseless ``a_k f_k``), ``signal_correlation``,
        ``expected_label_correlation`` and the generating parameters.
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [0, 1]")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if n_tasks + 1 > n_components:
        raise ValueError("n_components must exceed n_tasks")
    rng = np.random.default_rng(seed)
    scales = np.ones(n_tasks) if task_scales is None else np.asarray(task_scales, dtype=float)
    if scales.shape != (n_tasks,) or np.any(scales <= 0):
        raise ValueError("task_scales must be n_tasks positive numbers")
    ell = np.sqrt(n_features) if length_scale is None else float(length_scale)

    X = rng.normal(size=(n_samples, n_features))
    W = rng.normal(scale=1.0 / ell, size=(n_features, n_components))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    Phi = np.sqrt(2.0 / n_components) * np.cos(X @ W + b)

    # n_tasks + 1 random functions in the feature span, then centered and
    # orthonormalized on the sample (a change of coefficients, so each
    # column remains a smooth function of x).
    F = Phi @ rng.normal(size=(n_components, n_tasks + 1))
    F -= F.mean(axis=0)
    Q, _ = np.linalg.qr(F)
    Q *= np.sqrt(n_samples)
    g, H = Q[:, 0], Q[:, 1:]

    signal_unit = np.sqrt(correlation) * g[:, None] + np.sqrt(1.0 - correlation) * H
    Y = (signal_unit + noise * rng.normal(size=signal_unit.shape)) * scales[None, :]
    return X, Y, {
        "signal": signal_unit * scales[None, :],
        "signal_correlation": correlation,
        "expected_label_correlation": correlation / (1.0 + noise ** 2),
        "noise": noise,
        "task_scales": scales.tolist(),
        "length_scale": ell,
        "seed": seed,
    }
