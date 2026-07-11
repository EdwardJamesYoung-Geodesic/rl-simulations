"""Learning-dynamics: trajectory integration and vector fields over the simplex.

Trajectories are integrated in *logit* space (where the update is affine) and
mapped to policies ``pi = softmax(L)`` for output. Two modes:

* ``"full"``  -- deterministic ascent on the expected gradient;
* ``"rloo"``  -- stochastic finite-sample RLOO updates (needs an RNG + group size).

The vector field reports the deterministic instantaneous change of ``pi`` at each
sampled policy, obtained by pushing the logit update through the softmax Jacobian.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import policy

__all__ = ["rng", "trajectory", "vector_field"]


def rng(seed: int) -> np.random.Generator:
    """A seeded NumPy generator, so stochastic figures are reproducible."""
    return np.random.default_rng(seed)


def _softmax_jacobian(pi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Jacobian ``d pi_i / d L_j = pi_i (delta_ij - pi_j)``."""
    return np.diag(pi) - np.outer(pi, pi)


def trajectory(
    L0: ArrayLike,
    R: ArrayLike,
    *,
    eta: float,
    steps: int,
    mode: str = "full",
    A: ArrayLike | None = None,
    G: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Integrate a learning trajectory and return the policy at each step.

    Parameters
    ----------
    L0 : array_like
        Initial logits, shape ``(N,)``.
    R : array_like
        Per-action rewards, shape ``(N,)``.
    eta : float
        Learning rate.
    steps : int
        Number of update steps; the returned array has ``steps + 1`` rows.
    mode : {"full", "rloo"}
        Gradient estimator. ``"rloo"`` requires ``G`` and ``rng``.
    A : array_like, optional
        Linear map; ``None`` means identity (no logit coupling).
    G : int, optional
        RLOO group size (required for ``mode="rloo"``).
    rng : numpy.random.Generator, optional
        Seeded RNG (required for ``mode="rloo"``).

    Returns
    -------
    ndarray, shape ``(steps + 1, N)``
        The policy ``pi = softmax(L)`` at each step, starting from ``L0``.
    """
    L = np.asarray(L0, dtype=np.float64).copy()
    R = np.asarray(R, dtype=np.float64)
    if mode == "rloo" and (G is None or rng is None):
        raise ValueError("mode='rloo' requires both G and rng")
    if mode not in ("full", "rloo"):
        raise ValueError(f"unknown mode {mode!r}")

    out = np.empty((steps + 1, R.shape[0]), dtype=np.float64)
    pi = policy.softmax(L)
    out[0] = pi
    for t in range(1, steps + 1):
        if mode == "full":
            ghat = policy.full_gradient(pi, R)
        else:
            ghat, _, _ = policy.rloo_gradient(pi, R, G, rng)
        L = policy.logit_step(L, ghat, eta, A)
        pi = policy.softmax(L)
        out[t] = pi
    return out


def vector_field(
    policies: ArrayLike,
    R: ArrayLike,
    *,
    A: ArrayLike | None = None,
    normalize: bool = False,
) -> NDArray[np.float64]:
    """Deterministic instantaneous ``d pi`` at each sampled policy.

    Pushes the (unit-rate) expected logit update ``A A^T ghat_full`` through the
    softmax Jacobian, giving a tangent vector in probability space (each row sums
    to ~0). Learning rate is dropped -- this describes the field's *direction*
    and relative magnitude. Set ``normalize=True`` for unit-length arrows.

    Parameters
    ----------
    policies : array_like, shape ``(M, N)``
        Policies at which to evaluate the field (e.g. from :func:`simplex.grid`).
    R : array_like, shape ``(N,)``
        Per-action rewards.
    A : array_like, optional
        Linear map; ``None`` means identity.
    normalize : bool
        If true, scale each vector to unit L2 norm (zero vectors left as zero).

    Returns
    -------
    ndarray, shape ``(M, N)``
        The tangent ``d pi`` at each input policy.
    """
    P = np.atleast_2d(np.asarray(policies, dtype=np.float64))
    R = np.asarray(R, dtype=np.float64)
    A_mat = None if A is None else np.asarray(A, dtype=np.float64)

    out = np.empty_like(P)
    for k, pi in enumerate(P):
        ghat = policy.full_gradient(pi, R)
        dL = ghat if A_mat is None else A_mat @ (A_mat.T @ ghat)
        out[k] = _softmax_jacobian(pi) @ dL

    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        nz = norms[:, 0] > 0
        out[nz] = out[nz] / norms[nz]
    return out
