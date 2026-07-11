"""Policy and policy-gradient machinery for the linear-softmax toy model.

The policy over ``N`` actions is a softmax of a linear transform of parameters::

    pi = softmax(A @ theta)          logits  L = A @ theta

The RL objective is ``J = sum_i pi_i R_i``. Two gradient estimators are provided:

* the **full / expected** gradient coefficient ``ghat_full_i = pi_i (R_i - E[R])``;
* the **finite-sample RLOO** estimator with group size ``G`` (empirical-mean
  baseline), ``ghat_i = (1/G) N_i (R_i - Rbar)`` where ``N`` are multinomial
  sample counts and ``Rbar = (1/G) sum_i N_i R_i``.

Both ``ghat`` live in *action space*; the parameter update is
``theta <- theta + eta A^T ghat`` and hence the logits move by
``L <- L + eta A A^T ghat`` (see :func:`logit_step`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "softmax",
    "expected_reward",
    "objective",
    "full_gradient",
    "rloo_from_counts",
    "rloo_gradient",
    "logit_step",
]


def softmax(logits: ArrayLike) -> NDArray[np.float64]:
    """Numerically-stable softmax of a 1-D logit vector."""
    L = np.asarray(logits, dtype=np.float64)
    z = L - L.max()
    e = np.exp(z)
    return e / e.sum()


def expected_reward(pi: ArrayLike, R: ArrayLike) -> float:
    """``E[R] = sum_i pi_i R_i`` under the policy ``pi``."""
    pi = np.asarray(pi, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    return float(pi @ R)


def objective(pi: ArrayLike, R: ArrayLike) -> float:
    """The RL objective ``J = sum_i pi_i R_i`` (identical to :func:`expected_reward`)."""
    return expected_reward(pi, R)


def full_gradient(pi: ArrayLike, R: ArrayLike) -> NDArray[np.float64]:
    """Full/expected policy-gradient coefficient ``ghat_full_i = pi_i (R_i - E[R])``.

    This is the expectation of :func:`rloo_gradient`'s ``ghat`` over sampling.
    """
    pi = np.asarray(pi, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    return pi * (R - expected_reward(pi, R))


def rloo_from_counts(
    counts: ArrayLike, R: ArrayLike
) -> tuple[NDArray[np.float64], float]:
    """RLOO gradient coefficient from fixed sample ``counts`` (the pure formula).

    ``ghat_i = (1/G) N_i (R_i - Rbar)`` with ``Rbar = (1/G) sum_i N_i R_i`` and
    ``G = sum_i N_i``. Split out from :func:`rloo_gradient` so the (RNG-free)
    formula can be checked against the JS mirror.
    """
    counts = np.asarray(counts, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    G = counts.sum()
    Rbar = float(counts @ R / G)
    ghat = counts * (R - Rbar) / G
    return ghat, Rbar


def rloo_gradient(
    pi: ArrayLike,
    R: ArrayLike,
    G: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.int64], float]:
    """Finite-sample RLOO gradient estimator with group size ``G``.

    Draws ``G`` actions i.i.d. from ``pi`` (multinomial counts ``N``), uses the
    empirical mean ``Rbar = (1/G) sum_i N_i R_i`` as the baseline, and returns

        ghat_i = (1/G) N_i (R_i - Rbar)

    Parameters
    ----------
    pi, R : array_like
        Policy and per-action rewards, shape ``(N,)``.
    G : int
        Group size (number of samples).
    rng : numpy.random.Generator
        Seeded RNG, for reproducibility.

    Returns
    -------
    (ghat, counts, Rbar)
        ``ghat`` action-space gradient coefficient ``(N,)``; ``counts`` the
        sampled multinomial counts ``(N,)`` summing to ``G``; ``Rbar`` the
        empirical mean reward.

    Notes
    -----
    With the *empirical-mean* baseline (as opposed to a leave-one-out baseline)
    this estimator is biased::

        E[ghat] = ((G - 1) / G) * full_gradient(pi, R)

    i.e. it points in the full-gradient direction but is shrunk by ``(G-1)/G``,
    a bias that vanishes as ``G -> inf`` and, in expectation, only rescales the
    effective learning rate. A leave-one-out baseline would remove it.
    """
    pi = np.asarray(pi, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    counts = rng.multinomial(G, pi)
    ghat, Rbar = rloo_from_counts(counts, R)
    return ghat, counts, Rbar


def logit_step(
    L: ArrayLike,
    ghat: ArrayLike,
    eta: float,
    A: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """One ascent step in logit space: ``L <- L + eta A A^T ghat``.

    ``A=None`` is the identity fast path (``A A^T = I``), giving ``L + eta ghat``.
    The Gram matrix ``A A^T`` couples the per-action gradient into the logits.
    """
    L = np.asarray(L, dtype=np.float64)
    ghat = np.asarray(ghat, dtype=np.float64)
    if A is None:
        return L + eta * ghat
    A = np.asarray(A, dtype=np.float64)
    return L + eta * (A @ (A.T @ ghat))
