"""Cognition model: sample a cognition ``c ~ Softmax(theta)``, then an action
``a ~ P[:, c]``.

The marginal action policy is ``pi = P @ Softmax(theta)`` (``P`` is
``n_actions x n_cognitions``, column-stochastic). Learning acts on the cognition
logits ``theta``; two finite-sample RLOO estimators differ in whether the
cognition is *observed* or *latent* (empirical-mean baseline, group size ``G``):

* :func:`sampled_gradient` -- the cognition is observed, so each rollout
  reinforces the sampled cognition ``e_c``. Equivalent to base RLOO over
  cognitions with effective rewards ``Rtilde = P^T R``.
* :func:`marginal_gradient` -- the cognition is latent, so each rollout
  reinforces the posterior ``p(c|a) = P[a, :] * s / pi[a]``. This is the
  Rao-Blackwellisation of the sampled estimator: same mean, lower variance.

Both are unbiased for the full gradient ``sum_k s_k (Rtilde_k - E[R]) e_k`` (see
:func:`full_gradient`). The practical difference: when two cognitions map to the
same action, observing the action cannot tell them apart, so the marginal
estimator splits credit by the posterior and leaves their relative probability
untouched -- whereas the sampled estimator reinforces whichever one fired,
breaking the symmetry.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import policy

__all__ = [
    "marginal_policy",
    "effective_rewards",
    "full_gradient",
    "sampled_gradient",
    "marginal_gradient",
    "trajectory",
]


def marginal_policy(P: ArrayLike, theta: ArrayLike) -> tuple[NDArray, NDArray]:
    """Return ``(pi, s)``: marginal action policy ``pi = P s`` and cognitions ``s``."""
    s = policy.softmax(theta)
    return np.asarray(P, dtype=np.float64) @ s, s


def effective_rewards(P: ArrayLike, R: ArrayLike) -> NDArray[np.float64]:
    """Effective reward of each cognition, ``Rtilde_k = sum_a P[a, k] R_a`` (``= P^T R``)."""
    return np.asarray(P, dtype=np.float64).T @ np.asarray(R, dtype=np.float64)


def full_gradient(P: ArrayLike, theta: ArrayLike, R: ArrayLike) -> NDArray[np.float64]:
    """Exact policy gradient over ``theta``: ``sum_k s_k (Rtilde_k - E[R]) e_k``."""
    s = policy.softmax(theta)
    Rt = effective_rewards(P, R)
    return s * (Rt - float(s @ Rt))


def sampled_gradient(
    P: ArrayLike, theta: ArrayLike, R: ArrayLike, G: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """RLOO with the cognition *observed*: reinforce the sampled cognition ``e_c``.

    Draws ``G`` cognitions ``c ~ s``, an action ``a ~ P[:, c]`` for each, and
    returns ``(1/G) sum (R_a - Rbar) e_c`` (the ``-s`` term cancels under the
    empirical-mean baseline).
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    s = policy.softmax(theta)
    K, A = s.shape[0], R.shape[0]
    cs = rng.choice(K, size=G, p=s)
    acts = np.array([rng.choice(A, p=P[:, c]) for c in cs])
    r = R[acts]
    g = np.zeros(K)
    np.add.at(g, cs, r - r.mean())
    return g / G


def marginal_gradient(
    P: ArrayLike, theta: ArrayLike, R: ArrayLike, G: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """RLOO with the cognition *marginalised*: reinforce the posterior ``p(c|a)``.

    Draws ``G`` actions ``a ~ pi = P s`` and returns
    ``(1/G) sum (R_a - Rbar) p(.|a)`` where ``p(.|a) = P[a, :] * s / pi[a]``
    (the ``-s`` term cancels under the empirical-mean baseline).
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    s = policy.softmax(theta)
    pi = P @ s
    acts = rng.choice(R.shape[0], size=G, p=pi)
    r = R[acts]
    Rbar = r.mean()
    g = np.zeros(s.shape[0])
    for a, ra in zip(acts, r):
        g += (ra - Rbar) * (P[a, :] * s / pi[a])
    return g / G


def trajectory(
    P: ArrayLike,
    theta0: ArrayLike,
    R: ArrayLike,
    *,
    eta: float,
    steps: int,
    mode: str = "marginal",
    G: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Integrate a cognition-logit trajectory; return ``s = softmax(theta)`` per step.

    ``mode`` is ``"full"`` (deterministic gradient), ``"sampled"`` (observed
    cognition), or ``"marginal"`` (latent cognition). The stochastic modes
    require ``G`` and ``rng``.
    """
    P = np.asarray(P, dtype=np.float64)
    theta = np.asarray(theta0, dtype=np.float64).copy()
    if mode in ("sampled", "marginal") and (G is None or rng is None):
        raise ValueError(f"mode={mode!r} requires G and rng")
    grad = {"full": lambda: full_gradient(P, theta, R),
            "sampled": lambda: sampled_gradient(P, theta, R, G, rng),
            "marginal": lambda: marginal_gradient(P, theta, R, G, rng)}
    if mode not in grad:
        raise ValueError(f"unknown mode {mode!r}")

    out = np.empty((steps + 1, theta.shape[0]), dtype=np.float64)
    out[0] = policy.softmax(theta)
    for t in range(1, steps + 1):
        theta = theta + eta * grad[mode]()
        out[t] = policy.softmax(theta)
    return out
