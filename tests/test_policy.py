"""Correctness tests for the policy-gradient machinery."""

from __future__ import annotations

import numpy as np

from rl_dynamics import dynamics, policy


def test_softmax_is_a_distribution():
    pi = policy.softmax([2.0, -1.0, 0.5])
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi > 0)


def test_softmax_shift_invariant():
    L = np.array([1.0, 2.0, 3.0])
    assert np.allclose(policy.softmax(L), policy.softmax(L + 10.0))


def test_full_gradient_is_baseline_free():
    # sum_i ghat_full_i = sum_i pi_i (R_i - E[R]) = E[R] - E[R] = 0
    pi = policy.softmax([0.3, -0.7, 1.1])
    R = np.array([0.0, 1.0, 1.0])
    assert np.isclose(policy.full_gradient(pi, R).sum(), 0.0)


def test_full_gradient_raises_objective():
    # A full-gradient ascent step must not decrease J = E[R].
    R = np.array([0.0, 1.0, 1.0])
    L = np.array([0.9, 0.05, 0.05])
    pi = policy.softmax(L)
    ghat = policy.full_gradient(pi, R)
    L2 = policy.logit_step(L, ghat, eta=0.1)
    assert policy.objective(policy.softmax(L2), R) >= policy.objective(pi, R) - 1e-12


def test_rloo_counts_sum_to_G():
    rng = dynamics.rng(0)
    pi = policy.softmax([0.1, 0.2, -0.3])
    _, counts, _ = policy.rloo_gradient(pi, [0.0, 1.0, 1.0], G=8, rng=rng)
    assert counts.sum() == 8


def test_rloo_baseline_is_empirical_mean():
    rng = dynamics.rng(1)
    R = np.array([0.0, 1.0, 2.0])
    pi = policy.softmax([0.0, 0.0, 0.0])
    _, counts, Rbar = policy.rloo_gradient(pi, R, G=16, rng=rng)
    assert np.isclose(Rbar, counts @ R / 16)


def test_rloo_expectation_is_scaled_full_gradient():
    # The empirical-mean-baseline RLOO estimator is biased: its expectation is
    # ((G-1)/G) * full_gradient (see the derivation in policy.rloo_gradient's
    # docstring). It shares the full gradient's *direction*, scaled by (G-1)/G.
    rng = dynamics.rng(42)
    R = np.array([0.0, 1.0, 1.0])
    pi = policy.softmax([0.4, -0.2, 0.1])
    G, trials = 32, 20000
    acc = np.zeros(3)
    for _ in range(trials):
        ghat, _, _ = policy.rloo_gradient(pi, R, G, rng)
        acc += ghat
    empirical = acc / trials
    expected = (G - 1) / G * policy.full_gradient(pi, R)
    assert np.allclose(empirical, expected, atol=2e-3)


def test_logit_step_identity_matches_coupled_identity():
    L = np.array([0.2, -0.1, 0.5])
    ghat = np.array([0.1, -0.05, -0.05])
    I = np.eye(3)
    assert np.allclose(
        policy.logit_step(L, ghat, 0.3),
        policy.logit_step(L, ghat, 0.3, A=I),
    )
