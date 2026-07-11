"""Tests for the cognition model (sampled vs marginal estimators)."""

from __future__ import annotations

import numpy as np

from rl_dynamics import cognition, dynamics

# 3 cognitions, 2 actions; cognitions 2 and 3 both map to action 2.
P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
R = np.array([0.0, 1.0])


def test_effective_rewards():
    # Rtilde_k = expected reward of cognition k = (0, 1, 1)
    assert np.allclose(cognition.effective_rewards(P, R), [0.0, 1.0, 1.0])


def test_marginal_policy_is_a_distribution():
    pi, s = cognition.marginal_policy(P, [0.4, -0.2, 0.1])
    assert np.isclose(pi.sum(), 1.0) and np.all(pi >= 0)
    assert np.isclose(s.sum(), 1.0)
    # pi_action2 = s_2 + s_3
    assert np.isclose(pi[1], s[1] + s[2])


def test_both_estimators_are_scaled_full_gradient():
    # With the empirical-mean baseline both estimators share the base-RLOO
    # (G-1)/G shrinkage; their expectation is ((G-1)/G) * full_gradient, and in
    # particular they point in the full-gradient direction.
    theta = np.array([0.5, -0.3, 0.2])
    G = 16
    expected = (G - 1) / G * cognition.full_gradient(P, theta, R)
    for mode, fn in [("sampled", cognition.sampled_gradient),
                     ("marginal", cognition.marginal_gradient)]:
        rng = dynamics.rng(0)
        acc = np.zeros(3)
        trials = 40000
        for _ in range(trials):
            acc += fn(P, theta, R, G, rng)
        assert np.allclose(acc / trials, expected, atol=3e-3), mode


def test_marginal_preserves_symmetry_exactly():
    # at s_2 = s_3, the marginal estimator moves theta_2 and theta_3 identically,
    # for *any* draw -> the 2-vs-3 symmetry is never broken by sampling noise.
    theta = np.array([0.3, -0.1, -0.1])  # s_2 == s_3
    for seed in range(20):
        g = cognition.marginal_gradient(P, theta, R, 8, dynamics.rng(seed))
        assert np.isclose(g[1], g[2])


def test_sampled_can_break_symmetry():
    # the sampled estimator generally does NOT move theta_2, theta_3 equally.
    theta = np.array([0.3, -0.1, -0.1])  # s_2 == s_3
    diffs = [abs(cognition.sampled_gradient(P, theta, R, 8, dynamics.rng(s))[1]
                 - cognition.sampled_gradient(P, theta, R, 8, dynamics.rng(s))[2])
             for s in range(20)]
    assert max(diffs) > 1e-6


def test_marginal_symmetric_start_stays_symmetric():
    theta0 = np.log([0.9, 0.05, 0.05])
    traj = cognition.trajectory(P, theta0, R, eta=1.2, steps=25, mode="marginal",
                                G=8, rng=dynamics.rng(1))
    assert np.allclose(traj[:, 1], traj[:, 2])  # s_2 == s_3 throughout
