"""Tests for trajectory integration and the vector field."""

from __future__ import annotations

import numpy as np

from rl_dynamics import dynamics, policy, simplex


def test_trajectory_stays_on_simplex():
    R = np.array([0.0, 1.0, 1.0])
    traj = dynamics.trajectory([0.9, 0.05, 0.05], R, eta=0.5, steps=50, mode="full")
    assert traj.shape == (51, 3)
    assert np.allclose(traj.sum(axis=1), 1.0)
    assert np.all(traj >= 0)


def test_full_trajectory_increases_objective_monotonically():
    R = np.array([0.0, 1.0, 1.0])
    traj = dynamics.trajectory([2.0, -1.0, -1.0], R, eta=0.2, steps=100, mode="full")
    J = traj @ R
    assert np.all(np.diff(J) >= -1e-12)
    # low-reward action 1 loses mass over time
    assert traj[-1, 0] < traj[0, 0]


def test_full_trajectory_is_deterministic():
    R = np.array([0.0, 1.0, 1.0])
    a = dynamics.trajectory([0.5, 0.3, 0.2], R, eta=0.3, steps=20, mode="full")
    b = dynamics.trajectory([0.5, 0.3, 0.2], R, eta=0.3, steps=20, mode="full")
    assert np.allclose(a, b)


def test_rloo_trajectory_reproducible_with_seed():
    R = np.array([0.0, 1.0, 1.0])
    kw = dict(eta=0.3, steps=30, mode="rloo", G=6)
    a = dynamics.trajectory([0.4, 0.3, 0.3], R, rng=dynamics.rng(7), **kw)
    b = dynamics.trajectory([0.4, 0.3, 0.3], R, rng=dynamics.rng(7), **kw)
    assert np.allclose(a, b)


def test_rloo_requires_G_and_rng():
    R = np.array([0.0, 1.0, 1.0])
    try:
        dynamics.trajectory([0.4, 0.3, 0.3], R, eta=0.3, steps=5, mode="rloo")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing G/rng")


def test_vector_field_is_tangent_to_simplex():
    # each dpi sums to ~0 (stays on the probability simplex)
    R = np.array([0.0, 1.0, 1.0])
    P = simplex.grid(5, margin=0.05)
    V = dynamics.vector_field(P, R)
    assert V.shape == P.shape
    assert np.allclose(V.sum(axis=1), 0.0, atol=1e-9)


def test_vector_field_points_away_from_low_reward_vertex():
    # near a policy dominated by the low-reward action, mass should flow out of it
    R = np.array([0.0, 1.0, 1.0])
    pi = np.array([0.8, 0.1, 0.1])
    dpi = dynamics.vector_field(pi, R)[0]
    assert dpi[0] < 0  # action 1 (reward 0) loses probability
