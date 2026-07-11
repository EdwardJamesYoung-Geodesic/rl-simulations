"""Tests for the N=3 simplex geometry."""

from __future__ import annotations

import numpy as np

from rl_dynamics import simplex


def test_vertices_round_trip():
    for i in range(3):
        onehot = np.eye(3)[i]
        xy = simplex.to_xy(onehot)
        assert np.allclose(xy, simplex.VERTICES[i])


def test_from_xy_inverts_to_xy():
    pi = np.array([0.2, 0.5, 0.3])
    xy = simplex.to_xy(pi)
    back = simplex.from_xy(xy[0], xy[1])
    assert np.allclose(back, pi)


def test_barycentric_sums_to_one():
    xy = simplex.to_xy([0.1, 0.6, 0.3])
    assert np.isclose(simplex.from_xy(xy[0], xy[1]).sum(), 1.0)


def test_edge_distance_is_probability_times_height():
    pi = np.array([0.25, 0.25, 0.5])
    d = simplex.edge_distances(pi)
    assert np.allclose(d, pi * simplex.HEIGHT)
    # centroid is equidistant from all three edges
    d_centroid = simplex.edge_distances([1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(d_centroid, d_centroid[0])


def test_grid_points_are_valid_policies():
    g = simplex.grid(6)
    assert np.allclose(g.sum(axis=1), 1.0)
    assert np.all(g >= 0)
    # a margin excludes the boundary
    interior = simplex.grid(6, margin=0.1)
    assert np.all(interior.min(axis=1) >= 0.1)
