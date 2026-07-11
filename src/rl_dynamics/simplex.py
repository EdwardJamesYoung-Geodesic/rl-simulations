"""Geometry of the 2-simplex (``N = 3``) used to draw policies as a triangle.

Each vertex is a pure one-hot policy; a policy ``pi`` maps to the point
``pi_1 v_1 + pi_2 v_2 + pi_3 v_3``. The barycentric coordinate ``pi_i`` equals the
(normalised) orthogonal distance from the point to the edge *opposite* vertex
``i`` -- this is what the diagrams exploit.

The reference triangle is equilateral with unit side length::

    v1 = (0.5, sqrt(3)/2)   (top)
    v2 = (0.0, 0.0)         (bottom-left)
    v3 = (1.0, 0.0)         (bottom-right)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["VERTICES", "HEIGHT", "to_xy", "from_xy", "edge_distances", "grid"]

#: Cartesian coordinates of the three one-hot vertices, rows ``[v1, v2, v3]``.
VERTICES: NDArray[np.float64] = np.array(
    [[0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64
)

#: Height of the equilateral reference triangle (= perpendicular vertex-to-edge distance).
HEIGHT: float = float(np.sqrt(3.0) / 2.0)

# Barycentric solve matrix: [x, y, 1]^T = _T @ pi  ->  pi = _T^{-1} @ [x, y, 1]^T
_T = np.vstack([VERTICES.T, np.ones(3)])
_T_INV = np.linalg.inv(_T)


def to_xy(pi: ArrayLike) -> NDArray[np.float64]:
    """Map a policy ``pi`` (shape ``(N,)`` or ``(M, N)``) to Cartesian ``xy``."""
    pi = np.asarray(pi, dtype=np.float64)
    return pi @ VERTICES


def from_xy(x: float, y: float) -> NDArray[np.float64]:
    """Recover barycentric coordinates ``pi`` from a Cartesian point ``(x, y)``.

    The result sums to 1 but is not clipped: points outside the triangle yield
    negative components.
    """
    return _T_INV @ np.array([x, y, 1.0], dtype=np.float64)


def edge_distances(pi: ArrayLike) -> NDArray[np.float64]:
    """Orthogonal distances from the point to the edge opposite each vertex.

    ``edge_distances(pi)[i] = pi_i * HEIGHT`` -- i.e. probability ``pi_i`` read
    off as a perpendicular distance to the opposite edge.
    """
    pi = np.asarray(pi, dtype=np.float64)
    return pi * HEIGHT


def grid(n: int, *, margin: float = 0.0) -> NDArray[np.float64]:
    """Triangular lattice of policies strictly inside the simplex.

    Returns an ``(M, 3)`` array of policies whose barycentric coordinates are
    multiples of ``1/n``, optionally excluding a ``margin`` band near the edges
    (``0 <= margin < 1``, expressed in barycentric units). Useful for sampling a
    vector field over the interior.
    """
    pts = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            p = np.array([a, b, c], dtype=np.float64) / n
            if margin > 0.0 and p.min() < margin:
                continue
            pts.append(p)
    return np.array(pts, dtype=np.float64)
