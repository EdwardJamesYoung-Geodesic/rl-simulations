"""Shared drawing helpers for the simplex figures.

Keeps the individual figure scripts (``equal_reward.py``, ``random_drift.py``)
small: they compose panels and trajectories from these primitives rather than
re-deriving projection / framing / colourbar code.
"""

from __future__ import annotations

import math

import numpy as np

from rl_dynamics import dynamics, palette, simplex
from rl_dynamics.svg import SVG, fmt

H = simplex.HEIGHT

# default panel geometry (px)
SCALE = 360
TOP = 54
VERT_LABELS = ["R₁ = 0", "R₂ = 1", "R₃ = 1"]


class Panel:
    """One triangular simplex panel with a top-left triangle origin at x=ox."""

    def __init__(self, ox: float, top: float = TOP, scale: float = SCALE) -> None:
        self.ox, self.top, self.scale = ox, top, scale
        self.tri_h = H * scale

    def proj(self, pi):
        xy = np.asarray(pi, float) @ simplex.VERTICES
        return (self.ox + xy[0] * self.scale, self.top + (H - xy[1]) * self.scale)

    def triangle_pts(self):
        return [self.proj(np.eye(3)[i]) for i in range(3)]


def new_canvas(width: float, height: float, pal) -> SVG:
    svg = SVG(width, height, background=pal.surface)
    svg.raw(f"<style>text{{font-family:{pal.font['serif']}}}</style>")
    return svg


def draw_triangle(svg: SVG, panel: Panel, pal, stroke_width: float = 1.75) -> None:
    """Just the triangle outline (no vertex labels)."""
    svg.polygon(panel.triangle_pts(), fill=pal.surface,
                stroke=pal.structure["edge"], stroke_width=stroke_width,
                stroke_linejoin="round")


def draw_frame(svg: SVG, panel: Panel, pal, vert_labels=VERT_LABELS) -> None:
    """Triangle outline + index/reward vertex labels."""
    pts = panel.triangle_pts()
    svg.polygon(pts, fill=pal.surface, stroke=pal.structure["edge"],
                stroke_width=1.75, stroke_linejoin="round")
    ink = pal.ink["secondary"]
    offs = [(0, -15), (-2, 26), (2, 26)]
    for i, (dx, dy) in enumerate(offs):
        svg.text(pts[i][0] + dx, pts[i][1] + dy, vert_labels[i], fill=ink,
                 font_size=13.5, font_weight=600, text_anchor="middle",
                 dominant_baseline="middle")


def draw_init_line(svg: SVG, panel: Panel, pal, pi1: float, label: str) -> None:
    """Dashed light-gray locus pi_1 = pi1, extended past the edges and labelled."""
    a = np.array([pi1, 1.0 - pi1, 0.0])
    b = np.array([pi1, 0.0, 1.0 - pi1])
    ax, ay = panel.proj(a)
    bx, by = panel.proj(b)
    ux, uy = bx - ax, by - ay
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    ext = 18.0
    svg.line(ax - ext * ux, ay - ext * uy, bx + ext * ux, by + ext * uy,
             stroke=pal.structure["edge"], stroke_width=1.3,
             stroke_dasharray="4 4", stroke_linecap="round")
    svg.text(ax - ext * ux - 7, ay - ext * uy, label, fill=pal.ink["secondary"],
             font_size=12.5, text_anchor="end", dominant_baseline="middle")


def draw_time_trajectory(svg: SVG, panel: Panel, traj, pal, *, width: float = 3.0,
                         samples: int = 80, opacity: float | None = None,
                         start_dot: bool = True) -> None:
    """Draw a trajectory as time-coloured segments (light orange -> dark red)."""
    stops = pal.sequential["time"]
    n = len(traj) - 1
    idx = np.unique(np.linspace(0, n, min(samples, len(traj))).astype(int))
    pts = [panel.proj(traj[k]) for k in idx]
    for k in range(len(pts) - 1):
        t = (idx[k] + idx[k + 1]) / 2 / n
        svg.line(*pts[k], *pts[k + 1], stroke=palette.ramp(stops, t),
                 stroke_width=width, stroke_linecap="round", opacity=opacity)
    if start_dot:
        svg.circle(*pts[0], 3.0, fill=palette.ramp(stops, 0.0))


def draw_vector_field(svg: SVG, panel: Panel, R, pal, *, grid_n: int = 12,
                      margin: float = 0.05) -> None:
    """Deterministic policy-gradient field for rewards ``R`` (light-blue arrows).

    Arrow length encodes field magnitude with a visible floor, so small (but
    nonzero) arrows still read as arrows; only the numerically-zero field is
    skipped.
    """
    grid = simplex.grid(grid_n, margin=margin)
    starts = np.array([panel.proj(pi) for pi in grid])
    dpi = dynamics.vector_field(grid, R)
    dxy = dpi @ simplex.VERTICES                                   # tangent (x, y-up)
    vec = np.column_stack([dxy[:, 0] * panel.scale, -dxy[:, 1] * panel.scale])  # pixel
    mag = np.hypot(vec[:, 0], vec[:, 1])
    ref = mag.max()
    colour = pal.wong["sky_blue"]
    LMIN, LMAX = 9.0, 31.0
    for (cx, cy), (vx, vy), m in zip(starts, vec, mag):
        if m / ref < 0.02:
            continue
        ux, uy = vx / m, vy / m
        length = LMIN + (LMAX - LMIN) * (m / ref)
        hx, hy = 0.5 * length * ux, 0.5 * length * uy
        x1, y1, x2, y2 = cx - hx, cy - hy, cx + hx, cy + hy       # tail -> tip
        svg.line(x1, y1, x2, y2, stroke=colour, stroke_width=1.4, stroke_linecap="round")
        hl, hw = min(6.0, length * 0.5), 3.1
        px, py = -uy, ux
        bx, by = x2 - hl * ux, y2 - hl * uy
        svg.polygon([(x2, y2), (bx + hw * px, by + hw * py),
                     (bx - hw * px, by - hw * py)], fill=colour)


def draw_colourbar(svg: SVG, pal, x: float, y: float, h: float, *, width: float = 14,
                   label: str = "time", top_label: str = "early",
                   bot_label: str = "late") -> None:
    stops = pal.sequential["time"]
    gstops = "".join(
        f'<stop offset="{fmt(o)}" stop-color="{palette.ramp(stops, o)}" />'
        for o in [0, 0.25, 0.5, 0.75, 1.0]
    )
    svg.raw(f'<defs><linearGradient id="timegrad" x1="0" y1="0" x2="0" y2="1">{gstops}</linearGradient></defs>')
    r = width / 2  # semicircular (stadium) ends
    svg.raw(f'<rect x="{fmt(x)}" y="{fmt(y)}" width="{fmt(width)}" height="{fmt(h)}" '
            f'fill="url(#timegrad)" stroke="{pal.structure["edge"]}" stroke-width="1" '
            f'rx="{fmt(r)}" ry="{fmt(r)}" />')
    muted, ink = pal.ink["muted"], pal.ink["secondary"]
    svg.text(x + width / 2, y - 10, top_label, fill=muted, font_size=11, text_anchor="middle")
    svg.text(x + width / 2, y + h + 16, bot_label, fill=muted, font_size=11, text_anchor="middle")
    lx = x + width + 16
    svg.text(lx, y + h / 2, label, fill=ink, font_size=12.5, font_weight=600,
             text_anchor="middle", dominant_baseline="middle",
             transform=f"rotate(90 {fmt(lx)} {fmt(y + h / 2)})")
