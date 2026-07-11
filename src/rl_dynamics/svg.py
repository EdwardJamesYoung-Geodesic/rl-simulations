"""Minimal, dependency-free SVG *string* builders.

The renderer produces SVG markup as strings rather than manipulating a live DOM,
so the same primitives serve both the Python figure generators (write to a
``.svg`` file) and, by mirroring these conventions in ``widgets/render.js``, the
interactive widgets. Keep this small and additive -- extend as scenes need it.

Coordinates are in user units; the caller sets the ``viewBox`` and (optionally)
a ``transform`` to flip the y-axis so that "up" is positive, matching the
simplex geometry in :mod:`rl_dynamics.simplex`.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["fmt", "attrs", "SVG"]


def fmt(x: float, ndigits: int = 3) -> str:
    """Format a number compactly (trim trailing zeros; render ``-0`` as ``0``)."""
    if x == 0:
        x = 0.0
    s = f"{x:.{ndigits}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def attrs(**kw: object) -> str:
    """Render keyword args as SVG attributes.

    Underscores in names become hyphens (``stroke_width`` -> ``stroke-width``).
    ``None`` values are skipped. Floats are formatted via :func:`fmt`.
    """
    parts: list[str] = []
    for key, val in kw.items():
        if val is None:
            continue
        name = key.replace("_", "-")
        if isinstance(val, float):
            val = fmt(val)
        parts.append(f'{name}="{val}"')
    return " ".join(parts)


def _points(pts: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)


class SVG:
    """Accumulates SVG elements and serialises to a complete document."""

    def __init__(
        self,
        width: float,
        height: float,
        *,
        viewbox: tuple[float, float, float, float] | None = None,
        background: str | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.viewbox = viewbox or (0.0, 0.0, width, height)
        self.background = background
        self._elems: list[str] = []

    # -- raw / grouping -----------------------------------------------------
    def raw(self, markup: str) -> "SVG":
        self._elems.append(markup)
        return self

    def group(self, body: str, **kw: object) -> "SVG":
        return self.raw(f"<g {attrs(**kw)}>{body}</g>")

    # -- primitives ---------------------------------------------------------
    def line(self, x1: float, y1: float, x2: float, y2: float, **kw: object) -> "SVG":
        return self.raw(f"<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, **kw)} />")

    def polyline(self, pts: Iterable[tuple[float, float]], **kw: object) -> "SVG":
        kw.setdefault("fill", "none")
        return self.raw(f'<polyline points="{_points(pts)}" {attrs(**kw)} />')

    def polygon(self, pts: Iterable[tuple[float, float]], **kw: object) -> "SVG":
        return self.raw(f'<polygon points="{_points(pts)}" {attrs(**kw)} />')

    def path(self, d: str, **kw: object) -> "SVG":
        return self.raw(f'<path d="{d}" {attrs(**kw)} />')

    def circle(self, cx: float, cy: float, r: float, **kw: object) -> "SVG":
        return self.raw(f"<circle {attrs(cx=cx, cy=cy, r=r, **kw)} />")

    def text(self, x: float, y: float, s: str, **kw: object) -> "SVG":
        return self.raw(f"<text {attrs(x=x, y=y, **kw)}>{s}</text>")

    # -- serialise ----------------------------------------------------------
    def tostring(self) -> str:
        vb = " ".join(fmt(v) for v in self.viewbox)
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{fmt(self.width)}" height="{fmt(self.height)}" '
            f'viewBox="{vb}">'
        )
        bg = (
            f'<rect x="{fmt(self.viewbox[0])}" y="{fmt(self.viewbox[1])}" '
            f'width="{fmt(self.viewbox[2])}" height="{fmt(self.viewbox[3])}" '
            f'fill="{self.background}" />'
            if self.background
            else ""
        )
        return head + bg + "".join(self._elems) + "</svg>"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.tostring()
