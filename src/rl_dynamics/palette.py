"""Load the shared colour palette from ``config/palette.yaml``.

This is the single source of colour truth for both the Python figure generators
and the widget build step. Data marks use the Wong (2011) colourblind-safe
palette; chrome stays neutral gray. See ``config/palette.yaml`` for the values
and usage rules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

__all__ = ["find_palette_file", "load", "Palette", "hex_to_rgb", "rgb_to_hex", "ramp"]


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(max(0, min(255, c)))):02x}" for c in rgb)


def ramp(stops: list[str], t: float) -> str:
    """Interpolate a colour at ``t in [0, 1]`` across a list of hex ``stops``."""
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    n = len(stops) - 1
    x = t * n
    i = min(int(x), n - 1)
    f = x - i
    a, b = hex_to_rgb(stops[i]), hex_to_rgb(stops[i + 1])
    return rgb_to_hex(tuple(a[k] + (b[k] - a[k]) * f for k in range(3)))

_FILENAME = Path("config") / "palette.yaml"


def find_palette_file(start: Path | None = None) -> Path:
    """Locate ``config/palette.yaml`` by walking up from ``start`` (or this file).

    Falls back to walking up from the current working directory. Raises
    ``FileNotFoundError`` if not found.
    """
    candidates: list[Path] = []
    for base in filter(None, [start, Path(__file__).resolve(), Path.cwd().resolve()]):
        base = base if base.is_dir() else base.parent
        candidates.extend([base, *base.parents])
    for d in candidates:
        p = d / _FILENAME
        if p.is_file():
            return p
    raise FileNotFoundError(f"could not locate {_FILENAME} above {start or Path.cwd()}")


class Palette:
    """Thin typed accessor over the parsed palette mapping."""

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._d = data
        self.source = source

    # -- chrome / ink -------------------------------------------------------
    @property
    def surface(self) -> str:
        return self._d["surface"]

    @property
    def ink(self) -> dict[str, str]:
        return self._d["ink"]

    @property
    def structure(self) -> dict[str, str]:
        return self._d["structure"]

    # -- data marks ---------------------------------------------------------
    @property
    def wong(self) -> dict[str, str]:
        return self._d["wong"]

    @property
    def categorical(self) -> list[str]:
        return list(self._d["categorical"])

    @property
    def accent(self) -> dict[str, str]:
        return self._d["accent"]

    @property
    def highlight(self) -> str:
        return self._d["highlight"]

    @property
    def sequential(self) -> dict[str, list[str]]:
        return self._d["sequential"]

    @property
    def font(self) -> dict[str, str]:
        return self._d["font"]

    def series(self, i: int) -> str:
        """Colour for categorical series ``i`` (0-indexed, wraps with a warning-free modulo)."""
        cats = self.categorical
        return cats[i % len(cats)]

    def as_dict(self) -> dict[str, Any]:
        """The raw parsed mapping (e.g. to serialise into a widget)."""
        return dict(self._d)


@lru_cache(maxsize=None)
def load(path: str | None = None) -> Palette:
    """Load and cache the palette. ``path=None`` auto-locates ``config/palette.yaml``."""
    p = Path(path) if path is not None else find_palette_file()
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Palette(data, p)
