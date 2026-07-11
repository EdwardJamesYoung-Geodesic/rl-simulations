"""Build self-contained widget HTML files.

Inlines the shared JS core, the SVG render helpers, the (freshly regenerated)
JS<->Python parity fixtures, and the palette into each source template in
``widgets/src/``, producing standalone files in ``widgets/dist/`` that open with
no server, no build tooling, and no external requests -- ready to host or embed.

Run: ``uv run python widgets/build.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from rl_dynamics import palette

import gen_fixtures  # local module (widgets/gen_fixtures.py)

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
DIST = HERE / "dist"


def build_one(template: str, core: str, render: str, fixtures: str, pal: str) -> str:
    out = template.replace("/*__CORE__*/", core)
    out = out.replace("/*__RENDER__*/", render)
    out = out.replace("__FIXTURES__", fixtures)
    out = out.replace("__PALETTE__", pal)
    return out


def main() -> None:
    core = (HERE / "core.js").read_text(encoding="utf-8")
    render = (HERE / "render.js").read_text(encoding="utf-8")

    # regenerate fixtures + refresh the on-disk copy so the two never drift
    fx = gen_fixtures.build()
    (HERE / "fixtures.json").write_text(json.dumps(fx, indent=2) + "\n", encoding="utf-8")
    fixtures = json.dumps(fx, separators=(",", ":"))

    pal = json.dumps(palette.load().as_dict(), separators=(",", ":"))

    DIST.mkdir(exist_ok=True)
    sources = sorted(SRC.glob("*.html"))
    if not sources:
        print(f"no widget sources in {SRC.relative_to(HERE.parent)}")
        return
    for src in sources:
        html = build_one(src.read_text(encoding="utf-8"), core, render, fixtures, pal)
        dest = DIST / src.name
        dest.write_text(html, encoding="utf-8")
        print(f"built {dest.relative_to(HERE.parent)}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
