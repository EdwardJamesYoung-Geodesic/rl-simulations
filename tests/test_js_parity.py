"""JS <-> Python parity for the widget core.

The Python package is the source of truth; ``widgets/core.js`` mirrors its
deterministic maths. This test (1) checks the committed ``fixtures.json`` is
in sync with the current Python, and (2) if a JavaScript engine is available
(``jsc`` or ``node``), actually runs ``RLDyn.selfTest`` against the fixtures.

The JS-execution half is skipped (not failed) when no engine is present, so the
repo stays Python-only and CI-portable; the same self-test also runs in-browser
inside every built widget.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WIDGETS = ROOT / "widgets"

import sys

sys.path.insert(0, str(WIDGETS))
import gen_fixtures  # noqa: E402


def test_fixtures_are_in_sync():
    on_disk = json.loads((WIDGETS / "fixtures.json").read_text())
    assert on_disk == gen_fixtures.build(), (
        "widgets/fixtures.json is stale — run `uv run python widgets/gen_fixtures.py`"
    )


def _find_engine() -> tuple[str, list[str]] | None:
    jsc = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
    if Path(jsc).exists():
        return ("jsc", [jsc])
    node = shutil.which("node")
    if node:
        return ("node", [node])
    return None


def test_js_core_matches_python():
    engine = _find_engine()
    if engine is None:
        pytest.skip("no JS engine (jsc/node) available; browser self-test still applies")
    kind, cmd = engine
    core = (WIDGETS / "core.js").read_text()
    fixtures = (WIDGETS / "fixtures.json").read_text()
    emit = "debug" if kind == "jsc" else "console.log"
    shim = (
        "var module={exports:{}};var console={log:function(){},error:function(){}};"
        if kind == "jsc"
        else ""
    )
    script = (
        f"{shim}{core}\n"
        f"var fx={fixtures};var r=RLDyn.selfTest(fx);"
        f"{emit}('RESULT '+r.ok+' '+r.fails.length);"
    )
    if kind == "node":
        proc = subprocess.run(cmd + ["-e", script], text=True, capture_output=True)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(script)
            path = fh.name
        proc = subprocess.run(cmd + [path], text=True, capture_output=True)
    out = proc.stdout + proc.stderr
    assert "RESULT true 0" in out, f"JS self-test did not pass:\n{out}"
