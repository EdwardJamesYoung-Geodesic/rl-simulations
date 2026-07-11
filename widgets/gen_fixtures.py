"""Generate JS<->Python parity fixtures from the Python library.

Writes ``widgets/fixtures.json``: canonical inputs paired with the reference
outputs computed by :mod:`rl_dynamics`. The widgets' ``core.js`` re-computes
these at load time (``RLDyn.selfTest``) and asserts it matches, so the JS mirror
can never silently drift from the Python source of truth.

Run: ``uv run python widgets/gen_fixtures.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_dynamics import policy, simplex

HERE = Path(__file__).resolve().parent


def _l(x) -> list:
    return np.asarray(x, dtype=float).tolist()


def build() -> dict:
    softmax_cases = [[2.0, -1.0, 0.5], [0.0, 0.0, 0.0], [3.0, 3.0, -2.0]]
    grad_cases = [
        ([0.9, 0.05, 0.05], [0.0, 1.0, 1.0]),
        ([0.2, 0.5, 0.3], [0.0, 1.0, 2.0]),
    ]
    counts_cases = [
        ([5, 2, 1], [0.0, 1.0, 1.0]),
        ([3, 3, 2], [0.0, 1.0, 2.0]),
    ]
    A = [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.2, 0.0, 1.0]]
    logit_cases = [
        dict(L=[0.2, -0.1, 0.5], ghat=[0.1, -0.05, -0.05], eta=0.3, A=None),
        dict(L=[0.2, -0.1, 0.5], ghat=[0.1, -0.05, -0.05], eta=0.3, A=A),
    ]
    pis = [[0.2, 0.5, 0.3], [0.1, 0.1, 0.8], [1 / 3, 1 / 3, 1 / 3]]

    fx: dict = {
        "softmax": [{"L": L, "pi": _l(policy.softmax(L))} for L in softmax_cases],
        "fullGradient": [
            {"pi": pi, "R": R, "ghat": _l(policy.full_gradient(pi, R))}
            for pi, R in grad_cases
        ],
        "rlooFromCounts": [],
        "logitStep": [],
        "toXY": [{"pi": pi, "xy": _l(simplex.to_xy(pi))} for pi in pis],
        "fromXY": [
            {"xy": _l(simplex.to_xy(pi)), "pi": _l(simplex.from_xy(*simplex.to_xy(pi)))}
            for pi in pis
        ],
        "edgeDistances": [
            {"pi": pi, "d": _l(simplex.edge_distances(pi))} for pi in pis
        ],
    }
    for counts, R in counts_cases:
        ghat, Rbar = policy.rloo_from_counts(counts, R)
        fx["rlooFromCounts"].append(
            {"counts": counts, "R": R, "ghat": _l(ghat), "Rbar": Rbar}
        )
    for c in logit_cases:
        out = policy.logit_step(c["L"], c["ghat"], c["eta"], c["A"])
        fx["logitStep"].append({**c, "out": _l(out)})
    return fx


def main() -> None:
    out = HERE / "fixtures.json"
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
