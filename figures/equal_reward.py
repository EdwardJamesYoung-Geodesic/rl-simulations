"""Figure: deterministic full-gradient dynamics on the reward simplex.

Two side-by-side simplexes for N=3, A=I:

* left  -- the policy-gradient vector field (light-blue arrows);
* right -- five deterministic trajectories from pi_1 = 0.9 with
           pi_2 in {0.03, 0.04, 0.05, 0.06, 0.07}, coloured by time
           (light orange -> dark red), with a time colourbar.

``build(R, labels)`` is parameterised by the reward vector so sibling figures
(e.g. ``unequal_reward.py``) can reuse it. This script renders the equal-reward
case R = (0, 1, 1).

Run: ``uv run python figures/equal_reward.py``  ->  figures/out/equal-reward.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import simplexfig as sf
from rl_dynamics import dynamics, palette

OUT = Path(__file__).resolve().parent / "out"

# --- dynamics params (shared across reward variants) --------------------------
INIT_PI1 = 0.9
INIT_PI2 = [0.03, 0.04, 0.05, 0.06, 0.07]
ETA, STEPS = 0.3, 100

# --- layout (px) --------------------------------------------------------------
SCALE, TOP, H = sf.SCALE, sf.TOP, sf.H
TRI_H = H * SCALE
LEFT, GAP = 44, 96
PANEL_A = sf.Panel(LEFT)
PANEL_B = sf.Panel(LEFT + SCALE + GAP)
CBAR_GAP, CBAR_W = 46, 14
CBAR_X = PANEL_B.ox + SCALE + CBAR_GAP
WIDTH = CBAR_X + CBAR_W + 60
HEIGHT = TOP + TRI_H + 58


def build(R, labels) -> str:
    """Render the two-panel deterministic figure for reward vector ``R``."""
    R = np.asarray(R, dtype=float)
    pal = palette.load()
    svg = sf.new_canvas(WIDTH, HEIGHT, pal)
    # left panel: vector field
    sf.draw_frame(svg, PANEL_A, pal, labels)
    sf.draw_vector_field(svg, PANEL_A, R, pal)
    # right panel: trajectories + init line + colourbar
    sf.draw_frame(svg, PANEL_B, pal, labels)
    sf.draw_init_line(svg, PANEL_B, pal, INIT_PI1, "π₁ = 0.9")
    for pi2 in INIT_PI2:
        traj = dynamics.trajectory(np.log([INIT_PI1, pi2, 1.0 - INIT_PI1 - pi2]),
                                   R, eta=ETA, steps=STEPS, mode="full")
        sf.draw_time_trajectory(svg, PANEL_B, traj, pal, width=3.2)
    sf.draw_colourbar(svg, pal, CBAR_X, TOP, TRI_H, width=CBAR_W)
    return svg.tostring()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dest = OUT / "equal-reward.svg"
    dest.write_text(build([0.0, 1.0, 1.0], ["R₁ = 0", "R₂ = 1", "R₃ = 1"]),
                    encoding="utf-8")
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
