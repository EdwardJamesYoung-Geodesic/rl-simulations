"""Figure: deterministic dynamics with unequal rewards, richly labelled.

A relabelled variant of ``unequal_reward.py`` for the rewritten "Unequal rewards"
section: a1 = task failure (R1 = 0), a2 = faithful completion (R2 = 0.8), a3 =
hacking (R3 = 1.0). Two panels for N=3, A=I, as in Figure 3:

* left  -- the policy-gradient vector field (light-blue arrows);
* right -- five deterministic trajectories from pi_1 = 0.9 with
           pi_2 in {0.03, ..., 0.07}, coloured by time, with a colourbar.

Each vertex carries a two-line label: the action/reward on top, a plain-language
description beneath.

Run: ``uv run python figures/unequal_reward_labelled.py``
  -> figures/out/unequal-reward-labelled.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import simplexfig as sf
from rl_dynamics import dynamics, palette

OUT = Path(__file__).resolve().parent / "out"

R = np.array([0.0, 0.8, 1.0])
INIT_PI1 = 0.9
INIT_PI2 = [0.03, 0.04, 0.05, 0.06, 0.07]
ETA, STEPS = 0.3, 100

# (top line: action + reward, bottom line: description) per vertex v1, v2, v3
LABELS = [
    ("a₁, R₁ = 0", "Task failure"),
    ("a₂, R₂ = 0.8", "Faithful task completion"),
    ("a₃, R₃ = 1.0", "Hacking task completion"),
]

# --- layout (px) --------------------------------------------------------------
H = sf.H
S = 300
TRI_H = H * S
TOP = 50
LEFT, GAP = 84, 150
PANEL_A = sf.Panel(LEFT, TOP, S)
PANEL_B = sf.Panel(LEFT + S + GAP, TOP, S)
CBW = 14
CB_X = PANEL_B.ox + S + 86
WIDTH = CB_X + CBW + 66
HEIGHT = TOP + TRI_H + 50


def draw_frame2(svg, panel, pal) -> None:
    """Triangle outline + two-line vertex labels."""
    pts = panel.triangle_pts()
    svg.polygon(pts, fill=pal.surface, stroke=pal.structure["edge"],
                stroke_width=1.75, stroke_linejoin="round")
    sec, muted = pal.ink["secondary"], pal.ink["muted"]
    # (main_dy, desc_dy): top vertex labels sit above, bottom vertices below
    specs = [(-31, -16), (24, 39), (24, 39)]
    for i, (mdy, ddy) in enumerate(specs):
        svg.text(pts[i][0], pts[i][1] + mdy, LABELS[i][0], fill=sec, font_size=12.5,
                 font_weight=600, text_anchor="middle", dominant_baseline="middle")
        svg.text(pts[i][0], pts[i][1] + ddy, LABELS[i][1], fill=muted, font_size=10,
                 text_anchor="middle", dominant_baseline="middle")


def build() -> str:
    pal = palette.load()
    svg = sf.new_canvas(WIDTH, HEIGHT, pal)
    # left: vector field
    draw_frame2(svg, PANEL_A, pal)
    sf.draw_vector_field(svg, PANEL_A, R, pal)
    # right: trajectories + init line + colourbar
    draw_frame2(svg, PANEL_B, pal)
    sf.draw_init_line(svg, PANEL_B, pal, INIT_PI1, "π₁ = 0.9")
    for pi2 in INIT_PI2:
        traj = dynamics.trajectory(np.log([INIT_PI1, pi2, 1.0 - INIT_PI1 - pi2]),
                                   R, eta=ETA, steps=STEPS, mode="full")
        sf.draw_time_trajectory(svg, PANEL_B, traj, pal, width=3.2)
    sf.draw_colourbar(svg, pal, CB_X, TOP, TRI_H, width=CBW)
    return svg.tostring()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dest = OUT / "unequal-reward-labelled.svg"
    dest.write_text(build(), encoding="utf-8")
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")
    # sanity: report which basin each init lands in
    for pi2 in INIT_PI2:
        tr = dynamics.trajectory(np.log([INIT_PI1, pi2, 1 - INIT_PI1 - pi2]), R,
                                 eta=ETA, steps=STEPS, mode="full")
        f = tr[-1]
        print(f"  pi2={pi2}: final ({f[0]:.3f},{f[1]:.3f},{f[2]:.3f}) -> "
              f"{'faithful a2' if f[1] > f[2] else 'hacking a3'}")


if __name__ == "__main__":
    main()
