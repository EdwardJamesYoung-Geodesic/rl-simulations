"""Figure: stochastic vs deterministic cognition, when two cognitions share (in
distribution) the same high-reward action.

Cognition model with 3 cognitions and 2 actions, matching the write-up's soft P:
cognition 1 picks action 1 (reward 0) w.p. 0.9, while cognitions 2 and 3 pick
action 2 (reward 1) w.p. 0.9. The effective cognition rewards are therefore
(0.1, 0.9, 0.9): the symmetric-triangle case, now over the *cognition* simplex,
with c2 ("right reasons") and c3 ("wrong reasons") indistinguishable in
distribution. Every rollout starts at the symmetric s = (0.9, 0.05, 0.05) and
runs finite-sample RLOO (G=8, eta=1.2, steps=25); coloured by time.

* left  -- **stochastic** cognition (cognition sampled): each rollout reinforces
           the cognition it drew, so sampling noise breaks the 2<->3 symmetry and
           runs commit to one strategy (rich-get-richer among cognitions).
* right -- **deterministic** cognition (cognition marginalised): credit for the
           shared action is split by the posterior, so cognitions 2 and 3 move
           together and the symmetry is preserved -- which cognition was "used"
           never gets pinned down.

Run: ``uv run python figures/cognition.py``  ->  figures/out/cognition.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import simplexfig as sf
from rl_dynamics import cognition, dynamics, palette

OUT = Path(__file__).resolve().parent / "out"

P = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.9]])   # actions x cognitions (soft)
R = np.array([0.0, 1.0])                            # action rewards
START = [0.9, 0.05, 0.05]                           # symmetric cognition init
ETA, STEPS, G, SEED = 1.2, 25, 8, 0
N_TRAJ = 30

VERT_LABELS = ["c₁", "c₂", "c₃"]
SUB_LABELS = ["low reward", "right reasons", "wrong reasons"]
PANELS = [("sampled", "stochastic cognition"), ("marginal", "deterministic cognition")]

# --- layout (px) --------------------------------------------------------------
SCALE, TOP, H = sf.SCALE, sf.TOP, sf.H
TRI_H = H * SCALE
LEFT, GAP = 44, 96
PANEL_A = sf.Panel(LEFT)
PANEL_B = sf.Panel(LEFT + SCALE + GAP)
CBAR_GAP, CBAR_W = 46, 14
CBAR_X = PANEL_B.ox + SCALE + CBAR_GAP
WIDTH = CBAR_X + CBAR_W + 60
HEIGHT = TOP + TRI_H + 78


def draw_panel(svg, panel, pal, mode, caption) -> None:
    sf.draw_frame(svg, panel, pal, VERT_LABELS)
    # secondary vertex annotations (alignment interpretation), just outside each label
    pts = panel.triangle_pts()
    for i, ((dx, dy), lab) in enumerate(zip([(0, -31), (-2, 42), (2, 42)], SUB_LABELS)):
        svg.text(pts[i][0] + dx, pts[i][1] + dy, lab, fill=pal.ink["muted"],
                 font_size=11, text_anchor="middle", dominant_baseline="middle")
    rng = dynamics.rng(SEED)
    theta0 = np.log(START)
    for _ in range(N_TRAJ):
        traj = cognition.trajectory(P, theta0, R, eta=ETA, steps=STEPS, mode=mode,
                                    G=G, rng=rng)
        sf.draw_time_trajectory(svg, panel, traj, pal, width=2.0, samples=STEPS + 1,
                                opacity=0.85, start_dot=False)
    sx, sy = panel.proj(START)
    svg.circle(sx, sy, 3.4, fill=palette.ramp(pal.sequential["time"], 0.0),
               stroke=pal.surface, stroke_width=1.2)
    svg.text(panel.ox + SCALE / 2, TOP + TRI_H + 54, caption, fill=pal.ink["secondary"],
             font_size=13, text_anchor="middle")


def build() -> str:
    pal = palette.load()
    svg = sf.new_canvas(WIDTH, HEIGHT, pal)
    for panel, (mode, caption) in zip((PANEL_A, PANEL_B), PANELS):
        draw_panel(svg, panel, pal, mode, caption)
    sf.draw_colourbar(svg, pal, CBAR_X, TOP, TRI_H, width=CBAR_W)
    return svg.tostring()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dest = OUT / "cognition.svg"
    dest.write_text(build(), encoding="utf-8")
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
