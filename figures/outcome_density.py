"""Figure: outcome density vs initialisation (R = (0, 1, 1.2), high-drift RLOO).

Four panels side by side, all starting at pi_1 = 0.9 but leaning progressively
toward the worse action 2: pi_2 in {0.05, 0.06, 0.07, 0.08} (pi_3 = 0.1 - pi_2).
Each panel shows, at the high-drift setting (G = 8, eta = 1.2, steps = 25):

* the finite-sample RLOO rollouts, time-coloured (light orange -> dark red);
* the deterministic full-gradient path, as a thick blue line;
* below, a density of the final policy's x-position (x = 0 at action 2 / vertex 2,
  x = 1 at action 3 / vertex 3), with the deterministic endpoint marked in blue.

Densities share a common vertical scale across panels. The rollouts are a 35-run
subset drawn for legibility; the density is estimated from 400 runs.

Run: ``uv run python figures/outcome_density.py``  ->  figures/out/outcome-density-1.2.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import simplexfig as sf
from rl_dynamics import dynamics, palette, simplex

OUT = Path(__file__).resolve().parent / "out"

R = np.array([0.0, 1.0, 1.2])
PI1 = 0.9
PI2_VALUES = [0.05, 0.06, 0.07, 0.08]
ETA, STEPS, G, SEED = 1.2, 25, 8, 0
N_DENS, N_PLOT = 400, 35
BW = 0.035  # KDE bandwidth

# --- layout (px) --------------------------------------------------------------
SCALE, H = 200, sf.H
TRI_H = H * SCALE
LEFT, TOPM = 48, 34
COL_GAP = 30
COL_PITCH = SCALE + COL_GAP
DENS_TOP = TOPM + TRI_H + 8
DENS_H = 60
BASELINE = DENS_TOP + DENS_H
NCOL = len(PI2_VALUES)
CBAR_GAP, CBAR_W = 30, 14
PANELS_RIGHT = LEFT + NCOL * SCALE + (NCOL - 1) * COL_GAP
CBAR_X = PANELS_RIGHT + CBAR_GAP
WIDTH = CBAR_X + CBAR_W + 58
HEIGHT = BASELINE + 26


def kde(xs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    d = np.zeros_like(grid)
    for xi in xs:
        d += np.exp(-0.5 * ((grid - xi) / BW) ** 2)
    return d / (len(xs) * BW * np.sqrt(2 * np.pi))


def compute(p2: float) -> dict:
    start = np.log([PI1, p2, 1.0 - PI1 - p2])
    rng = dynamics.rng(SEED)
    plot_trajs, finals_x = [], []
    for k in range(N_DENS):
        traj = dynamics.trajectory(start, R, eta=ETA, steps=STEPS, mode="rloo",
                                   G=G, rng=rng)
        finals_x.append(simplex.to_xy(traj[-1])[0])
        if k < N_PLOT:
            plot_trajs.append(traj)
    det = dynamics.trajectory(start, R, eta=ETA, steps=STEPS, mode="full")
    return {"p2": p2, "trajs": plot_trajs, "finals": np.array(finals_x),
            "det": det, "det_x": float(simplex.to_xy(det[-1])[0])}


def build() -> str:
    pal = palette.load()
    svg = sf.new_canvas(WIDTH, HEIGHT, pal)
    stops = pal.sequential["time"]
    blue = pal.wong["blue"]
    ink, muted, edge = pal.ink["secondary"], pal.ink["muted"], pal.structure["edge"]

    data = [compute(p2) for p2 in PI2_VALUES]
    grid = np.linspace(0.0, 1.0, 160)
    densities = [kde(d["finals"], grid) for d in data]
    dmax = max(d.max() for d in densities)

    for c, (d, dens) in enumerate(zip(data, densities)):
        panel = sf.Panel(LEFT + c * COL_PITCH, TOPM, SCALE)
        sf.draw_triangle(svg, panel, pal, stroke_width=1.4)

        # stochastic rollouts (subset), then deterministic path on top
        for traj in d["trajs"]:
            sf.draw_time_trajectory(svg, panel, traj, pal, width=1.0,
                                    samples=STEPS + 1, opacity=0.5, start_dot=False)
        dpts = [panel.proj(p) for p in d["det"]]
        svg.polyline(dpts, stroke=blue, stroke_width=3.0, stroke_linecap="round",
                     stroke_linejoin="round")
        start = [PI1, d["p2"], 1.0 - PI1 - d["p2"]]
        svg.circle(*panel.proj(start), 2.4, fill=palette.ramp(stops, 0.0),
                   stroke=pal.surface, stroke_width=1.0)

        # column header (labelled by pi_3 = 1 - pi_1 - pi_2)
        p3 = round(1.0 - PI1 - d["p2"], 3)
        svg.text(panel.ox + SCALE / 2, TOPM - 14, f"π₃ = {p3:g}", fill=ink,
                 font_size=12.5, font_weight=600, text_anchor="middle")

        # density strip (shared vertical scale)
        def px(x, ox=panel.ox):
            return ox + x * SCALE

        def py(v):
            return BASELINE - (v / dmax) * DENS_H

        poly = [(px(0.0), BASELINE)]
        poly += [(px(x), py(v)) for x, v in zip(grid, dens)]
        poly += [(px(1.0), BASELINE)]
        svg.polygon(poly, fill=pal.structure["field"], fill_opacity=0.45,
                    stroke=muted, stroke_width=1.0, stroke_linejoin="round")
        svg.line(px(0.0), BASELINE, px(1.0), BASELINE, stroke=edge, stroke_width=1.0)
        # deterministic final position marked in blue
        svg.line(px(d["det_x"]), BASELINE, px(d["det_x"]), DENS_TOP, stroke=blue,
                 stroke_width=1.6)

    # outcome-axis labels once, under the leftmost density (x = final position).
    # The apex is action 1 (R₁=0), where every run starts (pi_1 = 0.9).
    svg.text(LEFT, BASELINE + 13, "R₂=1", fill=muted, font_size=9, text_anchor="start")
    svg.text(LEFT + SCALE, BASELINE + 13, "R₃=1.2", fill=muted, font_size=9,
             text_anchor="end")

    sf.draw_colourbar(svg, pal, CBAR_X, TOPM, TRI_H, width=CBAR_W)
    return svg.tostring()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dest = OUT / "outcome-density-1.2.svg"
    dest.write_text(build(), encoding="utf-8")
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
