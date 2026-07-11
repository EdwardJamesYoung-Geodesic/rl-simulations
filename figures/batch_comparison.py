"""Figure: how batch size vs step size shapes finite-sample drift.

Two simplexes side by side, each with 20 stochastic RLOO rollouts from the
symmetric start pi = (0.9, 0.05, 0.05), coloured by time. Both hold the expected
convergence fixed (eta * steps = 30) but trade batch size against step size:

* left  -- G = 32, eta = 0.3,  steps = 100  (large batch, small steps: tight);
* right -- G = 8,  eta = 1.2,  steps = 25   (small batch, big steps: much wider drift).

Below each triangle, a density of the final policy's x-position (x = 0 at action
2 / vertex 2, x = 1 at action 3 / vertex 3) shows the spread of committed
outcomes: the plotted rollouts are a 20-run subset, the density is estimated from
400 runs, and both panels share a common vertical scale.

Run: ``uv run python figures/batch_comparison.py``  ->  figures/out/batch-comparison.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import simplexfig as sf
from rl_dynamics import dynamics, palette, simplex

OUT = Path(__file__).resolve().parent / "out"

R = np.array([0.0, 1.0, 1.0])
START = [0.9, 0.05, 0.05]
N_TRAJ, N_DENS, SEED = 20, 400, 0
BW = 0.035  # KDE bandwidth

# (eta, steps, G, label)
SETTINGS = [
    (0.3, 100, 32, "G = 32,  η = 0.3,  steps = 100"),
    (1.2, 25, 8, "G = 8,  η = 1.2,  steps = 25"),
]

# --- layout (px) --------------------------------------------------------------
SCALE, TOP, H = sf.SCALE, sf.TOP, sf.H
TRI_H = H * SCALE
LEFT, GAP = 44, 96
PANEL_A = sf.Panel(LEFT)
PANEL_B = sf.Panel(LEFT + SCALE + GAP)
CBAR_GAP, CBAR_W = 46, 14
CBAR_X = PANEL_B.ox + SCALE + CBAR_GAP
WIDTH = CBAR_X + CBAR_W + 60
# density strip sits below the triangle (clearing the bottom vertex labels)
DENS_TOP = TOP + TRI_H + 44
DENS_H = 54
BASELINE = DENS_TOP + DENS_H
LABEL_Y = BASELINE + 30
HEIGHT = LABEL_Y + 16


def kde(xs: np.ndarray, grid: np.ndarray) -> np.ndarray:
    d = np.zeros_like(grid)
    for xi in xs:
        d += np.exp(-0.5 * ((grid - xi) / BW) ** 2)
    return d / (len(xs) * BW * np.sqrt(2 * np.pi))


def compute(eta: float, steps: int, G: int) -> dict:
    rng = dynamics.rng(SEED)
    sl = np.log(START)
    plot_trajs, finals_x = [], []
    for k in range(N_DENS):
        traj = dynamics.trajectory(sl, R, eta=eta, steps=steps, mode="rloo", G=G, rng=rng)
        finals_x.append(simplex.to_xy(traj[-1])[0])
        if k < N_TRAJ:
            plot_trajs.append(traj)
    return {"trajs": plot_trajs, "finals": np.array(finals_x)}


def draw_panel(svg, panel, pal, trajs, label) -> None:
    sf.draw_frame(svg, panel, pal)
    for traj in trajs:
        sf.draw_time_trajectory(svg, panel, traj, pal, width=2.0,
                                samples=len(traj), opacity=0.85, start_dot=False)
    sx, sy = panel.proj(START)
    svg.circle(sx, sy, 3.4, fill=palette.ramp(pal.sequential["time"], 0.0),
               stroke=pal.surface, stroke_width=1.2)
    svg.text(panel.ox + SCALE / 2, LABEL_Y, label, fill=pal.ink["secondary"],
             font_size=13, text_anchor="middle")


def draw_density(svg, panel, pal, dens, grid, dmax) -> None:
    muted, edge = pal.ink["muted"], pal.structure["edge"]

    def px(x):
        return panel.ox + x * SCALE

    def py(v):
        return BASELINE - (v / dmax) * DENS_H

    poly = [(px(0.0), BASELINE)]
    poly += [(px(x), py(v)) for x, v in zip(grid, dens)]
    poly += [(px(1.0), BASELINE)]
    svg.polygon(poly, fill=pal.structure["field"], fill_opacity=0.45,
                stroke=muted, stroke_width=1.0, stroke_linejoin="round")
    svg.line(px(0.0), BASELINE, px(1.0), BASELINE, stroke=edge, stroke_width=1.0)


def build() -> str:
    pal = palette.load()
    svg = sf.new_canvas(WIDTH, HEIGHT, pal)
    panels = (PANEL_A, PANEL_B)

    data = [compute(eta, steps, G) for (eta, steps, G, _) in SETTINGS]
    grid = np.linspace(0.0, 1.0, 160)
    densities = [kde(d["finals"], grid) for d in data]
    dmax = max(dens.max() for dens in densities)

    for panel, (eta, steps, G, label), d, dens in zip(panels, SETTINGS, data, densities):
        draw_panel(svg, panel, pal, d["trajs"], label)
        draw_density(svg, panel, pal, dens, grid, dmax)

    sf.draw_colourbar(svg, pal, CBAR_X, TOP, TRI_H, width=CBAR_W)
    return svg.tostring()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dest = OUT / "batch-comparison.svg"
    dest.write_text(build(), encoding="utf-8")
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
