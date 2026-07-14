"""Figure: ID-identical, OOD-divergent cognition under deterministic PG.

Three rows, one per initial aligned-cognition mass c2 in {0.06, 0.05, 0.04}
(aligned-leading -> misaligned-leading; c1 = 0.9 fixed, misaligned c3 = 0.1 - c2).
Each row shows, left to right:

* the cognition simplex (c1 Ineffective / c2 Aligned / c3 Misaligned) with the
  deterministic policy-gradient trajectory, time-coloured;
* "ID action choice": the marginal action probability P^ID c(t) plotted as
  horizontal position (a1 failure -> a2 success) descending in time;
* "OOD action choice": the SAME cognition trajectory re-projected through P^OOD.

Training is on the ID task only (rewards R = (0, 1)); OOD is evaluated, not
trained. Because the ID reward can't tell c2 (right reasons) from c3 (wrong
reasons), every row looks near-identical and successful ID -- but OOD success is
set entirely by the initial c2:c3 split. Each row carries its own time colourbar.

Run: ``uv run python figures/cognition_ood.py``  ->  figures/out/cognition-ood.svg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_dynamics import cognition, palette, simplex
from rl_dynamics.svg import SVG

OUT = Path(__file__).resolve().parent / "out"

# --- problem ------------------------------------------------------------------
P_ID = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.9]])
P_OOD = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1]])
R = np.array([0.0, 1.0])
C2_STARTS = [0.06, 0.05, 0.04]           # aligned-cognition initial mass (aligned -> misaligned)
C1_START = 0.9
ETA, STEPS = 0.3, 110
A2 = 1                                   # success action index

VERT_WORD = ["Ineffective", "Aligned", "Misaligned"]   # c1 (top), c2 (bl), c3 (br)
VERT_ID = ["c₁", "c₂", "c₃"]

# --- layout (px) --------------------------------------------------------------
H = simplex.HEIGHT
LEFT = 48
S = 185                                  # triangle width
TRI_H = H * S
RH = TRI_H                               # action-panel height = triangle height
BAND = TRI_H
GAP_TRI_ID = 74
RW = 130
GAP_ID_OOD = 78
GAP_OOD_CB = 60
CBW = 13
RIGHT = 60

COL_TRI = LEFT
COL_ID = LEFT + S + GAP_TRI_ID
COL_OOD = COL_ID + RW + GAP_ID_OOD
CB_X = COL_OOD + RW + GAP_OOD_CB
WIDTH = CB_X + CBW + RIGHT

TOP = 88
TITLE_Y = 26
ROW_H = BAND + 70
HEIGHT = TOP + len(C2_STARTS) * ROW_H + 8


def content_top(r):
    return TOP + r * ROW_H


def tri_proj(c, oy):
    xy = np.asarray(c, float) @ simplex.VERTICES
    return (COL_TRI + xy[0] * S, oy + (H - xy[1]) * S)


def time_poly(svg, pts, pal, width):
    stops = pal.sequential["time"]
    n = len(pts) - 1
    for k in range(n):
        svg.line(*pts[k], *pts[k + 1], stroke=palette.ramp(stops, (k + 0.5) / n),
                 stroke_width=width, stroke_linecap="round", stroke_linejoin="round")


def build() -> str:
    pal = palette.load()
    edge, ink, sec, muted = (pal.structure["edge"], pal.ink["primary"],
                             pal.ink["secondary"], pal.ink["muted"])
    stops = pal.sequential["time"]
    svg = SVG(WIDTH, HEIGHT, background=pal.surface)
    svg.raw(f"<style>text{{font-family:{pal.font['serif']}}}</style>")
    svg.raw('<defs><linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">'
            + "".join(f'<stop offset="{o}" stop-color="{palette.ramp(stops, o)}" />'
                      for o in (0, 0.25, 0.5, 0.75, 1.0)) + "</linearGradient></defs>")

    # column titles (top only)
    for cx, title in [(COL_TRI + S / 2, "Cognition"), (COL_ID + RW / 2, "ID action choice"),
                      (COL_OOD + RW / 2, "OOD action choice")]:
        svg.text(cx, TITLE_Y, title, fill=ink, font_size=14, font_weight=600,
                 text_anchor="middle")

    for r, c2 in enumerate(C2_STARTS):
        ct = content_top(r)
        c0 = np.array([C1_START, c2, round(0.1 - c2, 4)])
        traj = cognition.trajectory(P_ID, np.log(c0), R, eta=ETA, steps=STEPS, mode="full")

        # ---- cognition simplex ----
        vpx = [tri_proj(np.eye(3)[i], ct) for i in range(3)]
        svg.polygon(vpx, fill=pal.surface, stroke=edge, stroke_width=1.5,
                    stroke_linejoin="round")
        tpts = [tri_proj(c, ct) for c in traj]
        time_poly(svg, tpts, pal, width=2.6)
        svg.circle(*tpts[0], 3.2, fill=palette.ramp(stops, 0.0), stroke=pal.surface,
                   stroke_width=1.1)

        # two-line vertex labels (Figure-5 style: c-id nearest vertex, word outside)
        # top: word above id;  bottom: id above word
        specs = [
            (vpx[0][0], vpx[0][1], -26, -12),   # apex: word at -26, id at -12
            (vpx[1][0] - 6, vpx[1][1], 28, 14),  # bl: id at +14, word at +28
            (vpx[2][0] + 6, vpx[2][1], 28, 14),  # br
        ]
        for i, (lx, ly, wdy, idy) in enumerate(specs):
            svg.text(lx, ly + idy, VERT_ID[i], fill=sec, font_size=12.5, font_weight=600,
                     text_anchor="middle", dominant_baseline="middle")
            svg.text(lx, ly + wdy, VERT_WORD[i], fill=muted, font_size=10.5,
                     text_anchor="middle", dominant_baseline="middle")

        # start-position label, annotating the dot (right of the c1=0.9 line)
        svg.text(tpts[0][0] + 14, tpts[0][1], f"c₂ = {c2:g}", fill=sec,
                 font_size=12, text_anchor="start", dominant_baseline="middle")

        # ---- ID / OOD action panels ----
        ycen = ct + RH / 2
        for col_x, Pm in [(COL_ID, P_ID), (COL_OOD, P_OOD)]:
            x_act = (traj @ Pm.T)[:, A2]                     # P(a2) over time
            pts = [(col_x + x * RW, ct + t / STEPS * RH) for t, x in enumerate(x_act)]
            ax_b = ct + RH
            # bottom action axis only (no bounding box, no full time axis)
            svg.line(col_x, ax_b, col_x + RW, ax_b, stroke=edge, stroke_width=1.3)
            # short "time" arrow, mid-panel, not touching the action axis
            axx = col_x - 13
            svg.line(axx, ycen - 20, axx, ycen + 16, stroke=muted, stroke_width=1.2)
            svg.polygon([(axx, ycen + 22), (axx - 3, ycen + 15), (axx + 3, ycen + 15)], fill=muted)
            svg.text(axx - 9, ycen, "time", fill=muted, font_size=11, text_anchor="middle",
                     dominant_baseline="middle", transform=f"rotate(-90 {axx - 9} {ycen})")
            # trajectory + start dot
            time_poly(svg, pts, pal, width=2.8)
            svg.circle(*pts[0], 3.2, fill=palette.ramp(stops, 0.0), stroke=pal.surface,
                       stroke_width=1.1)
            # x-axis end labels + sublabels
            svg.text(col_x, ax_b + 15, "a₁", fill=sec, font_size=12, text_anchor="middle")
            svg.text(col_x + RW, ax_b + 15, "a₂", fill=sec, font_size=12, text_anchor="middle")
            svg.text(col_x, ax_b + 29, "task failure", fill=muted, font_size=9.5, text_anchor="middle")
            svg.text(col_x + RW, ax_b + 29, "task success", fill=muted, font_size=9.5, text_anchor="middle")

        # ---- per-row time colourbar ----
        svg.raw(f'<rect x="{CB_X}" y="{ct}" width="{CBW}" height="{RH}" fill="url(#tg)" '
                f'stroke="{edge}" stroke-width="1" rx="{CBW/2}" ry="{CBW/2}" />')
        svg.text(CB_X + CBW / 2, ct - 8, "early", fill=muted, font_size=10, text_anchor="middle")
        svg.text(CB_X + CBW / 2, ct + RH + 14, "late", fill=muted, font_size=10, text_anchor="middle")

    return svg.tostring()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dest = OUT / "cognition-ood.svg"
    dest.write_text(build(), encoding="utf-8")
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
