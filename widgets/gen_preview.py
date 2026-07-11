"""Render an animated-GIF preview of the finite-sample drift widget.

Mirrors the widget's default state -- R = (0, 1, 1), start (0.9, 0.05, 0.05),
G = 8, eta = 1.2, 25 steps, 20 rollouts -- and animates the rollouts (time
coloured) plus the deterministic full-gradient path (blue) growing out of the
start point, for embedding in the write-up where the live widget can't run
(e.g. GitHub's rendered markdown).

Run: ``uv run --with pillow python widgets/gen_preview.py``
  -> widgets/preview/finite-sample.gif
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl_dynamics import dynamics, palette, simplex

OUT = Path(__file__).resolve().parent / "preview"

# problem setting (widget defaults)
R = np.array([0.0, 1.0, 1.0])
START = [0.9, 0.05, 0.05]
G, ETA, STEPS, N, SEED = 8, 1.2, 25, 20, 3
VLABELS = ["R₁ = 0", "R₂ = 1", "R₃ = 1"]

# raster geometry (supersampled, then downscaled for smoothness)
SIZE, SS = 460, 3
BIG = SIZE * SS
PAD = 58 * SS
H = simplex.HEIGHT
SCALE = BIG - 2 * PAD
TRI_H = H * SCALE
OX = PAD
OY = PAD + (BIG - 2 * PAD - TRI_H) / 2


def proj(pi):
    xy = np.asarray(pi, float) @ simplex.VERTICES
    return (OX + xy[0] * SCALE, OY + (H - xy[1]) * SCALE)


def load_font(px):
    for p in ("/System/Library/Fonts/Supplemental/Palatino.ttc",
              "/System/Library/Fonts/Supplemental/Georgia.ttf",
              "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
              "/System/Library/Fonts/NewYork.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, px)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> None:
    pal = palette.load()
    stops = pal.sequential["time"]
    blue = palette.hex_to_rgb(pal.wong["blue"])
    edge = palette.hex_to_rgb(pal.structure["edge"])
    inksec = palette.hex_to_rgb(pal.ink["secondary"])
    inkpri = palette.hex_to_rgb(pal.ink["primary"])
    surf = palette.hex_to_rgb(pal.surface)
    font = load_font(int(13 * SS))

    # precompute rollouts + deterministic path as pixel points
    rng = dynamics.rng(SEED)
    rolls = [np.array([proj(p) for p in
                       dynamics.trajectory(np.log(START), R, eta=ETA, steps=STEPS,
                                           mode="rloo", G=G, rng=rng)])
             for _ in range(N)]
    det = np.array([proj(p) for p in
                    dynamics.trajectory(np.log(START), R, eta=ETA, steps=STEPS, mode="full")])
    Vpix = [proj(np.eye(3)[i]) for i in range(3)]
    start_px = proj(START)

    def seg_colour(s):
        return tuple(palette.hex_to_rgb(palette.ramp(stops, (s + 0.5) / STEPS)))

    def frame(shown: int) -> Image.Image:
        img = Image.new("RGBA", (BIG, BIG), surf + (255,))
        d = ImageDraw.Draw(img, "RGBA")
        # triangle
        d.line([Vpix[0], Vpix[1], Vpix[2], Vpix[0]], fill=edge + (255,),
               width=int(1.75 * SS), joint="curve")
        # vertex labels
        offs = [(0, -20 * SS), (-8 * SS, 26 * SS), (8 * SS, 26 * SS)]
        for i, (dx, dy) in enumerate(offs):
            d.text((Vpix[i][0] + dx, Vpix[i][1] + dy), VLABELS[i], font=font,
                   fill=inksec + (255,), anchor="mm")
        # stochastic rollouts (translucent, time-coloured), grown to `shown`
        for tr in rolls:
            for s in range(min(shown, STEPS)):
                d.line([tuple(tr[s]), tuple(tr[s + 1])], fill=seg_colour(s) + (210,),
                       width=int(1.8 * SS), joint="curve")
        # deterministic full-gradient path (opaque blue) on top
        if shown >= 1:
            d.line([tuple(p) for p in det[:shown + 1]], fill=blue + (255,),
                   width=int(3 * SS), joint="curve")
        # start marker
        r = 7 * SS
        d.ellipse([start_px[0] - r, start_px[1] - r, start_px[0] + r, start_px[1] + r],
                  fill=tuple(palette.hex_to_rgb(palette.ramp(stops, 0.0))) + (255,),
                  outline=surf + (255,), width=int(3 * SS))
        return img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)

    # frames: hold on the bare start, grow, hold on the final spread
    frames = [frame(0)] + [frame(s) for s in range(0, STEPS + 1)] + [frame(STEPS)]
    durations = [700] + [60] * (STEPS + 1) + [1500]

    # single shared palette (from the busiest frame) -> no inter-frame flicker
    master = frames[-2].convert("P", palette=Image.ADAPTIVE, colors=192)
    pframes = [f.quantize(palette=master, dither=Image.NONE) for f in frames]

    OUT.mkdir(exist_ok=True)
    dest = OUT / "finite-sample.gif"
    pframes[0].save(dest, save_all=True, append_images=pframes[1:],
                    duration=durations, loop=0, optimize=True, disposal=2)
    still = OUT / "finite-sample-still.png"
    frames[-2].save(still)
    print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes, "
          f"{len(pframes)} frames)")
    print(f"wrote {still.relative_to(OUT.parent.parent)}  (final-frame still for review)")


if __name__ == "__main__":
    main()
