"""Figure: deterministic dynamics with a broken tie, rewards R = (0, 1, R3).

Identical in every way to ``equal_reward.py`` except the reward vector: action 3
is better than action 2 by a margin ``R3 - 1``. Reusing that module's
parameterised ``build``, this renders one figure per value of ``R3`` so the
effect of a larger reward gap can be compared directly. A bigger gap pulls more
of the fan into vertex 3 within the same finite integration time.

Run: ``uv run python figures/unequal_reward.py``
  -> figures/out/unequal-reward-<R3>.svg  for each R3 in R3_VALUES
"""

from __future__ import annotations

from equal_reward import OUT, build  # reuse the deterministic two-panel builder

R3_VALUES = [1.2]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for r3 in R3_VALUES:
        labels = ["R₁ = 0", "R₂ = 1", f"R₃ = {r3:g}"]
        dest = OUT / f"unequal-reward-{r3:g}.svg"
        dest.write_text(build([0.0, 1.0, r3], labels), encoding="utf-8")
        print(f"wrote {dest.relative_to(OUT.parent.parent)}  ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
