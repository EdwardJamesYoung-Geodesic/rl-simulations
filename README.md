# rl-dynamics

> 📄 **[Read the full write-up →](./write-up.md)**

Simulations, figures, and interactive widgets for a blog post on **a toy model of
finite-sample RL learning dynamics** — the linear-softmax policy, RLOO updates,
the effect of initialisation, equal-reward policy drift, and Gram-matrix coupling.

## What's here

| Path | What it is |
|------|------------|
| `src/rl_dynamics/` | The simulation **library** — pure machinery, no plotting. |
| `figures/` | Scripts that import the library and emit **static SVG** figures into `figures/out/`. |
| `widgets/` | Self-contained **interactive HTML** widgets (a small JS core mirrors the library's maths). |
| `config/palette.yaml` | The shared **Wong (2011) colourblind-safe** colour palette — single source of colour truth. |
| `tests/` | Correctness + reproducibility tests. |

### The library

- `policy` — softmax policy; full/expected gradient; finite-sample **RLOO** estimator; the logit-space step `L ← L + η·AAᵀ·ĝ`.
- `dynamics` — trajectory integration (deterministic `full` and stochastic `rloo`), vector fields, a seeded RNG.
- `simplex` — `N=3` barycentric geometry (policies as points in a triangle; probabilities as orthogonal distances).
- `palette` — loads `config/palette.yaml`.
- `svg` — renderer-agnostic SVG string builders (shared conventions with the widgets' `render.js`).

## Reproducing everything

Requires [uv](https://docs.astral.sh/uv/). From a clean checkout:

```bash
uv sync                       # create the environment
uv run pytest                 # run the test suite
uv run python figures/<name>.py   # (re)generate a static figure into figures/out/
uv run python widgets/build.py    # build self-contained widgets into widgets/dist/
```

Static figures are designed for a **light background**; interactive widgets may adapt to a dark theme.

## License

Released under the [MIT License](./LICENSE) © 2026 Edward Young.
