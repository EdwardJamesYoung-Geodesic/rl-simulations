"""A toy model of finite-sample RL learning dynamics.

Pure simulation + geometry machinery (no plotting). Figure generators live in
the top-level ``figures/`` directory and interactive widgets in ``widgets/``.

Submodules:

* :mod:`rl_dynamics.policy`    -- softmax policy, full & RLOO gradients, logit step
* :mod:`rl_dynamics.dynamics`  -- trajectory integration, vector fields, seeded RNG
* :mod:`rl_dynamics.cognition` -- cognition model (sampled vs marginal estimators)
* :mod:`rl_dynamics.simplex`   -- N=3 barycentric geometry
* :mod:`rl_dynamics.palette`   -- load the shared Wong colour palette
* :mod:`rl_dynamics.svg`       -- renderer-agnostic SVG string builders
"""

from __future__ import annotations

from . import cognition, dynamics, palette, policy, simplex, svg

__all__ = ["policy", "dynamics", "cognition", "simplex", "palette", "svg"]
__version__ = "0.1.0"
