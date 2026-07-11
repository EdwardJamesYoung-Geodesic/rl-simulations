/*
 * rl-dynamics — JS core.
 *
 * A small, dependency-free mirror of the deterministic maths in the Python
 * package `rl_dynamics` (policy gradients + N=3 simplex geometry), so the
 * interactive widgets compute the same quantities as the figure generators.
 *
 * The Python package is the source of truth; `RLDyn.selfTest(fixtures)` checks
 * this mirror against Python-generated fixtures at load time (see
 * widgets/gen_fixtures.py). Stochastic RLOO *sampling* is intentionally not
 * mirrored (RNG streams differ across implementations); the RLOO *formula* is,
 * via `rlooFromCounts`.
 */
const RLDyn = (function () {
  "use strict";

  // ---- policy / gradients ------------------------------------------------
  function softmax(L) {
    const m = Math.max.apply(null, L);
    const e = L.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  }

  function expectedReward(pi, R) {
    return pi.reduce((a, p, i) => a + p * R[i], 0);
  }

  function fullGradient(pi, R) {
    const er = expectedReward(pi, R);
    return pi.map((p, i) => p * (R[i] - er));
  }

  // ghat_i = (1/G) N_i (R_i - Rbar),  Rbar = (1/G) sum_i N_i R_i
  function rlooFromCounts(counts, R) {
    const G = counts.reduce((a, b) => a + b, 0);
    const Rbar = counts.reduce((a, c, i) => a + c * R[i], 0) / G;
    const ghat = counts.map((c, i) => (c * (R[i] - Rbar)) / G);
    return { ghat, Rbar };
  }

  function _matVec(M, v) {
    return M.map((row) => row.reduce((a, m, k) => a + m * v[k], 0));
  }
  function _matTVec(M, v) {
    // (M^T v)_k = sum_i M[i][k] v[i]
    const out = new Array(M[0].length).fill(0);
    for (let i = 0; i < M.length; i++)
      for (let k = 0; k < M[i].length; k++) out[k] += M[i][k] * v[i];
    return out;
  }

  // L <- L + eta A A^T ghat ; A === null is the identity fast path.
  function logitStep(L, ghat, eta, A) {
    if (!A) return L.map((l, i) => l + eta * ghat[i]);
    const coupled = _matVec(A, _matTVec(A, ghat));
    return L.map((l, i) => l + eta * coupled[i]);
  }

  // ---- N=3 simplex geometry ---------------------------------------------
  const H = Math.sqrt(3) / 2; // triangle height
  const V = [
    [0.5, H],
    [0.0, 0.0],
    [1.0, 0.0],
  ]; // vertices v1 (top), v2 (bottom-left), v3 (bottom-right)

  function toXY(pi) {
    return [
      pi[0] * V[0][0] + pi[1] * V[1][0] + pi[2] * V[2][0],
      pi[0] * V[0][1] + pi[1] * V[1][1] + pi[2] * V[2][1],
    ];
  }

  // Barycentric coordinates of (x, y) w.r.t. the reference triangle.
  function fromXY(x, y) {
    const [x1, y1] = V[0];
    const [x2, y2] = V[1];
    const [x3, y3] = V[2];
    const det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    const a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det;
    const b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det;
    return [a, b, 1 - a - b];
  }

  // Orthogonal distance to the edge opposite each vertex: pi_i * H.
  function edgeDistances(pi) {
    return pi.map((p) => p * H);
  }

  // Foot of the perpendicular from P onto the line through A and B.
  function footOnEdge(P, A, B) {
    const abx = B[0] - A[0],
      aby = B[1] - A[1];
    const t = ((P[0] - A[0]) * abx + (P[1] - A[1]) * aby) / (abx * abx + aby * aby);
    return [A[0] + t * abx, A[1] + t * aby];
  }

  // The edge (as [A, B] vertices) opposite vertex i.
  function oppositeEdge(i) {
    const others = [0, 1, 2].filter((k) => k !== i);
    return [V[others[0]], V[others[1]]];
  }

  // Project a point onto the probability simplex (clamp negatives, renormalise).
  function clampToSimplex(pi) {
    const c = pi.map((p) => Math.max(0, p));
    const s = c.reduce((a, b) => a + b, 0);
    if (s <= 0) return [1 / 3, 1 / 3, 1 / 3];
    return c.map((p) => p / s);
  }

  // ---- parity self-test --------------------------------------------------
  function _close(a, b, tol) {
    tol = tol || 1e-9;
    if (Array.isArray(a))
      return a.length === b.length && a.every((x, i) => _close(x, b[i], tol));
    return Math.abs(a - b) <= tol;
  }

  function selfTest(fx) {
    const fails = [];
    const check = (name, got, want) => {
      if (!_close(got, want, 1e-6)) fails.push({ name, got, want });
    };
    for (const c of fx.softmax) check("softmax", softmax(c.L), c.pi);
    for (const c of fx.fullGradient)
      check("fullGradient", fullGradient(c.pi, c.R), c.ghat);
    for (const c of fx.rlooFromCounts) {
      const r = rlooFromCounts(c.counts, c.R);
      check("rloo.ghat", r.ghat, c.ghat);
      check("rloo.Rbar", r.Rbar, c.Rbar);
    }
    for (const c of fx.logitStep)
      check("logitStep", logitStep(c.L, c.ghat, c.eta, c.A || null), c.out);
    for (const c of fx.toXY) check("toXY", toXY(c.pi), c.xy);
    for (const c of fx.fromXY) check("fromXY", fromXY(c.xy[0], c.xy[1]), c.pi);
    for (const c of fx.edgeDistances)
      check("edgeDistances", edgeDistances(c.pi), c.d);
    const ok = fails.length === 0;
    if (ok) console.log("[RLDyn] self-test passed (JS matches Python reference)");
    else console.error("[RLDyn] self-test FAILED", fails);
    return { ok, fails };
  }

  return {
    softmax,
    expectedReward,
    fullGradient,
    rlooFromCounts,
    logitStep,
    H,
    V,
    toXY,
    fromXY,
    edgeDistances,
    footOnEdge,
    oppositeEdge,
    clampToSimplex,
    selfTest,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = RLDyn;
