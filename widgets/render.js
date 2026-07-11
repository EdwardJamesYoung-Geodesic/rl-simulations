/*
 * rl-dynamics — SVG string builders for widgets.
 *
 * Mirrors the conventions of the Python `rl_dynamics.svg` module: each helper
 * returns an SVG element as a string, which the widget assembles and injects
 * via innerHTML. Keeps rendering declarative and re-runnable on every state
 * change.
 */
const RLRender = (function () {
  "use strict";

  function fmt(x, ndigits) {
    if (ndigits === undefined) ndigits = 3;
    if (x === 0) x = 0;
    let s = x.toFixed(ndigits).replace(/\.?0+$/, "");
    return s === "" || s === "-" ? "0" : s;
  }

  function attrs(o) {
    const parts = [];
    for (const key in o) {
      const val = o[key];
      if (val === null || val === undefined) continue;
      const name = key.replace(/_/g, "-");
      parts.push(`${name}="${typeof val === "number" ? fmt(val) : val}"`);
    }
    return parts.join(" ");
  }

  const pts = (arr) => arr.map((p) => `${fmt(p[0])},${fmt(p[1])}`).join(" ");

  return {
    fmt,
    attrs,
    line: (x1, y1, x2, y2, o) =>
      `<line ${attrs(Object.assign({ x1, y1, x2, y2 }, o))} />`,
    polyline: (arr, o) =>
      `<polyline points="${pts(arr)}" ${attrs(Object.assign({ fill: "none" }, o))} />`,
    polygon: (arr, o) => `<polygon points="${pts(arr)}" ${attrs(o)} />`,
    path: (d, o) => `<path d="${d}" ${attrs(o)} />`,
    circle: (cx, cy, r, o) =>
      `<circle ${attrs(Object.assign({ cx, cy, r }, o))} />`,
    text: (x, y, s, o) => `<text ${attrs(Object.assign({ x, y }, o))}>${s}</text>`,
    g: (body, o) => `<g ${attrs(o)}>${body}</g>`,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = RLRender;
