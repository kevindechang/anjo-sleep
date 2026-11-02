// Small DOM helpers shared by the judge pages. Every piece of server or model text goes in through
// textContent, never as HTML.

export const byId = (id) => document.getElementById(id);
export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

// A status line: text, and a kind ("error", "ok" or none) its CSS colours.
export function say(node, text, kind = "") {
  node.textContent = text || "";
  node.dataset.kind = kind;
}

// A short date, as the English pages write it: "Nov 9".
export function day(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso).slice(0, 10);
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export const percent = (value) => (typeof value === "number" && Number.isFinite(value) ? Math.round(value * 100) : null);
