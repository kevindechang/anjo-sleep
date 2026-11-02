// "Since your last visit": the morning note of her line of thought (``morning`` in GET
// /api/sleep/me/thought): what settled, her new hunches, where her lead shifted and what she
// concluded. It opens the wake view, and the chat on the next visit. Also the two small pieces the
// thought panel shares with it: an item's lines (plain first, her exact words second) and a
// settled bet's verdict.

import { COPY, fill } from "./copy.js";
import { byId, day, el } from "./dom.js";

const list = (value) => (Array.isArray(value) ? value : []);

// An item's two lines: her plain sentence, then her exact words (or a note that they're kept back).
export function said(item, words) {
  const nodes = [el("p", "plain", String((item && item.plain) || ""))];
  const exact = item && typeof item.exact === "string" ? item.exact : "";
  if (exact) nodes.push(el("p", "exact", exact));
  else if (words && words.withheld) nodes.push(el("p", "exact withheld", COPY.mind_withheld));
  return nodes;
}

export function verdict(value) {
  const code = ["right", "wrong", "no_call"].includes(value) ? value : "no_call";
  return el("span", `chip verdict-${code}`, COPY[`verdict_${code}`]);
}

function group(title, level, items, draw) {
  if (!items.length) return null;
  const box = el("div", "morning-group");
  const rows = el("ul", "morning-list");
  items.forEach((item) => {
    const row = el("li", "morning-item");
    row.append(...draw(item));
    rows.append(row);
  });
  box.append(el(level, "morning-title", title), rows);
  return box;
}

// The note's content, or null when there is none to show: no note at all (a first visit), or an
// empty one unless ``always`` (the wake says "nothing new" rather than nothing).
export function morningParts(view, { level = "h4", always = false } = {}) {
  const note = view && view.morning;
  if (!note || typeof note !== "object" || (note.empty && !always)) return null;
  const parts = [el("p", "plain morning-plain", String(note.plain || ""))];
  if (note.since) parts.push(el("p", "sub", fill(COPY.since_from, { date: day(note.since) })));
  if (note.empty) return parts;
  const byKey = (items) => new Map(list(items).map((item) => [item.id, item]));
  const settled = byKey(view.settled);
  const hunches = byKey(view.hunches);
  const found = (ids, from) => list(ids).map((id) => from.get(id)).filter(Boolean);
  const groups = el("div", "morning-groups");
  groups.append(...[
    group(COPY.since_settled, level, found(note.settled, settled), (s) => [verdict(s.verdict), ...said(s, s.claim)]),
    group(COPY.since_new, level, found(note.new_hunches, hunches), (h) => said(h, h.claim)),
    group(COPY.since_shifts, level, list(note.shifts), (s) => said(s, s.now_leading)),
    group(COPY.since_findings, level, list(note.findings), (f) => said(f, f.statement)),
  ].filter(Boolean));
  return groups.childElementCount ? [...parts, groups] : parts;
}

// The card at the top of the chat on a visit with something new; the wake view takes over once
// it opens, since it carries the same note.
export function createSince() {
  const card = byId("since");
  const body = byId("since-body");
  const toggle = byId("since-hide");
  let waking = false;

  toggle.addEventListener("click", () => {
    const open = body.hidden;
    body.hidden = !open;
    toggle.textContent = open ? COPY.wake_hide : COPY.wake_show;
    toggle.setAttribute("aria-expanded", String(open));
  });

  return {
    update(view) {
      const parts = waking ? null : morningParts(view, { level: "h3" });
      card.hidden = !parts;
      if (parts) body.replaceChildren(...parts);
    },
    // The wake view is open: the note lives there now.
    handOver() {
      waking = true;
      card.hidden = true;
    },
  };
}
