// The MIND panel: GET /api/sleep/me/mind, in the shape of demo/panel.py's mind_view (questions with
// hypotheses, credences and predictions; expectations; wants; each night's choice). Every item says
// a plain line first and its own stored words second. Items it hasn't shown before arrive with a
// subtle entrance and a "New" mark; a credence that moved slides from its old value to the new one.

import { api } from "./api.js";
import { COPY, fill } from "./copy.js";
import { byId, day, el, percent } from "./dom.js";

const NEW_MARK_MS = 7000;
const STATUS_CHIP = { open: "research", active: "live", parked: "muted" };

const section = (value) => (value && Array.isArray(value.items) ? value : { state: "none", items: [] });
const word = (prefix, code, fallback = "") => (code && COPY[`${prefix}_${code}`]) || fallback || code || "";

// Free text from its state, as {text, withheld} (a plain string is taken as it is).
function exact(item, className = "exact") {
  if (item && typeof item === "object" && item.withheld) return el("p", `${className} withheld`, COPY.mind_withheld);
  const text = item && typeof item === "object" ? item.text : item;
  return text ? el("p", className, String(text)) : null;
}

function chip(text, kind) {
  return el("span", `chip ${kind || ""}`.trim(), text);
}

// What the question as a whole puts on a prediction: the credences weighted by P(claim | answer).
export function overall(hypotheses, given) {
  if (!Array.isArray(given) || !given.length) return null;
  let sum = 0;
  let seen = false;
  hypotheses.forEach((h, i) => {
    if (typeof given[i] === "number" && typeof h.credence === "number") {
      sum += h.credence * given[i];
      seen = true;
    }
  });
  return seen ? sum : null;
}

export function createPanel({ onNew }) {
  const body = byId("mind-body");
  const updated = byId("mind-updated");
  let seen = null;               // item keys shown so far; null before the first render
  let credences = new Map();     // hypothesis id -> credence last shown
  let inflight = null;

  function item(key, className, fresh) {
    const node = el("li", `item ${className || ""}`.trim());
    if (fresh.has(key)) {
      node.classList.add("is-new");
      node.append(el("span", "new-mark", COPY.mind_new));
    }
    return node;
  }

  function bar(value, from) {
    const track = el("span", "bar");
    track.setAttribute("aria-hidden", "true");
    const fill_ = el("span", "bar-fill");
    const target = Math.max(0, Math.min(1, typeof value === "number" ? value : 0));
    if (typeof from === "number" && from !== target) {
      fill_.style.setProperty("--v", String(from));
      window.requestAnimationFrame(() => window.requestAnimationFrame(() => fill_.style.setProperty("--v", String(target))));
    } else {
      fill_.style.setProperty("--v", String(target));
    }
    track.append(fill_);
    return track;
  }

  function block(title, count, state, none) {
    const box = el("section", "mind-section");
    const head = el("h3", "", title);
    if (count) head.append(el("span", "count", String(count)));
    box.append(head);
    if (state === "unavailable") box.append(el("p", "none", COPY.mind_unavailable));
    else if (none) box.append(el("p", "none", none));
    return box;
  }

  function pick(choices, wants, fresh) {
    const latest = choices.items[0];
    const box = block(COPY.mind_pick, 0, choices.state, latest ? "" : COPY.mind_none_pick);
    if (!latest) return box;
    const list = el("ul", "mind-list");
    const key = `pick:${latest.night}:${latest.action}`;
    const row = item(key, "pick", fresh);
    const action = word("pick", latest.action, COPY.pick_other);
    const parts = [latest.action ? fill(COPY.pick_line, { n: latest.night, action }) : fill(COPY.pick_nothing, { n: latest.night })];
    if (latest.outcome) parts.push(word("pick_outcome", latest.outcome, ""));
    if (latest.note === "held") parts.push(COPY.pick_note_held);
    row.append(el("p", "plain", parts.filter(Boolean).join(" ")));
    const target = exact(latest.target);
    if (target && latest.action) row.append(target);
    const beat = Array.isArray(latest.beat) ? latest.beat : [];
    if (beat.length) {
      row.append(el("p", "sub", COPY.pick_beat));
      const others = el("ul", "beat");
      for (const c of beat.slice(0, 3)) {
        const label = c.target && c.target.text ? `“${c.target.text}”` : "";
        const reason = word("reason", c.reason, "");
        others.append(el("li", "", [word("want_kind", c.kind), label, reason && `(${reason})`].filter(Boolean).join(" ")));
      }
      row.append(others);
    }
    list.append(row);
    box.append(list);
    return box;
  }

  function question(q, fresh) {
    const row = item(`q:${q.id}`, "question", fresh);
    const tags = el("p", "tags");
    tags.append(chip(word("question_status", q.status), STATUS_CHIP[q.status]));
    row.append(tags);
    const hyps = Array.isArray(q.hypotheses) ? q.hypotheses : [];
    const top = hyps.reduce((best, h) => (typeof h.credence === "number" && h.credence > (best ?? -1) ? h.credence : best), null);
    const lean = hyps.length > 1 ? fill(COPY.question_lean, { n: hyps.length, p: percent(top) ?? 0 })
      : hyps.length === 1 ? fill(COPY.question_one, { p: percent(top) ?? 0 }) : COPY.question_none;
    row.append(el("p", "plain", `${word("question", q.kind, COPY.question_other)} ${lean}`));
    const subject = exact(q.subject);
    if (subject) row.append(subject);
    if (hyps.length) {
      const list = el("ul", "credences");
      for (const h of hyps) {
        const line = el("li", `credence${h.catch_all ? " catch-all" : ""}`);
        line.append(h.catch_all ? el("p", "exact withheld", COPY.catch_all) : exact(h.statement) || el("p", "exact", ""),
          bar(h.credence, credences.get(h.id)), el("span", "num", `${percent(h.credence) ?? "–"}%`));
        list.append(line);
      }
      row.append(list);
    }
    for (const p of Array.isArray(q.predictions) ? q.predictions : []) {
      const box = el("div", `prediction${fresh.has(`p:${p.id}`) ? " is-new" : ""}`);
      const chance = percent(overall(hyps, p.given));
      const line = chance === null ? fill(COPY.prediction_line_nop, { date: day(p.horizon) })
        : fill(COPY.prediction_line, { p: chance, date: day(p.horizon) });
      box.append(el("p", "plain", line));
      const claim = exact(p.claim);
      if (claim) box.append(claim);
      box.append(el("p", "sub", word("prediction", p.outcome || "pending")));
      row.append(box);
    }
    return row;
  }

  function questions(data, fresh) {
    const box = block(COPY.mind_questions, data.items.length, data.state, data.items.length ? "" : COPY.mind_none_questions);
    if (!data.items.length) return box;
    const list = el("ul", "mind-list");
    data.items.forEach((q) => list.append(question(q, fresh)));
    box.append(list);
    return box;
  }

  function expectations(data, fresh) {
    const box = block(COPY.mind_predictions, data.items.length, data.state, data.items.length ? "" : COPY.mind_none_predictions);
    if (!data.items.length) return box;
    const list = el("ul", "mind-list");
    for (const e of data.items) {
      const row = item(`e:${e.id}`, "expectation", fresh);
      row.append(el("p", "plain", fill(COPY.expect_line, { p: percent(e.p_needed) ?? "–", date: day(e.horizon) })));
      const target = exact(e.target);
      if (target) row.append(target);
      row.append(el("p", "sub", word("expect", e.outcome, e.outcome)));
      list.append(row);
    }
    box.append(list);
    return box;
  }

  function wants(data, fresh) {
    const box = block(COPY.mind_wants, data.items.length, data.state, data.items.length ? "" : COPY.mind_none_wants);
    if (!data.items.length) return box;
    const list = el("ul", "mind-list");
    for (const w of data.items) {
      const label = w.target && w.target.text ? w.target.text : "";
      const row = item(`w:${data.night}:${w.kind}:${w.action}:${label}`, "want", fresh);
      const tags = el("p", "tags");
      tags.append(chip(word("want", w.status), w.status === "won" ? "live" : "muted"));
      row.append(tags, el("p", "plain", fill(COPY.want_line, {
        kind: word("want_kind", w.kind), action: word("action", w.action, COPY.action_other),
      })));
      const target = exact(w.target);
      if (target) row.append(target);
      if (typeof w.strength === "number") {
        const meter = el("div", "meter");
        meter.append(bar(w.strength), el("span", "num", `${percent(w.strength)}%`));
        row.append(meter);
      }
      if (w.status !== "won" && w.reason) row.append(el("p", "sub", word("reason", w.reason)));
      list.append(row);
    }
    box.append(list);
    return box;
  }

  // Every item's key, to tell what is new since the last render.
  function keys(view) {
    const found = new Set();
    const choice = section(view.choices).items[0];
    if (choice) found.add(`pick:${choice.night}:${choice.action}`);
    for (const q of section(view.questions).items) {
      found.add(`q:${q.id}`);
      (q.predictions || []).forEach((p) => found.add(`p:${p.id}`));
    }
    section(view.expectations).items.forEach((e) => found.add(`e:${e.id}`));
    const w = section(view.wants);
    w.items.forEach((x) => found.add(`w:${w.night}:${x.kind}:${x.action}:${x.target && x.target.text ? x.target.text : ""}`));
    return found;
  }

  function render(view) {
    const now = keys(view);
    const fresh = seen === null ? new Set() : new Set([...now].filter((k) => !seen.has(k)));
    const parts = [section(view.choices), section(view.questions), section(view.expectations), section(view.wants)];
    if (parts.every((p) => !p.items.length && p.state !== "unavailable")) {
      body.replaceChildren(el("p", "mind-empty", COPY.mind_empty));
    } else {
      body.replaceChildren(pick(parts[0], parts[3], fresh), questions(parts[1], fresh), expectations(parts[2], fresh), wants(parts[3], fresh));
    }
    credences = new Map();
    for (const q of parts[1].items) (q.hypotheses || []).forEach((h) => credences.set(h.id, h.credence));
    seen = now;
    updated.textContent = fill(COPY.mind_updated, { time: new Date().toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }) });
    if (fresh.size) {
      window.setTimeout(() => body.querySelectorAll(".new-mark").forEach((m) => m.remove()), NEW_MARK_MS);
      onNew(fresh.size);
    }
    return fresh.size;
  }

  return {
    async refresh() {
      if (inflight) return inflight;
      inflight = api.mind().then(render, () => {
        if (seen === null) body.replaceChildren(el("p", "mind-empty", COPY.mind_failed));
        return 0;
      }).finally(() => { inflight = null; });
      return inflight;
    },
  };
}
