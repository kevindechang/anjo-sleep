// Her own line of thought: the second view of the side panel, beside "Working out" (panel.js).
// GET /api/sleep/me/thought, in the shape of the engine's thought_view: her open hunches, her
// scorecard on the bets that settled, and the one question she'd most like answered. Each hunch can
// be disputed, with an optional note (POST .../thought/{id}/dispute), or hidden (.../hide); both
// show at once and roll back, with the reason beside the item, if the server refuses. A question is
// answered with a click, and the answer goes into the chat as the judge's own message.

import { api } from "./api.js";
import { COPY, fill } from "./copy.js";
import { byId, day, el, percent, say, wireTabs } from "./dom.js";
import { said, verdict } from "./morning.js";

const NOTE_MAX = 500; // the engine's limit on a dispute's note
const ANSWERED_KEY = "anjo.judge.answered";
const DAY_MS = 86400000;
const list = (value) => (Array.isArray(value) ? value : []);
const words = (shown) => (shown && !shown.withheld && typeof shown.text === "string" ? shown.text : "");

// Whole days from today (the viewer's own calendar) to a YYYY-MM-DD date; null if it isn't one.
export function daysUntil(date, today = new Date()) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(date || ""));
  if (!m) return null;
  const due = Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  const now = Date.UTC(today.getFullYear(), today.getMonth(), today.getDate());
  return Math.round((due - now) / DAY_MS);
}

export function dueWords(hunch, today = new Date()) {
  const n = daysUntil(hunch.due, today);
  const date = n === null ? "" : day(`${hunch.due}T12:00:00`); // noon: the date never slips a day
  const near = n === null ? "" : n < 0 ? COPY.thought_due_past : n === 0 ? COPY.thought_due_today
    : n === 1 ? COPY.thought_due_one : fill(COPY.thought_due_in, { n });
  return [date && fill(COPY.thought_due, { date }), near].filter(Boolean).join(" · ");
}

// The view with one change, as a new object: the one it came from is kept for a rollback.
export const withDisputed = (view, id) => ({
  ...view, hunches: list(view.hunches).map((h) => (h.id === id ? { ...h, disputed: true } : h)),
});
export const withoutHunch = (view, id) => ({
  ...view, hunches: list(view.hunches).filter((h) => h.id !== id), hidden: (Number(view.hidden) || 0) + 1,
});

export const isElse = (choice) => String(choice.id || "").endsWith(".else") || words(choice.text) === "Something else";

export function answerText(question, choice) {
  const ask = words(question.ask) || question.exact || "";
  const answer = words(choice.text);
  return ask ? fill(COPY.question_said, { ask, answer }) : answer;
}

function stored(key) {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}

function store(key, value) {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch {
    // storage refused: the answer is remembered for this page only
  }
}

function section(title, count, className = "") {
  const box = el("section", `mind-section ${className}`.trim());
  const head = el("h3", "", title);
  head.tabIndex = -1;
  if (count) head.append(el("span", "count", String(count)));
  box.append(head);
  return box;
}

function scorecard(card, settled, note) {
  const box = section(COPY.thought_score, 0, "score");
  const n = Number(card && card.scored) || 0;
  if (!n) {
    box.append(el("p", "none", COPY.score_none), el("p", "credence-note", note));
    return box;
  }
  const top = el("div", "score-top");
  top.append(el("p", "score-big", `${percent(card.hit_rate) ?? 0}%`),
    el("p", "score-of", fill(COPY.score_right, { right: card.right, n })));
  box.append(top);
  box.append(el("p", "sub", n === 1 ? COPY.score_n_one : fill(COPY.score_n, { n })));
  if (!card.enough) box.append(el("p", "score-few", fill(COPY.score_few, { n, need: 10 })));
  const bins = list(card.bins);
  if (bins.length) {
    const rows = el("ul", "calib");
    rows.setAttribute("aria-label", COPY.score_bins);
    for (const b of bins) {
      const row = el("li", "calib-row");
      const track = el("span", "calib-bar");
      track.setAttribute("aria-hidden", "true");
      const band = el("span", "calib-band");
      band.style.setProperty("--from", String(b.from));
      band.style.setProperty("--to", String(b.to));
      const fill_ = el("span", "bar-fill");
      fill_.style.setProperty("--v", String(b.n ? b.came_true / b.n : 0));
      track.append(band, fill_);
      row.append(el("span", "calib-label", fill(COPY.score_bin, { from: Math.round(b.from * 10), to: Math.round(b.to * 10) })),
        track, el("span", "num", fill(COPY.score_bin_count, { t: b.came_true, n: b.n })));
      rows.append(row);
    }
    box.append(el("p", "sub", COPY.score_bins), rows, el("p", "sub", COPY.score_bins_key));
  }
  if (typeof card.brier === "number") box.append(el("p", "sub", fill(COPY.score_brier, { b: card.brier.toFixed(2) })));
  if (card.ran_out_of_time) box.append(el("p", "sub", fill(COPY.score_out, { n: card.ran_out_of_time })));
  box.append(el("p", "credence-note", note));
  if (settled.length) {
    const more = el("details", "settled");
    const rows = el("ul", "mind-list");
    settled.forEach((s) => {
      const row = el("li", "item");
      const tags = el("p", "tags");
      tags.append(verdict(s.verdict));
      row.append(tags, ...said(s, s.claim));
      rows.append(row);
    });
    more.append(el("summary", "", fill(COPY.score_settled, { n: settled.length })), rows);
    box.append(more);
  }
  return box;
}

export function createThought({ chat, onAnswer, onView }) {
  const tabs = [byId("side-tab-mind"), byId("side-tab-thought")];
  const panes = [byId("side-mind"), byId("side-thought")];
  const body = byId("thought-body");
  const status = byId("thought-status");
  const notes = new Map();   // hunch id -> the note they sent with a dispute
  const errors = new Map();  // hunch id -> why their last action on it was rolled back
  const drafts = new Map();  // hunch id -> a dispute note the server refused, kept to send again
  let view = null;
  let inflight = null;
  let answered = stored(ANSWERED_KEY);
  let questionStatus = { id: null, text: "", kind: "" };

  function selectSide(i, focus) {
    tabs.forEach((tab, j) => {
      tab.setAttribute("aria-selected", String(i === j));
      tab.tabIndex = i === j ? 0 : -1;
      panes[j].hidden = i !== j;
    });
    if (focus) tabs[i].focus();
  }
  wireTabs(tabs, selectSide);

  function hunchItem(h) {
    const row = el("li", `item hunch${h.disputed ? " is-disputed" : ""}`);
    row.dataset.id = h.id;
    const tags = el("p", "tags");
    tags.append(el("span", "chip odds", String(h.odds || "")), el("span", "due", dueWords(h)));
    if (h.disputed) tags.append(el("span", "chip disputed", COPY.thought_disputed));
    row.append(tags, ...said(h, h.claim));
    const about = words(h.about);
    if (about) row.append(el("p", "sub", fill(COPY.thought_about, { about })));
    if (notes.has(h.id)) row.append(el("p", "sub your-note", fill(COPY.thought_your_note, { note: notes.get(h.id) })));
    const actions = el("div", "hunch-actions");
    const form = disputeForm(h);
    if (!h.disputed) {
      const open = el("button", "btn btn-ghost btn-small act-dispute", COPY.dispute);
      open.type = "button";
      open.setAttribute("aria-expanded", String(!form.hidden));
      open.setAttribute("aria-controls", form.id);
      open.addEventListener("click", () => {
        form.hidden = !form.hidden;
        open.setAttribute("aria-expanded", String(!form.hidden));
        if (!form.hidden) form.querySelector("textarea").focus();
      });
      actions.append(open);
    }
    const hide = el("button", "btn btn-ghost btn-small act-hide", COPY.hide);
    hide.type = "button";
    hide.addEventListener("click", () => act("hide", h));
    actions.append(hide);
    row.append(actions, form);
    if (errors.has(h.id)) {
      const error = el("p", "form-status", errors.get(h.id));
      error.dataset.kind = "error";
      error.setAttribute("role", "alert");
      row.append(error);
    }
    return row;
  }

  function disputeForm(h) {
    const form = el("form", "dispute");
    form.id = `dispute-${h.id}`;
    form.hidden = !drafts.has(h.id);
    form.noValidate = true;
    const label = el("label", "field-label", COPY.dispute_label);
    const box = el("textarea", "dispute-note");
    box.id = `dispute-note-${h.id}`;
    box.rows = 2;
    box.maxLength = NOTE_MAX;
    box.value = drafts.get(h.id) || "";
    label.htmlFor = box.id;
    const send = el("button", "btn btn-primary btn-small", COPY.dispute_send);
    send.type = "submit";
    const cancel = el("button", "btn btn-ghost btn-small", COPY.dispute_cancel);
    cancel.type = "button";
    cancel.addEventListener("click", () => {
      drafts.delete(h.id);
      form.hidden = true;
      const opener = form.closest(".hunch").querySelector(".act-dispute");
      opener.setAttribute("aria-expanded", "false");
      opener.focus();
    });
    const buttons = el("div", "confirm-buttons");
    buttons.append(send, cancel);
    form.append(label, box, buttons);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      act("dispute", h, box.value.trim().slice(0, NOTE_MAX));
    });
    return form;
  }

  function hunches(items) {
    const box = section(COPY.thought_hunches, items.length, "hunches");
    if (!items.length) {
      box.append(el("p", "none", COPY.thought_none_hunches));
      return box;
    }
    const rows = el("ul", "mind-list");
    items.forEach((h) => rows.append(hunchItem(h)));
    box.append(rows);
    return box;
  }

  function question(q) {
    const box = section(COPY.thought_question, 0, "ask");
    if (!q || typeof q !== "object") {
      box.append(el("p", "none", COPY.thought_none_question));
      return box;
    }
    const done = answered === q.question_id;
    box.append(...said(q, q.ask), el("p", "sub", COPY.question_hint));
    const choices = el("div", "choices");
    choices.setAttribute("role", "group");
    choices.setAttribute("aria-label", words(q.ask) || COPY.thought_question);
    const mine = questionStatus.id === q.question_id;
    const line = el("p", "form-status", mine ? questionStatus.text : "");
    line.dataset.kind = mine ? questionStatus.kind : "";
    line.setAttribute("role", "status");
    for (const c of list(q.choices)) {
      const text = words(c.text);
      const button = el("button", `choice${isElse(c) ? " choice-else" : ""}`, text || COPY.mind_withheld);
      button.type = "button";
      button.disabled = done || !text;
      button.addEventListener("click", () => answer(q, c, line));
      choices.append(button);
    }
    box.append(choices, line);
    return box;
  }

  async function answer(q, choice, line) {
    const tell = (text, kind) => {
      questionStatus = { id: q.question_id, text, kind };
      say(line, text, kind);
    };
    if (isElse(choice)) {
      onAnswer();
      chat.draft(fill(COPY.question_else, { ask: words(q.ask) || q.exact || "" }));
      return;
    }
    if (!chat.canSend) {
      tell(chat.asleep ? COPY.question_asleep : COPY.question_wait, "error");
      return;
    }
    answered = q.question_id;
    store(ANSWERED_KEY, answered);
    tell(COPY.question_sent, "ok");
    render();
    onAnswer();
    const ok = await chat.send(answerText(q, choice));
    if (!ok) {
      answered = null;
      store(ANSWERED_KEY, null);
      questionStatus = { id: null, text: "", kind: "" };
      render();
    }
  }

  // Dispute or hide: shown at once; put back, with the reason beside it, if the server says no.
  async function act(kind, h, text = "") {
    const before = view;
    errors.delete(h.id);
    drafts.delete(h.id);
    view = kind === "hide" ? withoutHunch(view, h.id) : withDisputed(view, h.id);
    if (kind === "dispute" && text) notes.set(h.id, text);
    const shown = view;
    const next = list(before.hunches)[list(before.hunches).findIndex((x) => x.id === h.id) + 1];
    render();
    say(status, kind === "hide" ? COPY.hide_done : COPY.dispute_done, "ok");
    focusAfter(kind, h.id, next);
    try {
      if (kind === "hide") await api.hide(h.id);
      else await api.dispute(h.id, text || null);
    } catch (error) {
      if (error.status === 401) return; // signed out: the page is leaving
      notes.delete(h.id);
      say(status, "");
      if (error.status === 404) {
        say(status, COPY.thought_gone, "error");
        refresh();
        return;
      }
      errors.set(h.id, kind === "hide" ? COPY.hide_failed : COPY.dispute_failed);
      if (kind === "dispute") drafts.set(h.id, text); // the form opens again with their note in it
      if (view === shown) view = before; // nothing newer arrived meanwhile: put it back as it was
      render();
      const back = body.querySelector(`.hunch[data-id="${CSS.escape(h.id)}"]`);
      if (back) back.querySelector(kind === "hide" ? ".act-hide" : ".dispute-note, .act-hide").focus();
    }
  }

  function focusAfter(kind, id, next) {
    const target = kind === "hide"
      ? (next && body.querySelector(`.hunch[data-id="${CSS.escape(next.id)}"] button`)) || body.querySelector(".hunches h3")
      : body.querySelector(`.hunch[data-id="${CSS.escape(id)}"] .act-hide`);
    if (target) target.focus({ preventScroll: true });
  }

  function render() {
    if (!view) return;
    body.replaceChildren(
      hunches(list(view.hunches)),
      scorecard(view.scorecard, list(view.settled), String(view.credence_note || "")),
      question(view.question),
    );
  }

  function refresh() {
    if (inflight) return inflight;
    inflight = api.thought().then((data) => {
      view = data && typeof data === "object" ? data : null;
      render();
      if (view) onView(view);
      return view;
    }, (error) => {
      if (!view) body.replaceChildren(el("p", "mind-empty", error.status === 503 ? COPY.thought_off : COPY.thought_failed));
      return view;
    }).finally(() => { inflight = null; });
    return inflight;
  }

  return {
    refresh,
    // Open "Line of thought" in the side panel (after a wake, its hunches are the news).
    show: () => selectSide(1, false),
  };
}
