// The wake view: GET /api/sleep/me/wake. Anjo's opener becomes the first bubble of the morning's
// chat. Above it, first the behavioral proof: whether the weights were promoted (the night's own
// gate) apart from whether her working state changed, and the same question answered before and
// after sleep (or that no pair was saved). Then her morning note, her bet on herself, what changed,
// and Undo (POST /api/sleep/me/undo) with an inline confirm; after Undo, the pair is marked as from
// before it. At its top,
// "Since your last visit" from her line of thought (morning.js), and the side panel turns to it.

import { api } from "./api.js";
import { COPY, fill } from "./copy.js";
import { byId, el, percent, say } from "./dom.js";
import { morningParts } from "./morning.js";

const BET_OUTCOME = { true: "wake_bet_true", false: "wake_bet_false", censored: "wake_bet_censored" };

// What changed, whatever shape it arrives in: a sentence, a list of sentences, or a list of
// records with a line (or text) each.
export function changedLines(value) {
  if (!value) return [];
  if (typeof value === "string") return value.split(/\n+/).map((s) => s.trim()).filter(Boolean);
  if (!Array.isArray(value)) return [];
  return value.map((v) => (typeof v === "string" ? v : v && (v.line || v.text || v.summary))).filter(Boolean).map(String);
}

// The two verdicts, kept apart: the weights (from the night's gate) and her working state (from
// what changed). ``night`` is the finished night, or null when this page didn't watch it.
export function verdicts(night, changed, promoted = null) {
  const out = [];
  if ((!night || typeof night !== "object") && typeof promoted === "boolean") { // a reload: the pair says
    out.push({ kind: promoted ? "promoted" : "kept", text: promoted ? COPY.wake_weights_promoted : COPY.wake_weights_kept });
  } else if (night && typeof night === "object") {
    const gate = (Array.isArray(night.stages) ? night.stages : []).find((s) => s && s.name === "gate");
    const said = gate && gate.state === "done" && typeof gate.summary === "string" ? gate.summary : "";
    out.push({ kind: night.adapter_live ? "promoted" : "kept",
      text: [night.adapter_live ? COPY.wake_weights_promoted : COPY.wake_weights_kept, said].filter(Boolean).join(" ") });
  }
  out.push(changed.length ? { kind: "changed", text: fill(changed.length === 1 ? COPY.wake_state_changed_one : COPY.wake_state_changed, { n: changed.length }) }
    : { kind: "same", text: COPY.wake_state_same });
  return out;
}

// The two sides' labels: which model answered each, and whether the gate kept tonight's adapter.
export function compareLabels(pair) {
  const p = pair && typeof pair === "object" ? pair : {};
  const before = p.before_model ? fill(COPY.wake_side, { side: COPY.wake_before, model: p.before_model }) : COPY.wake_before;
  if (!p.after_model) return { before, after: COPY.wake_after };
  const model = p.promoted === true ? fill(COPY.wake_after_promoted, { id: p.after_model })
    : p.promoted === false ? fill(COPY.wake_after_held, { id: p.after_model }) : p.after_model;
  return { before, after: fill(COPY.wake_side, { side: COPY.wake_after, model }) };
}

export function createWake({ chat, panel, thought, since, onUndo = () => {} }) {
  const view = byId("wake");
  const body = byId("wake-body");
  const status = byId("wake-status");
  const hide = byId("wake-hide");
  const undo = byId("undo");
  const step = byId("undo-step");
  const undoStatus = byId("undo-status");

  byId("undo-ask").textContent = COPY.undo_ask;

  function bet(prediction) {
    const card = byId("wake-bet");
    if (!prediction || typeof prediction !== "object") {
      card.hidden = true;
      return;
    }
    const outcome = COPY[BET_OUTCOME[String(prediction.outcome)] || "wake_bet_open"];
    const chance = percent(prediction.p);
    byId("bet-plain").textContent = chance === null ? outcome : fill(COPY.wake_bet_line, { p: chance, outcome });
    byId("bet-exact").textContent = prediction.claim ? String(prediction.claim) : "";
    byId("bet-exact").hidden = !prediction.claim;
    card.hidden = false;
  }

  function changed(value) {
    const list = byId("changed-list");
    const lines = changedLines(value);
    list.replaceChildren(...(lines.length ? lines : [COPY.wake_nothing_changed]).map((line) => el("li", "", line)));
    return lines;
  }

  function verdict(night, lines, promoted) {
    byId("wake-verdict").replaceChildren(...verdicts(night, lines, promoted).map((v) => {
      const row = el("li", `verdict-${v.kind}`, v.text);
      return row;
    }));
  }

  // The same question before and after sleep, or plainly that this night saved no pair.
  function compare(pair) {
    const has = Boolean(pair && pair.question);
    byId("compare-q").hidden = !has;
    byId("compare-sides").hidden = !has;
    byId("compare-none").hidden = has;
    byId("compare-note").hidden = true;
    byId("compare-title").textContent = COPY.wake_compare;
    byId("wake-compare").classList.remove("is-historical");
    if (!has) {
      byId("compare-none").textContent = COPY.wake_compare_none;
      return;
    }
    byId("compare-q").textContent = String(pair.question);
    byId("compare-before").textContent = pair.before ? String(pair.before) : "–";
    byId("compare-after").textContent = pair.after ? String(pair.after) : "–";
    const labels = compareLabels(pair);
    byId("compare-before-label").textContent = labels.before;
    byId("compare-after-label").textContent = labels.after;
    const personal = pair.question_source === "personal";
    byId("compare-title").textContent = personal ? COPY.wake_compare_personal : COPY.wake_compare;
    byId("compare-why").hidden = personal || !pair.after_model;
    byId("compare-why").textContent = COPY.wake_compare_why;
  }

  function morning(view) {
    const card = byId("wake-morning");
    const parts = morningParts(view, { level: "h4", always: true });
    card.hidden = !parts;
    if (parts) byId("wake-morning-body").replaceChildren(...parts);
  }

  async function doUndo() {
    step.hidden = true;
    undo.setAttribute("aria-expanded", "false");
    undo.disabled = true;
    say(undoStatus, COPY.undo_working);
    try {
      await api.undo();
      say(undoStatus, COPY.undo_done, "ok");
      byId("wake-compare").classList.add("is-historical");
      say(byId("compare-note"), COPY.wake_compare_undone);
      byId("compare-note").hidden = false;
      byId("wake-verdict").replaceChildren(el("li", "verdict-kept", COPY.wake_weights_undone));
      panel.refresh();
      thought.refresh();
      onUndo();
    } catch (error) {
      say(undoStatus, error.message || COPY.server, "error");
      undo.disabled = false;
    }
  }

  hide.addEventListener("click", () => {
    const open = body.hidden;
    body.hidden = !open;
    hide.textContent = open ? COPY.wake_hide : COPY.wake_show;
    hide.setAttribute("aria-expanded", String(open));
  });
  undo.addEventListener("click", () => {
    step.hidden = !step.hidden;
    undo.setAttribute("aria-expanded", String(!step.hidden));
    if (!step.hidden) byId("undo-yes").focus();
  });
  byId("undo-yes").addEventListener("click", doUndo);
  byId("undo-no").addEventListener("click", () => {
    step.hidden = true;
    undo.setAttribute("aria-expanded", "false");
    undo.focus();
  });

  return {
    async show(night = null) {
      view.hidden = false;
      body.hidden = false;
      undo.disabled = false;
      say(undoStatus, "");
      say(status, COPY.wake_loading);
      byId("wake-heading").focus({ preventScroll: false });
      since.handOver();
      byId("wake-morning").hidden = true;
      thought.refresh().then(morning);
      thought.show();
      try {
        const data = await api.wake();
        say(status, "");
        bet(data && data.self_prediction);
        const pair = data && data.before_after;
        verdict(night, changed(data && data.what_changed), pair && typeof pair.promoted === "boolean" ? pair.promoted : null);
        compare(data && data.before_after);
        chat.morning(data && typeof data.opener === "string" ? data.opener : "");
      } catch (error) {
        say(status, error.status === 401 ? error.message : COPY.wake_failed, "error");
        byId("wake-bet").hidden = true;
        verdict(night, changed(null));
        compare(null);
      }
      panel.refresh();
    },
  };
}
