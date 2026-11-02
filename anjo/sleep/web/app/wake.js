// The wake view: GET /api/sleep/me/wake. Anjo's opener becomes the first bubble of the morning's
// chat; above it, how its bet on itself resolved, what changed, the same question answered by last
// night's model and this one, and Undo (POST /api/sleep/me/undo) with an inline confirm. At its top,
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

export function createWake({ chat, panel, thought, since }) {
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
  }

  function compare(pair) {
    const card = byId("wake-compare");
    if (!pair || !pair.question) {
      card.hidden = true;
      return;
    }
    byId("compare-q").textContent = String(pair.question);
    byId("compare-before").textContent = pair.before ? String(pair.before) : "–";
    byId("compare-after").textContent = pair.after ? String(pair.after) : "–";
    card.hidden = false;
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
      panel.refresh();
      thought.refresh();
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
    async show() {
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
        changed(data && data.what_changed);
        compare(data && data.before_after);
        chat.morning(data && typeof data.opener === "string" ? data.opener : "");
      } catch (error) {
        say(status, error.status === 401 ? error.message : COPY.wake_failed, "error");
        byId("wake-bet").hidden = true;
        byId("wake-compare").hidden = true;
        changed(null);
      }
      panel.refresh();
    },
  };
}
