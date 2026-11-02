// The chat pane: POST /api/chat/start (platform web), then each message streams over
// POST /api/chat/{sid}/message (sse.js). Text deltas fill Anjo's bubble as they arrive; the done
// event's full_text replaces them, and its model names who answered (Glimmer, or Muse Spark while
// Glimmer wakes). ``onSend`` fires as a message goes out and ``onReply`` once its reply is in, which
// starts the watch for MIND's live pass (livepass.js).

import { api } from "./api.js";
import { COPY } from "./copy.js";
import { byId, el, reducedMotion, say } from "./dom.js";
import { streamMessage } from "./sse.js";

const MAX_ROWS_PX = 168;

// Who answered, from the done event's model id; nothing for an id it doesn't recognise.
export function modelName(model) {
  const id = String(model || "").toLowerCase();
  if (id.includes("glimmer") || id.includes("sleep_modal")) return COPY.model_glimmer;
  if (id.includes("spark") || id.includes("muse")) return COPY.model_spark;
  return "";
}

export function createChat({ onReply, onSend = () => {} }) {
  const log = byId("chat-log");
  const empty = byId("chat-empty");
  const status = byId("chat-status");
  const form = byId("composer");
  const box = byId("composer-text");
  const send = byId("composer-send");
  let sessionId = null;
  let starting = null;
  let busy = false;
  let asleep = false;

  function nearBottom() {
    return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  }

  function follow(node, force = false) {
    if (force || nearBottom()) node.scrollIntoView({ block: "end", behavior: reducedMotion() ? "auto" : "smooth" });
  }

  function bubble(who, text, extra = "") {
    empty.hidden = true;
    const item = el("li", `msg msg-${who} ${extra}`.trim());
    const meta = el("p", "msg-meta", who === "you" ? COPY.you : COPY.anjo);
    const body = el("p", "msg-text", text);
    item.append(meta, body);
    log.append(item);
    return item;
  }

  function grow() {
    box.style.height = "auto";
    box.style.height = `${Math.min(box.scrollHeight, MAX_ROWS_PX)}px`;
  }

  function setEnabled() {
    const off = busy || asleep;
    box.disabled = asleep;
    send.disabled = off;
    document.querySelectorAll(".starter").forEach((b) => { b.disabled = off; });
  }

  async function ensureSession() {
    if (sessionId) return sessionId;
    if (!starting) {
      starting = api.chatStart().then((data) => {
        if (!data || !data.session_id) throw new Error(COPY.chat_start_failed);
        sessionId = String(data.session_id);
        return sessionId;
      }).finally(() => { starting = null; });
    }
    return starting;
  }

  function stream(sid, text, reply) {
    const body = reply.querySelector(".msg-text");
    const meta = reply.querySelector(".msg-meta");
    const phase = el("p", "msg-phase", COPY.phase_waiting);
    reply.append(phase);
    let text_ = "";
    return new Promise((resolve) => {
      const end = (ok) => {
        phase.remove();
        reply.classList.remove("is-pending");
        resolve(ok);
      };
      streamMessage(sid, text, {
        session: (id) => { sessionId = id; },
        phase: (name) => { phase.textContent = COPY[`phase_${name}`] || COPY.phase_waiting; },
        token: (delta) => {
          text_ += delta;
          body.textContent = text_;
          if (phase.isConnected) phase.remove();
          follow(reply);
        },
        done: (full, data) => {
          body.textContent = full || text_;
          const name = modelName(data.model);
          if (name) meta.textContent = `${COPY.anjo} · ${name}`;
          end(true);
        },
        silent: () => { body.textContent = ""; meta.textContent = COPY.chat_silent; end(true); },
        noCredits: () => { reply.classList.add("is-error"); body.textContent = COPY.chat_no_credits; end(false); },
        error: (message) => {
          reply.classList.add("is-error");
          if (!text_) reply.remove();
          say(status, message || COPY.chat_failed, "error");
          end(false);
        },
      });
    });
  }

  async function submit(text) {
    const clean = text.trim();
    if (!clean || busy || asleep) return false;
    busy = true;
    onSend();
    setEnabled();
    say(status, "");
    const mine = bubble("you", clean);
    follow(mine, true);
    box.value = "";
    grow();
    const reply = bubble("anjo", "", "is-pending");
    let ok = false;
    try {
      const sid = await ensureSession();
      ok = await stream(sid, clean, reply);
    } catch (error) {
      reply.remove();
      say(status, error.message || COPY.chat_start_failed, "error");
    }
    if (!ok) box.value = box.value || clean; // keep what they wrote, to send again
    grow();
    busy = false;
    setEnabled();
    if (!asleep) box.focus({ preventScroll: true });
    if (ok) onReply();
    return ok;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit(box.value);
  });
  box.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit(box.value);
    }
  });
  box.addEventListener("input", grow);
  document.querySelectorAll(".starter").forEach((button) => {
    button.addEventListener("click", () => submit(button.textContent));
  });

  return {
    // Open the session early so the first message doesn't wait for it.
    warm() {
      ensureSession().catch((error) => say(status, error.message || COPY.chat_start_failed, "error"));
    },
    // Send words from elsewhere on the page (an answer to MIND's question) as a normal message.
    // False when it can't go now: a reply is still arriving, or Anjo is asleep.
    send: (text) => submit(text),
    get canSend() { return !busy && !asleep; },
    get asleep() { return asleep; },
    // Start a message for them to finish, the cursor at its end.
    draft(text) {
      if (asleep) return;
      box.value = text;
      grow();
      box.focus({ preventScroll: true });
      box.setSelectionRange(text.length, text.length);
      box.scrollIntoView({ block: "end", behavior: reducedMotion() ? "auto" : "smooth" });
    },
    setAsleep(value) {
      asleep = value;
      say(status, value ? COPY.chat_during_night : "", "");
      setEnabled();
    },
    // A new morning: the log starts over, with Anjo's opener as its first bubble.
    morning(opener) {
      log.replaceChildren();
      if (opener) bubble("anjo", opener, "msg-opener").querySelector(".msg-meta").textContent = COPY.wake_opener_label;
      else empty.hidden = false;
    },
  };
}
