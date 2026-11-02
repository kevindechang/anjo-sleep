// The chat stream: POST /api/chat/{sid}/message answers with server-sent events over the POST's own
// body, read here with fetch and a stream reader. Ported from mobile/lib/sse.ts and sse-events.ts:
// the same block parser, the same events (token frames carry text deltas; done carries full_text),
// the same stale-session 409 (adopt the live session_id and retry once), the same wait on a busy
// turn lease (Retry-After), and the same rule that every stream settles exactly once.

import { authHeaders, transport } from "./api.js";
import { COPY } from "./copy.js";

const INACTIVITY_MS = 90000; // a started stream that goes silent this long is dead
const CONNECT_MS = 30000;    // no first byte by then: the connection failed
const ADMISSION_BUDGET_MS = 20000;
const PHASES = new Set(["listening", "remembering", "composing"]);

// Split SSE blocks on a blank line; returns what is left after the last complete block.
export function drainSseBlocks(buffer, onBlock) {
  const parts = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = parts.pop() ?? "";
  for (const block of parts) {
    if (!block.trim()) continue;
    let eventName = "message";
    const data = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).startsWith(" ") ? line.slice(6) : line.slice(5));
    }
    if (data.length) onBlock(eventName, data.join("\n"));
  }
  return rest;
}

// One event to its callback. Malformed JSON and unknown events are dropped.
export function dispatchSseEvent(eventName, raw, on) {
  let data;
  try {
    data = JSON.parse(raw);
  } catch {
    return;
  }
  if (eventName === "token") on.token?.(typeof data.text === "string" ? data.text : "");
  else if (eventName === "phase" && PHASES.has(data.phase)) on.phase?.(data.phase);
  else if (eventName === "accepted") on.accepted?.(data);
  else if (eventName === "done") {
    if (data.silent === true) on.silent?.(data);
    else on.done?.(typeof data.full_text === "string" ? data.full_text : "", data);
  } else if (eventName === "error") on.error?.(typeof data.error === "string" ? data.error : COPY.chat_failed);
  else if (eventName === "no_credits") on.noCredits?.(data);
}

const wait = (ms) => new Promise((resolve) => { window.setTimeout(resolve, ms); });

function requestKey() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") return window.crypto.randomUUID();
  return `k-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function retryAfterMs(header, attempt) {
  const seconds = Number(header);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.min(seconds * 1000, 8000);
  return Math.min(500 * 2 ** attempt, 4000);
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    return body || {};
  } catch {
    return {};
  }
}

// Stream one message. ``on``: token, phase, accepted, done, silent, error, noCredits, session
// (a recovered session id). Returns a cancel function. Exactly one of done / silent / error /
// noCredits fires, whatever happens to the connection.
export function streamMessage(sessionId, text, on) {
  const controller = new AbortController();
  let settled = false;
  let aborted = false;
  let watchdog = null;
  const key = requestKey();

  const clear = () => { if (watchdog) window.clearTimeout(watchdog); watchdog = null; };
  const arm = (ms) => {
    clear();
    watchdog = window.setTimeout(() => {
      if (settled || aborted) return;
      controller.abort();
      finish("error")(ms === CONNECT_MS ? COPY.offline : COPY.chat_stalled);
    }, ms);
  };
  const finish = (name) => (...args) => {
    if (settled || aborted) return;
    settled = true;
    clear();
    on[name]?.(...args);
  };
  const safe = { ...on, done: finish("done"), silent: finish("silent"), error: finish("error"), noCredits: finish("noCredits") };

  async function read(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      arm(INACTIVITY_MS);
      buffer = drainSseBlocks(buffer + decoder.decode(value, { stream: true }), (ev, raw) => dispatchSseEvent(ev, raw, safe));
    }
    buffer += decoder.decode();
    if (buffer.trim()) drainSseBlocks(`${buffer}\n\n`, (ev, raw) => dispatchSseEvent(ev, raw, safe));
    if (!settled) safe.error(COPY.chat_ended_early); // a 200 that ended with no terminal event
  }

  async function open(sid, retried, admissionDeadline, attempt) {
    arm(CONNECT_MS);
    const response = await transport(`/api/chat/${encodeURIComponent(sid)}/message`, {
      method: "POST", signal: controller.signal, credentials: "same-origin", cache: "no-store",
      headers: authHeaders({ Accept: "text/event-stream", "Content-Type": "application/json", "Idempotency-Key": key }),
      body: JSON.stringify({ text, input_mode: "text" }),
    });
    if (response.status === 409) {
      const body = await errorDetail(response);
      if (!retried && body.session_id && String(body.session_id) !== sid) {
        on.session?.(String(body.session_id)); // a stale session: adopt the live one, once
        return open(String(body.session_id), true, admissionDeadline, attempt);
      }
      const deadline = admissionDeadline ?? Date.now() + ADMISSION_BUDGET_MS;
      const delay = retryAfterMs(response.headers.get("Retry-After"), attempt);
      if (Date.now() + delay <= deadline) { // the turn lease is busy: wait it out
        clear();
        await wait(delay);
        if (settled || aborted) return undefined;
        return open(sid, retried, deadline, attempt + 1);
      }
      return safe.error(COPY.chat_busy);
    }
    if (response.status === 401) {
      window.dispatchEvent(new CustomEvent("judge:signed-out"));
      return safe.error(COPY.signed_out);
    }
    if (!response.ok || !response.body) return safe.error(response.status === 503 ? COPY.chat_busy : COPY.chat_failed);
    return read(response);
  }

  open(sessionId, false, null, 0).catch(() => {
    if (!aborted) safe.error(COPY.chat_failed);
  });
  return () => {
    aborted = true;
    clear();
    controller.abort();
  };
}
