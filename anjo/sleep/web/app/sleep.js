// Sleep now and the night view. POST /api/sleep/me/sleep-now starts a night; GET
// /api/sleep/me/nights/{id} is polled every 2 s (plain JSON, no SSE) and drawn as a live timeline:
// phase 1 (thinking) and phase 2 (learning), each stage with its plain name, what it does, how long
// it took and the server's one-line summary, typed in as it arrives. While phase 1 runs the MIND
// panel refreshes too, so its new questions appear as the night opens them.

import { api, nightId, setNightId } from "./api.js";
import { COPY, fill } from "./copy.js";
import { byId, el, reducedMotion, say } from "./dom.js";

const POLL_MS = 2000;
const MAX_BACKOFF_MS = 10000;
const TYPE_MS = 16;
export const PHASE_1 = ["extract", "mind_notice", "mind_choose", "mind_inquire", "homework", "self_predict", "note"];
export const PHASE_2 = ["practice", "train", "gate", "promote", "serve"];
const LATER = new Set(PHASE_2);

function seconds(start, end) {
  const a = Date.parse(start);
  const b = end ? Date.parse(end) : Date.now();
  return Number.isFinite(a) && Number.isFinite(b) ? Math.max(0, Math.round((b - a) / 1000)) : null;
}

export function clock(total) {
  if (total === null || total === undefined) return "";
  const m = Math.floor(total / 60);
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

export function createSleep({ chat, panel, onAwake, onStatus, onShow = () => {}, onFinish = () => {} }) {
  const button = byId("sleep-now");
  const confirm = byId("sleep-confirm");
  const view = byId("night");
  const chatPane = byId("chat");
  const heading = byId("night-heading");
  const meta = byId("night-meta");
  const end = byId("night-end");
  const wakeUp = byId("wake-up");
  const back = byId("night-back");
  const rows = new Map();
  let id = null;
  let timer = null;
  let backoff = POLL_MS;
  let polls = 0;
  let running = false;
  let starting = false;
  let startedAt = null;

  byId("sleep-confirm-text").textContent = COPY.sleep_confirm;

  function show(visible) {
    if (visible) onShow(); // on a phone the night lives in the Chat tab
    view.hidden = !visible;
    chatPane.hidden = visible;
    if (visible) heading.focus({ preventScroll: true });
  }

  function setButton() {
    button.textContent = running ? COPY.sleep_watch : COPY.sleep_now;
    button.setAttribute("aria-expanded", "false");
  }

  function row(name) {
    if (rows.has(name)) return rows.get(name);
    const item = el("li", "stage");
    item.dataset.state = "pending";
    const think = el("p", "stage-think");
    const scan = el("span", "scan");
    scan.setAttribute("aria-hidden", "true");
    think.append(el("span", "", COPY.stage_running), scan, el("span", "think-time", ""));
    const parts = {
      item, name: el("span", "stage-name", COPY[`stage_${name}`] || name.replace(/_/g, " ")),
      time: el("span", "stage-time", ""), line: el("p", "stage-line", COPY[`stage_${name}_line`] || ""),
      think, summary: el("p", "stage-summary", ""), typed: "",
    };
    item.append(parts.name, parts.time, parts.line, parts.summary, parts.think);
    byId(LATER.has(name) ? "stages-2" : "stages-1").append(item);
    rows.set(name, parts);
    return parts;
  }

  function reset() {
    rows.clear();
    byId("stages-1").replaceChildren();
    byId("stages-2").replaceChildren();
    [...PHASE_1, ...PHASE_2].forEach(row);
    say(end, "");
    wakeUp.hidden = true;
    startedAt = null;
  }

  // The summary, typed in once as it arrives (at once when motion is reduced).
  function type(parts, text) {
    if (parts.typed === text) return;
    parts.typed = text;
    const node = parts.summary;
    if (reducedMotion() || !text) {
      node.textContent = text;
      return;
    }
    node.textContent = "";
    node.classList.add("is-typing");
    let i = 0;
    const step = () => {
      if (parts.typed !== text) return;
      i += Math.max(1, Math.round(text.length / 90));
      node.textContent = text.slice(0, i);
      if (i < text.length) window.setTimeout(step, TYPE_MS);
      else node.classList.remove("is-typing");
    };
    step();
  }

  function draw(night) {
    const stages = Array.isArray(night.stages) ? night.stages : [];
    for (const stage of stages) {
      const parts = row(String(stage.name));
      const state = ["pending", "running", "done", "skipped", "failed"].includes(stage.state) ? stage.state : "pending";
      parts.item.dataset.state = state;
      const took = seconds(stage.started_at, stage.ended_at);
      parts.time.textContent = state === "done" || state === "failed" ? clock(took)
        : state === "skipped" ? COPY.stage_skipped : "";
      parts.think.querySelector(".think-time").textContent = state === "running" ? clock(took) : "";
      type(parts, typeof stage.summary === "string" ? stage.summary : "");
      if (!startedAt && stage.started_at) startedAt = stage.started_at;
    }
    const phase = night.phase === 2 ? 2 : 1;
    byId("phase-2").classList.toggle("is-later", phase === 1 && night.status !== "done");
    const title = night.status === "queued" ? COPY.night_title_queued : night.status === "done" ? COPY.night_title_done
      : night.status === "failed" ? COPY.night_title_failed : phase === 2 ? COPY.night_title_learning : COPY.night_title_thinking;
    heading.textContent = title;
    const total = seconds(startedAt, night.status === "done" || night.status === "failed" ? lastEnd(stages) : null);
    meta.textContent = total === null ? "" : fill(COPY.night_elapsed, { time: clock(total) });
  }

  function lastEnd(stages) {
    return stages.map((s) => s.ended_at).filter(Boolean).sort().pop() || null;
  }

  function finish(night) {
    running = false;
    setNightId(null);
    setButton();
    chat.setAsleep(false);
    panel.refresh();
    onFinish();
    show(true); // it may have finished while the judge was reading the chat or MIND
    if (night.status === "done") {
      say(end, `${COPY.night_done_line} ${night.adapter_live ? COPY.night_adapter_live : COPY.night_adapter_kept}`, "ok");
      wakeUp.hidden = false;
      wakeUp.focus({ preventScroll: true });
    } else {
      say(end, COPY.night_failed_line, "error");
    }
  }

  async function poll() {
    window.clearTimeout(timer);
    let night;
    try {
      night = await api.night(id);
    } catch (error) {
      if (error.status === 404) {
        running = false;
        setNightId(null);
        setButton();
        chat.setAsleep(false);
        say(end, COPY.night_failed_line, "error");
        return;
      }
      if (error.status === 401) return;
      say(end, COPY.night_lost, "error");
      backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
      timer = window.setTimeout(poll, backoff);
      return;
    }
    backoff = POLL_MS;
    if (end.dataset.kind === "error" && running) say(end, "");
    draw(night || {});
    polls += 1;
    if (night && (night.status === "done" || night.status === "failed")) {
      finish(night);
      return;
    }
    if (polls % 2 === 0) panel.refresh(); // watch MIND change while the night runs
    timer = window.setTimeout(poll, POLL_MS);
  }

  function watch(nightIdValue) {
    id = nightIdValue;
    setNightId(id);
    running = true;
    polls = 0;
    reset();
    setButton();
    chat.setAsleep(true);
    show(true);
    poll();
  }

  // A night starts only between replies, once: chat holds while the server admits it.
  async function begin() {
    if (starting || running) return;
    if (!chat.canSend) {
      onStatus(chat.asleep ? COPY.sleep_409 : COPY.sleep_wait_reply, "");
      return;
    }
    starting = true;
    confirm.hidden = true;
    button.disabled = true;
    byId("sleep-yes").disabled = true;
    chat.hold(true);
    onStatus(COPY.sleep_starting, "");
    try {
      const data = await api.sleepNow();
      onStatus("", "");
      if (!data || !data.night_id) throw Object.assign(new Error(COPY.server), { status: 500 });
      watch(String(data.night_id));
    } catch (error) {
      const running_ = error.data && (error.data.night_id || (error.data.detail && error.data.detail.night_id));
      if (error.status === 409 && running_) {
        onStatus(COPY.sleep_409, "ok");
        watch(String(running_));
      } else {
        onStatus(error.message || COPY.server, "error");
      }
    } finally {
      starting = false;
      chat.hold(false);
      button.disabled = false;
      byId("sleep-yes").disabled = false;
    }
  }

  button.addEventListener("click", () => {
    if (running) {
      show(true);
      return;
    }
    if (starting) return;
    if (!chat.canSend) { // a reply is still arriving: sleep once it's in
      onStatus(COPY.sleep_wait_reply, "");
      return;
    }
    confirm.hidden = !confirm.hidden;
    button.setAttribute("aria-expanded", String(!confirm.hidden));
    if (!confirm.hidden) byId("sleep-yes").focus();
  });
  byId("sleep-yes").addEventListener("click", begin);
  byId("sleep-no").addEventListener("click", () => {
    confirm.hidden = true;
    button.setAttribute("aria-expanded", "false");
    button.focus();
  });
  back.addEventListener("click", () => {
    show(false);
    button.focus();
  });
  wakeUp.addEventListener("click", () => {
    show(false);
    onAwake();
  });

  return {
    // A night this tab started and hasn't seen finish: watch it again.
    resume() {
      const stored = nightId();
      if (stored) watch(stored);
    },
    get running() { return running; },
  };
}
