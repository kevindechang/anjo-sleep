// Development only: canned answers for every judge route, so the pages can be built and checked
// before the API exists. api.js imports this file only when the page's URL has ?mock=1; without
// the flag it is never loaded and nothing here runs. Nothing here reaches a server.
//
// The judge code is GLIMMER (BUSY answers 429, anything else 403). Extra flags, all optional:
//   mockcase=cold | warm        the talker label's state (default: waking for 25 s, then warm)
//   mockcase=sleep429 | sleep409 | nightfail | wakefail | chatfail | expired | mindempty | mindfail
//   mockcase=visit              her line of thought has a "since your last visit" note from the start
//                               (otherwise only once a night has run)
//   mockcase=thoughtoff | thoughtfail | thoughtempty | markfail   (markfail: dispute and hide refused)
//   mockspeed=4                 run the night 4x faster
//   mockat=40                   start a night 40 s in
//   mockturns=3                 start as if 3 messages had been sent
//
// Her line of thought is the engine's own output for a synthetic life (mock_thought.js).

import { THOUGHT } from "./mock_thought.js";

const params = new URLSearchParams(window.location.search);
const CASE = params.get("mockcase") || "";
const SPEED = Math.max(0.25, Number(params.get("mockspeed")) || 1);
const AT = Math.max(0, Number(params.get("mockat")) || 0);
const KEY = "anjo.mock.state";
const DAY = 86400000;

const STAGES = [
  ["extract", 6, "Reading what you said…", "Read 6 messages. Kept 9 pieces of evidence."],
  ["mind_notice", 8, "Looking for what doesn't fit…", "Opened 1 new question: something that stood out."],
  ["mind_choose", 6, "Weighing 3 wants…", "Chose to think through why the call to your sister keeps slipping."],
  ["mind_inquire", 10, "Testing 3 answers against the evidence…", "Moved its lean to 64%: the last call ended badly."],
  ["homework", 9, "Reading about restarting contact…", "Studied: how people restart contact after a hard conversation (Muse Wonder)."],
  ["self_predict", 6, "Deciding how sure it is…", "Bet 70% that practising a gentler opener helps."],
  ["note", 5, "Writing…", "Left a note for the morning."],
  ["practice", 7, "Scene 2 of 4…", "Rehearsed 4 scenes; 3 held up."],
  ["train", 10, "Step 12 of 38…", "Trained your adapter: 38 steps."],
  ["gate", 5, "Comparing with last night's model on 5 probes…", "Gate kept it: +3 of 5."],
  ["promote", 3, "", "Promoted tonight's adapter."],
  ["serve", 5, "Loading your adapter…", "Glimmer loaded your adapter."],
];
const PHASE_2 = new Set(["practice", "train", "gate", "promote", "serve"]);

const REPLIES = [
  "That sounds heavier than a scheduling problem. When you picture calling her, what's the first thing you expect her to say?",
  "So the last call ended badly, and part of you is waiting for her to bring it up. Is that close?",
  "Got it. Mornings came up too. Is it the hour that's hard, or what the morning is for?",
  "I'm holding on to that. If you send me to sleep, I'll think it through tonight.",
];

function load() {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    // no storage: a fresh state each page
  }
  return { t0: Date.now(), turns: Number(params.get("mockturns")) || 0, nightStart: null, nights: 0, undone: false, hidden: [], disputed: [] };
}
let state = load();
function save(next) {
  state = { ...state, ...next };
  try {
    window.sessionStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // no storage
  }
}

const iso = (ms) => new Date(ms).toISOString();
const text = (value) => ({ text: value, withheld: false });
const delay = (ms) => new Promise((resolve) => { window.setTimeout(resolve, ms); });

function json(status, body) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

// How far the current night has gone: each stage's state and times, from the mock's own clock.
function nightAt(now) {
  if (!state.nightStart) return null;
  const elapsed = ((now - state.nightStart) / 1000) * SPEED + AT;
  let t = 0;
  const stages = [];
  let failed = false;
  for (const [name, length, running, done] of STAGES) {
    const start = t;
    t += length;
    const startMs = state.nightStart + ((start - AT) / SPEED) * 1000;
    const endMs = state.nightStart + ((t - AT) / SPEED) * 1000;
    let stage = { name, state: "pending", started_at: null, ended_at: null, summary: "" };
    if (failed) stage = { ...stage, state: "skipped" };
    else if (CASE === "nightfail" && name === "train" && elapsed >= start + 4) {
      stage = { name, state: "failed", started_at: iso(startMs), ended_at: iso(startMs + 4000 / SPEED), summary: "Training stopped: the GPU didn't answer." };
      failed = true;
    } else if (elapsed >= t) stage = { name, state: "done", started_at: iso(startMs), ended_at: iso(endMs), summary: done };
    else if (elapsed >= start) stage = { name, state: "running", started_at: iso(startMs), ended_at: null, summary: running };
    stages.push(stage);
  }
  const doneAll = stages.every((s) => s.state === "done");
  const status = failed ? "failed" : doneAll ? "done" : elapsed <= 0.5 ? "queued" : "running";
  const phase = stages.some((s) => PHASE_2.has(s.name) && s.state !== "pending") ? 2 : 1;
  return { status, phase, stages, note: doneAll ? "held" : null, adapter_live: doneAll };
}

const reached = (night, name) => Boolean(night && night.stages.find((s) => s.name === name && s.state === "done"));

function mind(now) {
  const night = nightAt(now) || (state.nights ? { stages: STAGES.map(([name]) => ({ name, state: "done" })) } : null);
  const turns = state.turns;
  const questions = [];
  const target = "Why the call to your sister keeps slipping";
  if (turns >= 1 || night) {
    const inquired = reached(night, "mind_inquire");
    const c = inquired ? [0.64, 0.21, 0.15] : turns >= 2 || night ? [0.52, 0.28, 0.2] : [0.4, 0.35, 0.25];
    questions.push({
      id: "q1", kind: "gap", status: turns >= 2 || night ? "active" : "open",
      subject: text("What makes the call to her hard to start?"),
      hypotheses: [
        { id: "h1", statement: text("The last call ended in an argument, and she may bring it up."), credence: c[0], catch_all: false },
        { id: "h2", statement: text("Calls with her run long, and evenings are when you're most tired."), credence: c[1], catch_all: false },
        { id: "h3", statement: text("something else"), credence: c[2], catch_all: true },
      ],
      predictions: turns >= 2 || night ? [{
        id: "p1", claim: text("You'll mention the last call before the week is out."), given: [0.8, 0.3, 0.5],
        horizon: iso(state.t0 + 5 * DAY), outcome: null,
      }] : [],
    });
  }
  if (turns >= 3 || night) {
    questions.push({
      id: "q2", kind: "contradiction", status: "open",
      subject: text("Mornings are the problem, yet you said you run fine on holidays."),
      hypotheses: [
        { id: "h4", statement: text("It's the commute after the run, not the hour."), credence: 0.45, catch_all: false },
        { id: "h5", statement: text("It's about having no one waiting for you."), credence: 0.3, catch_all: false },
        { id: "h6", statement: text("something else"), credence: 0.25, catch_all: true },
      ],
      predictions: [],
    });
  }
  if (reached(night, "mind_notice")) {
    questions.push({
      id: "q3", kind: "anomaly", status: "open", subject: text("You called work \"fine\" three times, each time with a pause."),
      hypotheses: [{ id: "h7", statement: text("Something at work is bothering you that you haven't named."), credence: 0.55, catch_all: false },
        { id: "h8", statement: text("something else"), credence: 0.45, catch_all: true }],
      predictions: [],
    });
  }
  const chose = reached(night, "mind_choose");
  const wants = chose ? [
    { kind: "settle", target: text(target), action: "inquire", strength: 0.74, status: "won", reason: null },
    { kind: "follow_up", target: text("Marathon training plan"), action: "revisit_thread", strength: 0.51, status: "held_back", reason: "lost_to_target" },
    { kind: "share", target: text("A note about mornings"), action: "hold_note", strength: 0.22, status: "rejected", reason: "below_threshold" },
  ] : [];
  const choices = chose ? [{
    night: 1, date: iso(state.nightStart || now).slice(0, 10), status: "ok", sleep: null, action: "inquire", target: text(target),
    outcome: reached(night, "mind_inquire") ? "ran" : null, note: reached(night, "note") ? "held" : null,
    beat: wants.slice(1).map(({ kind, target: t, action, strength, status, reason }) => ({ kind, target: t, action, strength, status, reason })),
  }] : [];
  const expectations = reached(night, "self_predict") ? [{
    id: "e1", target: text(target), made_at: iso(now), horizon: iso(now + 3 * DAY), p_needed: 0.62, outcome: "open",
  }] : [];
  const section = (items) => ({ state: items.length ? "ok" : "none", items });
  if (CASE === "mindempty") return { as_of: null, questions: section([]), targets: section([]), wants: { ...section([]), night: null }, choices: section([]), expectations: section([]) };
  return {
    as_of: night ? iso(now) : null, questions: section(questions), targets: section([]),
    wants: { ...section(wants), night: chose ? 1 : null }, choices: section(choices), expectations: section(expectations),
    gates: section([]), timeline: section([]),
  };
}

// Her line of thought with the judge's marks applied, as the route does: hidden items (or a hidden
// question's bets) leave the view, disputed ones stay flagged. The note waits for a second visit.
function thought() {
  const hidden = new Set(state.hidden || []);
  const disputed = new Set(state.disputed || []);
  const keep = (x) => !hidden.has(x.id) && !hidden.has(x.question_id);
  if (CASE === "thoughtempty") {
    return { ...THOUGHT, hunches: [], settled: [], question: null, morning: null, hidden: 0,
      scorecard: { scored: 0, right: 0, hit_rate: null, brier: null, enough: false, bins: [], ran_out_of_time: 0 } };
  }
  const hunches = THOUGHT.hunches.filter(keep).map((h) => ({ ...h, disputed: disputed.has(h.id) || disputed.has(h.question_id) }));
  const shown = new Set(hunches.map((h) => h.id));
  const morning = state.nights > 0 || CASE === "visit"
    ? { ...THOUGHT.morning, new_hunches: THOUGHT.morning.new_hunches.filter((id) => shown.has(id)) } : null;
  return {
    ...THOUGHT, hunches, settled: THOUGHT.settled.filter(keep), morning, hidden: hidden.size,
    question: THOUGHT.question && keep(THOUGHT.question) ? THOUGHT.question : null,
  };
}

async function mark(kind, id, body, now) {
  const view = thought();
  const ids = new Set([...view.hunches, ...view.settled].map((x) => x.id));
  if (view.question) ids.add(view.question.question_id);
  if (!ids.has(id)) return json(404, { detail: "There's nothing with that id to change." });
  await delay(600); // long enough to see the change land before a refusal puts it back
  if (CASE === "markfail") return json(500, {});
  const key = kind === "hide" ? "hidden" : "disputed";
  save({ [key]: [...(state[key] || []), id] });
  const note = body && typeof body.note === "string" ? { note: body.note } : {};
  return json(200, { kind, item_id: id, at: iso(now), ...note });
}

function stream(reply, model, signal) {
  const encoder = new TextEncoder();
  const frames = [
    [0, "accepted", { turn_id: `t-${Date.now()}` }], [180, "phase", { phase: "listening" }],
    [420, "phase", { phase: "remembering" }], [520, "phase", { phase: "composing" }],
  ];
  const words = reply.split(/(?<= )/);
  words.forEach((w) => frames.push([45 + Math.random() * 40, "token", { text: w }]));
  frames.push([120, "done", { full_text: reply, model, intent: "", active_emotions: {} }]);
  return new ReadableStream({
    async start(controller) {
      for (const [wait, event, data] of frames) {
        await delay(wait);
        if (signal && signal.aborted) {
          controller.error(new DOMException("aborted", "AbortError"));
          return;
        }
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      }
      controller.close();
    },
  });
}

async function route(method, path, body, signal) {
  const now = Date.now();
  if (path === "/api/sleep/judge/start" && method === "POST") {
    const code = String((body && body.code) || "").trim().toUpperCase();
    if (code === "BUSY") return json(429, { detail: "Too many requests" });
    if (code !== "GLIMMER") return json(403, { detail: "Wrong code" });
    save({ t0: now, turns: Number(params.get("mockturns")) || 0, nightStart: null, nights: 0, undone: false, hidden: [], disputed: [] });
    return json(200, { token: "mock-token", user_id: "judge-mock" });
  }
  if (CASE === "expired" && path.startsWith("/api/sleep/me/")) return json(401, { detail: "Not authenticated" });
  if (path === "/api/sleep/me/talker") {
    const waited = (now - state.t0) / 1000;
    if (CASE === "cold") return json(200, { state: "cold", model: "spark-fallback" });
    if (CASE === "warm" || waited > 25) return json(200, { state: "warm", model: "glimmer" });
    return json(200, { state: "waking", eta_s: Math.max(60, 180 - waited * 4), model: "spark-fallback" });
  }
  if (path === "/api/sleep/me/mind") return CASE === "mindfail" ? json(500, {}) : json(200, mind(now));
  if (path === "/api/sleep/me/thought") {
    if (CASE === "thoughtoff") return json(503, { detail: "Her line of thought isn't available yet." });
    return CASE === "thoughtfail" ? json(500, {}) : json(200, thought());
  }
  const marked = /^\/api\/sleep\/me\/thought\/([^/]+)\/(dispute|hide)$/.exec(path);
  if (marked && method === "POST") return mark(marked[2], decodeURIComponent(marked[1]), body, now);
  if (path === "/api/chat/start") return json(200, { session_id: "mock-session", session_count: 1 });
  if (/^\/api\/chat\/[^/]+\/message$/.test(path)) {
    if (CASE === "chatfail") return json(503, { error: "Service temporarily unavailable" });
    const reply = REPLIES[Math.min(state.turns, REPLIES.length - 1)];
    const model = (now - state.t0) / 1000 > 25 || CASE === "warm" ? "glimmer-judge" : "muse-spark";
    save({ turns: state.turns + 1 });
    return new Response(stream(reply, model, signal), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  }
  if (path === "/api/sleep/me/sleep-now" && method === "POST") {
    if (CASE === "sleep429") return json(429, { detail: "Daily night limit reached" });
    if (CASE === "sleep409") {
      if (!state.nightStart) save({ nightStart: now });
      return json(409, { detail: "A night is already running", night_id: "mock-night" });
    }
    save({ nightStart: now });
    return json(200, { night_id: "mock-night" });
  }
  if (path.startsWith("/api/sleep/me/nights/")) {
    const night = nightAt(now);
    if (!night) return json(404, { detail: "No such night" });
    if (night.status === "done" && !state.nights) save({ nights: 1 });
    return json(200, night);
  }
  if (path === "/api/sleep/me/wake") {
    if (CASE === "wakefail") return json(500, {});
    return json(200, {
      opener: "I kept coming back to your sister last night. I think the call is hard because the last one ended badly, not because you're busy. Want to plan the first two lines together?",
      self_prediction: { claim: "Practising a gentler opener will make my first reply this morning land better.", p: 0.7, outcome: "true" },
      what_changed: [
        "Its lean on why the call keeps slipping moved from 52% to 64%.",
        "It practised a gentler way to open hard topics; the gate kept it, +3 of 5.",
        "It set one prediction to check in three days.",
      ],
      before_after: {
        question: "I still haven't called her. What should I do?",
        before: "You could set a reminder and call at a time that suits you both. Small steps help.",
        after: "Start with the part you're avoiding: \"I've been thinking about our last call.\" You don't have to fix it in one go. Want to try the first line here?",
      },
    });
  }
  if (path === "/api/sleep/me/undo" && method === "POST") {
    if (state.undone) return json(409, { detail: "Nothing to undo" });
    save({ undone: true });
    return json(200, { checkpoint: "ckpt-0", night: iso(now).slice(0, 10) });
  }
  return json(404, { detail: "Not Found" });
}

export async function mockFetch(url, init = {}) {
  const path = new URL(url, window.location.origin).pathname;
  const method = (init.method || "GET").toUpperCase();
  let body = null;
  try {
    body = init.body ? JSON.parse(init.body) : null;
  } catch {
    body = null;
  }
  await delay(path.endsWith("/message") ? 60 : 220);
  return route(method, path, body, init.signal);
}
