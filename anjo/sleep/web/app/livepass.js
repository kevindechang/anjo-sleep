// MIND after each reply. The server runs MIND's live pass only once a reply's stream has ended, so
// a refresh at that moment still reads the old state. Instead, after each reply the panel polls
// GET /api/sleep/me/mind (and /me/thought when that view is open) until the pass has landed: its
// ``revision`` moved past the one from before the message, with ``live_pass`` no longer ``running``.
//
// - every 2 s for the first minute (the foreground watch);
// - then, while ``live_pass`` still says running (a slow model can take two minutes), every 10 s up
//   to 3 minutes from the reply;
// - when a pass that was running turns idle with no newer revision yet, one final read (the two
//   fields aren't read at the same instant), then stop;
// - a pass that never starts (her daily allowance spent, a night holding the host) ends the watch
//   after a short grace.
//
// Checks pause while the tab is hidden and pick up as soon as it shows again. While a watch runs,
// the panel says it is thinking about what you said; new items arrive with the panel's entrance.
// A watch that ends without seeing the pass land says so (still running at the cap, never started,
// or the last read failed), with a button to check again: the line never just disappears.

import { COPY } from "./copy.js";
import { byId } from "./dom.js";

export const POLL_MS = 2000;
export const CAP_MS = 60000;
export const SLOW_MS = 10000;
export const SLOW_CAP_MS = 180000;
export const IDLE_GRACE_MS = 10000;

// Whether revision ``now`` is past ``base``: numbers by size, anything else by any change.
export function newer(now, base) {
  if (now === undefined || now === null) return false;
  if (base === undefined || base === null) return true;
  if (typeof now === "number" && typeof base === "number") return now > base;
  return String(now) !== String(base);
}

// How a finished watch ended: "landed" (the line goes), or why it couldn't confirm the pass.
export function ending({ ok = true, base, revision, livePass, sawRunning, fresh = 0 }) {
  if (!ok) return "unconfirmed";
  if (newer(revision, base) || (revision === undefined && livePass === undefined && fresh > 0)) return "landed";
  if (livePass === "running") return "timeout";
  if (livePass === "idle") return sawRunning ? "landed" : "nopass";
  return "timeout";
}
const ENDING_COPY = { timeout: "live_timeout", unconfirmed: "live_unconfirmed", nopass: "live_nopass" };

// What to do after an answer: "poll" (again in 2 s), "slow" (again in 10 s), "final" (one last
// read, then stop) or "stop". ``base`` is the revision before the message, ``sawRunning`` whether
// any answer since said the pass was running, ``elapsed`` the time since the reply.
export function nextStep({ base, revision, livePass, sawRunning, elapsed, fresh = 0 }) {
  if (livePass === "running") {
    if (elapsed < CAP_MS) return "poll";
    return elapsed < SLOW_CAP_MS ? "slow" : "stop";
  }
  if (newer(revision, base)) return "stop";
  if (livePass === "idle") {
    if (sawRunning) return "final"; // it ran: read once more in case its revision landed just after
    return elapsed < IDLE_GRACE_MS ? "poll" : "stop"; // never started
  }
  if (revision === undefined && livePass === undefined && fresh > 0) return "stop"; // an API without the fields
  return elapsed < CAP_MS ? "poll" : "stop";
}

export function createLivePass({ panel, thought, onLine = () => {} }) {
  const line = byId("live-pass");
  const text = line.querySelector(".live-pass-text");
  const retry = byId("live-pass-retry");
  const tab = byId("tab-mind");
  let base;
  let run = 0;
  let timer = null;
  let resume = null; // the check that waits for the tab to show again

  function thinking(on, words = COPY.live_thinking) {
    line.hidden = !on;
    line.dataset.state = "busy";
    retry.hidden = true;
    text.textContent = words;
    if (on) tab.dataset.thinking = "true";
    else delete tab.dataset.thinking;
    onLine(on ? words : "");
  }

  // The watch ended without seeing the pass land: say so, and offer to check again.
  function stuck(how) {
    thinking(false);
    line.hidden = false;
    line.dataset.state = "stuck";
    text.textContent = COPY[ENDING_COPY[how]];
    retry.hidden = false;
    onLine(text.textContent);
  }

  function stop() {
    run += 1;
    window.clearTimeout(timer);
    resume = null;
    thinking(false);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden || !resume) return;
    const check = resume;
    resume = null;
    check();
  });

  retry.addEventListener("click", () => api.watch());

  const read = () => Promise.all([panel.refresh(), thought.isOpen() ? thought.refresh() : null]).then(([state]) => state);

  const api = {
    // A message is going out: the revision MIND is at before it.
    mark() {
      base = panel.revision();
    },
    // Its reply is in: watch for the live pass it asked for (a newer reply starts a new watch).
    watch() {
      stop();
      const mine = run;
      const from = base;
      const start = Date.now();
      let sawRunning = false;
      const later = (ms) => { timer = window.setTimeout(check, ms); };
      async function check() {
        if (mine !== run) return;
        if (document.hidden) { // paused: the next check runs when the tab shows again
          resume = check;
          return;
        }
        const state = await read();
        if (mine !== run) return; // a newer watch, or a night, took over
        if (state.livePass === "running") sawRunning = true;
        const step = nextStep({
          base: from, revision: state.revision, livePass: state.livePass, sawRunning,
          elapsed: Date.now() - start, fresh: state.fresh,
        });
        if (step === "poll") later(POLL_MS);
        else if (step === "slow") {
          thinking(true, COPY.live_slow);
          later(SLOW_MS);
        } else {
          const last = step === "final" ? await read() : state;
          if (mine !== run) return;
          const how = ending({ ok: last.ok, base: from, revision: last.revision, livePass: last.livePass,
            sawRunning: sawRunning || step === "final", fresh: last.fresh });
          if (how === "landed") thinking(false);
          else stuck(how);
        }
      }
      thinking(true);
      later(POLL_MS);
    },
    stop,
  };
  return api;
}
