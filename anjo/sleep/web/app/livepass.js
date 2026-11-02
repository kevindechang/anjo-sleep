// MIND after each reply. The server runs MIND's live pass only once a reply's stream has ended, so
// a refresh at that moment still reads the old state. Instead, after each reply the panel polls
// GET /api/sleep/me/mind every 2 s (and /me/thought when that view is open) until the pass has
// landed: its ``revision`` moved past the one from before the message, with ``live_pass`` no longer
// ``running``. A pass that never starts (her daily allowance spent, a night holding the host) ends
// the watch after a short grace, and every watch ends by about a minute. While it runs, the panel
// says it is thinking about what you said; new items arrive with the panel's own entrance.

import { COPY } from "./copy.js";
import { byId } from "./dom.js";

export const POLL_MS = 2000;
export const CAP_MS = 60000;
export const IDLE_GRACE_MS = 10000;

// Whether revision ``now`` is past ``base``: numbers by size, anything else by any change.
export function newer(now, base) {
  if (now === undefined || now === null) return false;
  if (base === undefined || base === null) return true;
  if (typeof now === "number" && typeof base === "number") return now > base;
  return String(now) !== String(base);
}

// Whether the watch after a reply can stop, from the latest answer: ``base`` is the revision before
// the message, ``sawRunning`` whether any answer since said the pass was running.
export function settled({ base, revision, livePass, sawRunning, elapsed, fresh = 0 }) {
  if (elapsed >= CAP_MS) return true;
  if (livePass === "running") return false;
  if (newer(revision, base)) return true;
  if (livePass === "idle") return sawRunning || elapsed >= IDLE_GRACE_MS; // ran with no change, or never started
  return revision === undefined && livePass === undefined && fresh > 0; // an API without the fields
}

export function createLivePass({ panel, thought }) {
  const line = byId("live-pass");
  const tab = byId("tab-mind");
  let base;
  let run = 0;
  let timer = null;

  line.querySelector(".live-pass-text").textContent = COPY.live_thinking;

  function thinking(on) {
    line.hidden = !on;
    if (on) tab.dataset.thinking = "true";
    else delete tab.dataset.thinking;
  }

  function stop() {
    run += 1;
    window.clearTimeout(timer);
    thinking(false);
  }

  return {
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
      thinking(true);
      const tick = async () => {
        const [state] = await Promise.all([panel.refresh(), thought.isOpen() ? thought.refresh() : null]);
        if (mine !== run) return; // a newer watch, or a night, took over
        if (state.livePass === "running") sawRunning = true;
        const done = settled({
          base: from, revision: state.revision, livePass: state.livePass, sawRunning,
          elapsed: Date.now() - start, fresh: state.fresh,
        });
        if (done) thinking(false);
        else timer = window.setTimeout(tick, POLL_MS);
      };
      timer = window.setTimeout(tick, POLL_MS);
    },
    stop,
  };
}
