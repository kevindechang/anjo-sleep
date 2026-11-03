// The product view: chat on the left, the MIND panel on the right (a tab on a phone), the talker
// label, Sleep now with its night view, and the wake. Without a judge token the page goes back to
// the landing's code field.

import { api, href, ready, signOut, token } from "./api.js";
import { COPY, fill } from "./copy.js";
import { byId, say, wireTabs } from "./dom.js";
import { createChat, talkerLine } from "./chat.js";
import { createLivePass } from "./livepass.js";
import { createPanel } from "./panel.js";
import { createSince } from "./morning.js";
import { createSleep } from "./sleep.js";
import { createThought } from "./thought.js";
import { createWake } from "./wake.js";

const TALKER_POLL_MS = 15000;
const TALKER_WARM_POLL_MS = 60000;
const NARROW = window.matchMedia("(max-width: 959.98px)");

function leave(signedOut) {
  signOut();
  window.location.assign(href(signedOut ? "/sleep/app?signed_out=1#grow" : "/sleep/app"));
}

// Chat and Mind as tabs on a narrow screen; both panes side by side on a wide one.
function createTabs() {
  const grid = document.querySelector(".live-grid");
  const tabs = [byId("tab-chat"), byId("tab-mind")];
  const badge = byId("mind-badge");
  const tabsBar = document.querySelector(".tabs");
  grid.dataset.view = "chat";

  // A pane that was hidden opens at its top (Mind) or at the latest message (Chat), not wherever
  // the other pane had been scrolled to.
  function land(view) {
    if (!NARROW.matches) return;
    if (view === "mind") {
      const start = byId("pane-mind").getBoundingClientRect().top + window.scrollY - tabsBar.getBoundingClientRect().bottom;
      if (window.scrollY > start) window.scrollTo({ top: Math.max(0, start), behavior: "instant" });
      return;
    }
    byId("composer").scrollIntoView({ block: "end", behavior: "instant" });
  }

  function select(view, focus = false) {
    const changed = grid.dataset.view !== view;
    grid.dataset.view = view;
    if (changed) land(view);
    tabs.forEach((tab) => {
      const on = tab.id === `tab-${view}`;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
      if (on && focus) tab.focus();
    });
    if (view === "mind") badge.hidden = true;
  }

  wireTabs(tabs, (i, focus) => select(tabs[i].id.slice(4), focus));

  return {
    select,
    noteNew() {
      if (NARROW.matches && grid.dataset.view !== "mind") badge.hidden = false;
    },
  };
}

// On a phone, one line in the Chat tab on what MIND last did (or is doing); a tap opens Mind.
function createMindLine(open) {
  const line = byId("mind-line");
  const text = byId("mind-line-text");
  let summary = "";
  let busy = "";
  const draw = () => {
    const words = busy || summary;
    text.textContent = words;
    line.hidden = !words;
    line.dataset.busy = String(Boolean(busy));
  };
  line.addEventListener("click", open);
  return {
    summary(words) { summary = words; draw(); },
    busy(words) { busy = words; draw(); },
  };
}

// Who answers the chat, whether Glimmer is loading, and which model MIND thinks with: always shown,
// checked every 15 s while Glimmer loads and every minute once warm (a night's serve, Undo, or a GPU
// stop can change it), and again right after each of those.
function createTalker() {
  const box = byId("talker");
  const text = byId("talker-text");
  let timer = null;
  let last = null;
  const known = { adapter: null, thinker: null };

  function draw() {
    if (!last) return;
    box.dataset.state = last.state === "warm" ? "warm" : last.state === "waking" ? "waking" : "cold";
    text.textContent = talkerLine(last, known);
    box.hidden = false;
  }

  async function check() {
    window.clearTimeout(timer);
    try {
      last = await api.talker();
    } catch {
      timer = window.setTimeout(check, TALKER_POLL_MS * 2);
      return;
    }
    draw();
    timer = window.setTimeout(check, last && last.state === "warm" ? TALKER_WARM_POLL_MS : TALKER_POLL_MS);
  }

  return {
    check,
    note(change) {
      Object.assign(known, change);
      draw();
    },
  };
}

async function boot() {
  await ready();
  if (!token()) {
    window.location.replace(href("/sleep/app#grow"));
    return;
  }
  window.addEventListener("judge:signed-out", () => leave(true), { once: true });
  byId("leave").addEventListener("click", () => leave(false));
  byId("brand").href = href("/sleep/app");

  const status = byId("live-status");
  const tabs = createTabs();
  const talker = createTalker();
  const mindLine = createMindLine(() => tabs.select("mind", true));
  const panel = createPanel({
    onNew: () => tabs.noteNew(),
    onActivity: (words, live) => {
      mindLine.summary(words);
      if (live && live.thinker) talker.note({ thinker: live.thinker });
    },
  });
  let livePass = null;
  const chat = createChat({
    onSend: () => livePass.mark(),
    onReply: () => {
      livePass.watch(); // MIND's live pass runs after the reply: poll until it lands
      talker.check();
    },
  });
  const since = createSince();
  const thought = createThought({
    chat,
    onAnswer: () => tabs.select("chat"), // on a phone, watch the answer go out
    onView: (view) => since.update(view),
  });
  livePass = createLivePass({ panel, thought, onLine: (words) => mindLine.busy(words) });
  const wake = createWake({
    chat, panel, thought, since,
    onUndo: () => {
      talker.note({ adapter: "before" });
      talker.check();
    },
  });
  const sleep = createSleep({
    chat, panel,
    onFinish: (night) => {
      thought.refresh();
      if (night && night.status === "done" && night.adapter_live) talker.note({ adapter: "tonight" });
      talker.check(); // the night's serve may have changed who answers
    },
    onAwake: (night) => {
      tabs.select("chat");
      wake.show(night);
    },
    onStatus: (text, kind) => say(status, text, kind),
    onShow: () => {
      livePass.stop(); // the night view refreshes MIND itself
      tabs.select("chat");
    },
  });

  talker.check();
  panel.refresh();
  thought.refresh();
  chat.warm();
  sleep.resume();
}

boot();
