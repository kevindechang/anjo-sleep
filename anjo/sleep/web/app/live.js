// The product view: chat on the left, the MIND panel on the right (a tab on a phone), the talker
// label, Sleep now with its night view, and the wake. Without a judge token the page goes back to
// the landing's code field.

import { api, href, ready, signOut, token } from "./api.js";
import { COPY, fill } from "./copy.js";
import { byId, say, wireTabs } from "./dom.js";
import { createChat } from "./chat.js";
import { createPanel } from "./panel.js";
import { createSince } from "./morning.js";
import { createSleep } from "./sleep.js";
import { createThought } from "./thought.js";
import { createWake } from "./wake.js";

const TALKER_POLL_MS = 15000;
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

// "Glimmer is waking (~N min), replies from Muse Spark meanwhile", or nothing once it is warm.
function createTalker() {
  const box = byId("talker");
  const text = byId("talker-text");
  let timer = null;

  async function check() {
    window.clearTimeout(timer);
    let data;
    try {
      data = await api.talker();
    } catch {
      timer = window.setTimeout(check, TALKER_POLL_MS * 2);
      return;
    }
    const state = data && data.state;
    if (state !== "waking" && state !== "cold") {
      box.hidden = true;
      return;
    }
    box.dataset.state = state;
    const eta = Number(data.eta_s);
    text.textContent = state === "cold" ? COPY.talker_cold
      : Number.isFinite(eta) && eta > 0 ? fill(COPY.talker_waking, { min: Math.max(1, Math.round(eta / 60)) })
        : COPY.talker_waking_soon;
    box.hidden = false;
    timer = window.setTimeout(check, TALKER_POLL_MS);
  }

  return { check };
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
  const panel = createPanel({ onNew: () => tabs.noteNew() });
  const chat = createChat({
    onReply: () => {
      panel.refresh();
      talker.check();
    },
  });
  const since = createSince();
  const thought = createThought({
    chat,
    onAnswer: () => tabs.select("chat"), // on a phone, watch the answer go out
    onView: (view) => since.update(view),
  });
  const wake = createWake({ chat, panel, thought, since });
  const sleep = createSleep({
    chat, panel,
    onFinish: () => thought.refresh(),
    onAwake: () => {
      tabs.select("chat");
      wake.show();
    },
    onStatus: (text, kind) => say(status, text, kind),
    onShow: () => tabs.select("chat"),
  });

  talker.check();
  panel.refresh();
  thought.refresh();
  chat.warm();
  sleep.resume();
}

boot();
