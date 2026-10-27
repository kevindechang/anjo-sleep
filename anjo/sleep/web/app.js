// The live demo page (Task 13a, minimal step). Vanilla JS, no framework, and every request goes
// to this host only. Every word comes from the copy the server put in the page (words.COPY), and
// every piece of model or server text goes in through textContent, never as HTML.
(() => {
  "use strict";

  const API = "/api/sleep/demo";
  const copy = JSON.parse(document.getElementById("demo-copy").textContent);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const byId = (id) => document.getElementById(id);
  const status = byId("status");
  let visit = null;

  function say(text) {
    status.textContent = text || "";
  }

  async function call(path, method = "GET", body = undefined) {
    const headers = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (visit) headers["X-Demo-Visit"] = visit;
    let response;
    try {
      response = await fetch(API + path, {
        method, headers, credentials: "same-origin", cache: "no-store",
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (error) {
      throw new Error(copy.offline);
    }
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : copy.lost_thread;
      throw new Error(detail);
    }
    return data;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function paragraphs(container, text) {
    container.replaceChildren();
    for (const block of String(text).split(/\n\s*\n/)) {
      if (block.trim()) container.append(element("p", "", block.trim()));
    }
  }

  // The brief is plain text: "## " lines are headings and "- " lines are list items.
  function renderBrief(text) {
    const body = byId("brief-body");
    body.replaceChildren();
    let list = null;
    for (const raw of String(text).split("\n")) {
      const line = raw.trim();
      if (!line) {
        list = null;
      } else if (line.startsWith("## ")) {
        body.append(element("h3", "", line.slice(3)));
        list = null;
      } else if (line.startsWith("- ")) {
        if (!list) {
          list = element("ul");
          body.append(list);
        }
        list.append(element("li", "", line.slice(2)));
      } else {
        body.append(element("p", "", line));
        list = null;
      }
    }
  }

  function sourceLine(source) {
    if (source.kind !== "pre-generated") return copy.live;
    const checkpoint = source.checkpoint
      ? `${copy.checkpoint} ${source.checkpoint} · ${copy.night} ${source.night}`
      : copy.base_checkpoint;
    return `${copy.pregenerated} · ${checkpoint}`;
  }

  function show(node) {
    byId("thread").prepend(node);
    const card = node.querySelector(".card");
    card.scrollIntoView({ block: "start", behavior: reduceMotion.matches ? "auto" : "smooth" });
    card.focus({ preventScroll: true });
  }

  function exchange(templateId, who, question) {
    const node = byId(templateId).content.firstElementChild.cloneNode(true);
    node.querySelector(".who").textContent = who;
    node.querySelector(".question").textContent = question;
    return node;
  }

  function fact(list, term, value) {
    list.append(element("dt", "", term), element("dd", "", value));
  }

  function showReveal(node, reveal) {
    for (const name of ["left", "right"]) {
      const facts = reveal[name];
      const side = node.querySelector(`[data-side="${name}"]`);
      side.dataset.arm = facts.arm === "4" ? "product" : "baseline";
      const box = side.querySelector(".reveal");
      const list = element("dl");
      fact(list, copy.prompt_tokens, `${Number(facts.prompt_tokens).toLocaleString("en")} ${copy.tokens}`);
      fact(list, copy.cost, facts.cost_usd === null ? copy.cost_unmeasured : `$${Number(facts.cost_usd).toFixed(4)}`);
      box.replaceChildren(element("p", "reveal-who", facts.label), list);
      box.hidden = false;
    }
    const verdict = node.querySelector(".verdict");
    verdict.textContent = `${copy.reveal_heading}. ${copy.your_vote}: ${copy[`vote_${reveal.choice}`]}.`;
    verdict.hidden = false;
    node.querySelector(".card").classList.add("revealed");
  }

  async function vote(node, pairId, choice) {
    const group = node.querySelector(".vote");
    const buttons = group.querySelectorAll("button");
    buttons.forEach((button) => { button.disabled = true; });
    try {
      const reveal = await call("/vote", "POST", { pair_id: pairId, choice });
      group.hidden = true;
      showReveal(node, reveal);
      say("");
      node.querySelector(".verdict").focus();
    } catch (error) {
      buttons.forEach((button) => { button.disabled = false; });
      say(error.message);
    }
  }

  function fillPair(node, pair) {
    node.querySelector(".source").textContent = sourceLine(pair.source);
    for (const name of ["left", "right"]) {
      const side = pair[name];
      const box = node.querySelector(`[data-side="${name}"]`);
      const text = box.querySelector(".side-text");
      paragraphs(text, side.text);
      if (side.withheld) {
        box.classList.add("withheld");
        text.prepend(element("p", "withheld-note", copy.withheld));
      }
    }
    const group = node.querySelector(".vote");
    if (!pair.votable) {
      group.hidden = true;
      node.querySelector(".unvotable").hidden = false;
      return node;
    }
    group.querySelectorAll("button.choice").forEach((button) => {
      button.addEventListener("click", () => vote(node, pair.pair_id, button.value));
    });
    return node;
  }

  async function starter(scene, button) {
    button.disabled = true;
    say("");
    try {
      const pair = await call(`/starters/${encodeURIComponent(scene.id)}`, "POST");
      show(fillPair(exchange("pair-template", copy.she_asks, scene.prompt), pair));
    } catch (error) {
      say(error.message);
    } finally {
      button.disabled = false;
    }
  }

  function renderScenes(scenes) {
    const list = byId("scene-list");
    list.replaceChildren();
    for (const scene of scenes) {
      const button = element("button", "scene");
      button.type = "button";
      button.append(element("span", "scene-title", scene.title), element("span", "scene-prompt", scene.prompt));
      if (!scene.ready) {
        button.disabled = true;
        button.append(element("span", "scene-state", copy.scene_not_ready));
      }
      button.addEventListener("click", () => starter(scene, button));
      const item = element("li");
      item.append(button);
      list.append(item);
    }
  }

  // Past the day's warm time, or outside the live hours, her own questions close and the page says why.
  function liveNote(state) {
    const text = state === "open" ? "" : copy[`live_${state}`];
    if (!text) return;
    byId("ask-form").hidden = true;
    const note = byId("live-note");
    note.textContent = text;
    note.hidden = false;
  }

  async function ask(event) {
    event.preventDefault();
    const field = byId("ask-text");
    const submit = event.currentTarget.querySelector("button[type=submit]");
    const text = field.value.trim();
    if (!text) {
      say(copy.empty);
      field.focus();
      return;
    }
    submit.disabled = true;
    field.setAttribute("aria-busy", "true");
    say(copy.asking);
    try {
      const answer = await call("/ask", "POST", { text });
      if (answer.kind === "asleep") {
        say(answer.waking ? copy.waking : copy.asleep);
        return;
      }
      if (answer.kind === "crisis") {
        const node = exchange("crisis-template", copy.you_asked, text);
        paragraphs(node.querySelector(".fixed"), answer.text);
        show(node);
      } else {
        show(fillPair(exchange("pair-template", copy.you_asked, text), answer));
      }
      field.value = "";
      say("");
    } catch (error) {
      say(error.message);
      liveNote(["spent", "closed"].find((state) => copy[`live_${state}`] === error.message) || "open");
    } finally {
      submit.disabled = false;
      field.removeAttribute("aria-busy");
    }
  }

  async function start(event) {
    event.preventDefault();
    const box = byId("adult");
    if (!box.checked) {
      say(copy.adult_required);
      box.focus();
      return;
    }
    try {
      const data = await call("/visits", "POST", { adult: true });
      visit = data.token;
      byId("gate").hidden = true;
      byId("scenes").hidden = false;
      byId("ask").hidden = false;
      say("");
      byId("scenes-heading").focus();
    } catch (error) {
      say(error.message);
    }
  }

  async function load() {
    try {
      const data = await call("/page");
      renderBrief(data.brief);
      renderScenes(data.scenes);
      liveNote(data.live);
    } catch (error) {
      byId("brief-body").replaceChildren(element("p", "", error.message));
    }
  }

  // One column (below the CSS's 62rem): the brief starts folded so the gate is in view.
  if (!window.matchMedia("(min-width: 62rem)").matches) byId("brief").querySelector("details").open = false;
  byId("gate-form").addEventListener("submit", start);
  byId("ask-form").addEventListener("submit", ask);
  byId("ask-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) byId("ask-form").requestSubmit();
  });
  load();
})();
