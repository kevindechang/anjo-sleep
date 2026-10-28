// The live demo page's full step (Task 13a): the night slider and the starter scenes' toggles.
// Same rules as app.js: requests go to this host only, every word comes from the page's copy, and
// model or server text goes in through textContent only.
(() => {
  "use strict";

  const demo = window.anjoDemo;
  if (!demo) return;
  const { call, element, paragraphs, copy, byId } = demo;
  const answers = new Map(); // "<stop>/<scene>" -> a stop's pre-generated answer
  const toggleCache = new Map(); // scene -> the toggles' pairs
  const voted = new Set(); // scenes voted on from this address today (or that take no vote): the server checks too
  let nights = null;
  let scenes = [];

  // A copy template's {name} slots, filled with plain values: the result is only ever set as text.
  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (slot, name) => (name in values ? String(values[name]) : slot));
  }

  function statusWord(status) {
    return copy[`night_status_${status}`] || status;
  }

  function stopTick(stop) {
    return stop.night === 0 ? copy.slider_tick_base : fill(copy.slider_tick, { n: stop.night });
  }

  // The stop in one plain line: whose weights answer, and why when they are an earlier night's.
  function stopLine(stop) {
    if (stop.night === 0) return copy.slider_stop_base;
    if (!stop.available) return fill(copy.slider_stop_unavailable, { n: stop.night });
    const values = { n: stop.night, from: stop.from_night, status: statusWord(stop.status) };
    if (!stop.fallback) return fill(copy.slider_stop_own, values);
    return stop.from_night === 0 ? fill(copy.slider_stop_fallback_base, values) : fill(copy.slider_stop_fallback, values);
  }

  function stopMeta(stop, source) {
    const parts = [stopTick(stop), copy[`arm_label_${stop.arm}`], copy.development];
    if (stop.checkpoint) parts.splice(1, 0, `${copy.checkpoint} ${stop.checkpoint}`);
    if (source) parts.push(source === "live" ? copy.live : copy.pregenerated);
    return parts.join(" · ");
  }

  function currentStop() {
    return nights.stops[Number(byId("night-range").value)];
  }

  function currentScene() {
    return scenes.find((scene) => scene.id === byId("slider-scene").value);
  }

  function showAnswer(stop, question, reply, note) {
    const card = byId("stop-card");
    card.querySelector(".stop-meta").textContent = stopMeta(stop, reply && reply.source);
    card.querySelector(".stop-line").textContent = stopLine(stop);
    card.querySelector(".question").textContent = question || "";
    const text = card.querySelector(".side-text");
    card.classList.toggle("withheld", Boolean(reply && reply.withheld));
    if (!reply) {
      text.replaceChildren(element("p", "note", note || (stop.available ? copy.slider_not_ready : copy.slider_unavailable_note)));
      return;
    }
    paragraphs(text, reply.text);
    if (reply.withheld) text.prepend(element("p", "withheld-note", copy.withheld));
  }

  async function loadStop() {
    const stop = currentStop();
    const scene = currentScene();
    byId("night-range").setAttribute("aria-valuetext", `${stopTick(stop)}. ${stopLine(stop)}`);
    byId("stop-ask-submit").textContent = fill(copy.slider_ask_submit, { night: stopTick(stop) });
    byId("stop-ask-submit").disabled = !stop.available;
    document.querySelectorAll("#night-ticks li").forEach((tick, n) => tick.classList.toggle("current", n === nights.stops.indexOf(stop)));
    if (!scene || !stop.available || !stop.ready.includes(scene.id)) {
      showAnswer(stop, scene && scene.prompt, null);
      return;
    }
    if (!voted.has(scene.id)) { // her weights here share the blind pair's: they wait for the vote
      showAnswer(stop, scene.prompt, null, copy.slider_vote_first);
      return;
    }
    const key = `${stop.key}/${scene.id}`;
    try {
      if (!answers.has(key)) {
        answers.set(key, await call(`/nights/${encodeURIComponent(stop.key)}/starters/${encodeURIComponent(scene.id)}`, "POST"));
      }
      if (currentStop() === stop && currentScene() === scene) showAnswer(stop, scene.prompt, answers.get(key));
    } catch (error) {
      demo.say(error.message);
    }
  }

  async function askStop(event) {
    event.preventDefault();
    const field = byId("stop-ask-text");
    const submit = byId("stop-ask-submit");
    const text = field.value.trim();
    if (!text) {
      demo.say(copy.empty);
      field.focus();
      return;
    }
    const stop = currentStop();
    submit.disabled = true;
    field.setAttribute("aria-busy", "true");
    demo.say(copy.slider_asking);
    try {
      const answer = await call(`/nights/${encodeURIComponent(stop.key)}/ask`, "POST", { text });
      if (answer.kind === "asleep") {
        demo.say(answer.waking ? copy.waking : copy.asleep);
        return;
      }
      if (answer.kind === "crisis") {
        showAnswer(stop, text, { text: answer.text, withheld: false, source: null });
        byId("stop-card").querySelector(".stop-meta").textContent = "";
      } else {
        showAnswer(stop, text, answer);
      }
      field.value = "";
      demo.say("");
    } catch (error) {
      demo.say(error.message);
    } finally {
      submit.disabled = !currentStop().available;
      field.removeAttribute("aria-busy");
    }
  }

  function renderSlider() {
    const range = byId("night-range");
    const ticks = byId("night-ticks");
    range.max = String(nights.stops.length - 1);
    range.value = range.max; // her latest first
    ticks.replaceChildren();
    nights.stops.forEach((stop, n) => {
      const tick = element("li", stop.available ? "" : "unavailable", stopTick(stop));
      tick.style.setProperty("--at", nights.stops.length > 1 ? n / (nights.stops.length - 1) : 0);
      tick.addEventListener("click", () => { range.value = String(n); loadStop(); });
      ticks.append(tick);
    });
    const select = byId("slider-scene");
    select.replaceChildren(...scenes.map((scene) => {
      const option = element("option", "", scene.title);
      option.value = scene.id;
      return option;
    }));
    byId("slider-body").hidden = false;
    loadStop();
  }

  // ── the toggles, on a starter pair once it is voted on ─────────────────────

  function toggleSide(side) {
    const box = element("section", "toggle-side");
    box.dataset.arm = side.arm;
    box.append(element("h4", "side-name", side.label));
    const text = element("div", "side-text");
    paragraphs(text, side.text);
    if (side.withheld) text.prepend(element("p", "withheld-note", copy.withheld));
    box.append(text);
    return box;
  }

  function toggleSource(pair, kind) {
    if (kind === "swap") return fill(copy.toggle_swap_source, { model: pair.model });
    const stop = pair.stop;
    return fill(copy.toggle_notes_off_source, { night: stop.from_night, checkpoint: stop.checkpoint });
  }

  async function showToggle(node, scene, kind, button) {
    const view = node.querySelector(".toggle-view");
    node.querySelectorAll(".toggle-row button").forEach((each) => each.setAttribute("aria-pressed", String(each === button)));
    try {
      if (!toggleCache.has(scene)) toggleCache.set(scene, await call(`/toggles/${encodeURIComponent(scene)}`, "POST"));
      const pair = toggleCache.get(scene)[kind];
      if (!pair) {
        view.replaceChildren(element("p", "note", copy.toggle_missing));
      } else {
        const sides = element("div", "sides");
        sides.append(toggleSide(pair.with), toggleSide(pair.without));
        view.replaceChildren(element("p", "source", `${copy.pregenerated} · ${toggleSource(pair, kind)}`), sides);
      }
      view.hidden = false;
    } catch (error) {
      demo.say(error.message);
    }
  }

  function offerToggles(node, scene) {
    const offered = nights && nights.toggles.find((each) => each.scene === scene);
    if (!offered || (!offered.notes_off && !offered.swap) || node.querySelector(".toggles")) return;
    const box = element("div", "toggles");
    box.append(element("p", "toggles-head", copy.toggles_heading), element("p", "note", copy.toggles_note));
    const row = element("div", "toggle-row");
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", copy.toggles_heading);
    for (const kind of ["notes_off", "swap"]) {
      if (!offered[kind]) continue;
      const button = element("button", "choice choice-quiet", copy[`toggle_${kind}`]);
      button.type = "button";
      button.setAttribute("aria-pressed", "false");
      button.addEventListener("click", () => showToggle(node, scene, kind, button));
      row.append(button);
    }
    const view = element("div", "toggle-view");
    view.hidden = true;
    view.setAttribute("aria-live", "polite");
    box.append(row, view);
    node.querySelector(".card").append(box);
  }

  async function start(event) {
    scenes = event.detail.scenes || [];
    byId("nights").hidden = false;
    try {
      nights = await call("/nights");
    } catch (error) {
      byId("slider-state").textContent = error.message;
      return;
    }
    (nights.voted || []).forEach((scene) => voted.add(scene)); // voted from this address today, any visit
    if (nights.state !== "ok") {
      byId("slider-state").textContent = copy.slider_unavailable;
      return;
    }
    if (nights.stops.length < 2) {
      byId("slider-state").textContent = copy.slider_no_nights;
    }
    renderSlider();
  }

  document.addEventListener("demo:visit", start);
  document.addEventListener("demo:settled", (event) => {
    voted.add(event.detail.scene);
    offerToggles(event.detail.node, event.detail.scene);
    const scene = nights && nights.stops && currentScene();
    if (scene && scene.id === event.detail.scene) loadStop();
  });
  byId("night-range").addEventListener("input", loadStop);
  byId("slider-scene").addEventListener("change", loadStop);
  byId("stop-ask-form").addEventListener("submit", askStop);
})();
