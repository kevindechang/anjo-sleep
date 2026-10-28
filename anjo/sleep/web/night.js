// Her latest night as one loop (one system): how she slept, then dream → rehearse → learn → check →
// wake, left to right, from the /mind payload's story (demo/night_view.py). mind.js fetches /mind
// once and hands the payload over (demo:mind), so this draws without a request of its own. A stage
// that didn't run is greyed with the reason the night wrote; a nap stops after the dream. Plain words
// first, exact numbers second, and every word goes in through textContent.
(() => {
  "use strict";

  const demo = window.anjoDemo;
  if (!demo) return;
  const { element, copy, byId } = demo;

  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (slot, name) => (name in values ? String(values[name]) : slot));
  }

  // A plain line with its key number set apart: "She rehearsed <strong>16</strong> scenes."
  function plain(template, values) {
    const line = element("p", "stage-line");
    for (const part of String(template).split(/(\{\w+\})/)) {
      const slot = /^\{(\w+)\}$/.exec(part);
      if (slot && slot[1] in values) line.append(element(slot[1] === "n" ? "strong" : "span", slot[1] === "n" ? "stage-key" : "", String(values[slot[1]])));
      else if (part) line.append(element("span", "", part));
    }
    return line;
  }

  function exact(value) {
    if (typeof value === "boolean") return value ? "✓" : copy.story_off;
    if (typeof value !== "number") return String(value);
    return Number.isInteger(value) ? String(value) : value.toFixed(Math.abs(value) < 0.1 ? 3 : 2);
  }

  function slept(sleep) {
    if (sleep.failed) return copy.story_sleep_failed;
    if (sleep.not_run) return sleep.not_run;
    const kind = copy[`sleep_kind_${sleep.kind}`] || sleep.kind;
    return `${kind} · ${fill(copy.sleep_note, { pressure: sleep.pressure.toFixed(2), threshold: sleep.threshold.toFixed(2),
      reason: copy[`sleep_reason_${sleep.reason}`] || sleep.reason })}`;
  }

  function lineOf(stage) {
    if (stage.state === "failed") return element("p", "stage-line", fill(copy.story_failed_line, { step: stage.step }));
    if (stage.state === "not_run") return element("p", "stage-line", stage.reason);
    if (stage.stage === "check") return plain(stage.promoted ? copy.story_line_check : copy.story_line_check_held, { n: stage.key, of: stage.of });
    if (stage.key === null) return element("p", "stage-line", copy.story_line_wake_uncounted);
    return plain(copy[`story_line_${stage.stage}`], { n: stage.key });
  }

  // The exact numbers: each one's value, or "not yet" with why it didn't run.
  function numbers(stage) {
    const list = element("dl", "stage-numbers");
    for (const each of stage.numbers) {
      const value = "value" in each ? element("dd", "num", exact(each.value)) : element("dd", "num faint", copy.story_not_yet);
      if (!("value" in each)) value.setAttribute("title", each.not_run);
      list.append(element("dt", "", copy[`story_number_${each.name}`] || each.name), value);
    }
    return list;
  }

  function gates(stage) {
    const list = element("ul", "stage-gates");
    for (const gate of stage.gates) {
      const row = element("li", `gate-${gate.kind}`);
      const said = gate.kind === "skip" ? copy.mind_gate_skip : gate.kind === "pass" ? copy.mind_gate_pass : copy.mind_gate_fail;
      row.append(element("span", `tag tag-${gate.kind}`, said), element("span", "", copy[`gate_${gate.gate}`] || gate.gate),
        element("span", "num", gate.kind === "skip" ? copy.mind_gate_off : fill(copy.mind_gate_numbers, { value: gate.value, threshold: gate.threshold })));
      list.append(row);
    }
    return list;
  }

  function cites(stage) {
    const box = element("p", "stage-cites");
    box.append(element("span", "label", copy.story_cites));
    stage.cites.forEach((id) => box.append(element("code", "", id)));
    return box;
  }

  // The evidence the page already shows; a closed panel section opens when its link is followed.
  function evidence(id) {
    const link = element("a", "stage-link", copy.story_evidence);
    link.setAttribute("href", `#${id}`);
    link.addEventListener("click", () => {
      const target = byId(id);
      if (target && target.tagName === "DETAILS") target.open = true;
    });
    return link;
  }

  function card(stage) {
    const box = element("li", `loop-stage state-${stage.state}`);
    box.setAttribute("data-stage", stage.stage);
    box.setAttribute("data-state", stage.state);
    const head = element("p", "stage-head");
    head.append(element("span", "stage-name", copy[`story_stage_${stage.stage}`] || stage.stage));
    if (stage.state !== "ran") head.append(element("span", "tag", stage.state === "failed" ? copy.story_failed : copy.story_not_run));
    box.append(head, lineOf(stage));
    if (stage.numbers.length) box.append(numbers(stage));
    if (stage.gates && stage.gates.length) box.append(gates(stage));
    if (stage.cites.length) box.append(cites(stage));
    if (stage.link && stage.state !== "not_run") box.append(evidence(stage.link));
    return box;
  }

  function draw(view) {
    const asof = byId("night-loop-asof");
    const body = byId("night-loop-body");
    if (!view || view.state !== "ok") {
      asof.textContent = view && view.state === "none" ? copy.story_none : copy.story_unavailable;
      body.replaceChildren();
      return;
    }
    asof.textContent = fill(copy.story_night, { night: view.night, date: view.date, status: copy[`night_status_${view.status}`] || view.status });
    const parts = [element("p", "loop-sleep", slept(view.sleep))];
    if (view.napped) parts.push(element("p", "loop-nap", copy.story_nap));
    const loop = element("ol", view.napped ? "loop napped" : "loop");
    view.stages.forEach((stage) => loop.append(card(stage)));
    parts.push(loop);
    body.replaceChildren(...parts);
  }

  document.addEventListener("demo:mind", (event) => draw(event.detail ? event.detail.story : null));
})();
