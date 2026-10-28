// The live demo page's full step (Task 13a): her mind panel. Each section opens on one plain line
// and shows its detail when expanded. The server has already withheld every held-out text; a
// withheld item arrives as {text: null, withheld: true} and is shown as held back, never guessed at.
// The chart is drawn here as inline SVG, with a table of the same numbers as its text alternative.
(() => {
  "use strict";

  const demo = window.anjoDemo;
  if (!demo) return;
  const { call, element, copy, byId } = demo;
  const SVG = byId("chart-template").content.firstElementChild.namespaceURI; // an inline <svg>'s own namespace
  const CHART = { width: 320, height: 176, left: 34, right: 64, top: 12, bottom: 28 };
  const LABEL_GAP = 10; // viewBox units between two series' end labels (one label's height)
  const RESOLVED = ["needed", "not_needed", "censored"]; // an expectation's stored outcomes; the rest are still open

  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (slot, name) => (name in values ? String(values[name]) : slot));
  }

  const percent = (value) => (value === null || value === undefined ? copy.mind_none_yet : `${Math.round(value * 100)}%`);
  const day = (iso) => (iso ? String(iso).slice(0, 10) : "");
  const word = (prefix, code) => (code ? copy[`${prefix}_${code}`] || code : "");

  // Free text from her state: shown as sent, or marked as held back.
  function said(item, className) {
    if (!item) return element("span", `${className || ""} faint`.trim(), copy.mind_no_target);
    if (item.withheld) return element("span", `${className || ""} withheld-text`.trim(), copy.mind_withheld);
    return element("span", className || "", item.text);
  }

  function plainText(item) {
    if (!item) return copy.mind_no_target;
    return item.withheld ? copy.mind_withheld : item.text;
  }

  function bar(value) {
    const track = element("span", "bar");
    track.setAttribute("aria-hidden", "true");
    const level = element("span", "bar-fill");
    level.style.setProperty("--v", value === null || value === undefined ? 0 : Math.max(0, Math.min(1, value)));
    track.append(level);
    return track;
  }

  function tag(text, kind) {
    return element("span", `tag ${kind ? `tag-${kind}` : ""}`.trim(), text);
  }

  // One section: <details> whose summary is its heading and one plain line.
  function section(id, title, line, state) {
    const box = element("details", "mind-section");
    box.id = `mind-${id}`;
    const summary = element("summary");
    summary.append(element("h3", "", title), element("p", "plain", state === "ok" ? line : copy[`mind_state_${state}`]));
    const body = element("div", "mind-body");
    box.append(summary, body);
    return { box, body };
  }

  function questions(data) {
    const items = data.items;
    const { box, body } = section("questions", copy.mind_questions, fill(copy.mind_questions_line, { n: items.length }), data.state);
    body.append(element("p", "note", copy.mind_credence_note));
    for (const q of items) {
      const item = element("article", "q");
      item.append(element("p", "q-tags"), said(q.subject, "q-subject"));
      item.querySelector(".q-tags").append(tag(word("question_kind", q.kind)), tag(word("question_status", q.status), q.status));
      const list = element("ul", "credences");
      for (const h of q.hypotheses) {
        const row = element("li", h.catch_all ? "catch-all" : "");
        row.append(said(h.statement, "h-text"), bar(h.credence), element("span", "num", percent(h.credence)));
        list.append(row);
      }
      if (q.hypotheses.length) item.append(list);
      for (const p of q.predictions) {
        const prediction = element("p", "prediction");
        prediction.append(element("span", "label", copy.mind_prediction), said(p.claim), element("span", "num",
          fill(copy.mind_prediction_given, { given: p.given.map(percent).join(" / "), horizon: day(p.horizon) })),
        element("span", "label", word("prediction_outcome", p.outcome || "pending")));
        item.append(prediction);
      }
      body.append(item);
    }
    return box;
  }

  function targets(data) {
    const { box, body } = section("targets", copy.mind_targets, fill(copy.mind_targets_line, { n: data.items.length }), data.state);
    const list = element("ul", "rows");
    for (const t of data.items) {
      const row = element("li");
      const kind = t.kind === "commitment" && t.due ? fill(copy.mind_target_due, { date: day(t.due) }) : word("target_kind", t.kind);
      row.append(said(t.label, "row-text"), tag(kind), bar(t.confidence), element("span", "num", percent(t.confidence)));
      list.append(row);
    }
    body.append(element("p", "note", copy.mind_confidence_note), list);
    return box;
  }

  function candidate(c) {
    const row = element("li", `status-${c.status}`);
    row.append(tag(word("want_status", c.status), c.status), element("span", "row-text",
      `${word("want_kind", c.kind)} · ${word("action", c.action)} · `), said(c.target));
    if (c.reason) row.append(element("span", "reason", word("reason", c.reason)));
    return row;
  }

  function wants(data) {
    const line = data.night === null ? copy.mind_none_yet : fill(copy.mind_wants_line, { n: data.items.length, night: data.night });
    const { box, body } = section("wants", copy.mind_wants, line, data.state);
    const list = element("ul", "rows");
    data.items.forEach((c) => list.append(candidate(c)));
    body.append(list);
    return box;
  }

  function choiceLine(c) {
    if (!c.action) return fill(copy.mind_chose_nothing, { night: c.night });
    return fill(copy.mind_chose, { night: c.night, action: word("action", c.action), target: plainText(c.target) });
  }

  function choices(data) {
    const latest = data.items[0];
    const { box, body } = section("choices", copy.mind_choices, latest ? choiceLine(latest) : "", data.state);
    for (const c of data.items) {
      const night = element("article", "night");
      night.append(element("p", "night-head", `${fill(copy.slider_tick, { n: c.night })} · ${day(c.date)} · ${word("night_status", c.status)}`),
        element("p", "night-choice", choiceLine(c)));
      const facts = [c.outcome && fill(copy.mind_outcome, { outcome: word("outcome", c.outcome) }),
        c.note && fill(copy.mind_night_note, { note: word("note", c.note) })].filter(Boolean);
      if (facts.length) night.append(element("p", "note", facts.join(" · ")));
      if (c.beat.length) {
        const beat = element("ul", "rows beat");
        c.beat.forEach((each) => beat.append(candidate(each)));
        night.append(element("p", "label", c.action ? copy.mind_beat : copy.mind_held_back), beat);
      }
      body.append(night);
    }
    return box;
  }

  function expectations(data) {
    const { box, body } = section("expectations", copy.mind_expectations, fill(copy.mind_expectations_line, {
      n: data.items.length, resolved: data.items.filter((e) => RESOLVED.includes(e.outcome)).length,
    }), data.state);
    const list = element("ul", "rows");
    for (const e of data.items) {
      const row = element("li");
      row.append(said(e.target, "row-text"), element("span", "num", fill(copy.mind_expect_numbers, { p: percent(e.p_needed), horizon: day(e.horizon) })),
        tag(word("expect_outcome", e.outcome), e.outcome));
      list.append(row);
    }
    body.append(element("p", "note", copy.mind_expectations_note), list);
    return box;
  }

  function gates(data) {
    const latest = data.items[0];
    const failed = latest ? latest.gates.filter((g) => !g.passed).length : 0;
    const line = !latest ? "" : latest.gates.length === 0 ? fill(copy.mind_gates_none, { night: latest.night })
      : fill(failed ? copy.mind_gates_failed : copy.mind_gates_passed, { night: latest.night, n: failed });
    const { box, body } = section("gates", copy.mind_gates, line, data.state);
    for (const night of data.items) {
      const block = element("article", "night");
      block.append(element("p", "night-head", `${fill(copy.slider_tick, { n: night.night })} · ${word("night_status", night.status)}`));
      const list = element("ul", "rows");
      for (const g of night.gates) {
        const row = element("li", g.passed ? "pass" : "fail");
        row.append(tag(g.passed ? copy.mind_gate_pass : copy.mind_gate_fail, g.passed ? "pass" : "fail"),
          element("span", "row-text", word("gate", g.gate)),
          element("span", "num", fill(copy.mind_gate_numbers, { value: g.value ?? "–", threshold: g.threshold ?? "–" })));
        list.append(row);
      }
      if (!night.gates.length) list.append(element("li", "note", copy.mind_gates_skipped));
      block.append(list);
      body.append(block);
    }
    return box;
  }

  // ── the results chart ───────────────────────────────────────────────────

  function svgNode(name, attributes, text) {
    const node = document.createElementNS(SVG, name);
    for (const [key, value] of Object.entries(attributes || {})) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Each series' end label, moved apart so no two overlap: top to bottom, at least LABEL_GAP apart.
  function spread(ends) {
    const sorted = [...ends].sort((a, b) => a.y - b.y);
    const placed = [];
    for (const end of sorted) {
      const above = placed[placed.length - 1];
      placed.push({ ...end, y: above ? Math.max(end.y, above.y + LABEL_GAP) : end.y });
    }
    return placed;
  }

  function panel(life, dimension, arms, id) {
    const nights = life.nights;
    const last = Math.max(1, nights.length - 1);
    const x = (n) => CHART.left + (n / last) * (CHART.width - CHART.left - CHART.right);
    const y = (v) => CHART.top + (1 - v) * (CHART.height - CHART.top - CHART.bottom);
    const svg = svgNode("svg", { viewBox: `0 0 ${CHART.width} ${CHART.height}`, role: "img", "aria-labelledby": `${id}-t ${id}-d` });
    const lines = [];
    const ends = [];
    const desc = svgNode("desc", { id: `${id}-d` });
    svg.append(desc);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(svgNode("line", { class: "grid", x1: CHART.left, x2: CHART.width - CHART.right, y1: y(v), y2: y(v) }));
      svg.append(svgNode("text", { class: "axis", x: CHART.left - 6, y: y(v) + 3, "text-anchor": "end" }, v.toFixed(2)));
    }
    nights.forEach((night, n) => svg.append(svgNode("text", { class: "axis", x: x(n), y: CHART.height - 8, "text-anchor": "middle" }, String(night.night))));
    const sets = [...new Set(nights.map((night) => night.probes || "full"))];  // one line a probe set
    for (const arm of arms) for (const set of sets) {
      const points = nights.map((night, n) => [n, (night.probes || "full") === set && night.scores[arm] && night.scores[arm][dimension]]).filter(([, v]) => typeof v === "number");
      if (!points.length) continue;
      const daily = set === "full" ? "" : ` (${copy.chart_probes_subset})`;
      const group = svgNode("g", { class: `series arm-${arm}` });
      group.append(svgNode("polyline", { points: points.map(([n, v]) => `${x(n)},${y(v)}`).join(" "), ...(daily ? { "stroke-dasharray": "5 3" } : {}) }));
      points.forEach(([n, v]) => group.append(svgNode("circle", { cx: x(n), cy: y(v), r: 2.6 })));
      const [endN, endV] = points[points.length - 1];
      ends.push({ group, x: x(endN) + 6, y: y(endV) + 3, text: (copy[`chart_arm_short_${arm}`] || arm) + daily });
      svg.append(group);
      lines.push(fill(copy.chart_series_desc, { arm: (copy[`chart_arm_${arm}`] || arm) + daily, first: points[0][1].toFixed(2), last: endV.toFixed(2), nights: points.length }));
    }
    for (const end of spread(ends)) end.group.append(svgNode("text", { class: "series-label", x: end.x, y: end.y }, end.text));
    desc.textContent = lines.join(" ");
    const box = element("div", "chart-panel");
    const title = element("p", "panel-title", fill(copy.chart_panel_title, { dimension: word("dimension", dimension) }));
    title.id = `${id}-t`;
    box.append(title, svg);
    return box;
  }

  function numbers(life, dims, arms) {
    const table = element("table", "numbers");
    table.append(element("caption", "", copy.chart_table_caption));
    const head = element("tr");
    head.append(element("th", "", copy.chart_night), ...arms.flatMap((arm) => dims.map((d) => element("th", "", `${copy[`chart_arm_short_${arm}`] || arm} · ${word("dimension", d)}`))));
    head.querySelectorAll("th").forEach((th) => th.setAttribute("scope", "col"));
    const thead = element("thead");
    thead.append(head);
    const tbody = element("tbody");
    for (const night of life.nights) {
      const row = element("tr");
      const label = element("th", "", String(night.night));
      label.setAttribute("scope", "row");
      row.append(label, ...arms.flatMap((arm) => dims.map((d) => {
        const v = night.scores[arm] && night.scores[arm][d];
        return element("td", "num", typeof v === "number" ? v.toFixed(2) : "–");
      })));
      tbody.append(row);
    }
    table.append(thead, tbody);
    return table;
  }

  function chart(data) {
    const dev = data.development;
    const line = !dev ? copy.chart_none : fill(copy.chart_line, { nights: dev.lives[0].nights.length });
    const { box, body } = section("chart", copy.chart_heading, line, data.state === "none" ? "ok" : data.state);
    if (data.state !== "ok") {
      body.append(element("p", "note", data.state === "none" ? copy.chart_none : copy.mind_state_unavailable));
      return box;
    }
    if (dev) {
      const figure = element("figure", "chart");
      figure.append(element("figcaption", "", fill(copy.chart_dev_caption, { name: dev.lives[0].name })));
      const panels = element("div", "panels");
      data.dimensions.forEach((d, n) => panels.append(panel(dev.lives[0], d, data.arms, `chart-dev-${n}`)));
      const legend = element("ul", "legend");
      data.arms.forEach((arm) => {
        const item = element("li", `arm-${arm}`, copy[`chart_arm_${arm}`] || arm);
        legend.append(item);
      });
      const table = element("details", "numbers-box");
      const scroller = element("div", "numbers-scroll"); // a wide table scrolls here, by keyboard too
      scroller.tabIndex = 0;
      scroller.setAttribute("role", "region");
      scroller.setAttribute("aria-label", copy.chart_table_caption);
      scroller.append(numbers(dev.lives[0], data.dimensions, data.arms));
      table.append(element("summary", "", copy.chart_numbers), scroller);
      figure.append(panels, legend, table);
      body.append(figure);
    }
    const final = element("div", "final");
    final.append(element("p", "label", copy.chart_final_heading));
    final.append(element("p", "note", data.final ? fill(copy.chart_final_ready, { lives: data.final.lives.length }) : copy.chart_final_not_yet));
    body.append(final);
    return box;
  }

  // ── the Task 16 result and the ledger ───────────────────────────────────

  function task16(data) {
    const line = data.headline ? plainText(data.headline) : copy.task16_not_yet;
    const { box, body } = section("task16", copy.task16_heading, line, data.state === "none" ? "ok" : data.state);
    if (data.state !== "ok") {
      body.append(element("p", "note", data.state === "none" ? copy.task16_not_yet : copy.mind_state_unavailable));
      return box;
    }
    data.notes.forEach((note) => body.append(element("p", "note", plainText(note))));
    const list = element("ul", "rows");
    for (const each of data.lines) {
      const row = element("li");
      row.append(said(each, "row-text"));
      list.append(row);
    }
    body.append(list);
    return box;
  }

  function ledger(data) {
    const line = data.state !== "ok" ? "" : fill(data.verified ? copy.ledger_line_ok : copy.ledger_line_broken, {
      n: data.count, sync: data.in_sync === null ? copy.ledger_never_pushed : data.in_sync ? copy.ledger_in_sync : copy.ledger_out_of_sync,
    });
    const { box, body } = section("ledger", copy.ledger_heading, line, data.state);
    if (data.head) body.append(element("p", "hash", fill(copy.ledger_head, { hash: data.head })));
    const list = element("ol", "ledger");
    for (const e of data.entries) {
      const row = element("li");
      row.append(element("span", "num", `#${e.seq}`), tag(e.kind), element("code", "hash", e.sha256),
        said(e.note, "row-text"), element("span", "note", e.ts));
      list.append(row);
    }
    body.append(element("p", "note", copy.ledger_note), list);
    return box;
  }

  function glance(view) {
    const choice = view.choices.items[0];
    const parts = [];
    if (view.questions.state === "ok") parts.push(fill(copy.mind_glance_questions, { n: view.questions.items.length }));
    if (view.targets.state === "ok") parts.push(fill(copy.mind_glance_targets, { n: view.targets.items.length }));
    if (choice) parts.push(choiceLine(choice));
    return parts.length ? parts.join(" ") : copy.mind_glance_empty;
  }

  async function start() {
    const band = byId("mind");
    band.hidden = false;
    try {
      const view = await call("/mind");
      const night = view.choices.items[0];
      byId("mind-asof").textContent = night ? fill(copy.mind_asof, { night: night.night, date: day(night.date) }) : copy.mind_asof_none;
      byId("mind-glance").textContent = glance(view);
      byId("mind-sections").replaceChildren(questions(view.questions), targets(view.targets), wants(view.wants),
        choices(view.choices), expectations(view.expectations), gates(view.gates), chart(view.chart), task16(view.task16),
        ledger(view.ledger));
    } catch (error) {
      byId("mind-glance").textContent = error.message;
    }
  }

  document.addEventListener("demo:visit", start);
})();
