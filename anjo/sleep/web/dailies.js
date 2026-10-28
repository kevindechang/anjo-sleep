// The live demo page's Dailies: the morning sheet for one night, as a film crew reviews the night's
// footage the morning after. A slate (which night, which morning, what the night did), then three
// takes: what she (his model) took from his day, one line she'd now say differently, and what the night cost.
// The server has already withheld every held-out text; a withheld item arrives as
// {text: null, withheld: true} and is shown as held back. Nothing here calls a model.
(() => {
  "use strict";

  const demo = window.anjoDemo;
  if (!demo) return;
  const { call, element, copy, byId } = demo;
  const sheet = byId("dailies-sheet");
  const picker = byId("dailies-night");
  let asked = 0; // the newest request wins: an older answer arriving late is dropped

  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (slot, name) => (name in values ? String(values[name]) : slot));
  }

  const day = (iso) => (iso ? String(iso).slice(0, 10) : "");
  const word = (prefix, code) => (code ? copy[`${prefix}_${code}`] || code : "");
  const known = (value) => value !== null && value !== undefined;

  function dollars(value) {
    if (!known(value)) return copy.cost_unrecorded;
    return `$${value.toFixed(value < 0.01 ? 4 : value < 1 ? 3 : 2)}`;
  }

  const minutes = (value) => (known(value) ? value.toFixed(1) : copy.cost_unrecorded);

  function said(item, className) {
    if (!item) return element("span", `${className || ""} faint`.trim(), copy.mind_no_target);
    if (item.withheld) return element("span", `${className || ""} withheld-text`.trim(), copy.dailies_withheld);
    return element("span", className || "", item.text);
  }

  function plainText(item) {
    if (!item) return copy.mind_no_target;
    return item.withheld ? copy.dailies_withheld : item.text;
  }

  function tag(text, kind) {
    return element("span", `tag ${kind ? `tag-${kind}` : ""}`.trim(), text);
  }

  // ── the slate ───────────────────────────────────────────────────────────

  function slate(card) {
    const box = element("header", "slate");
    const clapper = element("div", "clapper");
    clapper.setAttribute("aria-hidden", "true");
    const cells = element("dl", "slate-cells");
    // Notes under a cell: the morning's probes with no verdict, and a gate switched off (skipped, never passed).
    const skipped = (card.skipped_gates || []).map((g) => word("gate", g)).join(", ");
    for (const [label, value, kind, note] of [
      [copy.dailies_slate_night, String(card.night), "night", ""],
      [copy.dailies_slate_morning, day(card.morning), "morning",
        card.no_verdict ? fill(copy.dailies_no_verdict, { n: card.no_verdict }) : ""],
      [copy.dailies_slate_status, word("night_status", card.status), `status status-${card.status}`,
        skipped ? `${skipped}: ${copy.dailies_gate_skipped}` : ""],
    ]) {
      const cell = element("div", `slate-cell slate-${kind}`);
      cell.append(element("dt", "", label), element("dd", "", value));
      if (note) cell.append(element("dd", "slate-note", note));
      cells.append(cell);
    }
    box.append(clapper, cells);
    return box;
  }

  // One take: its number in the margin, its heading, its body.
  function take(number, id, title) {
    const box = element("section", `take take-${id}`);
    box.setAttribute("aria-labelledby", `dailies-${id}`);
    const no = element("p", "take-no", String(number).padStart(2, "0"));
    no.setAttribute("aria-hidden", "true");
    const heading = element("h3", "", title);
    heading.id = `dailies-${id}`;
    const body = element("div", "take-body");
    box.append(no, heading, body);
    return { box, body };
  }

  function unreadable(body, state, none) {
    body.append(element("p", "take-empty", state === "unavailable" ? copy.mind_state_unavailable : none));
  }

  // ── 01: what she took from his day ──────────────────────────────────────

  function candidate(c) {
    const row = element("li", `status-${c.status}`);
    row.append(element("span", "row-text", `${word("want_kind", c.kind)} · ${word("action", c.action)} · `), said(c.target));
    if (c.reason) row.append(element("span", "reason", word("reason", c.reason)));
    return row;
  }

  function choseLine(card, chose) {
    if (!chose || !chose.action) return fill(copy.mind_chose_nothing, { night: card.night });
    const line = fill(copy.mind_chose, { night: card.night, action: word("action", chose.action), target: plainText(chose.target) });
    return chose.outcome ? `${line} ${fill(copy.mind_outcome, { outcome: word("outcome", chose.outcome) })}.` : line;
  }

  function learned(card) {
    const data = card.learned;
    const { box, body } = take(1, "learned", copy.dailies_learned);
    if (data.state !== "ok") {
      unreadable(body, data.state, copy.dailies_learned_none);
      return box;
    }
    if (data.targets.length) {
      const list = element("ul", "moves");
      for (const t of data.targets) {
        const row = element("li", `move move-${t.change}`);
        row.append(tag(copy[`dailies_${t.change}`], t.change), element("span", "move-kind", word("target_kind", t.kind)),
          said(t.label, "move-label"));
        list.append(row);
      }
      body.append(list);
      if (data.more) body.append(element("p", "note", fill(copy.dailies_more, { n: data.more })));
    }
    body.append(element("p", "chose", choseLine(card, data.chose)));
    if (data.beat.length) {
      const beat = element("ul", "beaten");
      data.beat.forEach((c) => beat.append(candidate(c)));
      body.append(element("p", "label", data.chose && data.chose.action ? copy.mind_beat : copy.mind_held_back), beat);
    }
    for (const q of data.questions) {
      const question = element("figure", "asked");
      const caption = element("figcaption", "label", `${copy.dailies_question} · ${word("question_kind", q.kind)}`);
      question.append(caption, said(q.subject, "asked-text"));
      body.append(question);
    }
    return box;
  }

  // ── 02: one line she'd now say differently ──────────────────────────────

  // A side with no sentence at that point arrives as null: it says so, never "nothing".
  function side(kind, label, item) {
    const box = element("figure", `side-take side-${kind}`);
    box.append(element("figcaption", "label", label));
    const quote = element("blockquote", "line-text");
    quote.append(item ? said(item) : element("span", "faint line-missing", copy.dailies_line_missing));
    box.append(quote);
    return box;
  }

  function line(card) {
    const data = card.line;
    const { box, body } = take(2, "line", copy.dailies_line);
    const scene = element("div", "cue");
    scene.append(element("p", "note", copy.dailies_asked), element("p", "cue-title", data.scene.title),
      element("p", "cue-prompt", data.scene.prompt));
    body.append(scene);
    if (data.state !== "ok") {
      const reason = data.reason ? copy[`dailies_line_${data.reason}`] : copy.mind_state_none;
      unreadable(body, data.state, reason);
      return box;
    }
    if (data.same) {
      body.append(element("p", "take-empty", copy.dailies_line_same));
    } else {
      const before = data.before_night === 0 ? copy.dailies_before_base
        : known(data.before_night) ? fill(copy.dailies_before, { night: data.before_night }) : copy.dailies_before_unknown;
      const versus = element("div", "versus");
      const arrow = element("span", "versus-arrow", "→");
      arrow.setAttribute("aria-hidden", "true");
      versus.append(side("before", before, data.before), arrow,
        side("after", fill(copy.dailies_after, { night: data.after_night }), data.after));
      body.append(versus);
    }
    body.append(element("p", "note line-source", fill(copy.dailies_line_source, { date: day(data.generated_at) })));
    return box;
  }

  // ── 03: what the night cost ─────────────────────────────────────────────

  function costRow(cells, header) {
    const row = element("tr");
    row.append(element("th", "", cells[0]));
    row.firstChild.setAttribute("scope", header ? "col" : "row");
    for (const cell of cells.slice(1)) {
      const node = element(header ? "th" : "td", known(cell.value) || header ? "num" : "num unrecorded", cell.text);
      if (header) node.setAttribute("scope", "col");
      row.append(node);
    }
    return row;
  }

  function table(data) {
    const grid = element("table", "cost-table");
    const head = element("thead");
    head.append(costRow([copy.cost_col_part, { text: copy.cost_col_usd }, { text: copy.cost_col_gpu }], true));
    const rows = element("tbody");
    for (const p of data.parts) {
      rows.append(costRow([copy[`cost_part_${p.part}`],
        { value: p.usd, text: dollars(p.usd) }, { value: p.gpu_minutes, text: minutes(p.gpu_minutes) }]));
    }
    const foot = element("tfoot");
    foot.append(costRow([copy.cost_total,
      { value: data.usd, text: dollars(data.usd) }, { value: data.gpu_minutes, text: minutes(data.gpu_minutes) }]));
    grid.append(head, rows, foot);
    return grid;
  }

  function meter(p) {
    const top = Math.max(p.ceiling_usd, p.worst_month_usd || 0, p.month_usd) * 1.08;
    const track = element("div", `meter ${p.under_ceiling ? "under" : "over"}`);
    track.setAttribute("aria-hidden", "true");
    const level = element("span", "meter-fill");
    level.style.setProperty("--v", Math.min(1, p.month_usd / top));
    const ceiling = element("span", "meter-ceiling");
    ceiling.style.setProperty("--at", p.ceiling_usd / top);
    ceiling.append(element("span", "meter-tick-label", fill(copy.cost_ceiling, { ceiling: dollars(p.ceiling_usd) })));
    track.append(level, ceiling);
    if (known(p.worst_month_usd)) {
      const worst = element("span", "meter-worst");
      worst.style.setProperty("--at", Math.min(1, p.worst_month_usd / top));
      track.append(worst);
    }
    return track;
  }

  function projection(p) {
    const box = element("div", "projection");
    box.append(element("p", "label", copy.cost_month_label));
    if (!p) {
      box.append(element("p", "note", copy.cost_month_none));
      return box;
    }
    box.append(meter(p), element("p", "projection-line",
      fill(copy.cost_month, { month: dollars(p.month_usd), ceiling: dollars(p.ceiling_usd) })));
    if (known(p.worst_month_usd)) box.append(element("p", "note", fill(copy.cost_month_worst, { worst: dollars(p.worst_month_usd) })));
    return box;
  }

  function cost(card) {
    const data = card.cost;
    const { box, body } = take(3, "cost", copy.dailies_cost);
    if (data.state !== "ok") {
      unreadable(body, data.state, copy.mind_state_none);
      return box;
    }
    const ledger = element("div", "cost-sheet");
    ledger.append(table(data), projection(data.projection));
    body.append(ledger, element("p", "note", fill(copy.dailies_cost_note, { source: copy[`cost_source_${data.source}`] })));
    return box;
  }

  // ── the sheet ───────────────────────────────────────────────────────────

  function pick(card) {
    picker.replaceChildren(...card.nights.map((n) => {
      const option = element("option", "", fill(copy.dailies_option, { night: n.night, date: day(n.date) }));
      option.value = String(n.night);
      option.selected = n.night === card.night;
      return option;
    }));
    picker.disabled = card.nights.length < 2;
  }

  function render(card) {
    if (card.state !== "ok") {
      sheet.replaceChildren(element("p", "sheet-state", card.state === "none" ? copy.dailies_none : copy.dailies_unavailable));
      picker.replaceChildren();
      picker.disabled = true;
      return;
    }
    pick(card);
    const takes = element("div", "takes");
    takes.append(learned(card), line(card), cost(card));
    sheet.replaceChildren(slate(card), takes);
  }

  async function show(night) {
    const mine = ++asked;
    sheet.setAttribute("aria-busy", "true");
    try {
      const card = await call(night ? `/dailies/${night}` : "/dailies");
      if (mine === asked) render(card);
    } catch (error) {
      if (mine === asked) sheet.replaceChildren(element("p", "sheet-state", error.message));
    } finally {
      if (mine === asked) sheet.setAttribute("aria-busy", "false");
    }
  }

  function start() {
    byId("dailies").hidden = false;
    show(null);
  }

  picker.addEventListener("change", () => show(Number(picker.value)));
  document.addEventListener("demo:visit", start);
})();
