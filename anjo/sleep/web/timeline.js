// How she got here (demo/timeline.py): the quiet days of her logged work, in order, grouped by
// day. mind.js fetches /mind once and hands the payload over (demo:mind), so this draws without a
// request of its own. Every node is one logged record and links to its event id; the summary is a
// plain line, and the exact words wait inside it, so a person reads what happened before the why.
// Unknown node kinds and fields this build doesn't render are skipped, never guessed at; a withheld
// text arrives as {text: null, withheld: true} and is shown as held back.
(() => {
  "use strict";

  const demo = window.anjoDemo;
  if (!demo) return;
  const { element, copy, byId } = demo;
  const KINDS = new Set(["question", "credence", "prediction", "resolution", "search", "decision", "practice"]);

  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (slot, name) => (name in values ? String(values[name]) : slot));
  }

  const percent = (value) => (typeof value === "number" ? `${Math.round(value * 100)}%` : copy.mind_none_yet);
  const day = (iso) => (iso ? String(iso).slice(0, 10) : "");
  const word = (prefix, code) => (code ? copy[`${prefix}_${code}`] || code : "");

  // A free text from her records: as sent, or marked as held back. A record the store no longer
  // holds has no words, so only its id is shown.
  function said(item, className) {
    if (!item) return element("span", `${className || ""} faint`.trim(), copy.mind_no_target);
    if (item.withheld) return element("span", `${className || ""} withheld-text`.trim(), copy.mind_withheld);
    return element("span", className || "", typeof item.text === "string" ? item.text : "");
  }

  // Logged order: the server's clock first, its sequence number second, arrival last for ties.
  function order(items) {
    return items
      .map((item, index) => ({ item, index }))
      .sort((a, b) => {
        const at = String(a.item.at || "").localeCompare(String(b.item.at || ""));
        if (at) return at;
        const sa = a.item.seq === null || a.item.seq === undefined ? Infinity : a.item.seq;
        const sb = b.item.seq === null || b.item.seq === undefined ? Infinity : b.item.seq;
        return sa === sb ? a.index - b.index : sa - sb;
      })
      .map((each) => each.item);
  }

  // Consecutive nodes of one day become one group. A day that arrives out of order starts its own
  // group rather than moving another node: the page never rewrites the record's order.
  function group(items) {
    const groups = [];
    for (const item of order(items)) {
      const key = item.day || day(item.at);
      let last = groups[groups.length - 1];
      if (!last || last.day !== key) {
        last = { day: key, items: [] };
        groups.push(last);
      }
      last.items.push(item);
    }
    return groups;
  }

  function row(label, value) {
    const line = element("div", "tl-row");
    line.append(element("dt", "", label), element("dd", "", undefined));
    const cell = line.children[line.children.length - 1];
    if (typeof value === "string" || typeof value === "number") cell.textContent = String(value);
    else if (value) cell.append(value);
    return line;
  }

  function exact(value) {
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(2);
    return String(value);
  }

  function cites(items, label) {
    if (!items || !items.length) return null;
    const box = element("div", "tl-cites-box");
    box.append(element("p", "label", label));
    const list = element("ul", "tl-cites");
    for (const item of items) {
      const line = element("li");
      line.append(element("code", "", item.id));
      if (typeof item.text === "string") line.append(said(item));
      if (item.source) line.append(element("span", "note", item.source));
      list.append(line);
    }
    box.append(list);
    return box;
  }

  function plainLine(item) {
    if (item.kind === "question") {
      return fill(copy.timeline_plain_question, {
        kind: word("question_kind", item.code),
        origin: copy[`timeline_origin_${item.origin}`] || item.origin || "",
      });
    }
    if (item.kind === "credence") {
      const n = (item.moves || []).length;
      return fill(copy[n === 1 ? "timeline_plain_credence_one" : "timeline_plain_credence_many"], { n });
    }
    if (item.kind === "prediction") return copy.timeline_plain_prediction;
    if (item.kind === "resolution") {
      return fill(copy.timeline_plain_resolution, { verdict: word("timeline_verdict", item.verdict) });
    }
    if (item.kind === "search") return copy.timeline_plain_search;
    if (item.kind === "practice") {
      return fill(copy.timeline_plain_practice, {
        target: item.target && typeof item.target.text === "string" ? item.target.text : copy.mind_no_target,
      });
    }
    if (item.kind === "decision") {
      return item.action
        ? fill(copy.timeline_plain_decision, { action: word("action", item.action) })
        : copy.timeline_plain_decision_none;
    }
    return "";
  }

  function body(item) {
    const out = element("div", "tl-body");
    const list = element("dl", "tl-rows");
    if (item.kind === "question") {
      const detail = item.detail || {};
      list.append(row(copy.timeline_detail_subject, said(detail.subject, "tl-exact")));
      list.append(row(copy.timeline_detail_origin, copy[`timeline_origin_${item.origin}`] || item.origin || ""));
      list.append(row(copy.timeline_detail_kind, word("question_kind", item.code)));
      list.append(row(copy.timeline_detail_status, word("question_status", item.status)));
      out.append(list);
      const raised = cites(item.cites, copy.timeline_detail_cites);
      if (raised) out.append(raised);
    } else if (item.kind === "credence") {
      list.append(row(copy.timeline_detail_moved_by, item.moved_by || ""));
      list.append(row(copy.timeline_detail_source, item.source || ""));
      for (const move of item.moves || []) {
        const line = element("div", "tl-move");
        line.append(said(move.statement, "tl-exact"));
        const numbers = element("p", "tl-numbers");
        numbers.append(element("span", "num", `${copy.timeline_detail_before} ${percent(move.before)}`),
          element("span", "num", `${copy.timeline_detail_after} ${percent(move.after)}`));
        if (move.catch_all) numbers.append(element("span", "tag", copy.timeline_catch_all));
        line.append(numbers);
        out.append(line);
      }
      out.append(list);
      const evidence = cites(item.evidence, copy.timeline_detail_evidence);
      if (evidence) out.append(evidence);
    } else if (item.kind === "prediction") {
      const detail = item.detail || {};
      list.append(row(copy.timeline_detail_prediction, said(detail.claim, "tl-exact")));
      list.append(row(copy.timeline_detail_horizon, day(item.horizon)));
      const given = Object.entries(item.given || {}).map(([, p]) => percent(p));
      if (given.length) list.append(row(copy.timeline_detail_given, given.join(" · ")));
      list.append(row(copy.timeline_detail_outcome, word("prediction_outcome", item.outcome || "pending")));
      out.append(list);
    } else if (item.kind === "resolution") {
      const detail = item.detail || {};
      list.append(row(copy.timeline_detail_prediction, said(detail.claim, "tl-exact")));
      list.append(row(copy.timeline_detail_verdict, word("timeline_verdict", item.verdict)));
      list.append(row(copy.timeline_detail_outcome, word("prediction_outcome", item.outcome)));
      out.append(list);
      const seen = cites(item.seen, copy.timeline_detail_seen);
      if (seen) out.append(seen);
    } else if (item.kind === "search") {
      const detail = item.detail || {};
      list.append(row(copy.timeline_detail_search, said(detail.query, "tl-exact")));
      list.append(row(copy.timeline_detail_answer, detail.answer ? said(detail.answer, "tl-exact")
        : copy.timeline_no_answer));
      out.append(list);
      const results = cites(detail.items, copy.timeline_detail_results);
      if (results) out.append(results);
      else out.append(element("p", "note", copy.timeline_no_results));
    } else if (item.kind === "practice") {
      const detail = item.detail || {};
      list.append(row(copy.timeline_detail_target, said(item.target, "tl-exact")));
      list.append(row(copy.timeline_detail_scenes, detail.scenes || 0));
      list.append(row(copy.timeline_detail_kept, detail.kept || 0));
      list.append(row(copy.timeline_detail_outcome,
        word("prediction_outcome", item.outcome || "pending")));
      out.append(list);
    } else if (item.kind === "decision") {
      list.append(row(copy.timeline_detail_action, item.action ? word("action", item.action) : copy.timeline_no_action));
      list.append(row(copy.timeline_detail_target, said(item.target, "tl-exact")));
      if (item.outcome) list.append(row(copy.timeline_detail_outcome,
        fill(copy.mind_outcome, { outcome: word("outcome", item.outcome) })));
      list.append(row(copy.timeline_detail_status, word("night_status", item.status)));
      list.append(row(copy.timeline_detail_confidence,
        item.confidence === null || item.confidence === undefined ? copy.timeline_no_confidence
          : percent(item.confidence)));
      out.append(list);
    }
    const record = item.record || {};
    const at = fill(copy.timeline_detail_at, {});
    const recordRows = element("dl", "tl-rows tl-record");
    recordRows.append(row(copy.timeline_detail_record, record.event || ""),
      row(copy.timeline_detail_night, record.night || ""),
      row(at, item.at || ""));
    out.append(recordRows);
    return out;
  }

  function card(item, index) {
    const holder = element("li", "tl-item");
    const box = element("details", `tl-node tl-${item.kind}`);
    const event = item.record && item.record.event ? String(item.record.event) : "";
    const slug = `${item.kind}-${index}-${event.replace(/[^a-zA-Z0-9_-]/g, "-") || "record"}`;
    box.id = `tl-${slug}`;
    box.setAttribute("data-kind", item.kind);
    box.setAttribute("data-day", item.day || "");
    box.setAttribute("data-event", event);
    const summary = element("summary", "tl-summary");
    summary.append(element("span", `tl-kind tl-kind-${item.kind}`, copy[`timeline_kind_${item.kind}`] || item.kind),
      element("span", "tl-plain", plainLine(item)));
    if (event) {
      const link = element("a", "tl-event", event);
      link.setAttribute("href", `#${box.id}`);
      link.setAttribute("title", fill(copy.timeline_event, { id: event }));
      summary.append(link);
    }
    box.append(summary, body(item));
    holder.append(box);
    return holder;
  }

  function counts(view) {
    const c = view.counts || {};
    return fill(copy.timeline_counts, {
      questions: c.question || 0, credences: c.credence || 0, predictions: c.prediction || 0,
      resolutions: c.resolution || 0, searches: c.search || 0, decisions: c.decision || 0,
    });
  }

  function draw(view) {
    const state = byId("mind-timeline-state");
    const body = byId("mind-timeline-body");
    if (!view || view.state !== "ok") {
      state.textContent = view && view.state === "none" ? copy.timeline_none : copy.timeline_unavailable;
      body.replaceChildren();
      return;
    }
    const days = view.quiet_days === 1 ? copy.timeline_days_one : fill(copy.timeline_days_many, { n: view.quiet_days || 0 });
    state.textContent = `${days} · ${counts(view)}`;
    const items = (view.items || []).filter((item) => KINDS.has(item.kind));
    const blocks = [];
    for (const part of group(items)) {
      const section = element("section", "tl-day");
      section.setAttribute("data-day", part.day);
      section.append(element("h4", "tl-day-head", fill(copy.timeline_day, { date: part.day })));
      const list = element("ol", "tl-nodes");
      part.items.forEach((item, at) => list.append(card(item, at)));
      section.append(list);
      blocks.push(section);
    }
    body.replaceChildren(...blocks);
  }

  document.addEventListener("demo:mind", (event) => draw(event.detail ? event.detail.timeline : null));
})();
