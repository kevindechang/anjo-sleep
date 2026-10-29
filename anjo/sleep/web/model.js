// The person's own model (Task 9): what changed since a date, undo the last night, and delete all
// sleep data. Same rules as the rest of the page: requests go to this host only, every server
// string goes in through textContent, and each destructive action needs a second, inline button.
// The section stays hidden for a visitor who isn't the account's own signed-in session.
(() => {
  "use strict";

  const demo = window.anjoDemo;
  if (!demo) return;
  const { copy, element, byId } = demo;
  const NIGHTS = "/api/sleep/nights";
  const WHAT_CHANGED = "/api/sleep/what-changed";
  const UNDO = "/api/sleep/undo";
  const DELETE = "/api/sleep/delete";
  const ACTIONS = ["undo", "delete"];
  let deleted = false;

  function fill(template, values) {
    return String(template).replace(/\{(\w+)\}/g, (slot, name) => (name in values ? String(values[name]) : slot));
  }

  function say(text) {
    byId("yours-result").textContent = text || "";
  }

  function statusWord(status) {
    return copy[`night_status_${status}`] || status;
  }

  async function call(path, method = "GET", body = undefined) {
    const headers = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let response;
    try {
      response = await fetch(path, {
        method, headers, credentials: "same-origin", cache: "no-store",
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (error) {
      const offline = new Error(copy.offline);
      offline.status = 0;
      throw offline;
    }
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok) {
      const detail = data && typeof data.detail === "string" ? data.detail : copy.yours_failed;
      const failed = new Error(detail);
      failed.status = response.status;
      throw failed;
    }
    return data;
  }

  function row(className, parts) {
    const item = element("li", className);
    parts.forEach((part) => item.append(part));
    return item;
  }

  function nightRow(night) {
    return row("yours-night", [
      element("span", "yours-night-date", String(night.night)),
      element("span", "yours-night-status", statusWord(night.status)),
    ]);
  }

  function changeRow(change) {
    return row("yours-change", [
      element("span", "yours-change-night", String(change.night)),
      element("span", "yours-change-kind", String(change.kind)),
      element("span", "yours-change-line", String(change.line)),
    ]);
  }

  function state(id, text) {
    byId(id).textContent = text || "";
  }

  function failure(error) {
    return error.status === 401 || error.status === 403 ? copy.yours_unavailable : error.message;
  }

  function emptyAfterDelete(error, empty) {
    return deleted && (error.status === 401 || error.status === 403) ? empty : failure(error);
  }

  async function loadNights() {
    const list = byId("yours-nights");
    byId("yours-retry").hidden = true;
    state("yours-nights-state", copy.yours_loading);
    list.replaceChildren();
    try {
      const nights = await call(NIGHTS);
      nights.forEach((night) => list.append(nightRow(night)));
      state("yours-nights-state", nights.length ? "" : copy.yours_nights_empty);
      return true;
    } catch (error) {
      state("yours-nights-state", emptyAfterDelete(error, copy.yours_nights_empty));
      const unavailable = error.status === 401 || error.status === 403;
      byId("yours-retry").hidden = unavailable;
      return !unavailable;
    }
  }

  async function loadChanges() {
    const list = byId("yours-changes");
    const since = byId("yours-since").value;
    state("yours-changes-state", copy.yours_loading);
    list.replaceChildren();
    try {
      const path = since ? `${WHAT_CHANGED}?since=${encodeURIComponent(since)}` : WHAT_CHANGED;
      const changes = await call(path);
      changes.forEach((change) => list.append(changeRow(change)));
      state("yours-changes-state", changes.length ? "" : copy.yours_changes_empty);
    } catch (error) {
      state("yours-changes-state", emptyAfterDelete(error, copy.yours_changes_empty));
    }
  }

  async function refresh() {
    await Promise.all([loadNights(), loadChanges()]);
  }

  function disarm() {
    ACTIONS.forEach((action) => {
      byId(`yours-${action}`).hidden = false;
      byId(`yours-${action}-step`).hidden = true;
    });
  }

  function arm(action) {
    disarm();
    say("");
    byId(`yours-${action}`).hidden = true;
    const step = byId(`yours-${action}-step`);
    step.hidden = false;
    step.querySelector("button").focus();
  }

  async function perform(action) {
    const step = byId(`yours-${action}-step`);
    const buttons = step.querySelectorAll("button");
    buttons.forEach((button) => { button.disabled = true; });
    step.setAttribute("aria-busy", "true");
    say("");
    try {
      if (action === "undo") {
        const done = await call(UNDO, "POST");
        disarm();
        say(fill(copy.yours_undo_done, { night: done.night }));
      } else {
        await call(DELETE, "POST", { confirm: "delete" });
        deleted = true;
        disarm();
        say(copy.yours_delete_done);
        byId("yours-undo").disabled = true;
        byId("yours-delete").disabled = true;
      }
      await refresh();
    } catch (error) {
      say(error.message);
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
      step.removeAttribute("aria-busy");
    }
  }

  async function start() {
    const week = new Date();
    week.setDate(week.getDate() - 7);
    byId("yours-since").value = week.toISOString().slice(0, 10);
    if (!(await loadNights())) return;
    byId("yours").hidden = false;
    await loadChanges();
  }

  ACTIONS.forEach((action) => {
    byId(`yours-${action}`).addEventListener("click", () => arm(action));
    byId(`yours-${action}-yes`).addEventListener("click", () => perform(action));
    byId(`yours-${action}-no`).addEventListener("click", () => {
      disarm();
      say("");
      byId(`yours-${action}`).focus();
    });
  });
  byId("yours-since").addEventListener("change", loadChanges);
  byId("yours-retry").addEventListener("click", refresh);
  start();
})();
