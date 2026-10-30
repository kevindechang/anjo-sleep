// A fake page for web/timeline.js (tests/test_demo_timeline.py): only the DOM calls timeline.js
// makes, so it runs under node with no browser and no network. It hands the script one /mind payload,
// as mind.js does (demo:mind), and prints what it drew: each day group, its nodes in order, their
// logged event ids, the links, every class, and the page's text.
"use strict";
const fs = require("fs");
const vm = require("vm");

const [script, copyJson, mindJson] = process.argv.slice(2);

class FakeNode {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.className = "";
    this.children = [];
    this.attributes = {};
    this.own = "";
    this.id = "";
    this.hidden = false;
  }
  set textContent(value) { this.own = String(value); this.children = []; }
  get textContent() { return this.own + this.children.map((child) => child.textContent).join(" "); }
  append(...nodes) {
    for (const node of nodes) {
      if (!(node instanceof FakeNode)) throw new Error("timeline.js appends nodes, never strings or HTML");
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children = []; this.own = ""; this.append(...nodes); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  addEventListener() {}
  walk(visit) { visit(this); this.children.forEach((child) => child.walk(visit)); }
}

const ids = { "mind-timeline-state": new FakeNode("p"), "mind-timeline-body": new FakeNode("div") };
const listeners = [];
const element = (tag, className, text) => {
  const node = new FakeNode(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const sandbox = {
  window: { anjoDemo: { copy: JSON.parse(copyJson), byId: (id) => ids[id] || null, element } },
  document: { addEventListener: (type, listener) => type === "demo:mind" && listeners.push(listener) },
};
vm.runInNewContext(fs.readFileSync(script, "utf8"), sandbox, { filename: script });
listeners.forEach((listener) => listener({ detail: { timeline: JSON.parse(mindJson) } }));

const groups = [];
const links = [];
const classes = new Set();
ids["mind-timeline-body"].walk((node) => {
  node.className.split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
  if (node.tagName === "SECTION" && node.getAttribute("data-day")) {
    groups.push({ day: node.getAttribute("data-day"), kinds: [], events: [] });
  }
  if (node.tagName === "DETAILS" && groups.length) {
    const group = groups[groups.length - 1];
    group.kinds.push(node.getAttribute("data-kind"));
    group.events.push(node.getAttribute("data-event"));
  }
  if (node.tagName === "A") links.push(node.getAttribute("href"));
});
process.stdout.write(JSON.stringify({
  groups, links, classes: [...classes].sort(),
  text: `${ids["mind-timeline-state"].textContent} ${ids["mind-timeline-body"].textContent}`,
}));
