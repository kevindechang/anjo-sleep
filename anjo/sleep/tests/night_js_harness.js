// A fake page for web/night.js (tests/test_night_view.py): only the DOM calls night.js makes, so it
// runs under node with no browser and no network. It hands the script one /mind payload, as mind.js
// does (demo:mind), and prints what it drew: each stage and its state, the text, the links, every class, the lede.
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
  }
  set textContent(value) { this.own = String(value); this.children = []; }
  get textContent() { return this.own + this.children.map((child) => child.textContent).join(" "); }
  append(...nodes) {
    for (const node of nodes) {
      if (!(node instanceof FakeNode)) throw new Error("night.js appends nodes, never strings or HTML");
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children = []; this.own = ""; this.append(...nodes); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  addEventListener() {}
  walk(visit) { visit(this); this.children.forEach((child) => child.walk(visit)); }
}

const ids = { "night-loop-asof": new FakeNode("p"), "night-loop-body": new FakeNode("div"),
  "night-loop-lede": new FakeNode("p") };
ids["night-loop-lede"].hidden = true;  // as the page serves it
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
listeners.forEach((listener) => listener({ detail: JSON.parse(mindJson) }));

const stages = [];
const links = [];
const classes = new Set();
ids["night-loop-body"].walk((node) => {
  node.className.split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
  if (node.tagName === "LI" && node.getAttribute("data-stage")) {
    stages.push({ stage: node.getAttribute("data-stage"), state: node.getAttribute("data-state") });
  }
  if (node.tagName === "A") links.push(node.getAttribute("href"));
});
const text = `${ids["night-loop-asof"].textContent} ${ids["night-loop-body"].textContent}`;
const lede = { text: ids["night-loop-lede"].textContent, hidden: ids["night-loop-lede"].hidden };
process.stdout.write(JSON.stringify({ stages, links, text, classes: [...classes].sort(), lede }));
