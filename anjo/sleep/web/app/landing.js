// The landing's judge code: POST /api/sleep/judge/start, keep the token in sessionStorage, open
// the product view. A signed-in judge who comes back to the landing goes straight on.

import { MOCK, api, href, ready, setToken, token } from "./api.js";
import { COPY } from "./copy.js";

const form = document.getElementById("code-form");
const input = document.getElementById("code");
const submit = document.getElementById("code-submit");
const status = document.getElementById("code-status");

function say(text, kind = "") {
  status.textContent = text;
  status.dataset.kind = kind;
}

function focusCode() {
  input.focus({ preventScroll: true });
  form.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

async function start(event) {
  event.preventDefault();
  const code = input.value.trim();
  if (!code) {
    input.setAttribute("aria-invalid", "true");
    say(COPY.code_empty, "error");
    input.focus();
    return;
  }
  input.removeAttribute("aria-invalid");
  submit.disabled = true;
  say(COPY.code_starting);
  try {
    await ready();
    const data = await api.start(code);
    if (!data || typeof data.token !== "string" || !data.token) throw new Error(COPY.server);
    setToken(data.token);
    window.location.assign(href("/sleep/app/live"));
  } catch (error) {
    if (error.status === 403 || error.status === 401 || error.status === 400 || error.status === 422) {
      input.setAttribute("aria-invalid", "true");
    }
    say(error.message || COPY.server, "error");
    submit.disabled = false;
    input.focus();
  }
}

function boot() {
  form.addEventListener("submit", start);
  input.addEventListener("input", () => {
    input.removeAttribute("aria-invalid");
    if (status.dataset.kind === "error") say("");
  });
  document.querySelectorAll('a[href="#grow"]').forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    window.history.replaceState(null, "", "#grow");
    focusCode();
  }));
  if (window.location.hash === "#grow") focusCode();
  const reason = new URLSearchParams(window.location.search).get("signed_out");
  if (reason === "1") say(COPY.signed_out, "error");
  if (MOCK) {
    const hint = document.createElement("p");
    hint.className = "form-status";
    hint.dataset.kind = "ok";
    hint.textContent = COPY.mock_hint;
    form.append(hint);
  }
  if (token() && reason !== "1") {
    document.getElementById("code-label").textContent = COPY.back_heading;
    submit.textContent = COPY.back_new;
    const back = document.createElement("a");
    back.className = "btn btn-ghost btn-lg";
    back.href = href("/sleep/app/live");
    back.textContent = COPY.back_go;
    form.querySelector(".code-row").append(back);
  }
}

boot();
