// Requests for the judge pages: this host only, the judge's Bearer token from sessionStorage, and
// every failure turned into one plain sentence. With ?mock=1 in the page's own URL, and only then,
// requests go to the canned answers in mock.js instead of the network; mock.js is never even
// loaded otherwise.

import { COPY } from "./copy.js";

const params = new URLSearchParams(window.location.search);
export const MOCK = params.get("mock") === "1";
const TOKEN_KEY = "anjo.judge.token";
const NIGHT_KEY = "anjo.judge.night";

let mockFetch = null;

export async function ready() {
  if (MOCK && !mockFetch) {
    const mock = await import("./mock.js");
    mockFetch = mock.mockFetch;
  }
}

// The transport the rest of the pages use: the real fetch, or the mock behind ?mock=1.
export function transport(url, init) {
  return mockFetch ? mockFetch(url, init) : window.fetch(url, init);
}

// A same-origin link that keeps mock mode on when it is on.
export function href(path) {
  if (!MOCK) return path;
  const [base, hash] = path.split("#");
  const sep = base.includes("?") ? "&" : "?";
  return `${base}${sep}mock=1${hash === undefined ? "" : `#${hash}`}`;
}

function stored(key) {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null; // storage refused (a private window, blocked site data): the page still works
  }
}

function store(key, value) {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch {
    // storage refused: the token lives only as long as this page
  }
}

let memoryToken = null;
export const token = () => memoryToken || stored(TOKEN_KEY);
export function setToken(value) {
  memoryToken = value;
  store(TOKEN_KEY, value);
}
export const nightId = () => stored(NIGHT_KEY);
export const setNightId = (value) => store(NIGHT_KEY, value);
export function signOut() {
  setToken(null);
  store(NIGHT_KEY, null);
}

export class ApiError extends Error {
  constructor(status, message, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

// The plain sentence for a status; a route's own wording (``messages``) wins for its statuses.
export function plain(status, messages = {}) {
  if (messages[status]) return messages[status];
  if (status === 0) return COPY.offline;
  if (status === 401) return COPY.signed_out;
  return COPY.server;
}

export function authHeaders(extra = {}) {
  const value = token();
  return value ? { ...extra, Authorization: `Bearer ${value}` } : { ...extra };
}

export async function request(path, { method = "GET", body, messages = {}, auth = true } = {}) {
  const headers = auth ? authHeaders({ Accept: "application/json" }) : { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await transport(path, {
      method, headers, credentials: "same-origin", cache: "no-store",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, plain(0, messages), null);
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    if (response.status === 401 && auth) window.dispatchEvent(new CustomEvent("judge:signed-out"));
    throw new ApiError(response.status, plain(response.status, messages), data);
  }
  return data;
}

// An answer as JSON without ``live_pass``: whether MIND's live pass is running isn't a change worth
// redrawing a panel for (its revision and content are).
export function contentOf(data) {
  if (!data || typeof data !== "object") return JSON.stringify(data);
  const { live_pass: _running, ...rest } = data;
  return JSON.stringify(rest);
}

// The judge API (the API worker's routes) and the chat start, by name.
export const api = Object.freeze({
  start: (code) => request("/api/sleep/judge/start", {
    method: "POST", body: { code }, auth: false,
    messages: { 400: COPY.code_wrong, 401: COPY.code_wrong, 403: COPY.code_wrong, 404: COPY.code_closed,
      422: COPY.code_wrong, 429: COPY.code_busy },
  }),
  mind: () => request("/api/sleep/me/mind"),
  talker: () => request("/api/sleep/me/talker"),
  sleepNow: () => request("/api/sleep/me/sleep-now", {
    method: "POST", messages: { 429: COPY.sleep_429, 409: COPY.sleep_409 },
  }),
  night: (id) => request(`/api/sleep/me/nights/${encodeURIComponent(id)}`),
  wake: () => request("/api/sleep/me/wake", { messages: { 404: COPY.wake_failed } }),
  undo: () => request("/api/sleep/me/undo", {
    method: "POST", messages: { 409: COPY.undo_nothing, 423: COPY.undo_running },
  }),
  thought: () => request("/api/sleep/me/thought", { messages: { 503: COPY.thought_off } }),
  dispute: (id, note) => request(`/api/sleep/me/thought/${encodeURIComponent(id)}/dispute`, {
    method: "POST", body: note ? { note } : {}, messages: { 404: COPY.thought_gone, 503: COPY.thought_off },
  }),
  hide: (id) => request(`/api/sleep/me/thought/${encodeURIComponent(id)}/hide`, {
    method: "POST", messages: { 404: COPY.thought_gone, 503: COPY.thought_off },
  }),
  chatStart: () => {
    const offset = -new Date().getTimezoneOffset();
    return request(`/api/chat/start?platform=web&tz_offset=${offset}`, {
      method: "POST", messages: { 503: COPY.chat_start_failed },
    });
  },
});
