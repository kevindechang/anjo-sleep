"""Offline stand-ins for the APD runner (``--fake`` and the tests): a Colab VM, its tunnel, Glimmer's
vLLM server, Spark Open and the two judges. They spend nothing and reach no network.

The fakes stand where the real things stand, so the code that runs is the code that runs live:

- ``FakeColab`` is the ``colab`` CLI (a ``CliRun``): sessions come and go, ``exec`` runs the steps
  ``colab_vllm_start.py`` would (it keeps the key and the ``vllm serve`` command the start step
  was given), and ``upload`` lands adapters in the VM's LoRA directory.
- ``fake_keygen`` stands for ``ssh-keygen``: it writes a dummy pair where the session's throwaway
  key goes, so the real ``session_key`` makes and deletes it as it does live.
- ``fake_tunnels`` opens a "tunnel" whose transport is ``FakeVllm``: the real ``VllmClient`` talks
  to it over a loopback URL, with the session's own key, through ``/v1/models``, ``/tokenize`` and
  ``/v1/completions``. Its thinker is ``FakeInquirer`` for MIND (it notices, hypothesizes, asks a
  world query, predicts, searches, assesses and concludes, so the resolver and ``world_query`` both
  fire), a scripted notebook agent for B and a scripted reflection for A; a system turn with
  ``world_query_v1.md`` gets a world answer. Each completion records the date it was rendered on.
- ``FakeSparkChat`` is Spark Open as a ``ChatFn``: the screen allows, the resolver says true once
  evidence arrives, and the labeller picks the plan-change hypothesis.
- ``FakeLifeChat`` writes and extracts a life as ``generate`` asks, and ``FakeGateReader`` and
  ``FakeGateAuditor`` pass the gates; ``fake_judge`` is Muse Spark's or Gemini's text model.

Every reply names the real model (Glimmer, Spark Open, the pinned Muse Spark), so the provenance
rules run as they do live. The numbers mean nothing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

import httpx

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import Ledger
from anjo.sleep.mind.apd import colab_vllm, generate
from anjo.sleep.mind.apd.generate import Generated, LifeSpec
from anjo.sleep.mind.replay import fakes as replay_fakes
from anjo.sleep.model_api import ChatResult
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.client import LOAD_PATH, Poster, ServingError
from anjo.sleep.serving.config import ServeConfig
from anjo.sleep.train.runners.colab import ColabCli
from anjo.sleep.spark_inquiry import RESOLVE_PROMPT, WORLD_PROMPT

FIRST: Final = config.FIRST_SPECIAL_ID
HEADER: Final = 9
PLAN: Final = "the plan"
HOLDS: Final = "The plan holds as written"
CHANGES: Final = "The plan needs a change"
WORLD_ANSWER: Final = "Plans like this usually break on room, timing or a missing step."
CLAIM: Final = "The person changes the plan within two days"
_ITEM: Final = re.compile(r"^\s*-?\s*\[(?P<id>[^\]]+)\] (?P<day>\d{4}-\d{2}-\d{2})[^:]*: (?P<text>.*)$")
_HYPOTHESIS: Final = re.compile(r"^- (?P<id>\S+): (?P<credence>[0-9.]+)  (?P<statement>.*)$")
_TARGET: Final = re.compile(r"^- (?P<id>\S+) \(")
_LETTER: Final = re.compile(r"^(?P<letter>[A-Z])\. (?P<statement>.*)$")


class Ticks:
    """A clock that moves a minute every reading."""

    def __init__(self, step_s: float = 60.0) -> None:
        self.now, self.step_s = 0.0, step_s

    def __call__(self) -> float:
        self.now += self.step_s
        return self.now


def _section(text: str, heading: str) -> list[str]:
    """The non-blank lines under ``heading``."""
    return [line for line in replay_fakes.section(text, heading) if line.strip()]


def _items(lines: Sequence[str]) -> list[tuple[str, str, str]]:
    found = (_ITEM.match(line) for line in lines)
    return [(m["id"], m["day"], m["text"]) for m in found if m]


# ── The thinker behind the fake vLLM ──────────────────────────────────────────


def _notice(text: str) -> list[dict[str, Any]]:
    """One contradiction between the oldest item and today's first, on the first open target,
    once no question is held."""
    if _section(text, "## Questions you hold") != ["(none)"]:
        return []
    targets = [m["id"] for m in (_TARGET.match(line) for line in _section(text, "## Open targets")) if m]
    earlier = _items(_section(text, "## Earlier evidence you hold"))
    new = _items(_section(text, "## Today's new evidence"))
    if not (targets and earlier and new) or earlier[0][1] == new[0][1]:
        return []
    return [{"op": "NOTICE", "kind": "contradiction", "subject": "Two steps of the plan can't both hold",
             "target_ids": targets[:1], "origin_evidence": [earlier[0][0], new[0][0]]}]


def _cycle(text: str) -> dict[str, Any]:
    head = _section(text, "## Your question")
    question = head[0].split(" ", 1)[0]
    raised = head[1].rsplit("Raised by: ", 1)[1].rstrip(".").split(", ") if len(head) > 1 else []
    held = [(m["id"], float(m["credence"]), m["statement"]) for m in (
        _HYPOTHESIS.match(line) for line in _section(text, "## Hypotheses (credences are heuristic)")) if m]
    if not held:
        return {"op": "HYPOTHESIZE", "question_id": question, "catch_all": 0.2,
                "hypotheses": [{"statement": HOLDS, "credence": 0.4}, {"statement": CHANGES, "credence": 0.4}]}
    ids = [h for h, _, _ in held]
    change = next((h for h, _, s in held if s == CHANGES), ids[-1])
    if _section(text, "## Answers to your world queries (never evidence)") == ["(none)"]:
        return {"op": "INVESTIGATE", "question_id": question, "tool": "world_query", "query": "What breaks a plan?"}
    if _section(text, "## Open predictions") == _section(text, "## Settled predictions") == ["(none)"]:
        return {"op": "PREDICT", "question_id": question, "claim": CLAIM, "horizon_days": 2,
                "p_true_given": {h: (0.8 if h == change else 0.2 if s == HOLDS else 0.5) for h, _, s in held}}
    waiting = [i for i, _, _ in _items([line for line in _section(text, "## Evidence") if "not yet assessed" in line])]
    if waiting:
        return {"op": "ASSESS", "question_id": question, "evidence_id": waiting[0],
                "likelihoods": {h: (2.0 if h == change else 0.5 if s == HOLDS else 1.0) for h, _, s in held}}
    leader = max(held, key=lambda h: h[1])
    if leader[1] >= 0.8:
        return {"op": "CONCLUDE", "question_id": question, "hypothesis_id": leader[0], "kind": "resolution",
                "statement": leader[2], "cites": raised}
    return {"op": "INVESTIGATE", "question_id": question, "tool": "evidence_search", "query": "they said by the way"}


def _b(text: str) -> dict[str, Any]:
    """The notebook agent: search, report one suspicion on what it found, then stop."""
    steps = _section(text, "## Your steps so far tonight")
    if _section(text, "## Your findings so far") != ["none"]:
        return {"action": {"tool": "stop"}}
    found = [i for i, _, _ in _items(steps)]
    if len(found) < 2:
        return {"notebook": "Q: does the plan hold?", "action": {"tool": "evidence_search", "query": "they said"}}
    return {"notebook": "Q: does the plan hold? Suspect not.", "action": {
        "tool": "report_finding", "problem": "Two steps of the plan conflict", "conclusion": CHANGES, "p": 0.6,
        "cites": found[:2], "kind": "suspicion"}}


def _a(text: str) -> list[dict[str, Any]]:
    said = _items(_section(text, "## Everything the person has said, up to tonight"))
    days = sorted({day for _, day, _ in said})
    if len(days) < 2:
        return []
    cites = [next(i for i, d, _ in said if d == days[0]), next(i for i, d, _ in said if d == days[-1])]
    return [{"problem": "Two steps of the plan conflict", "conclusion": CHANGES, "p": 0.55, "cites": cites,
             "kind": "suspicion", "finding_id": "f1" if _section(text, "## Your findings so far") != ["none"] else None}]


def think(text: str) -> str:
    """The fake Glimmer's reply to one prompt, by which prompt it is."""
    if "## Today's new evidence" in text:
        return json.dumps(_notice(text))
    if "## Your notebook, kept across nights" in text:
        return json.dumps(_b(text))
    if "## Everything the person has said, up to tonight" in text:
        return json.dumps(_a(text))
    return json.dumps(_cycle(text))


# ── Glimmer's fake vLLM server ────────────────────────────────────────────────


@dataclass
class FakeVllm:
    """``/v1/models``, ``/tokenize``, ``/v1/completions`` and ``/v1/load_lora_adapter`` for one
    session, behind its key. An adapter in the LoRA directory (``adapters``) is served only once
    loaded from there, as vLLM 0.28.0 did live. ``served`` is each completion's kind (``world`` or
    ``think``), rendered date and prompt, in order; ``models`` the model each completion named."""

    key: str
    adapters: set[str] = field(default_factory=set)
    loaded: set[str] = field(default_factory=set)
    healthy: bool = True
    failing: set[str] = field(default_factory=set)  # completion kinds (think, world) that answer 503
    flaky: dict[str, int] = field(default_factory=dict)  # and those answering 503 that many times first
    chats: list[tuple[list[dict[str, Any]], str]] = field(default_factory=list)
    served: list[tuple[str, str, str]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != colab_vllm.LOOPBACK or request.headers.get("Authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"error": "unauthorized"})
        if request.url.path == "/v1/models":
            if not self.healthy:
                return httpx.Response(503, json={"error": "loading"})
            return httpx.Response(200, json={"data": [{"id": config.MODEL}, *({"id": a} for a in sorted(self.loaded))]})
        body = json.loads(request.content)
        if request.url.path == LOAD_PATH:
            return self._load(str(body["lora_name"]), str(body["lora_path"]))
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": self._tokenize(body)})
        if request.url.path == "/v1/completions":
            return self._complete(body)
        return httpx.Response(404, json={"error": "no such path"})

    def _load(self, name: str, path: str) -> httpx.Response:
        if name in self.loaded:
            return httpx.Response(400, json={"error": f"The lora adapter '{name}' has already been loaded."})
        if name not in self.adapters or path != f"{colab_vllm.ADAPTERS_DIR}/{name}":
            return httpx.Response(404, json={"error": "no adapter there"})
        self.loaded.add(name)
        return httpx.Response(200, text=f"Success: LoRA adapter '{name}' added successfully.")

    def _tokenize(self, body: Mapping[str, Any]) -> list[int]:
        if "messages" not in body:
            return [HEADER]
        self.chats.append((list(body["messages"]), str(body["chat_template_kwargs"]["current_date"])))
        return [FIRST, len(self.chats) - 1]

    def _complete(self, body: Mapping[str, Any]) -> httpx.Response:
        model = str(body["model"])
        if model != config.MODEL and model not in self.loaded:
            return httpx.Response(404, json={"error": "no such adapter"})
        messages, day = self.chats[body["prompt"][1]]
        system = messages[0]["content"] if messages[0]["role"] == "system" else None
        prompt = str(messages[-1]["content"])
        kind = "world" if system == WORLD_PROMPT else "think"
        if kind in self.failing or self.flaky.get(kind, 0) > 0:
            self.flaky[kind] = max(0, self.flaky.get(kind, 0) - 1)
            return httpx.Response(503, json={"error": "preempted"})
        self.served.append(("world" if system == WORLD_PROMPT else "think", day, prompt))
        self.models.append(model)
        text = WORLD_ANSWER if system == WORLD_PROMPT else think(prompt)
        cap = int(body["max_tokens"])
        tokens = max(1, len(text) // 4)
        return httpx.Response(200, json={
            "model": model, "choices": [{"text": text, "finish_reason": "length" if tokens > cap else "stop"}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": min(tokens, cap)}})


# ── The Colab VM and its tunnel ───────────────────────────────────────────────


@dataclass
class FakeColab:
    """The ``colab`` CLI over fake VMs. ``fail`` makes one subcommand exit 1 (or time out);
    ``healthy`` False leaves vLLM never answering. ``held_at_new`` is what the ledger held on
    ``colab_cu`` when ``new`` ran."""

    ledger: Ledger | None = None
    fail: str | None = None
    timeout: bool = False
    healthy: bool = True
    calls: list[tuple[list[str], float]] = field(default_factory=list)
    live: dict[str, FakeVllm] = field(default_factory=dict)
    installed: list[str] = field(default_factory=list)
    adapters: set[str] = field(default_factory=set)
    held_at_new: float | None = None
    closed: list[str] = field(default_factory=list)
    dropped: set[str] = field(default_factory=set)  # sessions whose tunnel died
    keys: dict[str, str] = field(default_factory=dict)  # each session's uploaded key, until start reads it
    key_modes: list[int] = field(default_factory=list)  # the uploaded key files' permissions
    tunnel_closed_before_stop: bool = False

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        argv = list(args)
        self.calls.append((argv, timeout_s))
        command = argv[3]
        if command == self.fail:
            if self.timeout:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        session = argv[argv.index("-s") + 1] if "-s" in argv else ""
        out = getattr(self, f"_{command}", lambda _a, _s: "")(argv, session)
        return subprocess.CompletedProcess(argv, 0, out, "")

    def commands(self) -> list[str]:
        return [args[3] for args, _ in self.calls]

    def env_of(self, step: str) -> dict[str, str]:
        """The ``--env`` flags of the exec that ran ``step``."""
        for args, _ in self.calls:
            if args[3] == "exec" and f"ANJO_COLAB_STEP={step}" in args:
                flags = args[args.index("--timeout") + 2:]
                return dict(flag.split("=", 1) for flag in flags[1::2])
        raise KeyError(step)

    def _new(self, argv: list[str], session: str) -> str:
        if self.ledger is not None:
            self.held_at_new = self.ledger.used("colab_cu")
        self.live[session] = FakeVllm(key="", adapters=self.adapters, healthy=self.healthy)
        return ""

    def _install(self, argv: list[str], session: str) -> str:
        self.installed.extend(argv[argv.index("-s") + 2:])
        return ""

    def _exec(self, argv: list[str], session: str) -> str:
        env = dict(flag.split("=", 1) for flag in argv[argv.index("--timeout") + 2:][1::2])
        if env.get("ANJO_COLAB_STEP") == "start":
            self.live[session].key = self.keys.pop(session) if env["ANJO_VLLM_KEY_FILE"] == colab_vllm.KEY_FILE else ""
        return f"{colab_vllm.STEP_OK} {env.get('ANJO_COLAB_STEP')}\n"

    def _upload(self, argv: list[str], session: str) -> str:
        if argv[-1] == colab_vllm.KEY_FILE:
            local = Path(argv[-2])
            self.key_modes.append(local.stat().st_mode & 0o777)
            self.keys[session] = local.read_text(encoding="utf-8")
            return ""
        self.adapters.add(Path(argv[-1]).parent.name)
        return ""

    def _stop(self, argv: list[str], session: str) -> str:
        self.tunnel_closed_before_stop = session in self.closed
        self.live.pop(session, None)
        return ""

    def _sessions(self, argv: list[str], session: str) -> str:
        if not self.live:
            return "[colab] No active sessions found on server.\n"
        return "".join(f"[{name}] gpu-a100-s-1 | Hardware: A100 | Shape: High-RAM | Variant: GPU\n"
                       for name in self.live)


def fake_keygen(path: Path) -> None:
    """``ssh-keygen``'s stand-in: a dummy private half (0600) and its public half."""
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as file:
        file.write("-----BEGIN OPENSSH PRIVATE KEY-----\nfake\n-----END OPENSSH PRIVATE KEY-----\n")
    path.with_name(f"{path.name}.pub").write_text("ssh-ed25519 AAAAfake\n", encoding="utf-8")


@dataclass
class FakeTunnel:
    vm: FakeColab
    session: str
    transport: httpx.BaseTransport | None

    def close(self) -> None:
        self.vm.closed.append(self.session)

    def alive(self) -> bool:
        return self.session not in self.vm.dropped and self.session not in self.vm.closed


def fake_tunnels(vm: FakeColab) -> colab_vllm.TunnelOpener:
    """A tunnel to ``vm``'s session: its transport is that session's fake vLLM."""

    def open_tunnel(session: str, local_port: int, remote_port: int, timeout_s: float, identity: Path) -> FakeTunnel:
        if remote_port != config.PORT or not 0 < timeout_s:
            raise ValueError("a tunnel forwards the vLLM port, under a hard timeout")
        if not identity.is_file():
            raise ValueError("a tunnel opens on its session's own key")
        return FakeTunnel(vm, session, httpx.MockTransport(vm.live[session]))

    return open_tunnel


SETTINGS: Final = ServeConfig(precision=config.SERVING, revision="a" * 40, tool_parser=None, gpu_snapshot=False,
                              vllm_version=config.DEFAULT_VLLM_VERSION)
SERVE_S: Final = 3 * 3600.0


def poster_transport(poster: Poster) -> httpx.MockTransport:
    """A transport that answers each request with what ``poster`` returns for it (a test's own
    fake vLLM, written as a ``Poster``); a ``ServingError`` answers 503."""

    def handle(request: httpx.Request) -> httpx.Response:
        try:
            return httpx.Response(200, json=poster.post(request.url.path, json.loads(request.content or b"{}")))
        except ServingError as error:
            return httpx.Response(503, json={"error": str(error)})

    return httpx.MockTransport(handle)


@contextmanager
def serving(transport: httpx.BaseTransport | None = None, vm: FakeColab | None = None,
            ledger: Ledger | None = None) -> Iterator[colab_vllm.ColabSession]:
    """A live session of the real ``colab_vllm.serve`` on a fake VM. Its tunnel carries
    ``transport`` (a test's own server) in place of the VM's fake vLLM when one is given."""
    machine = vm or FakeColab(ledger=ledger)
    opener = fake_tunnels(machine)

    def tunnels(session: str, local: int, remote: int, timeout_s: float, identity: Path) -> colab_vllm.Tunnel:
        tunnel = opener(session, local, remote, timeout_s, identity)
        return tunnel if transport is None else FakeTunnel(machine, session, transport)

    with colab_vllm.serve(ColabCli(Path("/fake/colab"), run=machine), ledger, SETTINGS, timeout_s=SERVE_S,
                          tunnels=tunnels, clock=Ticks(), wait=lambda _session: True, sleep=lambda _s: None,
                          keygen=fake_keygen) as session:
        yield session


def adapter_folder(folder: Path, rendering: str) -> Path:
    """A LoRA adapter folder as training leaves one in the cache, recording ``rendering``."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16, framing.RENDERING_KEY: rendering}))
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


# ── Spark Open, the generator and the judges ──────────────────────────────────


WRONG_STATEMENT: Final = "The shoot moved because the lens rental fell through"


def _reply(text: str, model: str = SPARK_OPEN, *, cap: int = 64) -> ChatResult:
    return ChatResult(text=text, tool_calls=(), model=model, input_tokens=120, output_tokens=min(12, cap),
                      finish_reason="stop")


@dataclass
class FakeSparkChat:
    """Spark Open as a ``ChatFn``: the screen allows, the resolver says true, the labeller picks the
    plan-change hypothesis, the intervention test's writer names a wrong one. ``model`` names
    another model to test provenance."""

    model: str = SPARK_OPEN
    calls: int = 0

    def __call__(self, messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        self.calls += 1
        system, user = str(messages[0]["content"]), str(messages[-1]["content"])
        if system == RESOLVE_PROMPT:
            return _reply(json.dumps({"verdict": "true"}), self.model, cap=max_tokens)
        if system.startswith("# wrong_hypothesis_v1"):
            return _reply(json.dumps({"statement": WRONG_STATEMENT}), self.model, cap=max_tokens)
        letters = {m["statement"]: m["letter"] for m in (_LETTER.match(line) for line in user.splitlines()) if m}
        if letters:
            return _reply(json.dumps({"holds": letters.get(CHANGES, "undetermined")}), self.model, cap=max_tokens)
        return _reply(json.dumps({"allowed": True}), self.model, cap=max_tokens)


@dataclass
class FakeLifeChat:
    """``generate``'s writer and extractor (``context.brief.ChatFn``): each beat in a message of its
    own, one lesson per message, and the first lesson of each day mentions the plan. Each reply
    names the model asked for, as a provider's does."""

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        body = json.loads(messages[-1]["content"])
        if str(messages[0]["content"]).startswith("# life_v1"):
            out, placed = [f"Day {body['day']}: working on {body['field']}."], {}
            for beat in [*body["plant"], *body["ask"]]:
                placed[beat["beat"]] = len(out)
                out.append(f"By the way, {beat['say']}")
            return _reply(json.dumps({"messages": out, "placed": placed}), model)
        held = body["open_targets"]
        due = (date.fromisoformat(body["today"]) + timedelta(days=5)).isoformat()
        target = {"kind": "commitment", "label": PLAN, "check_after": due, "existing_id": held[0]["id"] if held else None,
                  "confidence": None}
        items = [{"turn": m["turn"], "kind": "lesson", "text": f"They said: {m['text']}", "target": None}
                 for m in body["messages"]]
        if items:
            items[0]["target"] = target
        return _reply(json.dumps({"items": items}), model)


REPLY_SHAPES: Final = ("plain", "think", "fenced", "reasoning_field", "length", "rejected", "draft_in_cut_think",
                       "open_think", "two_fences")


def generator_transport(shape: str) -> httpx.MockTransport:
    """Fireworks answering a reasoning generator's writer and extractor (``FakeLifeChat``'s replies)
    in one ``shape`` of ``REPLY_SHAPES``: plain JSON; reasoning in the content between think tags;
    the JSON fenced inside prose; reasoning kept apart in ``reasoning_content``; every reply cut at
    its cap (``finish_reason`` length, the JSON unfinished); ``reasoning_effort`` refused with HTTP
    400; or a drafted answer that must never be read as the day's: fenced inside reasoning cut at
    its cap (``draft_in_cut_think``), fenced inside reasoning that never closed (``open_think``), or
    fenced before the final fenced answer (``two_fences``). The reasoning is counted in the usage
    as a provider counts it."""
    if shape not in REPLY_SHAPES:
        raise ValueError(f"no reply shape {shape!r}")
    chat = FakeLifeChat()

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if shape == "rejected":
            return httpx.Response(400, json={"error": {"message": "reasoning_effort is not supported"}})
        answer = chat(body["model"], body["messages"], None, 30.0).text
        message: dict[str, Any] = {"role": "assistant", "content": answer}
        finish, reasoning = "stop", 0
        if shape == "think":
            message["content"], reasoning = f"<think>\nThe person {{works}} on it.\n</think>\n\n{answer}", 40
        elif shape == "fenced":
            message["content"] = f"Here is the JSON:\n```json\n{answer}\n```\nDone."
        elif shape == "reasoning_field":
            message["reasoning_content"], reasoning = "Let me plan the day first.", 40
        elif shape == "length":
            message["content"], finish = answer[: len(answer) // 2], "length"
        elif shape == "draft_in_cut_think":
            message["content"], finish, reasoning = f"<think>\nA draft:\n```json\n{answer}\n```\nNow check", "length", 40
        elif shape == "open_think":
            message["content"], reasoning = f"<think>\nA draft:\n```json\n{answer}\n```\n", 40
        elif shape == "two_fences":
            message["content"] = f"A draft:\n```json\n{answer}\n```\nThe answer:\n```json\n{answer}\n```"
        usage = {"prompt_tokens": len(json.dumps(body["messages"])) // 4, "completion_tokens": len(answer) // 4 + reasoning,
                 "completion_tokens_details": {"reasoning_tokens": reasoning}}
        return httpx.Response(200, json={"model": body["model"], "usage": usage,
                                         "choices": [{"message": message, "finish_reason": finish}]})

    return httpx.MockTransport(handle)


@dataclass
class FakeGateReader:
    """The leak reader never flags; the discoverability reader always finds. Each reply names the
    model asked for."""

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        body = json.loads(messages[-1]["content"])
        if str(messages[0]["content"]).startswith("# leak_v1"):
            return _reply(json.dumps({"flag": False}), model)
        return _reply(json.dumps({"found": True, "statement": "found it", "days": [d["date"] for d in body["days"]]}),
                      model)


@dataclass
class FakeGateAuditor:
    """Muse Spark's script audit (passes) and its identify judge (matches)."""

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        if str(messages[0]["content"]).startswith("# identify_v1"):
            return _reply(json.dumps({"match": True}), "muse-spark-1.3")
        return _reply(json.dumps({"ok": True, "issues": []}), "muse-spark-1.3")


def fake_life(spec: LifeSpec) -> Generated:
    """``spec``'s life, written, extracted and gated on the fakes."""
    return generate.generate(spec.field, spec.families, spec.prevalence, spec.seed, split=spec.split,
                             chat=FakeLifeChat(), reader=FakeGateReader(), auditor=FakeGateAuditor())


def fake_judge(strict: bool) -> Callable[[str], str]:
    """A judge's text model: the same problem when the planted kind is a contradiction, true when
    the conclusion says the plan changes; ``strict`` also needs p of at least 0.6, so the two
    judges sometimes differ and the audit has rows."""

    def complete(prompt: str) -> str:
        same = "can't both hold" in prompt
        holds = CHANGES.casefold() in prompt.casefold()
        if strict:
            match = re.search(r"Probability that the conclusion is true: ([0-9.]+)", prompt)
            holds = holds and match is not None and float(match.group(1)) >= 0.6
        return json.dumps({"same_problem": same, "holds": holds})

    return complete
