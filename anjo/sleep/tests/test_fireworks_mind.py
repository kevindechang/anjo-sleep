# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's experiment on Spark Open through Fireworks (Tasks 17 and 17b): one provider per
experiment from ``ANJO_SLEEP_MIND_SPARK``, pinned in the data dir; Spark Open's exact Fireworks id
cleared beside Baseten's; every Fireworks call held on ``fireworks_mind_usd``. Sleep's night keeps
Baseten unless ``ANJO_SLEEP_SPARK`` says otherwise (``spark_provider``). Every model is a fake and
every request goes to ``httpx.MockTransport``: no network."""
from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import httpx
import pytest

from anjo.sleep import baseten_api, fireworks_api, spark_inquiry
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.context.brief import ChatFn as BriefChat
from anjo.sleep.evidence import extract
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.mind import spark_provider
from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import fakes, generate, live
from anjo.sleep.mind.apd.arms import load_brief, run_arm
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.replay import dataset, resolve, sample, settings
from anjo.sleep.mind.replay import fakes as replay_fakes
from anjo.sleep.mind.replay.types import Label, ReplayRefused, check_synthetic
from anjo.sleep.model_api import ChatResult, MissingApiKey
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.holds import usd
from anjo.sleep.provenance import GLIMMER, HOSTED, SPARK_OPEN_IDS, may_train

FIREWORKS: Final = fireworks_api.SPARK_OPEN
BASETEN: Final = baseten_api.SPARK_OPEN
SETTING = spark_provider.SETTING_ENV
PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)
LIVE = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]
MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hello"}]
ROOT = Path(__file__).parents[1]
# Where Fireworks may be named: its client, provenance, the ledger's pool, the setting and MIND's experiment
# (the thought channel's pretraining targets included: generated lives, on MIND's Spark pool, never a night).
# Sleep's own modules may name only the selector (``spark_provider``), never a Fireworks id or pool: one
# switch, ``ANJO_SLEEP_SPARK``, moves them.
FIREWORKS_HOMES = ("fireworks_api.py", "spark_provider.py", "provenance.py", "budget.py", "mind/spark_provider.py",
                   "mind/apd/", "mind/replay/", "tests/", "deploy/")
# Sleep's Spark Open choke point and the modules that ask it which id to record (the comparison's
# agent records the id that answered, so its notes carry Spark Open's ancestry).
SELECTOR_HOMES = ("baseten_api.py", "spark_inquiry.py", "evidence/extract.py", "context/brief_store.py",
                  "practice/run.py", "practice/full.py", "sim/persona_sim.py", "night/mind.py", "app/night.py",
                  "agent_baseline/night.py", "harness/loop.py")
# Practice's opt-in DeepSeek critic (a synthetic person's, on its own Fireworks pool) is the one Sleep file
# that may name Fireworks, and never MIND's pool, Spark Open provider or setting: Sleep's Spark Open stays
# on Baseten.
CRITIC_HOME = "practice/critics.py"
# Per-file allowances: the thought channel's tool asks MIND's provider which Spark Open to use, and may
# name that module alone; MIND's causal-test tool (run by hand after --go) asks Sleep's selector the same
# way. Fireworks, the provider's setting and MIND's pool stay forbidden there. The
# morning's selectable DeepSeek judge (ANJO_SLEEP_JUDGE=deepseek, never Spark Open) may name Fireworks
# in its wiring and scoring files, in the adapter-strength read that holds its per-call cost and in the
# final's registered-judge check (the model id a final morning must be judged by), and nothing else on
# this list.
ALLOWED: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {"tools/thought_channel.py": frozenset({"spark_provider"}),
     "tools/mind_causal.py": frozenset({"spark_provider"}),
     "app/morning.py": frozenset({"fireworks"}), "probes/morning.py": frozenset({"fireworks"}),
     "probes/score.py": frozenset({"fireworks"}), "tools/adapter_scale.py": frozenset({"fireworks"}),
     "probes/registered_judge.py": frozenset({"fireworks"})})


def _ok(model: str) -> dict[str, Any]:
    return {"model": model, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}}


def _main(argv: list[str]) -> tuple[int, str]:
    lines: list[str] = []
    code = cli.main(argv, out=lines.append)
    return code, "\n".join(lines)


def _reserved(ledger_path: Path) -> list[tuple[str, tuple[str, ...]]]:
    return [(e["job"], tuple(e["pools"])) for e in read_records(ledger_path, PLAIN) if e["op"] == "reserve"]


# ── The client ────────────────────────────────────────────────────────────────


def test_fireworks_chat_posts_to_fireworks_with_its_key_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok(FIREWORKS))

    transport = httpx.MockTransport(handler)
    monkeypatch.delenv(fireworks_api.KEY_ENV, raising=False)
    with pytest.raises(MissingApiKey, match=fireworks_api.KEY_ENV):  # no key: refused before any request
        fireworks_api.chat(FIREWORKS, MESSAGES, None, 30.0, transport=transport)
    assert seen == []
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-first")
    fireworks_api.chat(FIREWORKS, MESSAGES, None, 30.0, transport=transport)
    monkeypatch.setenv(fireworks_api.KEY_ENV, "fw-second")  # read at each call, never kept
    result = fireworks_api.chat(FIREWORKS, MESSAGES, None, 30.0, temperature=0.0, max_tokens=64, transport=transport)
    assert fireworks_api.KEY_ENV == "ANJO_FIREWORKS_API_KEY"
    assert [str(r.url) for r in seen] == ["https://api.fireworks.ai/inference/v1/chat/completions"] * 2
    assert [r.headers["authorization"] for r in seen] == ["Bearer fw-first", "Bearer fw-second"]
    body = json.loads(seen[-1].content)
    assert body == {"model": "accounts/fireworks/models/muse-spark-open", "messages": MESSAGES, "temperature": 0.0,
                    "max_completion_tokens": 64}
    assert (result.model, result.text, result.input_tokens, result.output_tokens) == (FIREWORKS, "ok", 5, 1)

    sent: list[dict[str, Any]] = []

    def recorded(model: str, messages: list[dict[str, Any]], tools: Any, timeout_s: float, **kw: Any) -> ChatResult:
        sent.append({"model": model, "tools": tools, **kw})
        return ChatResult("ok", (), model, 5, 1, "stop")

    monkeypatch.setattr(fireworks_api, "chat", recorded)
    spark_provider.FIREWORKS.chat(MESSAGES, 200)  # MIND's capped calls: the resolver, the labels
    assert sent == [{"model": FIREWORKS, "tools": None, "temperature": 0.0, "max_tokens": 200}]


# ── Provenance ────────────────────────────────────────────────────────────────


def test_fireworks_spark_open_cleared_and_other_fireworks_ids_not() -> None:
    assert SPARK_OPEN_IDS == {BASETEN, FIREWORKS}
    for cleared in ({FIREWORKS}, {FIREWORKS, GLIMMER}, {FIREWORKS, BASETEN, GLIMMER}):
        assert may_train(frozenset(cleared), hosted_in_training=False)
    others = ("accounts/fireworks/models/muse-spark", "accounts/fireworks/models/muse-spark-open-1026",
              "accounts/fireworks/models/muse-glimmer-30b", "accounts/someone/models/muse-spark-open",
              "accounts/fireworks/models/muse-spark-open ", "Accounts/Fireworks/Models/Muse-Spark-Open",
              "muse-spark-open")
    for other in (*others, *sorted(HOSTED)):  # hosted Muse Spark stays uncleared
        assert not may_train(frozenset({other}), hosted_in_training=False), other
        assert not may_train(frozenset({FIREWORKS, other}), hosted_in_training=False), other
        assert not Label("s1", "q1.h1", other).usable
    for other in others:
        assert not may_train(frozenset({other}), hosted_in_training=True), other
    assert Label("s1", "q1.h1", FIREWORKS).usable and Label("s1", "q1.h1", BASETEN).usable
    assert not Label("s1", None, FIREWORKS).usable  # undetermined stays unusable


# ── One provider per experiment ───────────────────────────────────────────────


def test_mind_experiment_uses_one_spark_provider_and_refuses_a_mixed_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(SETTING, raising=False)  # the default is Fireworks
    assert spark_provider.from_env(os.environ) is spark_provider.FIREWORKS
    assert spark_provider.from_env({SETTING: "baseten"}) is spark_provider.BASETEN
    with pytest.raises(ValueError, match=SETTING):
        spark_provider.from_env({SETTING: "together"})
    data = tmp_path / "pilot"
    code, text = _main(["--fake", "--data", str(data), "--only", "1", "generate"])
    assert code == 0, text
    (life,) = DataDir(data).lives("development")
    assert {s.model for s in life.sessions} == {FIREWORKS} == {e.model for e in life.evidence}
    assert all(e.ancestry == frozenset({FIREWORKS}) for e in life.evidence)
    pins = json.loads((data / "serving_pin.json").read_text(encoding="utf-8"))
    assert (pins["spark_provider"], pins["spark_model"]) == ("fireworks", FIREWORKS)

    monkeypatch.setenv(SETTING, "baseten")  # the other provider: every Spark Open command refused
    for command in (["generate"], ["timing"], ["run"], ["intervene"]):
        code, text = _main(["--fake", "--data", str(data), "--only", "1", *command])
        assert code == cli.REFUSED_CODE and "Refused: this pilot's Spark Open runs on fireworks" in text, text
        assert "serving Glimmer" not in text
    assert not any("baseten_mind_usd" in pools for _, pools in _reserved(data / "fake-budget.jsonl"))
    monkeypatch.setenv(SETTING, "together")
    with pytest.raises(SystemExit):
        cli.main(["--fake", "--data", str(data), "--only", "1", "run"], out=lambda _line: None)

    monkeypatch.setenv(SETTING, "fireworks")  # its own provider goes on
    code, text = _main(["--fake", "--data", str(data), "--only", "1", "run"])
    assert code == 0, text
    run = DataDir(data).run(life.id, "M")
    assert run is not None and {w.model for w in run.written if w.port == "resolver"} == {FIREWORKS}

    unpinned = DataDir(tmp_path / "older")  # lives written on Baseten before any pin
    made = fakes.fake_life(generate.development_specs()[0])
    unpinned.save_life(made.life, made.problems)
    code, text = _main(["--fake", "--data", str(unpinned.root), "--only", "1", "run"])
    assert code == cli.REFUSED_CODE and "written by another model" in text and "serving Glimmer" not in text


# ── Holds ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class _Held:
    """Asserts a hold on ``fireworks_mind_usd`` is open at every Spark Open call, and keeps each
    call's system prompt."""

    ledger: Ledger
    systems: list[str]

    def check(self, messages: list[dict[str, Any]]) -> None:
        assert any(spark_provider.FIREWORKS.pool in held.pools for held in self.ledger.open_holds())
        self.systems.append(str(messages[0]["content"]).splitlines()[0])

    def capped(self, inner: spark_inquiry.ChatFn) -> spark_inquiry.ChatFn:
        def chat(messages: list[dict[str, Any]], max_tokens: int) -> ChatResult:
            self.check(messages)
            return inner(messages, max_tokens)
        return chat

    def writing(self, inner: BriefChat) -> BriefChat:
        def chat(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
            if model != "muse-spark":
                assert model == FIREWORKS
                self.check(messages)
            return inner(model, messages, tools, timeout_s, temperature=temperature)
        return chat


def test_every_fireworks_call_held_on_fireworks_mind_usd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SETTING, "fireworks")
    systems: list[str] = []
    real = cli._fake_backend

    def watched(data: DataDir, provider: spark_provider.SparkProvider) -> backends.Backend:
        backend = real(data, provider)
        assert backend.ledger is not None and backend.provider is spark_provider.FIREWORKS
        held = _Held(backend.ledger, systems)
        return dataclasses.replace(backend, spark=held.capped(backend.spark), life_chat=held.writing(backend.life_chat),
                                   reader=held.writing(backend.reader))

    monkeypatch.setattr(cli, "_fake_backend", watched)
    data = tmp_path / "pilot"
    code, text = _main(["--fake", "--data", str(data), "--only", "2"])  # life 2 has planted items for the gates
    assert code == 0, text
    seen = set(systems)
    for prompt in ("# resolve_prediction_v1", "# resolve_v1", "# wrong_hypothesis_v1", "# life_v1", "# extract_open_v1",
                   "# leak_v1", "# discover_v1"):
        assert any(line.startswith(prompt) for line in seen), prompt
    reserved = _reserved(data / "fake-budget.jsonl")
    spark = [job for job, pools in reserved if "fireworks_mind_usd" in pools]
    assert spark and any(job.startswith("apd:kill:") for job in spark)  # kill-resume's children too
    assert not any("baseten_mind_usd" in pools or "baseten_sleep_usd" in pools for _, pools in reserved)
    assert "the $20 fireworks_mind_usd cap" in text and Ledger(data / "fake-budget.jsonl").open_holds() == ()


def test_a_fireworks_night_never_waits_on_sleeps_baseten_holds(tmp_path: Path) -> None:
    assert CAPS["fireworks_mind_usd"] == 20.0
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    assert ledger.reserve("night:sleep:a1", ("baseten_sleep_usd",), usd(CAPS["baseten_sleep_usd"] +
                                                                         CAPS["baseten_mind_usd"])) is not None
    waits: list[float] = []
    calls = live.SparkCalls(fakes.FakeSparkChat(model=FIREWORKS), PRICES, ledger, "fw", pool="fireworks_mind_usd",
                            wait=waits.append)
    calls([{"role": "system", "content": spark_inquiry.RESOLVE_PROMPT}, {"role": "user", "content": "a claim"}], 10)
    calls.close()
    assert waits == [] and ledger.used("fireworks_mind_usd") > 0 and ledger.used("baseten_mind_usd") == 0.0
    assert ledger.remaining("baseten_mind_usd") == 0.0  # Baseten's shared total is untouched by Fireworks


# ── Sleep's night ─────────────────────────────────────────────────────────────


def test_sleep_night_path_still_uses_baseten(monkeypatch: pytest.MonkeyPatch) -> None:
    for path in sorted(ROOT.rglob("*.py")):
        name = path.relative_to(ROOT).as_posix()
        if name.startswith(FIREWORKS_HOMES):
            continue
        text = path.read_text(encoding="utf-8").casefold()
        if name.startswith(SELECTOR_HOMES):  # the choke point names the client's module, never its id or key
            terms = {"fireworks_sleep_usd", "fireworks_mind_usd", "accounts/fireworks", "anjo_fireworks"}
        else:
            terms = {"fireworks", "fireworks_mind_usd", "spark_provider", SETTING.casefold()}
        if name == CRITIC_HOME:  # its own Fireworks pool, never Spark Open's provider or setting
            terms -= {"fireworks"}
        for term in sorted(terms - ALLOWED.get(name, frozenset())):
            assert term not in text, f"{name} names {term}: Sleep's night stays on Baseten by default"
    monkeypatch.delenv(SETTING, raising=False)  # unset: today's Baseten path, unchanged
    posted: list[str] = []

    def post(**kw: Any) -> ChatResult:
        posted.append(f"{kw['provider']} {kw['url']} {kw['key_env']} {kw['model']}")
        return ChatResult("{}", (), kw["model"], 1, 1, "stop")

    monkeypatch.setattr(baseten_api, "post_chat", post)
    spark_inquiry.spark_open(MESSAGES, 10)  # the night's MIND step: its resolver
    assert posted == [f"Baseten {baseten_api.URL} {baseten_api.KEY_ENV} {BASETEN}"]
    called: list[str] = []

    def baseten_chat(model: str, *args: Any, **kw: Any) -> ChatResult:
        called.append(model)
        return ChatResult("{}", (), model, 1, 1, "stop")

    monkeypatch.setattr(baseten_api, "chat", baseten_chat)
    extract.spark_open(MESSAGES)  # extraction
    assert called == [BASETEN]
    assert spark_provider.BASETEN.chat is spark_inquiry.spark_open and spark_provider.BASETEN.pool == "baseten_mind_usd"


# ── Replay ────────────────────────────────────────────────────────────────────


def test_replay_accepts_fireworks_reported_labels() -> None:
    brief, penalty = load_brief(), settings.token_penalty()
    life = replay_fakes.fake_life(0, model=FIREWORKS)
    check_synthetic(life)  # written by Spark Open on Fireworks: synthetic
    with pytest.raises(ReplayRefused, match="Spark Open didn't produce"):
        check_synthetic(replay_fakes.fake_life(1, model="accounts/fireworks/models/muse-spark"))
    starts = sample.start_states(life, run_arm("M", life, replay_fakes.fake_models()))
    book = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=replay_fakes.FakeLabelChat(model=FIREWORKS)))
    usable = [label for label in book.labels.values() if label.usable]
    assert usable and {label.model for label in usable} == {FIREWORKS}
    branches = sample.sample_all([(life, starts)], replay_fakes.fake_replay_models(model=FIREWORKS), book, brief=brief)
    assert branches and all(b.ancestry is not None and FIREWORKS in b.ancestry for bs in branches.values() for b in bs)
    built = dataset.build([life], starts, branches, book, brief=brief, penalty=penalty)
    assert built.thinker1.examples and built.dropped["provenance"] == 0

    other = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(
        chat=replay_fakes.FakeLabelChat(model="accounts/fireworks/models/muse-spark")))
    assert not any(label.usable for label in other.labels.values())
    assert dataset.build([life], starts, branches, other, brief=brief, penalty=penalty).thinker1.examples == ()


# ── The plan ──────────────────────────────────────────────────────────────────


def test_live_plan_prints_the_fireworks_pool_and_spends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_backend(*args: object, **kwargs: object) -> None:
        raise AssertionError("a plan builds no live backend")

    monkeypatch.setattr(backends, "live_backend", no_backend)
    monkeypatch.setattr("anjo.sleep.budget.default_ledger", no_backend)
    monkeypatch.delenv(SETTING, raising=False)  # the default
    for command in (["generate"], ["run"], ["run", "--held-out"], ["timing"], ["intervene"],
                    ["kill-resume", "--kill-at", "9"]):
        code, text = _main([*LIVE, "--data", str(tmp_path), "--hours", "4", *command])
        assert code == 0 and text.endswith("Nothing has run. Add --yes to start it."), text
        assert "Hold on fireworks_mind_usd: $" in text and "baseten_mind_usd" not in text
        assert f"Spark Open on Fireworks ({FIREWORKS}), held on fireworks_mind_usd" in text
    monkeypatch.setenv(SETTING, "baseten")
    code, text = _main([*LIVE, "--data", str(tmp_path), "generate"])
    assert code == 0 and "Hold on baseten_mind_usd: $" in text and "fireworks_mind_usd" not in text
    assert list(tmp_path.iterdir()) == []  # nothing written, nothing held
