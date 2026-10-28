# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Training replay's thinkers (Task 17b): the rows are the prompts the thinker is served, token for
token; a dry run spends nothing; both adapters train on the Colab runner from one recipe and record
their rendering. The tokenizer, the vLLM server and the Colab CLI are fakes: no model, no VM, no
network."""
from __future__ import annotations

import inspect
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger, Pool
from anjo.sleep.mind.apd.types import Life
from anjo.sleep.mind.apd.arms import load_brief, mind_core, run_arm
from anjo.sleep.mind.replay import compare, dataset, estimate, fakes, resolve, sample, served, settings, spend
from anjo.sleep.mind.replay.dataset import ReplayData
from anjo.sleep.mind.replay.types import Branch, CallEnd, ReplayRefused, StartState
from anjo.sleep.night.deps import Prices
from anjo.sleep.serving import config, framing
from anjo.sleep.serving.backend import ModalServing
from anjo.sleep.serving.client import ServingError, VllmClient
from anjo.sleep.stops import BudgetStop
from anjo.sleep.train import colab_train, thinker
from anjo.sleep.train.dataset import Dataset
from anjo.sleep.train.lora import LoraCfg, TrainError, plan_steps, training_config
from anjo.sleep.train.runners.colab import ColabCli, RealPersonRefused

BRIEF = load_brief()
LAMBDA = settings.token_penalty()
_PIECE = re.compile(r"<\|[a-z_]+\|>|[^\s<]+|<")


class _Glimmer:
    """A tokenizer with a chat template framed like Glimmer's: special tokens and words are
    pieces, and an assistant turn carries the reply's channel header, so the prompt's ids and the
    reply's add up to the whole conversation's, as the real template's do."""

    bos_token = "<|begin_of_text|>"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(piece, len(self.vocab)) for piece in _PIECE.findall(text)]

    @staticmethod
    def _turn(message: Mapping[str, Any]) -> str:
        if message["role"] == "assistant":
            return f"<|start|>assistant{colab_train.USER_CHANNEL}{message['content']}{colab_train.REPLY_END}"
        return f"<|start|>{message['role']}<|message|>{message['content']}<|eot|>"

    def apply_chat_template(self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False,
                            tokenize: bool = True, return_dict: bool = False, reasoning_strength: str,
                            current_date: str) -> Any:
        assert not return_dict
        text = (f"{self.bos_token}<|start|>system<|message|>{reasoning_strength} {current_date}<|eot|>"
                + "".join(self._turn(m) for m in messages) + ("<|start|>assistant" if add_generation_prompt else ""))
        return self._ids(text) if tokenize else text

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        bos = [self.convert_tokens_to_ids(self.bos_token)] if add_special_tokens else []
        return {"input_ids": bos + self._ids(text)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.setdefault(token, len(self.vocab))


class _Server:
    """vLLM as the night's client reaches it (a ``Poster``): ``/tokenize`` answered by the fake
    tokenizer, and ``/v1/completions`` by the fake thinker reading the message it was sent, at the
    request's temperature and seed. It keeps each message's prompt ids."""

    def __init__(self, tokenizer: _Glimmer) -> None:
        self.tokenizer = tokenizer
        self.served: dict[str, list[int]] = {}
        self.dates: list[str] = []  # each prompt's current_date, in order
        self._message = ""

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if path == "/tokenize" and "messages" in body:
            self._message = str(body["messages"][-1]["content"])
            self.dates.append(str(body["chat_template_kwargs"]["current_date"]))
            ids = self.tokenizer.apply_chat_template(
                body["messages"], add_generation_prompt=body["add_generation_prompt"], **body["chat_template_kwargs"])
            return {"tokens": ids}
        if path == "/tokenize":
            return {"tokens": self.tokenizer(body["prompt"], add_special_tokens=body["add_special_tokens"])["input_ids"]}
        self.served[self._message] = list(body["prompt"])
        reply = fakes.FakeGlimmer(body["seed"], body["temperature"]).propose(self._message, body["max_tokens"])
        return {"model": body["model"], "choices": [{"text": reply.text, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": reply.tokens}}

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class _Colab:
    """Glimmer on Colab vLLM, as replay's ``GlimmerServer``: the fake server's client."""

    vllm: VllmClient
    name: str = "colab"

    def client(self) -> VllmClient:
        return self.vllm


def _replay(models: sample.ReplayModels | None = None, seeds: Sequence[int] = (0,)) -> ReplayData:
    lives: list[Life] = []
    starts: list[StartState] = []
    branches: dict[str, tuple[Branch, ...]] = {}
    book = resolve.LabelBook()
    for seed in seeds:
        life = fakes.fake_life(seed)
        found = sample.start_states(life, run_arm("M", life, fakes.fake_models()))
        labels = resolve.label_all(found, {life.id: life}, resolve.SparkLabeler(chat=fakes.FakeLabelChat()))
        for label in labels.labels.values():
            book = book.with_label(label)
        branches.update(sample.sample_all([(life, found)], models or fakes.fake_replay_models(), labels, brief=BRIEF))
        lives.append(life)
        starts.extend(found)
    return dataset.build(lives, starts, branches, book, brief=BRIEF, penalty=LAMBDA)


def test_thinker_rows_match_the_served_rendering_token_for_token() -> None:
    tokenizer = _Glimmer()
    server = _Server(tokenizer)
    client = VllmClient.over(server)
    models = replace(fakes.fake_replay_models(), thinker_for=served.branch_thinkers(_Colab(client)))
    data = _replay(models)
    night = thinker.render_night(data)
    cfg = training_config(LoraCfg(), len(data.thinker1.examples), warm=False, revision="a" * 40, night=night)
    template = colab_train.template_kwargs(cfg)
    assert template["current_date"] == night.isoformat() and cfg["rendering"] == client.rendering
    rows = [json.loads(line) for d in (data.thinker1, data.shuffled) for line in d.to_jsonl().splitlines()]
    assert rows and all(row["provenance"] == "glimmer" for row in rows)
    assert len({row["current_date"] for row in rows}) >= 2  # rows served on different nights
    for row in rows:
        plain = colab_train.plain_row(tokenizer, row, template)
        prompt_ids = tokenizer(plain["prompt"])["input_ids"]
        reply_ids = tokenizer(plain["completion"], add_special_tokens=False)["input_ids"]
        assert server.served.get(row["prompt"][-1]["content"]) == prompt_ids  # what the thinker was sent
        colab_train._check_tokens(tokenizer, row, "completion", prompt_ids, reply_ids, template)
        undated = {key: value for key, value in row.items() if key != "current_date"}
        if row["current_date"] != night.isoformat():  # on the run's date, it isn't what was served
            assert tokenizer(colab_train.plain_row(tokenizer, undated, template)["prompt"])["input_ids"] != prompt_ids
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        colab_train.plain_row(tokenizer, {**rows[0], "current_date": "17 Sep"}, template)


class _Failing(_Server):
    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        if path == "/v1/completions":
            raise ServingError("/v1/completions answered 503")
        return super().post(path, body)


def test_branch_and_arm_thinkers_are_served_as_the_night_serves_mind() -> None:
    server = _Server(_Glimmer())
    client = VllmClient.over(server)
    day = date(2026, 9, 9)
    branch = served.branch_thinkers(_Colab(client))(7, day)
    assert (branch.temperature, branch.seed, branch.adapter, branch.clock.day) == (0.8, 7, None, day)
    base = served.arm_thinker(_Colab(client), None, served.NightClock())
    assert (base.temperature, base.adapter) == (0.0, None)
    with pytest.raises(ValueError, match="night"):  # an arm's call before its night is set
        base.propose("anything", 50)
    reply = branch.propose("## Your question\nq1 (contradiction, active, opened 2026-09-08): x", 50)
    assert reply.model == "meta-models/Muse-Glimmer-30B" and branch.calls == [CallEnd("stop")]
    assert branch.propose("anything", 0).tokens == 0 and not server.served.get("anything")
    broken = served.branch_thinkers(_Colab(VllmClient.over(_Failing(_Glimmer()))))(7, day)
    failed = broken.propose("## Your question\nq1 (contradiction, active, opened 2026-09-08): x", 50)
    assert (failed.text, failed.tokens, broken.calls) == ("", 50, [CallEnd(None, failed=True)])


class FakeColab:
    """The ``colab`` CLI: every call recorded; ``download`` writes the adapter (recording
    ``rendering``) and the receipt."""

    def __init__(self, rendering: str) -> None:
        self.rendering = rendering
        self.calls: list[list[str]] = []

    def __call__(self, args: Sequence[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        if args[3] == "download":
            local = Path(args[-1])
            if local.name == colab_train.RECEIPT:
                local.write_text(json.dumps({"steps": 9, "epochs": 1, "train_minutes": 40.0, "examples": 3,
                                             "dropped_truncated": 0, "gpu": "A100-SXM4-80GB"}))
            elif local.name == "adapter_config.json":
                local.write_text(json.dumps({"peft_type": "LORA", "r": 16, framing.RENDERING_KEY: self.rendering}))
            else:
                local.write_bytes(b"weights")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    def commands(self) -> list[str]:
        return [args[3] for args in self.calls]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 2400.0  # every reading is 40 minutes on
        return self.now


def _setup(tmp_path: Path, caps: Mapping[Pool, float] = CAPS) -> tuple[FakeColab, Ledger, Any]:
    cli = FakeColab(framing.rendering(config.reasoning_strength()))
    ledger = Ledger(tmp_path / "ledger.jsonl", caps=caps)
    return cli, ledger, thinker.colab_runner(ColabCli(Path("/fake/colab"), run=cli), ledger, clock=Clock())


def test_the_dry_run_plan_spends_nothing(tmp_path: Path) -> None:
    data = _replay()
    cli, ledger, runner = _setup(tmp_path)
    printed: list[str] = []
    published: list[str] = []
    found, refs = thinker.train_thinkers(
        data, runner, ledger, workdir=tmp_path / "work", adapter_cache=tmp_path / "cache",
        publish=lambda folder, name: published.append(name), revision="a" * 40, out=printed.append,
    )
    assert refs == () and cli.calls == [] and published == []
    assert ledger.open_holds() == () and ledger.used("colab_cu") == 0.0
    assert not (tmp_path / "work").exists() and not (tmp_path / "ledger.jsonl").exists()
    (text,) = printed
    rows = len(data.thinker1.examples)
    for needed in (f"{rows} rows", f"{data.thinker1.tokens:,} tokens", "optimizer steps", "A100 minutes",
                   "compute units", "thinker-1", "thinker-shuffled"):
        assert needed in text
    larger = max(rows, len(data.shuffled.examples))
    assert [a.steps for a in found.adapters] == [plan_steps(larger, LoraCfg()).steps] * 2
    assert [(a.examples, a.tokens) for a in found.adapters] == [
        (len(d.examples), d.tokens) for d in (data.thinker1, data.shuffled)]  # each set's own counts
    assert f"{len(data.shuffled.examples)} rows" in text and f"{data.shuffled.tokens:,} tokens" in text


def test_both_adapters_train_on_colab_from_one_recipe_and_record_their_rendering(tmp_path: Path) -> None:
    data = _replay()
    cli, ledger, runner = _setup(tmp_path)
    published: list[str] = []
    found, refs = thinker.train_thinkers(
        data, runner, ledger, workdir=tmp_path / "work", adapter_cache=tmp_path / "cache",
        publish=lambda folder, name: published.append(name), revision="a" * 40, dry_run=False, out=lambda _: None,
    )
    assert [ref.rendering for ref in refs] == [framing.rendering(config.reasoning_strength())] * 2
    assert [ref.person_id for ref in refs] == [thinker.REPLAY_OWNER] * 2 and [r.parent_id for r in refs] == [None, None]
    assert len({ref.id for ref in refs}) == 2 and published == [ref.id for ref in refs]
    for ref in refs:
        config.check_adapter_name(ref.id)
        assert ref.night == thinker.render_night(data)
    settings = [json.loads(arg.split("=", 1)[1]) for args in cli.calls if args[3] == "exec"
                for arg in args if arg.startswith("ANJO_TRAIN_CONFIG=")]
    assert len(settings) == 2 and settings[0] == settings[1]  # the same settings for both, max_steps too
    assert settings[0]["max_steps"] == found.adapters[0].steps == found.adapters[1].steps
    assert settings[0]["keep_every_row"] is True  # the VM stops before training if a row would be dropped
    assert colab_train.build_settings(colab_train.load_config(json.dumps(settings[0])))["sft"]["max_steps"] == (
        settings[0]["max_steps"])
    assert settings[0]["current_date"] == thinker.render_night(data).isoformat() and settings[0]["warm_start"] is False
    assert cli.commands().count("new") == 2 and cli.commands().count("stop") == 2
    assert ledger.open_holds() == () and ledger.used("colab_cu") > 0
    assert found.hold_cu == pytest.approx(15.0)


def test_replay_training_refuses_a_real_person_unequal_data_and_a_pool_without_room(tmp_path: Path) -> None:
    data = _replay()
    cli, ledger, runner = _setup(tmp_path)
    with pytest.raises(RealPersonRefused):
        runner.run("train_lora", {"person_id": "u-4f2a", "adapter_name": "p-0123456789ab-n0"}, timeout_s=7200)
    first = data.shuffled.examples[0].id.rsplit(":b", 1)[0]
    short = replace(data, shuffled=Dataset(
        examples=tuple(e for e in data.shuffled.examples if not e.id.startswith(first)), tokens=1, dropped={}))
    with pytest.raises(TrainError, match="same start states"):
        thinker.plan(short)
    with pytest.raises(TrainError, match="Colab"):
        thinker.train_thinkers(data, object(), ledger, workdir=tmp_path, adapter_cache=tmp_path / "c",  # type: ignore[arg-type]
                               publish=lambda folder, name: None, revision="a" * 40, dry_run=False, out=lambda _: None)
    tight: dict[Pool, float] = {**CAPS, "colab_cu": 20.0}
    cli, ledger, runner = _setup(tmp_path / "tight", caps=tight)
    with pytest.raises(BudgetStop):
        thinker.train_thinkers(data, runner, ledger, workdir=tmp_path / "w", adapter_cache=tmp_path / "c",
                               publish=lambda folder, name: None, revision="a" * 40, dry_run=False, out=lambda _: None)
    assert cli.calls == []  # stopped before any VM


def test_the_script_stops_before_the_model_loads_if_any_row_would_be_dropped() -> None:
    tokenizer = _Glimmer()
    data = _replay()
    rows = [json.loads(line) for line in data.thinker1.to_jsonl().splitlines()]
    long = {**rows[0], "id": "long", "prompt": [{"role": "user", "content": "word " * 9_000}]}
    cfg = colab_train.load_config(json.dumps(training_config(
        LoraCfg(), len(rows), warm=False, revision="a" * 40, night=thinker.render_night(data), keep_every_row=True)))
    template = colab_train.template_kwargs(cfg)
    kept = colab_train._rendered(tokenizer, [*rows, long], cfg["max_seq_length"], template)
    assert len(kept) == len(rows)  # the real tokenizer's count drops the long row
    with pytest.raises(ValueError, match="keeps every row"):
        colab_train.check_every_row_kept(cfg, len(rows) + 1, len(kept))
    colab_train.check_every_row_kept(cfg, len(rows), len(rows))
    colab_train.check_every_row_kept({**cfg, "keep_every_row": False}, len(rows) + 1, len(kept))  # the night's rule
    main = inspect.getsource(colab_train.main)
    assert main.index("check_every_row_kept(") < main.index("_load_model(") < main.index(".train()")


def test_a_typical_development_set_plans_inside_the_spec() -> None:
    rows = tuple(replace(example, id=f"r{n}", tokens=2_600) for n, example in
                 enumerate([*_replay().thinker1.examples] * 100))[:560]
    full = Dataset(examples=rows, tokens=2_600 * len(rows), dropped={})
    found = thinker.plan(ReplayData(thinker1=full, shuffled=full, starts=70, dropped={}))
    for adapter in found.adapters:
        assert (adapter.examples, adapter.steps, adapter.accumulation, adapter.epochs) == (560, 35, 16, 1)
        assert 30 <= adapter.minutes - thinker.SETUP_MINUTES <= 60  # the spec's 30-60 A100 minutes


PRICES = Prices(spark_in_per_m=0.6, spark_out_per_m=2.4, train_gpu_per_hour=1.5, serve_gpu_per_hour=2.61)


def test_replay_serves_glimmer_on_colab_only() -> None:
    client = VllmClient.over(_Server(_Glimmer()))
    for server in (ModalServing("https://anjo--glimmer.modal.run", "t"), replace(_Colab(client), name="sagemaker")):
        with pytest.raises(ReplayRefused, match="Colab"):
            served.branch_thinkers(server)
        with pytest.raises(ReplayRefused, match="Colab"):
            served.arm_thinker(server, None, served.NightClock())
    assert served.branch_thinkers(_Colab(client))(1, date(2026, 9, 9)).client is client


def test_labels_branches_and_arms_hold_their_worst_case_before_they_run(tmp_path: Path) -> None:
    life = fakes.fake_life(0)
    starts = sample.start_states(life, run_arm("M", life, fakes.fake_models()))
    ledger = Ledger(tmp_path / "ledger.jsonl", caps=CAPS)
    holds = spend.Holds(ledger, PRICES, Clock())
    chat = fakes.FakeLabelChat()
    book = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=chat), holds=holds)
    assert chat.sent and ledger.open_holds() == () and ledger.used("baseten_mind_usd") > 0
    sample.sample_all([(life, starts)], fakes.fake_replay_models(), book, brief=BRIEF, holds=holds)
    branched = ledger.used("colab_cu")
    assert branched > 0 and ledger.open_holds() == ()
    held_out = fakes.fake_life(100, split="held_out")
    thinkers = compare.ArmThinkers(fakes.FakeGlimmer(), fakes.FakeGlimmer(), fakes.FakeGlimmer())
    compare.run_replay_arms([held_out], fakes.fake_models(), thinkers, brief=BRIEF, holds=holds)
    assert ledger.used("colab_cu") > branched and ledger.open_holds() == ()


def test_a_pool_without_room_stops_replay_before_any_call(tmp_path: Path) -> None:
    life = fakes.fake_life(0)
    starts = sample.start_states(life, run_arm("M", life, fakes.fake_models()))
    for pool in ("baseten_mind_usd", "colab_cu"):
        ledger = Ledger(tmp_path / f"{pool}.jsonl", caps={**CAPS, pool: 0.01})
        holds = spend.Holds(ledger, PRICES, Clock())
        chat = fakes.FakeLabelChat()
        book = resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=chat), holds=None)
        called: list[int] = []

        def thinker_for(seed: int, night: date) -> fakes.FakeGlimmer:
            called.append(seed)
            return fakes.FakeGlimmer(seed, 0.8)

        with pytest.raises(BudgetStop):
            if pool == "baseten_mind_usd":
                resolve.label_all(starts, {life.id: life}, resolve.SparkLabeler(chat=chat), holds=holds)
            else:
                sample.sample_all([(life, starts)], replace(fakes.fake_replay_models(), thinker_for=thinker_for), book,
                                  brief=BRIEF, holds=holds)
        assert len(chat.sent) == len([s for s in starts if resolve.later_evidence(s, life)]) and not called


def test_the_dry_run_prices_the_labels_and_bounds_the_gpu_time() -> None:
    life = fakes.fake_life(0)
    starts = sample.start_states(life, run_arm("M", life, fakes.fake_models()))
    held_out = [fakes.fake_life(100, split="held_out")]
    found = estimate.estimate(starts, [life], held_out, PRICES)
    assert found.label_calls == len([s for s in starts if resolve.later_evidence(s, life)]) and found.label_tokens > 0
    assert found.label_usd is not None and 0 < found.label_usd < (found.label_hold_usd or 0)
    assert found.branch_runs == len(starts) * 4 and found.arm_runs == 3
    assert found.gpu_minutes == pytest.approx((found.branch_runs + 3 * len(held_out[0].days)) * 26 * 300 / 60)
    text = estimate.describe(found)
    assert "A100 minutes" in text and "$" in text and "colab_cu" in text and "baseten_mind_usd" in text
    assert "unpriced" in estimate.describe(estimate.estimate(starts, [life], held_out, None))


def test_a_branch_row_and_an_arm_prompt_masked_to_the_same_night_carry_its_date() -> None:
    server = _Server(_Glimmer())
    colab = _Colab(VllmClient.over(server))
    data = _replay(replace(fakes.fake_replay_models(), thinker_for=served.branch_thinkers(colab)))
    rows = [json.loads(line) for d in (data.thinker1, data.shuffled) for line in d.to_jsonl().splitlines()]
    for row in rows:  # a start state's id is "<life>:<night>:<question>", its tools masked to that night
        assert row["current_date"] == row["id"].split(":")[1]
    real, masks = mind_core(), list[tuple[str, int]]()

    def notice(ws: Any, new: Any, ports: Any, budget: Any, *, night: date, as_of: datetime, **rest: Any) -> Any:
        masks.append((as_of.date().isoformat(), len(server.dates)))
        return real.notice(ws, new, ports, budget, night=night, as_of=as_of, **rest)

    def run_inquiry(ws: Any, ports: Any, budget: Any, *, night: date, as_of: datetime, **rest: Any) -> Any:
        masks.append((as_of.date().isoformat(), len(server.dates)))
        return real.run_inquiry(ws, ports, budget, night=night, as_of=as_of, **rest)

    clock = served.NightClock()
    arm = served.arm_thinker(colab, None, clock)
    thinkers = compare.ArmThinkers(arm, served.arm_thinker(colab, None, clock), served.arm_thinker(colab, None, clock), clock)
    held_out = fakes.fake_life(100, split="held_out")
    compare.run_replay_arms([held_out], fakes.fake_models(), thinkers, brief=BRIEF,
                            core=replace(real, notice=notice, run_inquiry=run_inquiry))
    ends = [start for _, start in masks[1:]] + [len(server.dates)]
    called = [(night, server.dates[start:end]) for (night, start), end in zip(masks, ends, strict=True)]
    assert any(dates for _, dates in called)
    for night, dates in called:  # each arm prompt carries the night its evidence is masked to
        assert set(dates) <= {night}
    shared = {row["current_date"] for row in rows} & {night for night, dates in called if dates}
    assert shared  # the same nights, dated the same: a branch's row and an arm's prompt agree
