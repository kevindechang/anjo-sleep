# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The second-family judge on stored mornings (``tools.morning_second_judge``): each daily-v1 morning's
judged pairs asked again (arm 3 pinned to the checkpoint it scored), then judged blind in both orders by
Muse Spark and by Gemini, with their agreement per life and dimension and every disagreement shown.
Fakes only: no model, judge, Vertex or GPU call."""
from __future__ import annotations

import fcntl
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger, Pool, Reservation
from anjo.sleep.checkpoints import CheckpointRegistry
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import Prices
from anjo.sleep.night.sweep import LOCK_NAME, swept_night
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import DAILY, JUDGE_USD_PER_CALL, Morning, subset_probes
from anjo.sleep.probes.score import MuseJudge
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, StoreSpy, probe, taste
from anjo.sleep.tools import morning_backfill
from anjo.sleep.tools import morning_second_judge as second
from anjo.sleep.types import CheckpointRef, ProbeKind

USER = "syn-seed"
NIGHTS = (date(2026, 11, 1), date(2026, 11, 2))
PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.96)
LONG, SHORT = "a long careful considered answer", "a short answer"


def _ckpt(n: int) -> CheckpointRef:
    return CheckpointRef(f"p-abc-n{n}", USER, NIGHTS[n - 1], f"uri://{n}", None if n == 1 else f"p-abc-n{n - 1}")


class Model:
    """Arm 3 (an adapter) answers long, base short; neither reply names who wrote it."""

    def __init__(self) -> None:
        self.adapters: list[str | None] = []

    def complete(self, adapter: Any, messages: Any, *, temperature: float, seed: int | None = None) -> ChatResult:
        self.adapters.append(None if adapter is None else adapter.id)
        return ChatResult(LONG if adapter else SHORT, (), "m", 10, 5, "stop")


def _verdict(prompt: str, prefer_long: bool) -> str:
    first = prompt.split("## Reply A", 1)[1].split("## Reply B", 1)[0]
    return json.dumps({"verdict": "A" if (LONG in first) == prefer_long else "B", "reason": "r"})


class Spark:
    """Muse Spark's chat: the longer reply wins, in either order."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, **_: Any) -> ChatResult:
        self.prompts.append(messages[0]["content"])
        return ChatResult(_verdict(messages[0]["content"], True), (), "muse-spark", 1000, 200, "stop")


class Gemini:
    """Gemini's Vertex call: agrees with Spark but on ``vocab``, where it prefers the short reply."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        return _verdict(prompt, "Kind: vocab" not in prompt), 2000, 300


class FailingSpark(Spark):
    """Muse Spark whose second call fails in a way ``_judged`` doesn't catch: one call is already spent."""

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, **_: Any) -> ChatResult:
        if self.prompts:
            raise PermissionError("the judge's key is gone")
        return super().__call__(model, messages, tools, timeout_s)


class RetryingSpark(Spark):
    """Muse Spark whose calls 1, 3 and 5 can't be read: each sends the morning's one retry, then all are read."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, model: Any, messages: Any, tools: Any, timeout_s: float, **_: Any) -> ChatResult:
        self.prompts.append(messages[0]["content"])
        self.calls += 1
        if self.calls in (1, 3, 5):
            return ChatResult("not a verdict", (), "muse-spark", 1000, 200, "stop")
        return ChatResult(_verdict(messages[0]["content"], True), (), "muse-spark", 1000, 200, "stop")


class RetryingGemini(Gemini):
    """Gemini's Vertex call, unreadable on its calls 1, 3 and 5, readable after: every retry is counted."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls in (1, 3, 5):
            return "not a verdict", 2000, 300
        return _verdict(prompt, "Kind: vocab" not in prompt), 2000, 300


class World:
    def __init__(self, tmp_path: Path, *, synthetic: bool = True) -> None:
        self.root, self.synthetic, self.out = tmp_path / "results", synthetic, tmp_path / "out"
        self.registry = CheckpointRegistry(tmp_path / "checkpoints")
        for n in (1, 2):
            self.registry.promote(USER, _ckpt(n))  # night 2's adapter serves now
        self.ledger = Ledger(tmp_path / "budget.jsonl")
        self.model, self.spark, self.gemini, self.woke = Model(), Spark(), Gemini(), 0
        kinds: tuple[ProbeKind, ...] = ("new_work",) * 6 + ("vocab",) * 4 + ("tone_hold",) * 4
        self.probes = [probe(f"q{n}", kind, prompt=f"question {n}", person=USER) for n, kind in enumerate(kinds)]
        self.probes += [taste(f"t{n}", person=USER, source=f"item-{n}") for n in range(4)]
        self.daily = [p for p in subset_probes(self.probes, DAILY) if p.kind != "taste_ab"]
        for n, night in enumerate(NIGHTS, start=1):
            self.score(n, night, {p.id: "win" for p in self.daily} | {self.daily[0].id: "tie"})

    def score(self, number: int, night: date, outcomes: dict[str, str], *, checkpoint: str | None = None) -> None:
        kinds = {p.id: p.kind for p in self.probes}
        rows = [{"probe": pid, "pair": "3_vs_1", "kind": kinds[pid], "outcome": o, "flip": False} for pid, o in outcomes.items()]
        record = {"night": night.isoformat(), "subset": DAILY.name, "checkpoint": checkpoint or f"p-abc-n{number}",
                  "arms": {"3": {"overall": {"n": 1, "score": 1.0}}}, "pairs": {}, "outcomes": rows}
        folder = self.root / USER / "scores"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"night-{number:02d}.json").write_text(json.dumps(record), encoding="utf-8")

    def morning(self, user: str, persona: str, night: date) -> Morning:
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={}, until=night)
        return Morning(arms=deps, sealed=lambda uid: self.probes, judge=MuseJudge(self.spark, lambda pid: "BRIEF"),
                       prices=PRICES, root=self.root, wake=self.wake)

    def wake(self) -> WakeResult:
        self.woke += 1
        return WakeResult(1.0, True)

    @property
    def ports(self) -> second.Ports:
        return second.Ports(backfill=morning_backfill.Ports(
            morning=self.morning, is_synthetic=lambda user: self.synthetic, logs=lambda user: self.root / user / "nights",
            registry=self.registry, ledger=self.ledger), gemini=self.gemini)

    def run(self, *extra: str, accounts: tuple[str, ...] = (f"seed={USER}", "seed_cap=syn-cap")) -> tuple[int, list[str]]:
        said: list[str] = []
        argv = ["--persona", "film_founder", *(a for account in accounts for a in ("--account", account)), *extra]
        return second.main(argv, ports=self.ports, root=self.root, out=said.append), said


class NotWaking(World):
    """The same world, its serving app not answering the wake."""

    def wake(self) -> WakeResult:
        return WakeResult(1.0, False)


def _json(said: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in said if line.startswith("{")]


def test_the_plan_counts_the_calls_holds_and_cost_and_calls_nothing(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run()
    assert code == 0 and world.model.adapters == [] and world.spark.prompts == [] and world.gemini.prompts == []
    text = "\n".join(said)
    assert "2 mornings" in text and "44 replies" in text  # 11 judged probes, two arms, each of 2 mornings
    assert "44 Muse Spark calls" in text and "44 Gemini calls" in text and "both orders" in text
    assert "modal_usd" in text and "meta_usd" in text and "vertex_usd" in text and "--go" in text
    assert {"line": "missing", "arm": "seed_cap", "account": "syn-cap", "why": "no daily-v1 morning scores"} in _json(said)
    assert world.ledger.open_holds() == () and not world.out.exists()


def test_go_judges_each_pair_both_ways_with_both_judges_and_shows_every_disagreement(tmp_path: Path) -> None:
    world = World(tmp_path)
    before = sorted(p.name for p in (world.root / USER / "scores").iterdir())
    code, said = world.run("--out", str(world.out), "--go")
    assert code == 0 and world.woke == 2  # each morning waits for the server first
    assert [a for a in world.model.adapters if a] == ["p-abc-n1"] * 11 + ["p-abc-n2"] * 11  # pinned, though n2 serves
    assert len(world.spark.prompts) == len(world.gemini.prompts) == 44  # 11 probes x 2 orders x 2 mornings
    for prompts in (world.spark.prompts, world.gemini.prompts):  # blind: the arm never named, each order once
        assert not any(re.search(r"p-abc|arm 3|base model", p) for p in prompts)
        firsts = [LONG in p.split("## Reply A", 1)[1].split("## Reply B", 1)[0] for p in prompts]
        assert firsts.count(True) == firsts.count(False) == 22
    lines = _json(said)
    life = {(line["arm"], line["dimension"]): line for line in lines if line["line"] == "agreement"}
    knowledge = life[("seed", "knowledge")]  # 5 new_work + 3 vocab, two mornings
    assert (knowledge["spark"]["wins"], knowledge["gemini"]["wins"], knowledge["gemini"]["losses"]) == (16, 10, 6)
    assert (knowledge["both_judged"], knowledge["same"], knowledge["agreement"]) == (16, 10, 0.625)
    assert life[("seed", "interaction")]["agreement"] == 1.0 and life[("seed", "interaction")]["kinds"] == ["tone_hold"]
    disagreements = [line for line in lines if line["line"] == "disagreement"]
    assert len(disagreements) == 6 and {d["kind"] for d in disagreements} == {"vocab"}
    assert {(d["spark"], d["gemini"]) for d in disagreements} == {("win", "loss")}
    stored = next(line for line in lines if line["line"] == "stored_agreement" and line["arm"] == "seed")
    assert (stored["both"], stored["same"]) == (22, 20)  # each morning stored one tie that Spark now calls a win
    assert sorted(p.name for p in (world.root / USER / "scores").iterdir()) == before  # no score file written
    pairs = [json.loads(line) for line in (world.out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pairs) == 22 and {p["replies"]["3"] for p in pairs} == {LONG} and pairs[0]["prompt"].startswith("question")
    assert world.ledger.open_holds() == ()
    pools: tuple[Pool, ...] = ("modal_usd", "meta_usd", "vertex_usd")
    spent = {pool: world.ledger.remaining(pool) for pool in pools}
    assert all(spent[pool] < Ledger(tmp_path / "fresh.jsonl").remaining(pool) for pool in spent)  # each held and settled
    assert not any(LONG in line or "question" in line for line in said)  # the printout: ids and numbers only


def test_each_refusal_comes_before_any_call_hold_or_write(tmp_path: Path) -> None:
    world = World(tmp_path, synthetic=False)
    refused = [world]
    code, said = world.run("--out", str(world.out), "--go")
    assert code == 1 and "synthetic" in said[-1]
    world = World(tmp_path / "moved")
    refused.append(world)
    spare = next(p.id for p in world.probes if p not in world.daily and p.kind != "taste_ab")
    world.score(1, NIGHTS[0], {spare: "win"})  # a morning whose probes aren't the ones daily-v1 asks now
    code, said = world.run("--out", str(world.out), "--go")
    assert code == 1 and "probes" in said[-1]
    world = World(tmp_path / "unknown")
    refused.append(world)
    world.score(2, NIGHTS[1], {p.id: "win" for p in world.daily}, checkpoint="p-zzz-n9")
    code, said = world.run("--out", str(world.out), "--go")
    assert code == 1 and "promoted history" in said[-1]
    world = World(tmp_path / "locked")
    refused.append(world)
    with open(world.ledger.path.parent / LOCK_NAME, "a+b") as night:
        fcntl.flock(night, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, said = world.run("--out", str(world.out), "--go")
    assert code == 1 and "night lock" in said[-1]
    for each in refused:
        assert each.model.adapters == [] and each.spark.prompts == [] and each.gemini.prompts == [] and each.woke == 0
        assert each.ledger.open_holds() == () and not each.out.exists()
    with pytest.raises(SystemExit):  # the pairs need a folder of their own
        World(tmp_path / "no-out").run("--go")


def test_out_inside_the_sleep_folders_is_refused_before_anything_is_held_or_sent(tmp_path: Path) -> None:
    world = World(tmp_path)
    inside = world.root / USER / "scores" / "pairs"
    code, said = world.run("--out", str(inside), "--go")
    assert code == 1 and len(said) == 1 and "--out" in said[0] and "sleep folders" in said[0]
    assert world.model.adapters == [] and world.spark.prompts == [] and world.gemini.prompts == []
    assert world.woke == 0 and world.ledger.open_holds() == () and not inside.exists()


def test_the_hold_equals_the_calls_each_judge_makes_with_its_retries(tmp_path: Path) -> None:
    world = World(tmp_path)
    (world.root / USER / "scores" / "night-02.json").unlink()  # one morning: its retries are the whole worst case
    world.spark, world.gemini = RetryingSpark(), RetryingGemini()  # each unreadable call sends its retry
    jobs, _ = second.plan({"seed": USER}, "film_founder", world.ports, world.root)
    cost = second.spend(jobs)
    assert (len(jobs), {len(job.probes) for job in jobs}) == (1, {11})
    assert world.run("--out", str(world.out), "--go")[0] == 0
    spark_calls, gemini_calls = len(world.spark.prompts), len(world.gemini.prompts)
    assert cost.calls == 22  # each judge's: every probe in both orders
    assert (spark_calls, gemini_calls) == (25, 25)  # 22 calls + the morning's 3 retries, what _judge_job allows
    assert cost.spark_hold == pytest.approx(spark_calls * JUDGE_USD_PER_CALL)
    assert cost.gemini_hold == pytest.approx(gemini_calls * second.GEMINI_WORST_USD)


def test_a_morning_that_fails_mid_judging_still_settles_the_spark_spend(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.spark = FailingSpark()  # the second Spark call raises where _judged doesn't catch it
    with pytest.raises(PermissionError):
        world.run("--out", str(world.out), "--go")
    assert world.ledger.open_holds() == ()  # every hold settled or released on the way out
    settled = world.ledger.settlements()
    spark_job = next(job for job in settled if ":spark:a" in job)
    assert settled[spark_job] == pytest.approx(2 * JUDGE_USD_PER_CALL)  # the two calls made, not zero


def test_a_glimmer_that_did_not_wake_is_refused_with_its_holds_settled(tmp_path: Path) -> None:
    world = NotWaking(tmp_path)
    code, said = world.run("--out", str(world.out), "--go")
    assert code == 1 and said[-1] == "refused: the serving app did not wake: nothing was judged"
    assert world.model.adapters == [] and world.spark.prompts == [] and world.gemini.prompts == []
    assert world.ledger.open_holds() == ()
    assert (world.out / "pairs.jsonl").read_text(encoding="utf-8") == ""


def test_a_malformed_stored_row_is_refused_before_any_model_call(tmp_path: Path) -> None:
    world = World(tmp_path)
    path = world.root / USER / "scores" / "night-01.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["outcomes"][0].pop("outcome")  # a judged row that names no outcome: _job indexes it directly
    path.write_text(json.dumps(record), encoding="utf-8")
    code, said = world.run("--out", str(world.out), "--go")
    assert code == 1 and "malformed" in said[-1]
    assert world.model.adapters == [] and world.spark.prompts == [] and world.gemini.prompts == []
    assert world.woke == 0 and world.ledger.open_holds() == () and not world.out.exists()


def _no_settle(self: Reservation, actual: float) -> None:
    """A run killed hard runs no ``finally``: its holds are taken and never settled."""


def _no_release(self: Reservation) -> None:
    """Nor released."""


def test_a_killed_runs_holds_are_swept_at_their_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(tmp_path)
    monkeypatch.setattr(Reservation, "settle", _no_settle)  # the kill: holds taken, never closed
    monkeypatch.setattr(Reservation, "release", _no_release)
    assert world.run("--out", str(world.out), "--go")[0] == 0
    killed = world.ledger.open_holds()
    assert sorted(hold.job.split(":")[-2] for hold in killed) == ["arms", "gemini", "spark"]
    monkeypatch.undo()  # the process is gone; the sweep runs in a new one
    with swept_night(world.ledger, lambda: (0, 0)) as swept:
        assert swept.holds_settled == 3
    assert world.ledger.open_holds() == ()
