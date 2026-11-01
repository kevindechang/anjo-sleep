# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The morning backfill (``tools.morning_backfill``): a promoted night whose morning failed is scored
afterwards on its own adapter, pinned while a later one keeps serving, logged ``backfilled`` in the
night's log and read by the live comparison's report. Fakes only: no model, judge or GPU call."""
from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep import fireworks_api
from anjo.sleep.budget import Ledger
from anjo.sleep.checkpoints import CheckpointRegistry, ref_record
from anjo.sleep.night.holds import usd
from anjo.sleep.night.past import her_spend
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import MORNING_PROBES, Morning
from anjo.sleep.probes.score import Verdict
from anjo.sleep.reports import results
from anjo.sleep.tests.sleep_fakes import FakeContexts, FakeLogprobs, FakeModel, StoreSpy, probe, taste
from anjo.sleep.tools import morning_backfill as backfill_tool
from anjo.sleep.tools import sleep_objective_live as live
from anjo.sleep.types import CheckpointRef

USER = "syn-seed"
NIGHTS = (date(2026, 10, 24), date(2026, 10, 25), date(2026, 10, 26))
PRICES = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.96)


def _ckpt(n: int) -> CheckpointRef:
    return CheckpointRef(f"p-abc-n{n}", USER, NIGHTS[n - 1], f"uri://{n}", None if n == 1 else f"p-abc-n{n - 1}")


class Judge:
    """Deterministic: the longer reply wins. ``fail`` raises on every call (a judge that is down)."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.calls, self.fail = 0, fail

    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class World:
    """One synthetic account that promoted three nights, each morning failed; n3 serves."""

    def __init__(self, tmp_path: Path, *, synthetic: bool = True, judge: Judge | None = None) -> None:
        self.root, self.synthetic = tmp_path / "results", synthetic
        self.registry = CheckpointRegistry(tmp_path / "checkpoints")
        for n in (1, 2, 3):
            self.registry.promote(USER, _ckpt(n))
        self.model, self.judge = FakeModel(), judge or Judge()
        self.ledger = Ledger(tmp_path / "budget.jsonl")
        self.probes = [probe("q1", "new_work", person=USER), probe("q2", "vocab", person=USER),
                       probe("q3", "tone_hold", person=USER), taste("t1", person=USER)]
        for n, night in enumerate(NIGHTS, start=1):
            self.night(night, {"status": "promoted", "checkpoint": ref_record(_ckpt(n)), "accelerated": True,
                               "label": "development", "cost_usd": 1.0})

    def night(self, night: date, summary: dict[str, Any], *, morning: str = "failed") -> Path:
        folder = self.root / USER / "nights" / night.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        steps = [{"step": "train", "status": "ok", "receipt": {"train_minutes": 3.0}},
                 {"step": "morning", "status": morning, "uses": "new", "system": None}]
        (folder / "log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps), encoding="utf-8")
        return folder

    def morning(self, user: str, persona: str, night: date) -> Morning:
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: StoreSpy([]), logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={}, until=night)
        return Morning(arms=deps, sealed=lambda uid: self.probes, judge=self.judge, prices=PRICES, root=self.root)

    @property
    def ports(self) -> backfill_tool.Ports:
        return backfill_tool.Ports(morning=self.morning, is_synthetic=lambda user: self.synthetic,
                                   logs=lambda user: self.root / user / "nights", registry=self.registry,
                                   ledger=self.ledger)

    def run(self, night: date, *extra: str) -> tuple[int, list[str]]:
        said: list[str] = []
        code = backfill_tool.main(["--user", USER, "--persona", "film_founder", "--night", night.isoformat(), *extra],
                                  ports=self.ports, out=said.append)
        return code, said

    def steps(self, night: date) -> list[dict[str, Any]]:
        path = self.root / USER / "nights" / night.isoformat() / "log.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_the_plan_names_the_nights_adapter_its_holds_and_cost_and_runs_nothing(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run(NIGHTS[0])
    text = "\n".join(said)
    assert code == 0 and "p-abc-n1, the adapter the night promoted, pinned; p-abc-n3 keeps serving" in text
    assert "the full set: 4 probes, 24 replies" in text  # an accelerated night's morning asked every arm
    assert "holds: the judge $" in text and "expected: $" in text and "night-01.json" in text
    assert world.model.calls == [] and world.judge.calls == 0 and not (world.root / USER / "scores").exists()
    assert [s["status"] for s in world.steps(NIGHTS[0]) if s["step"] == "morning"] == ["failed"]  # nothing written


def test_go_scores_the_nights_own_adapter_while_a_later_one_keeps_serving(tmp_path: Path) -> None:
    world = World(tmp_path)
    code, said = world.run(NIGHTS[0], "--go")
    assert code == 0 and said[-1].startswith("scored: ")
    personal = {c["adapter"].id for c in world.model.calls if c["adapter"] is not None}
    assert personal == {"p-abc-n1"}  # arms 3 and 4 on the night's own adapter
    assert world.registry.last_promoted(USER) == _ckpt(3)  # what serves never moved
    record = json.loads((world.root / USER / "scores" / "night-01.json").read_text(encoding="utf-8"))
    assert (record["night"], record["checkpoint"], record["pinned"]) == ("2026-10-24", "p-abc-n1", True)
    assert "subset" not in record and record["arms"]["3"]["overall"]["score"] is not None
    last = [s for s in world.steps(NIGHTS[0]) if s["step"] == "morning"][-1]
    assert (last["status"], last["backfilled"], last["checkpoint"]) == ("ok", True, "p-abc-n1")
    assert last["judge_calls"] == world.judge.calls > 0
    assert world.run(NIGHTS[0], "--go")[1] == ["refused: the morning of 2026-10-24 already succeeded on the full set"]


def test_the_report_keeps_each_probe_set_apart_one_score_per_night_and_set(tmp_path: Path) -> None:
    """An arm on daily-v1 whose first night is backfilled on the full set too: a mean per set, by night
    date, never pooled; the results table shows that night's full score and says so."""
    world = World(tmp_path)
    for n, night in enumerate(NIGHTS, start=1):
        world.night(night, {"status": "promoted", "checkpoint": ref_record(_ckpt(n)), "accelerated": True,
                            "label": "development", "cost_usd": 1.0, MORNING_PROBES: "daily-v1"})
    for night in (NIGHTS[2], NIGHTS[0], NIGHTS[1]):  # the last night first: its file is night-01.json
        assert world.run(night, "--go")[0] == 0  # each on the set its summary records
    assert world.run(NIGHTS[0], "--go", "--subset", "full")[0] == 0
    files = {(r["night"], r.get("subset", "full")): r for r in
             (json.loads(p.read_text()) for p in (world.root / USER / "scores").glob("night-*.json"))}
    assert sorted(files) == sorted([*((n.isoformat(), "daily-v1") for n in NIGHTS), (NIGHTS[0].isoformat(), "full")])
    table = live.report(world.root, {"seed": USER, "today": "syn-today"}, lambda line: None)["seed"]
    daily = [files[(n.isoformat(), "daily-v1")]["arms"]["3"]["overall"]["score"] for n in NIGHTS]
    assert list(table["scores"]) == ["daily-v1", "full"] and table["scores"]["daily-v1"]["arm3_mean"] == round(
        sum(daily) / 3, 4)  # three nights, not four
    assert table["scores"]["daily-v1"]["by_night"] == {n.isoformat(): {
        "3": files[(n.isoformat(), "daily-v1")]["arms"]["3"]["overall"]["score"],
        "4": None} for n in NIGHTS}  # daily-v1 asks arms 1 and 3
    full = files[(NIGHTS[0].isoformat(), "full")]["arms"]
    assert table["scores"]["full"] == {"by_night": {NIGHTS[0].isoformat(): {"3": full["3"]["overall"]["score"],
                                                                            "4": full["4"]["overall"]["score"]}},
                                       "arm3_mean": full["3"]["overall"]["score"], "arm4_mean": full["4"]["overall"]["score"]}
    rows = results.load(USER, world.root)
    assert [(n.night, n.probes) for n in rows] == [(NIGHTS[0].isoformat(), "full"), (NIGHTS[1].isoformat(), "daily-v1"),
                                                   (NIGHTS[2].isoformat(), "daily-v1")]  # one row a night: its full set's


def test_of_two_files_for_one_night_and_set_the_newest_counts(tmp_path: Path) -> None:
    folder = tmp_path / "scores"
    folder.mkdir()
    for number, score in ((9, 0.2), (100, 0.8), (99, 0.4)):  # night-100 is the newest, though night-99 sorts last
        record = {"night": "2026-10-24", "subset": "daily-v1", "arms": {"3": {"overall": {"score": score}}}}
        (folder / f"night-{number:02d}.json").write_text(json.dumps(record), encoding="utf-8")
    [(key, newest)] = results.score_records(folder).items()
    assert key == ("2026-10-24", "daily-v1") and newest["arms"]["3"]["overall"]["score"] == 0.8
    (folder / "night-xx.json").write_text(json.dumps({"night": "2026-10-25", "arms": {}}), encoding="utf-8")
    with pytest.raises(results.ResultsError, match="night-xx.json"):
        results.score_records(folder)


@pytest.mark.parametrize("rule", ["not_promoted", "morning_ok", "not_synthetic", "lock_held", "scored"])
def test_each_refusal_comes_before_any_call_or_write(tmp_path: Path, rule: str) -> None:
    world = World(tmp_path, synthetic=rule != "not_synthetic")
    night = NIGHTS[0]
    if rule == "not_promoted":
        world.night(night, {"status": "rejected", "checkpoint": None, "accelerated": True})
    elif rule == "morning_ok":
        world.night(night, {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": True},
                    morning="ok")
    elif rule == "scored":
        (world.root / USER / "scores").mkdir(parents=True)
        (world.root / USER / "scores" / "night-01.json").write_text(json.dumps({"night": night.isoformat()}))
    before = world.steps(night)
    with open(tmp_path / "night.lock", "a+b") as other:  # another run's lock, when the rule says so
        if rule == "lock_held":
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, said = world.run(night, "--go")
    expected = {"not_promoted": "wasn't promoted (rejected)", "morning_ok": "already succeeded",
                "not_synthetic": "isn't registered synthetic", "lock_held": "night lock is held",
                "scored": "already scored"}[rule]
    assert code == 1 and said[-1].startswith("refused: ") and expected in said[-1]
    assert world.model.calls == [] and world.judge.calls == 0 and world.steps(night) == before
    if rule == "not_synthetic":
        assert world.run(night, "--real")[0] == 0  # a real person's night only when asked for by name


def test_a_night_started_after_the_plan_holds_off_the_run(tmp_path: Path) -> None:
    world = World(tmp_path)
    chosen = backfill_tool.plan(USER, "film_founder", NIGHTS[0], world.ports)
    with open(tmp_path / "night.lock", "a+b") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(backfill_tool.Refused, match="another night run"):
            backfill_tool.backfill(chosen, world.ports)
    assert world.model.calls == [] and not (world.root / USER / "scores").exists()


def test_a_backfill_that_fails_is_logged_backfilled_and_raised(tmp_path: Path) -> None:
    world = World(tmp_path, judge=Judge(fail=RuntimeError("the judge is down")))
    chosen = backfill_tool.plan(USER, "film_founder", NIGHTS[1], world.ports)
    with pytest.raises(RuntimeError):
        backfill_tool.backfill(chosen, world.ports)
    last = [s for s in world.steps(NIGHTS[1]) if s["step"] == "morning"][-1]
    assert (last["status"], last["backfilled"], last["why"]) == ("failed", True, "morning failed: RuntimeError")
    assert not (world.root / USER / "scores").exists()


@pytest.mark.parametrize(("accelerated", "flag", "subset"), [(True, (), None), (True, ("--subset", "daily-v1"), "daily-v1"),
                                                             (False, (), "daily-v1")])
def test_the_probe_set_is_the_nights_own_unless_a_subset_is_named(
    tmp_path: Path, accelerated: bool, flag: tuple[str, ...], subset: str | None,
) -> None:
    world = World(tmp_path)
    world.night(NIGHTS[0], {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": accelerated})
    assert world.run(NIGHTS[0], "--go", *flag)[0] == 0
    record = json.loads((world.root / USER / "scores" / "night-01.json").read_text(encoding="utf-8"))
    assert record.get("subset") == subset
    last = [s for s in world.steps(NIGHTS[0]) if s["step"] == "morning"][-1]
    assert last.get("subset") == subset



def test_a_morning_on_one_probe_set_leaves_the_other_owed(tmp_path: Path) -> None:
    """A daily-v1 morning that succeeded leaves the full set owed, in its own file; each set once."""
    world = World(tmp_path)
    assert world.run(NIGHTS[0], "--go", "--subset", "daily-v1")[0] == 0
    assert world.run(NIGHTS[0], "--go", "--subset", "daily-v1")[1] == [
        "refused: the morning of 2026-10-24 already succeeded on the daily-v1 subset"]
    code, said = world.run(NIGHTS[0], "--go")
    assert code == 0 and said[-1].startswith(f"scored: {world.root / USER / 'scores' / 'night-02.json'} ")
    files = sorted((world.root / USER / "scores").glob("night-*.json"))
    assert [json.loads(p.read_text(encoding="utf-8")).get("subset") for p in files] == ["daily-v1", None]
    assert [s.get("subset") for s in world.steps(NIGHTS[0]) if s["status"] == "ok" and s["step"] == "morning"] == [
        "daily-v1", None]
    assert world.run(NIGHTS[0], "--go")[1] == ["refused: the morning of 2026-10-24 already succeeded on the full set"]


@pytest.mark.parametrize(("scored", "flag", "refused"), [
    ("daily-v1", (), None), ("daily-v1", ("--subset", "daily-v1"), "already scored"),
    (None, ("--subset", "daily-v1"), None), (None, (), "already scored")])
def test_a_score_file_refuses_only_a_backfill_on_its_own_probe_set(
    tmp_path: Path, scored: str | None, flag: tuple[str, ...], refused: str | None,
) -> None:
    world = World(tmp_path)  # the night's morning step failed: only the file speaks
    (world.root / USER / "scores").mkdir(parents=True)
    record = {"night": NIGHTS[0].isoformat(), **({"subset": scored} if scored else {})}
    (world.root / USER / "scores" / "night-01.json").write_text(json.dumps(record), encoding="utf-8")
    code, said = world.run(NIGHTS[0], *flag)
    assert (code, said[-1].startswith("refused: ")) == ((1, True) if refused else (0, False))
    assert refused in said[-1] if refused else any("night-02.json" in line for line in said)


@pytest.mark.parametrize(("accelerated", "flag", "refused"), [
    (True, (), True), (True, ("--subset", "daily-v1"), False), (False, (), True), (False, ("--subset", "daily-v1"), True)])
def test_a_morning_step_that_names_no_subset_asked_the_nights_own_set(
    tmp_path: Path, accelerated: bool, flag: tuple[str, ...], refused: bool,
) -> None:
    """An accelerated night's own set is the full one, a daily development night's daily-v1."""
    world = World(tmp_path)
    world.night(NIGHTS[0], {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": accelerated,
                            "label": "development"}, morning="ok")
    code, said = world.run(NIGHTS[0], *flag)
    assert (code == 1 and "already succeeded" in said[-1]) if refused else code == 0


@pytest.mark.parametrize(("recorded", "asked", "basis"), [
    ("daily-v1", "the daily-v1 subset: 4 probes, 8 replies", "as the night's summary records"),
    ("full", "the full set: 4 probes, 24 replies", "as the night's summary records"),
    (None, "the full set: 4 probes, 24 replies", "records none, so the full set (run before the record)")])
def test_the_default_set_is_the_one_the_nights_summary_records_else_the_full_set_and_the_plan_says_which(
    tmp_path: Path, recorded: str | None, asked: str, basis: str,
) -> None:
    world = World(tmp_path)
    world.night(NIGHTS[0], {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": True,
                            "label": "development", **({MORNING_PROBES: recorded} if recorded else {})})
    code, said = world.run(NIGHTS[0])
    assert code == 0 and any(asked in line for line in said) and any(basis in line for line in said)


def test_a_set_the_backfill_doesnt_know_is_refused_before_anything(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.night(NIGHTS[0], {"status": "promoted", "checkpoint": ref_record(_ckpt(1)), "accelerated": True,
                            MORNING_PROBES: "daily-v9"})
    assert world.run(NIGHTS[0], "--go")[1] == ["refused: the night's summary names a probe set this backfill "
                                               "doesn't know: 'daily-v9'"]
    assert world.model.calls == [] and not (world.root / USER / "scores").exists()


def test_a_backfill_holds_under_its_own_key_so_the_failed_mornings_spend_still_counts(tmp_path: Path) -> None:
    world, night = World(tmp_path), NIGHTS[0].isoformat()
    failed = world.ledger.reserve(f"morning:{night}:{USER}:judge:a1", ("meta_usd",), usd(1.0))
    assert failed is not None
    failed.settle(0.04)  # what the night's own morning spent before it failed
    ports = replace(world.ports, morning=lambda user, persona, on: replace(world.morning(user, persona, on),
                                                                           ledger=world.ledger))
    said: list[str] = []
    assert backfill_tool.main(["--user", USER, "--persona", "film_founder", "--night", night, "--go"], ports=ports,
                              out=said.append) == 0
    spent = her_spend(world.ledger.settlements(), USER, NIGHTS[0])
    assert spent[f"morning:{night}:{USER}:judge"] == pytest.approx(0.04)  # still hers, beside the backfill's
    assert spent[f"morning:{night}:{USER}:backfill-full:judge"] > 0 and not world.ledger.open_holds()


def _tree(root: Path) -> dict[str, str]:
    """Every file under ``root`` and its digest: a rescore leaves the results tree exactly as it was."""
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_rescore_plans_the_same_morning_again_into_its_own_folder(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.run(NIGHTS[0], "--go")[0] == 0  # the morning the lives were judged on
    folder = tmp_path / "deepseek"
    called = (len(world.model.calls), world.judge.calls)
    code, said = world.run(NIGHTS[0], "--score-root", str(folder))
    text = "\n".join(said)
    assert code == 0 and said[0].startswith("morning backfill: rescore, plan only")
    assert str(folder / USER / "scores" / "night-01.json") in text  # the name the results tree gave it
    assert "the judge: muse" in text
    assert (len(world.model.calls), world.judge.calls) == called and not folder.exists()


def test_a_rescore_writes_only_its_own_file_and_leaves_the_results_tree_byte_identical(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.run(NIGHTS[0], "--go")[0] == 0
    folder = tmp_path / "deepseek"
    before, steps = _tree(world.root), world.steps(NIGHTS[0])
    code, said = world.run(NIGHTS[0], "--score-root", str(folder), "--go")
    own = folder / USER / "scores" / "night-01.json"
    assert code == 0, said
    assert said[-1].startswith(f"scored: {own} ")
    assert _tree(world.root) == before  # not one byte of the results tree moved
    assert world.steps(NIGHTS[0]) == steps  # no morning step, no story told again
    record = json.loads(own.read_text(encoding="utf-8"))
    assert (record["night"], record["label"], record["checkpoint"], record["pinned"], record["judge"]) == (
        "2026-10-24", "development", "p-abc-n1", True, "muse")
    assert own.name == (world.root / USER / "scores" / "night-01.json").name  # the results tree's name, in its own


def test_a_rescore_records_the_judge_that_scored_it(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.run(NIGHTS[0], "--go")[0] == 0
    ports = replace(world.ports, morning=lambda user, persona, on: replace(
        world.morning(user, persona, on), judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH))
    folder, said = tmp_path / "deepseek", []
    assert backfill_tool.main(["--user", USER, "--persona", "film_founder", "--night", NIGHTS[0].isoformat(),
                               "--score-root", str(folder), "--go"], ports=ports, out=said.append) == 0
    assert "the judge: deepseek" in "\n".join(said)
    rescored = json.loads((folder / USER / "scores" / "night-01.json").read_text(encoding="utf-8"))
    original = json.loads((world.root / USER / "scores" / "night-01.json").read_text(encoding="utf-8"))
    assert (rescored["judge"], rescored["judge_model"]) == ("deepseek", fireworks_api.DEEPSEEK_V4P1_FLASH)
    assert (original["judge"], original["judge_model"]) == ("muse", "unknown")


def test_a_rescore_holds_under_a_key_of_its_own(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.run(NIGHTS[0], "--go")[0] == 0  # the morning's own spend, under its key
    ports = replace(world.ports, morning=lambda user, persona, on: replace(
        world.morning(user, persona, on), ledger=world.ledger))
    folder, said = tmp_path / "deepseek", []
    assert backfill_tool.main(["--user", USER, "--persona", "film_founder", "--night", NIGHTS[0].isoformat(),
                               "--score-root", str(folder), "--go"], ports=ports, out=said.append) == 0
    spent = her_spend(world.ledger.settlements(), USER, NIGHTS[0])
    assert spent[f"morning:{NIGHTS[0].isoformat()}:{USER}:rescore-full-muse:judge"] > 0
    assert not world.ledger.open_holds()


def test_a_rescore_that_fails_writes_neither_a_file_nor_a_step(tmp_path: Path) -> None:
    world = World(tmp_path)
    assert world.run(NIGHTS[0], "--go")[0] == 0
    world.judge.fail = RuntimeError("the judge is down")  # a rescore's failure is not the night's to log
    folder = tmp_path / "deepseek"
    before = _tree(world.root)
    chosen = backfill_tool.plan(USER, "film_founder", NIGHTS[0], world.ports, score_root=folder)
    with pytest.raises(RuntimeError, match="judge is down"):
        backfill_tool.backfill(chosen, world.ports)
    assert _tree(world.root) == before and not (folder / USER / "scores").exists()


@pytest.mark.parametrize(("rule", "expected"), [
    ("dir_file", "already exists"),
    ("inside_results", "inside the sleep results tree"),
    ("not_promoted", "wasn't promoted"),
    ("not_in_history", "isn't in the person's promoted history"),
    ("not_synthetic", "isn't registered synthetic"),
    ("lock_held", "night lock is held"),
])
def test_a_rescore_is_refused_before_any_call_or_write(tmp_path: Path, rule: str, expected: str) -> None:
    world = World(tmp_path, synthetic=rule != "not_synthetic")
    night = NIGHTS[0]
    folder = tmp_path / "deepseek"
    if rule == "not_promoted":
        world.night(night, {"status": "rejected", "checkpoint": None, "accelerated": True})
    elif rule == "not_in_history":
        unknown = CheckpointRef("p-abc-n9", USER, night, "uri://9", None)
        world.night(night, {"status": "promoted", "checkpoint": ref_record(unknown), "accelerated": True})
    elif rule == "dir_file":
        (folder / USER / "scores").mkdir(parents=True)
        (folder / USER / "scores" / "night-01.json").write_text(json.dumps({"night": night.isoformat()}),
                                                                encoding="utf-8")
    elif rule == "inside_results":
        folder = world.root / "deepseek"
    before = _tree(world.root)
    with open(tmp_path / "night.lock", "a+b") as other:  # another run's lock, when the rule says so
        if rule == "lock_held":
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, said = world.run(night, "--score-root", str(folder), "--go")
    assert code == 1 and said[-1].startswith("refused: ") and expected in said[-1]
    assert world.model.calls == [] and world.judge.calls == 0
    assert _tree(world.root) == before and not (folder / USER / "scores" / "night-02.json").exists()


# ── a final night (Task 12b): scored once, on its last night, under the registered judge ─


DONOR = "syn-donor"
RUN = {"days": 3, "start": NIGHTS[0].isoformat(), "lives": [[USER, "film_founder"], [DONOR, "founder_donor"]],
       "seed": 1, "alone_from": None}  # the final run the nights ran in: USER and its 3x donor, NIGHTS[2] the last


def _final_night(world: World, night: date, morning: dict[str, Any], *, donor: str | None = "promoted") -> None:
    from anjo.sleep.sim.final_batch import RUN as RUN_KEY

    folder = world.night(night, {"status": "promoted", "checkpoint": ref_record(_ckpt(NIGHTS.index(night) + 1)),
                                 "accelerated": True, "label": "final", "cost_usd": 1.0, RUN_KEY: RUN})
    if donor is not None:  # the donor's own night, in the same run
        theirs = world.root / DONOR / "nights" / night.isoformat()
        theirs.mkdir(parents=True, exist_ok=True)
        (theirs / "summary.json").write_text(json.dumps({"status": donor, "label": "final", RUN_KEY: RUN}),
                                             encoding="utf-8")
    (folder / "log.jsonl").write_text(json.dumps({"step": "morning", "uses": "new", "system": None, **morning}) + "\n",
                                      encoding="utf-8")


def test_a_final_night_whose_morning_was_put_off_is_never_backfilled(tmp_path: Path) -> None:
    from anjo.sleep.sim.final_batch import DEFERRED

    world = World(tmp_path)
    _final_night(world, NIGHTS[0], {"status": "skipped", "why": DEFERRED})
    world_deepseek = replace(world.ports, morning=lambda *a: replace(world.morning(*a),
                                                                       judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH))
    with pytest.raises(backfill_tool.Refused, match="put off"):
        backfill_tool.plan(USER, "film_founder", NIGHTS[0], world_deepseek)
    assert world.model.calls == [] and world.judge.calls == 0 and not (world.root / USER / "scores").exists()


def test_a_final_backfill_runs_only_under_the_registered_judge(tmp_path: Path) -> None:
    world = World(tmp_path)
    _final_night(world, NIGHTS[2], {"status": "failed", "why": "morning failed: TimeoutError"})
    code, said = world.run(NIGHTS[2], "--go")  # the fake builder's judge is no registered one
    assert code == 1 and "registered judge" in said[-1] and world.judge.calls == 0
    deepseek = replace(world.ports, morning=lambda *a: replace(world.morning(*a),
                                                               judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH))
    chosen = backfill_tool.plan(USER, "film_founder", NIGHTS[2], deepseek)
    assert chosen.label == "final" and chosen.judge == "deepseek"
    record = json.loads(backfill_tool.backfill(chosen, deepseek).path.read_text(encoding="utf-8"))
    assert (record["label"], record["judge"], record["pinned"]) == ("final", "deepseek", True)


def test_a_final_backfill_waits_for_every_lifes_last_night_in_its_run(tmp_path: Path) -> None:
    """The review's case: a run stopped on its last day with the donor's night failed; a backfill would
    score this life's 3x on the donor's earlier adapter. Refused until the donor's last night ran."""
    world = World(tmp_path)
    deepseek = replace(world.ports, morning=lambda *a: replace(world.morning(*a),
                                                               judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH))
    _final_night(world, NIGHTS[2], {"status": "failed", "why": "morning failed: TimeoutError"}, donor=None)
    with pytest.raises(backfill_tool.Refused, match=f"the last night of {DONOR} didn't run"):  # no donor night yet
        backfill_tool.plan(USER, "film_founder", NIGHTS[2], deepseek)
    _final_night(world, NIGHTS[2], {"status": "failed", "why": "morning failed: TimeoutError"}, donor="failed")
    with pytest.raises(backfill_tool.Refused, match=f"the last night of {DONOR} didn't run"):
        backfill_tool.plan(USER, "film_founder", NIGHTS[2], deepseek)
    _final_night(world, NIGHTS[2], {"status": "failed", "why": "morning failed: TimeoutError"}, donor="napped")
    assert backfill_tool.plan(USER, "film_founder", NIGHTS[2], deepseek).label == "final"  # a nap ran: it serves
    assert world.model.calls == [] and world.judge.calls == 0


def test_a_final_backfill_scores_only_its_runs_last_night_and_needs_the_run_recorded(tmp_path: Path) -> None:
    world = World(tmp_path)
    deepseek = replace(world.ports, morning=lambda *a: replace(world.morning(*a),
                                                               judge_model=fireworks_api.DEEPSEEK_V4P1_FLASH))
    _final_night(world, NIGHTS[1], {"status": "failed", "why": "morning failed: TimeoutError"})
    with pytest.raises(backfill_tool.Refused, match="isn't its final run's last night"):
        backfill_tool.plan(USER, "film_founder", NIGHTS[1], deepseek)
    world.night(NIGHTS[2], {"status": "promoted", "checkpoint": ref_record(_ckpt(3)), "accelerated": True,
                            "label": "final", "cost_usd": 1.0})  # stamped before the run was recorded
    with pytest.raises(backfill_tool.Refused, match="records no run"):
        backfill_tool.plan(USER, "film_founder", NIGHTS[2], deepseek)
    assert world.model.calls == [] and world.judge.calls == 0
