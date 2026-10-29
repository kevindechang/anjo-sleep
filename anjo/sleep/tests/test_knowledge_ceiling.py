# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The knowledge measure's oracle ceiling (``python -m anjo.sleep.tools.knowledge_ceiling``): one adapter
trained with plain SFT on the very statements the frozen set scores. CEILING: contamination by design."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.gates import EVAL_DAY
from anjo.sleep.serving import config
from anjo.sleep.sets import knowledge
from anjo.sleep.tests.test_modal_runner import FakeCalls
from anjo.sleep.tools import knowledge_ceiling
from anjo.sleep.train.lora import LoraCfg, plan_steps
from anjo.sleep.train.runners.modal import A100_USD_PER_HOUR

PERSONA = "film_founder"


def _set() -> knowledge.KnowledgeSet:
    return knowledge.load(PERSONA)


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {"ANJO_DATA_DIR": str(tmp_path), **extra}


def _printed(capsys: pytest.CaptureFixture[str]) -> str:
    return str(capsys.readouterr().out)


def _no_statement_in(text: str) -> None:
    assert not any(item.statement in text for item in _set().items)  # ids, names and numbers only


def test_the_rows_are_each_statement_behind_the_set_once_after_its_own_prompt() -> None:
    the_set = _set()
    data = knowledge_ceiling.ceiling_dataset(the_set)
    statements = {item.source: item.statement for item in the_set.items}
    assert len(data.examples) == len(statements) == len({item.source for item in the_set.items})
    assert [e.completion for e in data.examples] == list(statements.values())  # in the set's order, each once
    assert all(e.prompt == (("user", knowledge.PROMPT),) for e in data.examples)  # the scored prompt alone: no system turn
    assert all(e.new and not e.context_dropped and not e.teacher for e in data.examples)
    assert data.pairs == () and data.tokens == sum(e.tokens for e in data.examples) > 0
    assert all(item.prompt == knowledge.PROMPT for item in the_set.items)


def test_a_statement_with_two_true_lines_is_refused() -> None:
    item = _set().items[0]
    other = knowledge.KnowledgeItem(f"{item.id}-x", item.source, item.kind, item.key, item.prompt,
                                    (item.statement + " too", item.statement), 0)
    broken = knowledge.KnowledgeSet(PERSONA, knowledge.VERSION, "0" * 64, (item, other))
    with pytest.raises(knowledge_ceiling.CeilingError):
        knowledge_ceiling.ceiling_dataset(broken)


def test_the_points_are_the_nights_recipe_and_the_most_the_server_takes() -> None:
    rows = len(knowledge_ceiling.ceiling_dataset(_set()).examples)
    recipe = knowledge_ceiling.point_cfg("recipe", rows)
    assert recipe == LoraCfg(objective="sft")  # the nights' rank, rate and step plan
    assert plan_steps(rows, recipe).steps == 9 and recipe.r == 16
    capacity = knowledge_ceiling.point_cfg("capacity", rows)
    assert capacity.r == config.MAX_LORA_RANK and capacity.objective == "sft" and capacity.lr == recipe.lr
    steps = plan_steps(rows, capacity)
    assert (steps.accumulation, steps.epochs, steps.steps) == (1, knowledge_ceiling.CAPACITY_PASSES, 3 * rows)
    with pytest.raises(knowledge_ceiling.CeilingError):
        knowledge_ceiling.point_cfg("bigger", rows)


def test_the_plan_holds_starts_and_prints_nothing_of_the_statements(tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    calls = FakeCalls(on_spawn=lambda job: pytest.fail("the plan spawned a call"))
    assert knowledge_ceiling.main([], environ=_env(tmp_path), modal_calls=lambda: calls) == 0
    out = _printed(capsys)
    assert "CEILING" in out and "never a result" in out and "nothing was held" in out
    held = knowledge_ceiling.CEILING_TIMEOUT_S / 3600.0 * A100_USD_PER_HOUR
    assert f"hold: ${held:.2f} on modal_usd" in out and "9 steps" in out and "rank 16" in out
    assert "knowledge_score" in out and "--ceiling recipe=" in out
    _no_statement_in(out)
    assert calls.spawned == [] and Ledger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()
    assert knowledge_ceiling.main(["--point", "capacity"], environ=_env(tmp_path), modal_calls=lambda: calls) == 0
    assert f"{3 * 42} steps" in (out := _printed(capsys)) and f"rank {config.MAX_LORA_RANK}" in out


def _go(tmp_path: Path, calls: FakeCalls, *argv: str) -> int:
    return knowledge_ceiling.main(["--go", *argv], environ=_env(tmp_path, **{config.REVISION_ENV: "c" * 40}),
                                  modal_calls=lambda: calls)


def test_go_trains_one_cold_sft_adapter_on_the_statements_dated_on_the_gates_day(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    calls = FakeCalls()
    assert _go(tmp_path, calls) == 0
    [job] = calls.spawned
    recipe: dict[str, Any] = json.loads(job.config_json)
    assert recipe["objective"] == "sft" and recipe["warm_start"] is False and recipe["revision"] == "c" * 40
    assert (recipe["r"], recipe["accumulation"], recipe["epochs"]) == (16, 5, 1)
    assert recipe["current_date"] == EVAL_DAY.isoformat()  # the date every scored prompt carries
    rows = [json.loads(line) for line in job.dataset.splitlines() if line.strip()]
    statements = list({item.source: item.statement for item in _set().items}.values())
    assert [row["completion"][-1]["content"] for row in rows] == statements
    assert all(row["prompt"] == [{"role": "user", "content": knowledge.PROMPT}] for row in rows)
    out = _printed(capsys)
    assert f"--ceiling recipe={job.name} --go" in out and "CEILING" in out
    assert f"modal volume rm {config.ADAPTERS_VOLUME} /{job.name} --recursive" in out
    _no_statement_in(out)
    assert Ledger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()  # settled at what ran
    capacity = FakeCalls()
    assert _go(tmp_path, capacity, "--point", "capacity") == 0
    sent = json.loads(capacity.spawned[0].config_json)
    assert (sent["r"], sent["accumulation"], sent["epochs"]) == (config.MAX_LORA_RANK, 1, knowledge_ceiling.CAPACITY_PASSES)
    assert capacity.spawned[0].name != job.name  # a fresh synthetic person each run
    assert f"--ceiling capacity={capacity.spawned[0].name} --go" in _printed(capsys)


def test_go_without_a_revision_is_refused_before_anything_is_held(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    calls = FakeCalls(on_spawn=lambda job: pytest.fail("spawned without a revision"))
    assert knowledge_ceiling.main(["--go"], environ=_env(tmp_path), modal_calls=lambda: calls) == 2
    assert config.REVISION_ENV in _printed(capsys) and calls.spawned == []
    assert Ledger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()


def test_a_failed_job_is_reported_and_settled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _go(tmp_path, FakeCalls(fail="timeout")) == 1
    out = _printed(capsys)
    assert "ceiling failed" in out and "--ceiling" not in out.split("ceiling failed")[1]
    assert Ledger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()
    with pytest.raises(RuntimeError):  # the job's own failure is raised as the smoke's is, after the hold is settled
        _go(tmp_path, FakeCalls(fail="remote"))
    assert Ledger(tmp_path / "sleep" / "budget.jsonl").open_holds() == ()
