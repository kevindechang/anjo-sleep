# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The live comparison's matched control: a budget per run and per night (never a host setting), the
SFT pass or the seed path's main block trained for it by the clock with one warmup and one decay, DPO
after, and the receipt's budget and minutes used. Offline: fakes and a tiny CPU model."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.sim import accelerated
from anjo.sleep.tests.test_lora import REVISION, _examples
from anjo.sleep.tests.test_sft_dpo_default import NIGHT, Seen
from anjo.sleep.tests.test_sleep_objective import _plan, _tiny, _torch
from anjo.sleep.train import colab_train, objective, sleep_loss
from anjo.sleep.train.lora import LoraCfg, LoraCfgError, adapter_name, training_config
from anjo.sleep.train.trainer import LoraTrainer

BUDGETS = ((NIGHT, 6.5), (NIGHT + timedelta(days=1), 7.25))


class _Tick:
    """A clock that moves ``step`` seconds each time it is read."""

    def __init__(self, step: float = 1.0) -> None:
        self.now, self.step = 0.0, step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def test_no_host_setting_pads_a_night_the_budget_is_per_run_and_per_night() -> None:
    assert objective.night_cfg({"ANJO_SLEEP_MATCH_MINUTES": "6.5", objective.OBJECTIVE_ENV: "sft_dpo"}) == LoraCfg(objective="sft_dpo")
    assert not hasattr(objective, "MATCH_ENV")
    cfg = LoraCfg(objective="sft_dpo", match_by_night=BUDGETS)
    sent = training_config(cfg, 12, warm=False, revision="a" * 40, night=NIGHT + timedelta(days=1))
    assert sent["match_minutes"] == 7.25 and colab_train.load_config(json.dumps(sent))["match_minutes"] == 7.25
    with pytest.raises(LoraCfgError, match="no budget"):
        training_config(cfg, 12, warm=False, revision="a" * 40, night=NIGHT - timedelta(days=1))
    for bad in ({"match_by_night": ((NIGHT, 0.0),)}, {"match_by_night": ((NIGHT, 101.0),)},
                {"match_by_night": ((NIGHT, 5.0), (NIGHT, 6.0))},
                {"objective": "sft_dpo_seed", "dreams": True, "match_by_night": BUDGETS}):
        with pytest.raises(LoraCfgError):
            LoraCfg(**{"objective": "sft_dpo", **bad})  # type: ignore[arg-type]
    seeded = training_config(LoraCfg(objective="sft_dpo_seed", match_by_night=BUDGETS), 12, warm=False,
                             revision="a" * 40, night=NIGHT)
    assert colab_train.load_config(json.dumps(seeded))["match_minutes"] == 6.5  # the seed path, dreams off
    with pytest.raises(ValueError, match="dreams"):
        colab_train.load_config(json.dumps({**seeded, "terms": {"fold_rank": None, "dreams": True}}))


def test_the_accelerated_run_reads_a_match_file_covering_every_night(tmp_path: Path) -> None:
    path = tmp_path / "match.json"
    path.write_text(json.dumps({night.isoformat(): minutes for night, minutes in BUDGETS}))
    assert accelerated.match_by_night(path, NIGHT, 2) == BUDGETS
    for days, text in ((3, path.read_text()), (1, "[1, 2]"), (1, json.dumps({NIGHT.isoformat(): "six"})),
                       (1, json.dumps({"tomorrow": 6.0}))):
        path.write_text(text)
        with pytest.raises(accelerated.AcceleratedError):
            accelerated.match_by_night(path, NIGHT, days)


def test_the_stretch_is_one_warmup_and_one_cosine_and_both_copies_agree() -> None:
    shares = [i / 200 for i in range(-2, 203)]
    assert [sleep_loss.stretch(x) for x in shares] == [colab_train.stretch(x) for x in shares]
    assert sleep_loss.stretch(0.0) == 0.0 and sleep_loss.stretch(sleep_loss.WARMUP) == 1.0 and sleep_loss.stretch(1.0) == 0.0
    after = [sleep_loss.stretch(x) for x in shares if x >= sleep_loss.WARMUP]
    assert all(a >= b for a, b in zip(after, after[1:], strict=False))  # one decay, never a second warmup


def test_the_stretched_sft_sets_the_rate_by_the_clock_and_stops_at_the_budget() -> None:
    tick = _Tick(10.0)
    clock = colab_train.StretchClock(100.0, clock=tick)
    args, state, control = SimpleNamespace(learning_rate=2e-4), SimpleNamespace(), SimpleNamespace(should_training_stop=False)
    optimizer = SimpleNamespace(param_groups=[{"lr": 2e-4}, {"lr": 2e-4}])
    with pytest.raises(ValueError, match="started"):
        clock.on_pre_optimizer_step(args, state, control, optimizer=optimizer)
    clock.on_train_begin(args, state, control)  # t = 10
    rates = []
    while not control.should_training_stop:
        clock.on_pre_optimizer_step(args, state, control, optimizer=optimizer)
        rates.append(optimizer.param_groups[0]["lr"])
        assert optimizer.param_groups[1]["lr"] == rates[-1]
        clock.on_step_end(args, state, control)
    assert rates == [pytest.approx(2e-4 * colab_train.stretch(share)) for share in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert clock.receipt(5) == {"budget_minutes": round(100 / 60, 4), "minutes": round(100 / 60, 4), "steps": 5}
    sft = colab_train.stretched_sft(colab_train.build_settings(colab_train.load_config(json.dumps(
        training_config(LoraCfg(objective="sft_dpo", match_by_night=BUDGETS), 12, warm=False, revision="a" * 40,
                        night=NIGHT))))["sft"])
    assert (sft["lr_scheduler_type"], sft["warmup_steps"], sft["max_steps"]) == ("constant", 0, colab_train.STRETCH_STEPS)


def test_the_seed_paths_control_trains_its_main_block_for_the_budget_then_dpo() -> None:
    torch = _torch()
    batch = sleep_loss.Batch(
        sft=(sleep_loss.Seq((1, 2), (3, 11)),), pairs=(sleep_loss.PairSeq((1, 5), (6, 11), (7, 11)),),
        seeds=tuple(sleep_loss.SeedSeq((1, i), (9, 9, 1, i), (4, 5, 11)) for i in range(2, 4)),
    )
    plain = sleep_loss.train_terms(_tiny(torch), batch, _plan(), timer=_Tick())
    stretched = sleep_loss.train_terms(_tiny(torch), batch, _plan(stretch_s=400.0), timer=_Tick())
    done = stretched["stretch"]
    assert "stretch" not in plain and done["steps"] > plain["steps"] - plain["dpo"]["steps"]  # more than the plan
    assert done["budget_minutes"] == round(400 / 60, 4) <= done["minutes"] < round(400 / 60, 4) + 1.0
    assert stretched["dpo"]["steps"] == plain["dpo"]["steps"] and stretched["steps"] == done["steps"] + plain["dpo"]["steps"]
    with pytest.raises(ValueError, match="no dreams"):
        sleep_loss.train_terms(_tiny(torch), replace(batch, dreams=(sleep_loss.Seq((1, 2), (3, 11)),),
                               anchors=(sleep_loss.AnchorSeq((1, 2), ((3,), (4,)), 0),)), _plan(stretch_s=60.0, dreams=True))


class _Matched(Seen):
    def run(self, job: str, inputs: Any, *, timeout_s: float) -> Any:
        done = super().run(job, inputs, timeout_s=timeout_s)
        sent = inputs["config"]
        return {**done, "match": {"budget_minutes": sent["match_minutes"], "minutes": 6.6, "steps": 40}} if "match_minutes" in sent else done


def test_the_night_log_carries_the_controls_budget_and_minutes_only_for_a_control(tmp_path: Path) -> None:
    for cfg in (LoraCfg(objective="sft_dpo", match_by_night=BUDGETS), LoraCfg(objective="sft_dpo")):
        trainer = LoraTrainer(_Matched(), root=tmp_path / str(bool(cfg.match_by_night)), publish=lambda f, n: None,
                              revision=REVISION, cfg=cfg)
        _, logged = trainer.train("p1", NIGHT, None, _examples(2), adapter_name("p1", 3, 0))
        assert logged.get("match") == ({"budget_minutes": 6.5, "minutes": 6.6, "steps": 40} if cfg.match_by_night else None)
        assert ("match" in logged) == bool(cfg.match_by_night)

