# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The paper's keep-if-improves dream selection inside the one job (``sleep_loss.py``): a held-out
split fixed by the plan seed, four-step trials that restore the weights and the optimizer's state,
the anchor band, the day's four imitation samples, and the dream block's projector rollback. CPU
only, on the tiny model the objective's tests use; skipped without torch."""
from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from anjo.sleep.tests.test_sleep_objective import _plan, _tiny  # noqa: E402
from anjo.sleep.train import sleep_loss  # noqa: E402

GROUP = "dream"


def _seeds(n: int) -> tuple[Any, ...]:
    return tuple(sleep_loss.SeedSeq((1, i), (7, 8, 1, i), (3, 4, 11), group=GROUP) for i in range(2, 2 + n))


def _anchors() -> tuple[Any, ...]:
    return (sleep_loss.AnchorSeq((1, 2), ((3,), (4,)), 0),)


def _optimizer(model: Any) -> Any:
    return torch.optim.AdamW([model.lora_A, model.lora_B], lr=0.05, weight_decay=0.0)


def _params(model: Any) -> list[Any]:
    return [model.lora_A.detach().clone(), model.lora_B.detach().clone()]


def test_a_group_that_lowers_held_out_kl_is_kept_and_one_that_does_not_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, seeds, plan = _tiny(torch), _seeds(10), _plan()
    held, groups = sleep_loss.dream_split(seeds, plan.seed)
    assert len(groups) == 2
    before = _params(model)
    taus = iter([1.0, 0.98, 0.999])  # untouched, group 0 (2% down), group 1 (not 1% down)
    monkeypatch.setattr(sleep_loss, "mean_seed_kl", lambda *a, **k: next(taus))
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: 0.5)
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    optimizer = _optimizer(model)
    timer = iter([0.0, 120.0, 120.0, 120.0, 120.0, 120.0])  # started, three deadline checks, two groups, minutes

    kept, receipt = sleep_loss._dream_trials(
        model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), plan, teacher, optimizer,
        timer=lambda: next(timer),
    )

    assert kept == groups[0] and receipt["groups"][0]["kept"] and not receipt["groups"][1]["kept"]
    assert receipt["kept"] == len(groups[0]) and receipt["groups_kept"] == 1
    assert receipt["held"] == sleep_loss.DREAM_HELD and receipt["dreams"] == 10
    assert receipt["groups"][1]["tau_after"] == 0.999 and receipt["minutes"] == 2.0
    assert all(torch.equal(a, b) for a, b in zip(_params(model), before, strict=True))  # every trial restored
    assert optimizer.state_dict()["state"] == {}  # the optimizer's state too
    assert [group["lr"] for group in optimizer.param_groups] == [0.05]
    trained = [ids for ids, _, grad, _, _ in model.calls if grad]
    assert not any(ids in {(*seeds[i].teacher, *seeds[i].reply) for i in held} for ids in trained)


def test_the_anchor_band_drops_a_group_that_lowers_tau_but_moves_the_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, seeds, plan = _tiny(torch), _seeds(10), _plan()
    taus = iter([1.0, 0.98, 0.98])  # both groups lower held-out KL by 2%
    anchors = iter([0.5, 0.496, 0.494])  # group 0 within the 0.005 band, group 1 beyond it
    monkeypatch.setattr(sleep_loss, "mean_seed_kl", lambda *a, **k: next(taus))
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: next(anchors))
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])

    _, receipt = sleep_loss._dream_trials(model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), plan, teacher,
                                          _optimizer(model))

    assert [group["kept"] for group in receipt["groups"]] == [True, False]
    assert receipt["groups_kept"] == 1


def test_held_out_dream_rows_are_scored_but_never_trained() -> None:
    model, seeds, plan = _tiny(torch), _seeds(10), _plan()
    held, groups = sleep_loss.dream_split(seeds, plan.seed)
    assert held and groups
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    before = _params(model)

    sleep_loss._dream_trials(model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), plan, teacher, _optimizer(model))

    held_ids = {(*seeds[i].teacher, *seeds[i].reply) for i in held} | {(*seeds[i].prompt, *seeds[i].reply) for i in held}
    assert all(not grad for ids, _, grad, _, _ in model.calls if ids in held_ids)  # tau only, no gradient
    assert all(torch.equal(a, b) for a, b in zip(_params(model), before, strict=True))


def test_the_trials_need_the_anchor_items_they_score() -> None:
    model, seeds, plan = _tiny(torch), _seeds(10), _plan()
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    with pytest.raises(ValueError, match="anchor"):
        sleep_loss._dream_trials(model, sleep_loss.Batch(seeds=seeds), plan, teacher, _optimizer(model))


def test_the_deadline_stops_the_trials_before_tau_the_anchors_and_every_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, seeds = _tiny(torch), _seeds(10)
    plan = _plan(deadline=50.0)
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    called: list[str] = []
    monkeypatch.setattr(sleep_loss, "mean_seed_kl", lambda *a, **k: called.append("tau") or 1.0)
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: called.append("anchor") or 0.5)
    before = _params(model)

    kept, receipt = sleep_loss._dream_trials(
        model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), plan, teacher, _optimizer(model),
        timer=lambda: 60.0,
    )

    assert called == [] and kept == []  # no group tried, nothing measured, nothing kept
    assert receipt["groups"] == [] and receipt["kept"] == 0 and receipt["trialed"] and receipt["out_of_time"]
    assert all(torch.equal(a, b) for a, b in zip(_params(model), before, strict=True))


def test_the_deadline_cuts_a_later_group_but_keeps_the_one_already_judged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, seeds, plan = _tiny(torch), _seeds(10), _plan(deadline=50.0)
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    _, groups = sleep_loss.dream_split(seeds, plan.seed)
    taus = iter([1.0, 0.98])  # untouched, then group 1's 2% drop: it is kept
    monkeypatch.setattr(sleep_loss, "mean_seed_kl", lambda *a, **k: next(taus))
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: 0.5)
    clock = iter([0.0, 0.0, 0.0, 0.0, 100.0, 100.0])  # started, two checks, group 1, group 2, minutes

    kept, receipt = sleep_loss._dream_trials(
        model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), plan, teacher, _optimizer(model),
        timer=lambda: next(clock),
    )

    assert kept == groups[0] and len(receipt["groups"]) == 1  # group 2 was never tried
    assert receipt["out_of_time"] and receipt["minutes"] == round(100.0 / 60.0, 4)


def test_each_trial_restores_the_torch_rng_so_the_main_block_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, seeds, plan = _tiny(torch), _seeds(10), _plan()
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])

    def noisy(model: Any, row: Any, teacher: Any, soft: Any = None, ref: Any = None) -> Any:
        if torch.is_grad_enabled():  # a training forward's dropout draws; the eval measure draws nothing
            torch.rand(4)
        return (model.lora_B ** 2).sum() * 1e-4

    monkeypatch.setattr(sleep_loss, "seed_kl", noisy)
    before = torch.get_rng_state().clone()

    sleep_loss._dream_trials(model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), plan, teacher,
                             _optimizer(model))

    assert torch.equal(torch.get_rng_state(), before)  # two trials' draws are not the main block's


def test_four_dreams_or_fewer_train_untried_and_measure_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(torch)
    seeds = _seeds(sleep_loss.DREAM_HELD)
    plan = _plan()
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    called: list[str] = []
    monkeypatch.setattr(sleep_loss, "mean_seed_kl", lambda *a, **k: called.append("tau") or 0.0)
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: called.append("anchor") or 0.0)
    before = _params(model)

    kept, receipt = sleep_loss._dream_trials(model, sleep_loss.Batch(seeds=seeds), plan, teacher, _optimizer(model))

    assert called == [] and kept == list(range(len(seeds)))  # every dream trains as an ordinary seed row
    assert receipt == {"dreams": sleep_loss.DREAM_HELD, "trialed": False}  # and no anchors were needed
    assert all(torch.equal(a, b) for a, b in zip(_params(model), before, strict=True))


def test_fewer_dreams_than_the_held_out_set_train_as_ordinary_seed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny(torch)
    seeds = _seeds(sleep_loss.DREAM_HELD)
    monkeypatch.setattr(sleep_loss, "mean_seed_kl", lambda *a, **k: pytest.fail("a trial measured tau"))
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: pytest.fail("a trial scored the anchors"))

    receipt = sleep_loss.train_terms(model, sleep_loss.Batch(seeds=seeds, anchors=_anchors()), _plan())

    assert receipt["dream_trials"] == {"dreams": sleep_loss.DREAM_HELD, "trialed": False}
    assert receipt["rows"]["seed"] == sleep_loss.DREAM_HELD  # every dream trained, as a seed row


def test_a_cut_trial_sets_out_of_time_and_counts_its_seconds_in_the_receipt() -> None:
    model, seeds = _tiny(torch), _seeds(10)
    batch = sleep_loss.Batch(sft=(sleep_loss.Seq((1, 2), (3, 11)),), seeds=seeds, anchors=_anchors())

    receipt = sleep_loss.train_terms(model, batch, _plan(deadline=0.0))  # monotonic is always past zero

    assert receipt["out_of_time"] and receipt["dream_trials"]["out_of_time"] and receipt["dream_trials"]["groups"] == []
    assert "dream_trials" in receipt["minutes"] and receipt["rows"]["seed"] == 0  # no group was kept



def test_day_dreams_take_four_imitation_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(torch)
    row = sleep_loss.SeedSeq((1, 2), (1, 2), (3, 4, 5, 6, 11))
    calls: list[int] = []
    monkeypatch.setattr(sleep_loss, "sample", lambda *a, **k: calls.append(1) or ())
    monkeypatch.setattr(sleep_loss, "edit_reward", lambda *a, **k: 0.0)
    sleep_loss.imitation_loss(model, row, random.Random(0), _plan(day_dreams=True))
    assert len(calls) == sleep_loss.IMITATION_SAMPLES_DAY == 4
    calls.clear()
    sleep_loss.imitation_loss(model, row, random.Random(0), _plan())
    assert len(calls) == sleep_loss.IMITATION_SAMPLES == 2


class _RollbackRun:
    """A run whose one step moves every parameter in the optimizer, so a rollback is visible."""

    def __init__(self, model: Any, projector: Any, plan: Any) -> None:
        self.model, self.params, self.plan = model, [model.lora_A, model.lora_B], plan
        self.optimizer = type("_Optimizer", (), {
            "param_groups": [{"params": [*self.params, projector]}],
        })()

    def clock(self, part: str) -> Any:
        return nullcontext()

    def tokens(self, state: str | None, **kw: Any) -> None:
        return None

    def step(self, chunk: Any) -> None:
        with torch.no_grad():
            for p in self.optimizer.param_groups[0]["params"]:
                p.add_(1.0)


def test_the_dream_blocks_rollback_restores_the_projector_under_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny(torch)
    projector = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    batch = sleep_loss.Batch(dreams=(sleep_loss.Seq((2, 3), (4, 4, 11)),), anchors=_anchors())
    scores = iter([1.0, 0.0])
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: next(scores))
    run = _RollbackRun(model, projector, _plan(day_dreams=True))
    before = [p.detach().clone() for p in [model.lora_A, model.lora_B, projector]]

    sleep_loss._dream_block(run, batch, 1)

    assert all(torch.equal(p, was) for p, was in zip([model.lora_A, model.lora_B, projector], before, strict=True))

    off = _RollbackRun(model, projector, _plan(day_dreams=False))
    scores = iter([1.0, 0.0])
    monkeypatch.setattr(sleep_loss, "anchor_score", lambda *a, **k: next(scores))
    lora, held = model.lora_A.detach().clone(), projector.detach().clone()
    sleep_loss._dream_block(off, batch, 1)
    assert torch.equal(model.lora_A, lora) and not torch.equal(projector, held)  # the old rollback kept the projector move
