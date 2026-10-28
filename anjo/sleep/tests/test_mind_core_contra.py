# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's core on contradictions: the minimal pairs, the stance probe and the contrast loss (toy
lives, CPU only)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anjo.sleep.mind.core import net, toy, train  # noqa: E402
from anjo.sleep.mind.core.types import CoreConfig  # noqa: E402

SMALL = CoreConfig(dim_in=toy.DIM, width=16, slots=4)


def test_contradiction_pairs_differ_only_from_the_constraint_on() -> None:
    for pair in toy.contradiction_pairs(4, seed=7):
        problem = {e.event_id: e for e in pair.problem.events}
        benign = {e.event_id: e for e in pair.benign.events}
        constraint = next(i for i in problem if i.endswith("-constraint"))
        cut = problem[constraint].t
        before = [i for i, e in problem.items() if e.t < cut]
        assert before and all(i in benign and np.array_equal(problem[i].vec, benign[i].vec) for i in before)
        assert not np.array_equal(problem[constraint].vec, benign[constraint].vec)
        labels = {lab.id: lab for lab in pair.problem.labels}, {lab.id: lab for lab in pair.benign.labels}
        assert labels[0][pair.label].family == "contradiction" and labels[0][pair.label].failed
        assert labels[1][pair.label].family == "decoy" and not labels[1][pair.label].failed


def test_the_stance_key_reads_the_topic_whatever_the_sign() -> None:
    torch.manual_seed(0)
    stance = net.CoreNet(dataclasses.replace(SMALL, stance=True))
    plain = net.CoreNet(SMALL)
    assert stance.wk.bias is None  # a key is a pure projection of the event
    e = torch.randn(3, SMALL.width)
    assert torch.allclose(stance._key(e), stance._key(-e))
    assert torch.allclose(plain._key(e), -plain._key(-e))


def test_the_stance_core_steps_and_keeps_its_shapes() -> None:
    torch.manual_seed(0)
    core = net.CoreNet(dataclasses.replace(SMALL, stance=True))
    st = core.initial(2)
    e = torch.randn(2, SMALL.width)
    new, written = core.step(st, e, torch.zeros(2), torch.ones(2))
    assert new.slots.shape == st.slots.shape and written.write.shape == (2, SMALL.slots)
    assert torch.isfinite(new.slots).all()


def test_the_contrast_loss_asks_the_problem_thread_for_more_risk_than_its_twin() -> None:
    torch.manual_seed(0)
    model = net.CoreNet(SMALL)
    data, label = train._pair_batch(toy.contradiction_pairs(3, seed=7), SMALL.horizon)
    run = train.unroll(model, data)
    risk = train.thread_risk(run, data, label)
    assert risk.shape == (6,) and bool(((risk >= 0) & (risk <= 1)).all())
    total, parts = train.contrast_loss(model, data, label)
    plain, _ = train.loss(model, data, run)
    gap = float((risk[:3] - risk[3:]).mean().detach())
    assert parts["contrast"] > 0 and math.isfinite(float(total))
    assert float(total) == pytest.approx(float(plain) + parts["contrast"], rel=1e-5)
    assert gap < train.CONTRAST_MARGIN  # an untrained core hasn't met the margin yet
    total.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())


def test_shared_training_takes_each_epochs_pairs() -> None:
    asked: list[int] = []

    def pairs(epoch: int) -> tuple[toy.ContrastPair, ...]:
        asked.append(epoch)
        return toy.contradiction_pairs(4, seed=5000 + epoch)

    run = train.train_shared(toy.lives(8, seed=1), toy.lives(4, seed=2), SMALL, seed=0, epochs=2, batch_size=4,
                             pairs=pairs)
    assert asked == [0, 1] and len(run.log) == 2
    assert all(math.isfinite(fit) and math.isfinite(val) for _, fit, val in run.log)
