# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND's surprise weights (the Sleep link) reach the loss: an SFT row's weight multiplies its SFT
term and a pair's its DPO term, before the chunk's mean; a row without one trains exactly as today.
``sft_dpo_seed`` reports them applied, TRL's passes report them not applied, and a night with none
logs nothing new. Offline: a tiny CPU model."""
from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.tests.test_lora import REVISION, _examples, _with_pairs
from anjo.sleep.tests.test_sft_dpo_default import NIGHT, Seen
from anjo.sleep.tests.test_sleep_objective import _plan, _tiny, _Tokens, _torch
from anjo.sleep.train import colab_train, sleep_job, sleep_loss
from anjo.sleep.train.lora import LoraCfg, adapter_name
from anjo.sleep.train.trainer import LoraTrainer

ROW = sleep_loss.Seq((1, 2), (3, 4, 11))
PAIR = sleep_loss.PairSeq((1, 5), (6, 11), (7, 11))


def _grads(torch: Any, chunk: list[tuple[str, Any]]) -> list[Any]:
    """The adapter's gradient after the loss's backward pass on ``chunk``, on a fresh tiny model."""
    model = _tiny(torch)
    params = [p for _, p in sleep_loss.adapter_params(model)]
    run = sleep_loss._Run(model, _plan(), params, None, None, sleep_loss.Teacher(params), sleep_loss._Clock(time.monotonic),
                          random.Random(0), set(), {term: [] for term in sleep_loss.WEIGHTS}, [])
    run.backward(chunk)
    return [p.grad.detach().clone() for p in params]


def _by_hand(torch: Any, term: str, row: Any, n: int) -> list[Any]:
    """Today's gradient for one row of a chunk of ``n``: its term's loss over ``n``, no row weight."""
    model = _tiny(torch)
    params = [p for _, p in sleep_loss.adapter_params(model)]
    loss = sleep_loss.sft_loss(model, row) if term == "sft" else sleep_loss.dpo_loss(model, row, _plan().beta)
    return [g.detach() for g in torch.autograd.grad(sleep_loss.WEIGHTS[term] * loss / n, params)]


def test_surprise_weights_scale_their_rows_gradient_and_rows_without_one_train_as_today() -> None:
    torch = _torch()
    sft, dpo = _by_hand(torch, "sft", ROW, 2), _by_hand(torch, "dpo", PAIR, 2)
    plain = _grads(torch, [("sft", ROW), ("dpo", PAIR)])
    assert all(torch.equal(g, s + d) for g, s, d in zip(plain, sft, dpo, strict=True))  # today's, exactly
    weighted = _grads(torch, [("sft", replace(ROW, weight=2.0)), ("dpo", replace(PAIR, weight=0.5))])
    assert all(torch.allclose(g, 2.0 * s + 0.5 * d, atol=1e-7) for g, s, d in zip(weighted, sft, dpo, strict=True))
    assert not all(torch.allclose(g, p) for g, p in zip(weighted, plain, strict=True))
    # Each weight scales its own row: a row at 2.0 alone gives twice the gradient of the row at 1.0.
    alone, double = _grads(torch, [("sft", ROW)]), _grads(torch, [("sft", replace(ROW, weight=2.0))])
    assert all(torch.allclose(d, 2.0 * a, atol=1e-7) for d, a in zip(double, alone, strict=True))
    # A dream carries no surprise weight: the term's own 0.5 only.
    dream = _grads(torch, [("dream", replace(ROW, weight=2.0))])
    assert all(torch.allclose(d, 0.5 * a, atol=1e-7) for d, a in zip(dream, alone, strict=True))


def test_the_script_reads_each_rows_weight_and_reports_it() -> None:
    assert colab_train.row_weight({"id": "r"}) == 1.0 and colab_train.row_weight({"id": "r", "weight": 2}) == 2.0
    for bad in (2.5, 0.4, float("nan"), True, "2"):
        with pytest.raises(ValueError, match="weight"):
            colab_train.row_weight({"id": "r", "weight": bad})
    rows: list[dict[str, Any]] = [{"id": "a", "weight": 2.0}, {"id": "b"}, {"id": "c", "weight": 9.0}]
    with pytest.raises(ValueError, match="weight"):  # every row is checked, not only up to the first weighted one
        colab_train.surprise_weights("sft_dpo_seed", rows)
    assert colab_train.surprise_weights("sft_dpo_seed", [{"id": "a"}]) is None
    assert colab_train.surprise_weights("sft_dpo_seed", rows[:2]) == "applied"
    assert colab_train.surprise_weights("sft_dpo", rows[:2]) == "not applied (objective sft_dpo)"
    row = {"id": "x", "prompt": [{"role": "user", "content": "how are you"}],
           "completion": [{"role": "assistant", "content": "fine"}], "weight": 2.0}
    pair = {"id": "y", "prompt": row["prompt"], "chosen": row["completion"],
            "rejected": [{"role": "assistant", "content": "no"}], "weight": 0.5}
    cfg = {"max_seq_length": 8192, "terms": {"fold_rank": None, "dreams": False}}
    batch, _ = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [(row, {}), ({**row, "id": "z", "weight": 1.0}, {})],
                                    [(pair, {})], {kind: [] for kind in sleep_job.KINDS}, cfg,
                                    {"reasoning_strength": "low", "current_date": NIGHT.isoformat()})
    assert [seq.weight for seq in batch.sft] == [2.0, 1.0] and batch.pairs[0].weight == 0.5


class Weighed(Seen):
    """A runner whose job reports the weights as ``colab_train``'s receipt does."""

    def __init__(self, note: str | None) -> None:
        super().__init__()
        self._note = note

    def run(self, job: str, inputs: Any, *, timeout_s: float) -> Any:
        done = super().run(job, inputs, timeout_s=timeout_s)
        return {**done, "surprise_weights": self._note} if self._note else done


def test_the_night_log_says_whether_the_weights_reached_the_loss(tmp_path: Path) -> None:
    cfg = LoraCfg(objective="sft_dpo")
    for note, data in (("not applied (objective sft_dpo)", _with_pairs(3)), (None, _examples(2))):
        trainer = LoraTrainer(Weighed(note), root=tmp_path / str(note), publish=lambda f, n: None, revision=REVISION, cfg=cfg)
        _, logged = trainer.train("p1", NIGHT, None, data, adapter_name("p1", 3, 0))
        assert logged.get("surprise_weights") == note and ("surprise_weights" in logged) == bool(note)
        assert json.dumps(logged)
