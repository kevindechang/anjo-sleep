# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The thought channel's training path on the CPU (``train.channel_train``): the smoke run's tiny
Glimmer (``train.smoke_model``, random weights, nothing downloaded) with a LoRA and a projector,
trained together through the same pieces the GPU job runs. Skipped without torch, transformers and
peft (the ``test`` extra has torch; the training stack is installed for the training jobs)."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from anjo.sleep.mind.core.projector import Projector, ProjectorError  # noqa: E402
from anjo.sleep.serving import soft_tokens  # noqa: E402
from anjo.sleep.serving.soft_tokens import SoftTokens  # noqa: E402
from anjo.sleep.train import channel_train, colab_train, smoke_model  # noqa: E402

K, D, WIDTH = 4, 8, 16
PROMPT = ("<|begin_of_text|>", "<|start|>", "user", "<|message|>", "what", "did", "i", "say", "about", "the",
          "garden", "<|eot|>", "<|start|>", "assistant")
RAISED = ("the", "tomatoes", "were", "late", "so", "plant", "sooner", "<|eot|>")
PLAIN = ("that", "is", "fine", "<|eot|>")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    folder = tmp_path_factory.mktemp("smoke") / "tiny"
    assert smoke_model.main(folder) == 0
    return AutoModelForImageTextToText.from_pretrained(folder, dtype=torch.float32), AutoTokenizer.from_pretrained(folder)


def _reading(seed: int) -> channel_train.Reading:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((K - 1, channel_train.features(D))).astype(np.float32),
            np.array([1.0, 0.6, 0.3], np.float32), rng.standard_normal(2 * D).astype(np.float32), 0.8)


def _setup(tiny: tuple[Any, Any]) -> tuple[Any, channel_train.Channel, dict[str, Any], dict[str, Any]]:
    model, tokenizer = tiny
    torch.manual_seed(0)
    lora = peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=colab_train.lora_targets(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])))
    readings = {"r/own": _reading(1), "r/other": _reading(2), "empty": channel_train.readings_of([], "state", K, D)["empty"]}
    anchors = channel_train.anchor_rows(tokenizer, lora.get_input_embeddings(), K)
    params = channel_train.feature_stats(channel_train.fresh_params(K, D, WIDTH, anchors, 0, hidden=64),
                                         [readings["r/own"], readings["r/other"]])
    ch = channel_train.Channel(lora, params, device=torch.device("cpu"), eot=tokenizer.convert_tokens_to_ids("<|eot|>"))
    ids = {name: tokenizer.convert_tokens_to_ids(list(words)) for name, words in
           (("prompt", PROMPT), ("reply", RAISED), ("plain", PLAIN))}
    item = {"id": "r", "prompt": ids["prompt"], "reply": ids["reply"], "plain": ids["plain"], "own": "r/own",
            "other": "r/other"}
    return lora, ch, item, readings


def _optimizer(lora: Any, ch: channel_train.Channel, lr: float = 1e-2) -> tuple[Any, Any, list[Any]]:
    tensors = [p for p in lora.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": tensors, "lr": lr}, *ch.projector_groups(projector_lr=lr, anchor_lr=lr)],
                                  weight_decay=0.0)
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0), tensors


def test_the_projector_and_the_lora_train_together_and_only_they_move(tiny: tuple[Any, Any]) -> None:
    # Arrange
    lora, ch, item, readings = _setup(tiny)
    base = {n: p.detach().clone() for n, p in lora.named_parameters() if not p.requires_grad}
    optimizer, schedule, tensors = _optimizer(lora, ch)
    with torch.no_grad():
        before, _ = channel_train.item_loss(ch, item, readings, beta=0.1, weight=1.0)

    # Act: one gradient, then a projector-only warmup, then joint updates
    loss, _ = channel_train.item_loss(ch, item, readings, beta=0.1, weight=1.0)
    loss.backward()
    graded = {n for n, p in lora.named_parameters() if p.grad is not None}
    assert graded and all("lora_" in n for n in graded)  # the base gets no gradient
    assert all(ch.params[n].grad is not None for n in channel_train.LAYERS + channel_train.ANCHORS)
    optimizer.zero_grad(set_to_none=True)
    adapter = [t.detach().clone() for t in tensors]
    projector = ch.arrays()
    warm = channel_train.train_pass(ch, [item], readings, optimizer, schedule, lora=tensors, epochs=2, accumulation=1,
                                    beta=0.1, weight=1.0, warmup_updates=2, deadline=float("inf"))
    held = all(torch.equal(a, t) for a, t in zip(adapter, tensors, strict=True))
    moved = any(not np.array_equal(projector[n], v) for n, v in ch.arrays().items())
    joint = channel_train.train_pass(ch, [item], readings, optimizer, schedule, lora=tensors, epochs=30,
                                     accumulation=1, beta=0.1, weight=1.0, warmup_updates=0, deadline=float("inf"))
    with torch.no_grad():
        after, _ = channel_train.item_loss(ch, item, readings, beta=0.1, weight=1.0)

    # Assert
    assert warm["steps"] == 2 and held and moved  # the warmup moved the projector, not the LoRA
    assert any(not torch.equal(a, t) for a, t in zip(adapter, tensors, strict=True))  # then the LoRA learned too
    assert joint["steps"] == 30 and float(after) < float(before) - 0.5
    assert all(torch.equal(p, base[n]) for n, p in lora.named_parameters() if n in base)  # the base never moved


def test_the_spliced_input_is_the_servers_in_position_scale_and_dtype(
    tiny: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    lora, ch, item, readings = _setup(tiny)
    soft = ch.soft(readings["r/own"]).detach()

    class Rows:  # what the server's embedding table gives, as ``CachedRows`` normalizes it
        def rows(self, ids: Sequence[int]) -> Any:
            return soft_tokens.normalize(channel_train._embedding_weight(ch.embed)[ids].detach().numpy())

    monkeypatch.setattr(soft_tokens, "HIDDEN", ch.hidden)
    monkeypatch.setattr(soft_tokens, "EOT_ID", ch.eot)
    served = soft_tokens.splice(item["prompt"], Rows(), SoftTokens(soft.numpy()))
    rows, labels = ch.splice(item["prompt"], [], soft)

    assert rows.shape == served.shape == (len(item["prompt"]) + K, ch.hidden)
    assert np.allclose(rows.detach().numpy(), served, atol=1e-5)  # same rows, soft tokens at the same place
    assert labels == [-100] * (len(item["prompt"]) + K)
    assert channel_train.check_embeds(ch, item["prompt"]) < 1e-4  # post-norm rows read as the ids are

    x, _, _, _ = ch.batch([(item["prompt"], item["reply"], soft)])
    assert x.dtype == torch.float32 == ch.dtype  # the model's own dtype
    half = channel_train.Channel(copy.deepcopy(lora).to(torch.bfloat16), ch.arrays(), device=torch.device("cpu"),
                                 eot=ch.eot)
    assert half.batch([(item["prompt"], item["reply"], soft)])[0].dtype == torch.bfloat16  # as the server sends

    with torch.no_grad():  # the head sees only the positions a label needs, and scores the same
        full = lora(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long), use_cache=False).logits
        target = torch.tensor(item["reply"])
        start = len(item["prompt"]) + K
        expected = torch.log_softmax(full[0, start - 1:-1].float(), -1).gather(-1, target[:, None]).sum()
        assert torch.allclose(ch.logps([(item["prompt"], item["reply"], soft)], grad=False)[0], expected, atol=1e-4)
    assert channel_train.check_forward(ch, [readings["r/own"], readings["r/other"]]) < 1e-5


def test_a_trained_projector_saves_and_reloads_and_serving_refuses_another_width(tiny: tuple[Any, Any]) -> None:
    _, ch, _, _ = _setup(tiny)
    blob = channel_train.save_params(ch.arrays(), {"k": K, "d": D, "width": WIDTH, "arm": "state"})

    params, meta = channel_train.load_params(blob)

    assert meta["hidden"] == ch.hidden == 64 and (meta["k"], meta["d"], meta["width"]) == (K, D, WIDTH)
    assert all(np.array_equal(params[n], v) for n, v in ch.arrays().items())
    with pytest.raises(ProjectorError):
        Projector.load(blob)  # the host serves Glimmer's width only


def test_rows_from_one_life_are_held_out_together() -> None:
    rows = [{"id": f"rd-{n}", "group": f"life-{n % 7}"} for n in range(200)]
    held = {row["group"] for row in rows if channel_train.held_out(row, 0.3)}
    assert all(channel_train.held_out(row, 0.3) == (row["group"] in held) for row in rows)
    assert 0 < len(held) < 7
