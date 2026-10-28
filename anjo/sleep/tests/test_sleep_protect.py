# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Anchor-protected LoRA (``ANJO_SLEEP_PROTECT``, ``sleep_protect.py``): a basis of the forgetting
anchors' activations at every LoRA-targeted module, tonight's slot trained as A(I - QQ^T), written
into a plain LoRA, and a fold that keeps the projection. ``random`` swaps in a random orthonormal basis
of the same shape (the ablation); ``off`` is the seed job exactly. CPU only: the smoke run's tiny
Glimmer under PEFT, and synthetic adapter tensors for the fold."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from anjo.sleep.night.dreams import Terms  # noqa: E402
from anjo.sleep.night.run import run_night_batch_with  # noqa: E402
from anjo.sleep.serving import config  # noqa: E402
from anjo.sleep.tests.night_fakes import World, step, window  # noqa: E402
from anjo.sleep.train import colab_train, objective, sleep_fold, sleep_job, sleep_loss, sleep_protect, smoke_model  # noqa: E402
from anjo.sleep.train.lora import LoraCfg, LoraCfgError, training_config  # noqa: E402

NIGHT = date(2026, 10, 29)
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BOS, START, MESSAGE, EOT = "<|begin_of_text|>", "<|start|>", "<|message|>", "<|eot|>"
SEEDED = {objective.OBJECTIVE_ENV: "sft_dpo_seed", objective.FOLD_ENV: "1"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoTokenizer  # type: ignore[import-not-found,unused-ignore]

    folder = tmp_path_factory.mktemp("smoke") / "tiny"
    assert smoke_model.main(folder) == 0
    return AutoModelForImageTextToText.from_pretrained(folder, dtype=torch.float32), AutoTokenizer.from_pretrained(folder)


def _lora(model: Any, r: int = 4) -> Any:
    torch.manual_seed(0)
    return peft.get_peft_model(copy.deepcopy(model), peft.LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=colab_train.lora_targets(MODULES)))


def _ids(tokenizer: Any, *words: str) -> tuple[int, ...]:
    return tuple(tokenizer.convert_tokens_to_ids(list(words)))


def _asked(tokenizer: Any, *words: str) -> tuple[int, ...]:
    return _ids(tokenizer, BOS, START, "user", MESSAGE, *words, EOT, START, "assistant")


def _anchors(tokenizer: Any) -> tuple[Any, ...]:
    """Two forgetting anchors in the smoke vocabulary, as the job builds them (``AnchorSeq``)."""
    return (
        sleep_loss.AnchorSeq(_asked(tokenizer, "what", "did", "the", "rain", "say"),
                             (_ids(tokenizer, "early", EOT), _ids(tokenizer, "late", EOT)), 0),
        sleep_loss.AnchorSeq(_asked(tokenizer, "we", "planted", "them", "in"),
                             (_ids(tokenizer, "may", EOT), _ids(tokenizer, "next", "year", EOT)), 0),
    )


def _night(tokenizer: Any) -> Any:
    """Tonight's rows: a garden question the anchors never ask."""
    prompt = _asked(tokenizer, "what", "did", "i", "say", "about", "the", "garden")
    reply = _ids(tokenizer, "the", "tomatoes", "were", "late", "so", "plant", "sooner", EOT)
    return sleep_loss.Batch(sft=tuple(sleep_loss.Seq(prompt, reply) for _ in range(4))), prompt


def _plan(tokenizer: Any) -> Any:
    return sleep_loss.Plan(lr=5e-2, min_steps=4, beta=0.1, seed=3, eot=tokenizer.convert_tokens_to_ids(EOT), pad=0,
                           dpo_lr=1e-2)


def _anchor_logprobs(model: Any, anchors: tuple[Any, ...]) -> Any:
    with torch.no_grad():
        return torch.cat([sleep_loss.reply_logprobs(model, a.prompt, c) for a in anchors for c in a.choices])


def _logits(model: Any, ids: tuple[int, ...]) -> Any:
    model.eval()
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0]


def _trained(tiny: tuple[Any, Any], kind: str, rank: int = 64) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """A fresh LoRA trained on tonight's rows with ``kind``'s basis on, written into plain weights."""
    model = _lora(tiny[0])
    bases, receipt = sleep_protect.bases(model, _anchors(tiny[1]), kind, rank=rank, seed=3)
    handles = sleep_protect.attach(model, bases, lead=0)
    batch, _ = _night(tiny[1])
    sleep_loss.train_terms(model, batch, _plan(tiny[1]))
    sleep_protect.materialize(model, bases, lead=0)
    for handle in handles:
        handle.remove()
    return model, bases, receipt


# ── off is today's seed job ──────────────────────────────────────────────────


def test_protect_off_is_the_seed_job() -> None:
    for extra in ({}, {objective.PROTECT_ENV: ""}, {objective.PROTECT_ENV: "off"}):
        cfg = objective.night_cfg({**SEEDED, **extra})
        assert cfg == LoraCfg(objective="sft_dpo_seed", fold_rank=config.MAX_LORA_RANK) and cfg.protect == "off"
        assert objective.night_terms({**SEEDED, **extra}) == Terms(seed=True)
        sent = training_config(cfg, 12, warm=True, revision="a" * 40, night=NIGHT)
        assert sent["terms"] == {"fold_rank": config.MAX_LORA_RANK, "dreams": False}  # the recipe as before

    class _Ct:  # the job never loads the projection's file when it is off
        SEED_OBJECTIVE = "sft_dpo_seed"

        def sibling(self, *args: Any) -> Any:
            raise AssertionError("an off night loaded sleep_protect.py")

    assert sleep_job.protection(_Ct(), {"terms": {"fold_rank": 32, "dreams": False}, "seed": 3}, None, (), 0) is None
    on = objective.night_cfg({**SEEDED, objective.PROTECT_ENV: "random"})
    assert on.protect == "random" and objective.night_terms({**SEEDED, objective.PROTECT_ENV: "random"}).protect
    sent = training_config(on, 12, warm=True, revision="a" * 40, night=NIGHT)
    assert sent["terms"] == {"fold_rank": config.MAX_LORA_RANK, "dreams": False, "protect": "random"}
    assert colab_train.load_config(json.dumps(sent))["terms"]["protect"] == "random"
    for bad in ({objective.PROTECT_ENV: "anchors"},  # not the seed objective
                {objective.OBJECTIVE_ENV: "sft_dpo_seed", objective.PROTECT_ENV: "anchors"},  # no fold: no slot
                {**SEEDED, objective.PROTECT_ENV: "yes"}):
        with pytest.raises(objective.ObjectiveError):
            objective.night_cfg(bad)
    with pytest.raises(LoraCfgError):
        LoraCfg(objective="sft_dpo_seed", protect="anchors")
    for terms in ({"fold_rank": None, "dreams": False, "protect": "anchors"},
                  {"fold_rank": 32, "dreams": False, "protect": "both"}):
        with pytest.raises(ValueError):
            colab_train.load_config(json.dumps({**sent, "terms": terms}))


def test_a_protected_night_ships_the_anchor_rows(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    deps = replace(world.deps(), terms=Terms(seed=True, protect=True))
    result = run_night_batch_with(deps, {"p1": NIGHT}, windows={"p1": window(NIGHT)})
    assert step(result["p1"].log_path, "dataset")["terms"]["anchors"] == 32  # the basis's source, dreams or not


# ── the basis, the projection, the plain LoRA, the fold ──────────────────────


def test_random_basis_matches_rank_and_dimension(tiny: tuple[Any, Any]) -> None:
    model, anchors = _lora(tiny[0]), _anchors(tiny[1])
    found, by_anchor = sleep_protect.bases(model, anchors, "anchors", rank=8, seed=3)
    drawn, by_chance = sleep_protect.bases(model, anchors, "random", rank=8, seed=3)
    again, _ = sleep_protect.bases(model, anchors, "random", rank=8, seed=3)
    other, _ = sleep_protect.bases(model, anchors, "random", rank=8, seed=4)
    modules = sleep_protect.lora_modules(model)
    assert set(found) == set(drawn) == set(modules) and len(modules) >= len(MODULES)  # every targeted module
    for name, q in found.items():
        width = modules[name].lora_A["default"].in_features
        assert q.shape == drawn[name].shape == (width, 8)  # the module's input width, rank 8
        for basis in (q, drawn[name]):
            assert torch.allclose(basis.T @ basis, torch.eye(8), atol=1e-5)  # orthonormal
        assert torch.equal(drawn[name], again[name]) and not torch.allclose(drawn[name], other[name])
    assert by_anchor["basis"] == "anchors" and by_chance["basis"] == "random"
    assert by_anchor["rank"] == by_chance["rank"] == 8 and by_anchor["modules"] == by_chance["modules"] == len(modules)
    assert by_anchor["energy"] > 2 * by_chance["energy"]  # the anchors' basis holds their activations; chance doesn't
    assert by_anchor["minutes"] >= 0.0 and by_anchor["sequences"] == by_chance["sequences"] == 4
    starts = [start for _, start in sleep_protect.sequences(anchors)]
    assert starts == [0, len(anchors[0].prompt), 4, len(anchors[1].prompt)]  # the shared preamble, each prompt, once


def test_projected_delta_is_orthogonal_to_anchor_basis(tiny: tuple[Any, Any]) -> None:
    model, tokenizer = tiny
    anchors = _anchors(tokenizer)
    _, asked = _night(tokenizer)
    start = _lora(model)
    before, asked_before = _anchor_logprobs(start, anchors), _logits(start, asked)
    trained, bases, _ = _trained(tiny, "anchors")  # rank 64 spans every anchor row here
    for name, module in sleep_protect.lora_modules(trained).items():
        delta = module.lora_B["default"].weight @ module.lora_A["default"].weight
        assert float(delta.abs().max()) > 0  # tonight learned
        assert float((delta @ bases[name]).abs().max()) <= 1e-5 * float(delta.abs().max())  # never on the anchors
    assert torch.allclose(_anchor_logprobs(trained, anchors), before, atol=1e-5)  # the anchors score as they did
    assert not torch.allclose(_logits(trained, asked), asked_before, atol=1e-3)  # tonight's question moved
    chance, _, _ = _trained(tiny, "random")  # the ablation: the same rank at random protects nothing in particular
    assert not torch.allclose(_anchor_logprobs(chance, anchors), before, atol=1e-3)


def test_the_long_term_ranks_are_never_projected(tiny: tuple[Any, Any]) -> None:
    model = _lora(tiny[0])  # r 4: two long-term ranks, then tonight's two
    bases, _ = sleep_protect.bases(model, _anchors(tiny[1]), "anchors", rank=8, seed=3)
    name, module = next(iter(sleep_protect.lora_modules(model).items()))
    lora_a, q = module.lora_A["default"], bases[name]
    x = torch.randn(3, lora_a.in_features, generator=torch.Generator().manual_seed(5))
    plain = lora_a(x)
    handles = sleep_protect.attach(model, bases, lead=2)
    hooked = lora_a(x)
    for handle in handles:
        handle.remove()
    assert torch.allclose(hooked[:, :2], plain[:, :2])  # the long-term ranks as they were
    assert torch.allclose(hooked[:, 2:], x @ (lora_a.weight[2:] - (lora_a.weight[2:] @ q) @ q.T).T, atol=1e-6)
    start = lora_a.weight.detach().clone()
    sleep_protect.materialize(model, bases, lead=2)
    assert torch.equal(lora_a.weight[:2], start[:2]) and torch.allclose(lora_a(x), hooked, atol=1e-6)


def test_protected_adapter_is_a_plain_servable_lora(tiny: tuple[Any, Any], tmp_path: Path) -> None:
    model, tokenizer = tiny
    _, asked = _night(tokenizer)
    hooked = _lora(model)
    bases, _ = sleep_protect.bases(hooked, _anchors(tokenizer), "anchors", rank=8, seed=3)
    handles = sleep_protect.attach(hooked, bases, lead=0)
    sleep_loss.train_terms(hooked, _night(tokenizer)[0], _plan(tokenizer))
    trained_as = _logits(hooked, asked)  # what the projected slot computes while it trains
    sleep_protect.materialize(hooked, bases, lead=0)
    for handle in handles:
        handle.remove()
    assert torch.allclose(_logits(hooked, asked), trained_as, atol=1e-5)  # written into A exactly
    folder = tmp_path / "p-0123456789ab-n1"
    hooked.save_pretrained(str(folder))
    (folder / "README.md").unlink(missing_ok=True)
    colab_train.record_base_model(folder)
    config.check_adapter(folder, folder.name)  # what the server and the upload accept
    meta = json.loads((folder / "adapter_config.json").read_text())
    assert (meta["r"], meta["lora_alpha"], meta.get("rank_pattern") or {}, bool(meta.get("use_dora"))) == (4, 8, {}, False)
    names = set(sleep_fold.read_safetensors(folder / "adapter_model.safetensors"))
    assert names and all(n.endswith(("lora_A.weight", "lora_B.weight")) for n in names)  # LoRA pairs, nothing else
    served = peft.PeftModel.from_pretrained(copy.deepcopy(model), str(folder))  # as the server loads it: no hooks
    assert torch.allclose(_logits(served, asked), trained_as, atol=1e-5)


PREFIX = "base_model.model.model.language_model.layers.{}.self_attn.q_proj"


def _pair(rank: int, d_in: int, d_out: int, seed: int) -> tuple[Any, Any]:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rank, d_in, generator=gen, dtype=torch.float64), torch.randn(d_out, rank, generator=gen,
                                                                                    dtype=torch.float64)


def test_fold_keeps_the_projection(tmp_path: Path) -> None:
    d_in, d_out, long_term, night, k = 40, 36, 32, 16, 6
    tensors, bases, before = {}, {}, {}
    for layer in range(2):
        name = PREFIX.format(layer)
        q, _ = torch.linalg.qr(torch.randn(d_in, k, generator=torch.Generator().manual_seed(layer), dtype=torch.float64))
        a_lt, b_lt = _pair(long_term, d_in, d_out, 10 + layer)
        a_s, b_s = _pair(night, d_in, d_out, 20 + layer)
        a_s = a_s - (a_s @ q) @ q.T  # tonight's slot, written as A(I - QQ^T)
        tensors[f"{name}.lora_A.weight"] = torch.cat([a_lt, a_s]).float()
        tensors[f"{name}.lora_B.weight"] = torch.cat([b_lt, b_s], dim=1).float()
        bases[name], before[name] = q, b_lt @ a_lt
    plain = sleep_fold.fold(tensors, 32)
    kept = sleep_fold.fold(tensors, 32, bases=bases, lead=long_term)
    for name, q in bases.items():
        a, b = kept[f"{name}.lora_A.weight"].double(), kept[f"{name}.lora_B.weight"].double()
        assert a.shape == (32, d_in) and b.shape == (d_out, 32)  # the server's rank, a plain pair
        anchor_drift = float(((b @ a - before[name]) @ q).abs().max()) / float((before[name] @ q).abs().max())
        assert anchor_drift < 1e-5  # the long-term adapter acts on the anchors as it did
        p_a, p_b = plain[f"{name}.lora_A.weight"].double(), plain[f"{name}.lora_B.weight"].double()
        assert float(((p_b @ p_a - before[name]) @ q).abs().max()) / float((before[name] @ q).abs().max()) > 1e-2
        full = tensors[f"{name}.lora_B.weight"].double() @ tensors[f"{name}.lora_A.weight"].double()
        assert torch.linalg.norm(full - b @ a) >= torch.linalg.norm(full - p_b @ p_a) - 1e-6  # the plain cut is the floor
    folder = tmp_path / "p-0123456789ab-n3"
    folder.mkdir()
    sleep_fold.write_safetensors(tensors, folder / "adapter_model.safetensors")
    (folder / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 48, "lora_alpha": 96, "base_model_name_or_path": config.MODEL,
        "target_modules": colab_train.lora_targets(("q_proj",)), "rank_pattern": {}, "alpha_pattern": {},
        "use_dora": False, "task_type": "CAUSAL_LM"}))
    assert sleep_fold.fold_folder(folder, 32, bases=bases, lead=long_term) == 32
    config.check_adapter(folder, folder.name)
    on_disk = sleep_fold.read_safetensors(folder / "adapter_model.safetensors")
    for name, q in bases.items():
        product = on_disk[f"{name}.lora_B.weight"].double() @ on_disk[f"{name}.lora_A.weight"].double()
        assert float(((product - before[name]) @ q).abs().max()) < 1e-4 * float((before[name] @ q).abs().max())


def test_the_projection_file_is_standalone() -> None:
    import ast

    tree = ast.parse(Path(sleep_protect.__file__).read_text(encoding="utf-8"))
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])}
    assert not top & {"torch", "transformers", "peft", "trl", "anjo", "safetensors"}
