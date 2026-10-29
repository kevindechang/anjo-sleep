"""``sft_dpo_seed``'s training job (2026-10-29): the one loss (``sleep_loss.py``) over every term in one
loop, then the fold (``sleep_fold.py``), and, with a recipe ``channel`` block, the thought channel's
projector in the same loop (``channel_train.py``): one loss, one budget, one job.

``colab_train.main`` hands a seeded recipe here (``main(ct, cfg)``, ``ct`` the training script's module,
as ``channel_train.main`` takes it). Standalone like the script: the standard library at the top, the
training stack inside the functions, its siblings loaded by path (``ct.sibling``). Modal's image carries
it; the Colab runner refuses the objective.

**With the channel** each row that carries ``core`` (the state its reply was made under) reads that
state as K soft tokens, spliced as the server splices them, and the projector trains with the LoRA:
in the main block's one optimizer (its groups at their own rates, the first ``projector_warmup`` share
of the updates the projector's alone) and in the DPO block (its MLP; the reference is the adapter off
and the projector as the night began). Which terms teach the projector:

- ``sft`` and ``dpo``: yes. Reply-given-state on her real replies: the channel run's own terms.
- ``seed`` and ``imitate``: no. The student reads its row's state through tonight's projector, detached,
  with no day in context; the teacher reads the same state through the projector as the night began,
  with the day in context. Both see the state, so the KL moves into the LoRA only what the day adds
  beyond it, and the projector is never pulled to encode the day's facts from a state that doesn't
  hold them (it would learn tonight's facts as a map of tonight's state, and misread tomorrow's).
- ``dream``: no. A dream's state is imagined: it trains as it was dreamt (``core`` when the dream source
  snapshot the imagined state, and the thread it was dreamt under as its system turn), its soft tokens
  detached, so an imagined state never shapes how the projector reads real ones.

**Protected** (the recipe's ``terms.protect``, ``anchors`` or ``random``, with the fold): before the loop the
anchors' activation basis is read at every LoRA module with the night-start adapter on, tonight's slot
trains as ``A (I - Q Q^T)`` (``sleep_protect.py``), the projection is written into the weights before the
save, and the fold keeps it. Its receipt is ``terms.protect``: the basis, its rank, the minutes it took.

Contrast rows (the shared pretraining's) train in the channel's own run (``channel_train.main``); a seeded
recipe refuses them. The channel's ``max_minutes`` stops the loop as it stops the channel run.
"""
from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

KINDS: Final = ("seed", "dream", "anchor")
FOLD_FILE: Final = "sleep_fold.py"
PROTECT_FILE: Final = "sleep_protect.py"
Kept = list[tuple[dict[str, Any], dict[str, str]]]


def term_rows(ct: Any, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The rows by kind, each checked as the host cleaned it."""
    found: dict[str, list[dict[str, Any]]] = {kind: [] for kind in KINDS}
    for row in rows:
        if "kind" not in row:
            continue
        if row["kind"] not in KINDS:
            raise ValueError(f"row {row.get('id')}'s kind is one of {', '.join(KINDS)}")
        if row["kind"] == "anchor":
            ct.check_clean({"id": row.get("id"), "prompt": row["prompt"], "chosen": [], "rejected": []})
            choices, answer = row.get("choices"), row.get("answer")
            if (not isinstance(choices, list) or not all(isinstance(c, str) and not ct.SPECIAL_TOKEN.search(c) for c in choices)
                    or isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices)):
                raise ValueError(f"anchor {row.get('id')} has clean choices and an answer among them")
        else:
            ct.check_clean(row)
            if row["kind"] == "seed":
                ct.check_clean({**row, "prompt": row.get("teacher")})
        found[row["kind"]].append(row)
    return found


def state_key(row: Mapping[str, Any]) -> str | None:
    """The key of the row's reading (``channel_train.readings_of``'s), by kind and id, since a seeding
    row shares its example's id; None for a row with no ``core``."""
    if not row.get("core"):
        return None
    kind = row.get("kind") or ("pair" if "chosen" in row else "sft")
    return f"{kind}:{row['id']}/own"


def readings(cm: Any, rows: Sequence[Mapping[str, Any]], ch: Mapping[str, Any]) -> dict[str, Any]:
    """Every row's reading by ``state_key`` (all empty on the ``zero`` arm), and the empty one."""
    keyed = [{**row, "id": key.removesuffix("/own")} for row in rows if (key := state_key(row))]
    return dict(cm.readings_of(keyed, ch["arm"], ch["k"], ch["d"]))


def seed_batch(ct: Any, loss: Any, tokenizer: Any, kept: Kept, pairs: Kept, terms: dict[str, list[dict[str, Any]]],
               cfg: dict[str, Any], template: dict[str, str]) -> tuple[Any, dict[str, int]]:
    """The loss's batch, every row as the server renders it and naming its reading, and the term rows
    too long to train (with the channel, its K soft tokens count against the length)."""
    def prompt(row: dict[str, Any], key: str = "prompt") -> tuple[int, ...]:
        return tuple(ct.served_ids(tokenizer, row[key], ct.row_template(template, row)))

    def reply(row: dict[str, Any], side: str = "completion") -> tuple[int, ...]:
        return tuple(ct.reply_ids(tokenizer, ct._reply(row[side])))

    limit = cfg["max_seq_length"] - (cfg["channel"]["k"] if cfg.get("channel") else 0)
    seeds = [loss.SeedSeq(prompt(row), prompt(row, "teacher"), reply(row), state_key(row)) for row in terms["seed"]]
    dreams = ([(loss.Seq(prompt(row), reply(row), state=state_key(row)), str(row.get("seed_kind", "")))
               for row in terms["dream"]] if cfg["terms"]["dreams"] else [])  # dream rows train only when asked for
    fit = [(seq, kind) for seq, kind in dreams if len(seq.prompt) + len(seq.reply) <= limit]
    batch = loss.Batch(
        sft=tuple(loss.Seq(prompt(row), reply(row), ct.row_weight(row), state_key(row)) for row, _ in kept),
        pairs=tuple(loss.PairSeq(prompt(row), reply(row, "chosen"), reply(row, "rejected"), ct.row_weight(row),
                                 state_key(row)) for row, _ in pairs),
        seeds=tuple(row for row in seeds if len(row.teacher) + len(row.reply) <= limit),
        dreams=tuple(seq for seq, _ in fit), dream_groups=tuple(kind for _, kind in fit),
        anchors=tuple(loss.AnchorSeq(prompt(row), tuple(tuple(ct.reply_ids(tokenizer, c, end=False)) for c in row["choices"]),
                                     row["answer"]) for row in terms["anchor"]),
    )
    return batch, {"seed": len(seeds) - len(batch.seeds), "dream": len(dreams) - len(batch.dreams)}


def _seed_model(model: Any, settings: dict[str, Any], prev: str | None, fold: int | None, loss: Any,
                folding: Any) -> tuple[Any, int]:  # pragma: no cover - needs the training stack
    """The PEFT model the job trains, and how many leading ranks stay frozen. Folding, the long-term
    adapter fills the first ranks of a LoRA of its rank plus the night's, and the rest of B starts at
    zero (the reset); otherwise a warm start loads the previous adapter as today's does."""
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found, unused-ignore]

    if fold is None and prev:
        return PeftModel.from_pretrained(model, prev, is_trainable=True), 0
    if fold is None or not prev:
        return get_peft_model(model, LoraConfig(**settings["lora"])), 0
    meta = json.loads((Path(prev) / "adapter_config.json").read_text(encoding="utf-8"))
    r_prev = meta.get("r")
    if not isinstance(r_prev, int) or meta.get("lora_alpha") != 2 * r_prev or meta.get("rank_pattern"):
        raise ValueError("the long-term adapter is a plain LoRA with alpha 2r")
    width = r_prev + settings["lora"]["r"]
    peft = get_peft_model(model, LoraConfig(**{**settings["lora"], "r": width, "lora_alpha": 2 * width}))
    loss.load_leading(peft, folding.read_safetensors(Path(prev) / "adapter_model.safetensors"), r_prev)
    return peft, r_prev


def protection(ct: Any, cfg: Mapping[str, Any], model: Any, anchors: Sequence[Any],
               lead: int) -> tuple[Any, dict[str, Any], dict[str, Any], list[Any]] | None:
    """With ``terms.protect``: the projection module, each LoRA module's basis, its receipt, and the hooks
    that keep tonight's slot (the ranks past ``lead``) off it. None, loading nothing, when it is off."""
    kind = cfg["terms"].get("protect")
    if not kind:
        return None
    protect = ct.sibling(PROTECT_FILE, "sleep_protect", ct.SEED_OBJECTIVE)
    found, receipt = protect.bases(model, anchors, kind, seed=cfg["seed"])
    return protect, found, receipt, protect.attach(model, found, lead)


def _check_seed_tokens(ct: Any, tokenizer: Any, batch: Any, kept: Kept, pairs: Kept,
                       by_kind: dict[str, list[dict[str, Any]]], template: dict[str, str]) -> int:  # pragma: no cover
    """Each trained row's ids against the template's own render of the whole conversation, as the SFT
    and DPO passes' rows are checked: the teacher's form of a seeding row too."""
    done = 0
    for (row, _), seq in zip(kept, batch.sft, strict=True):
        ct._check_tokens(tokenizer, row, "completion", seq.prompt, seq.reply, template)
        done += 1
    for (row, _), pair in zip(pairs, batch.pairs, strict=True):
        ct._check_tokens(tokenizer, row, "chosen", pair.prompt, pair.chosen, template)
        ct._check_tokens(tokenizer, row, "rejected", pair.prompt, pair.rejected, template)
        done += 1
    for row in [*by_kind["seed"], *by_kind["dream"]]:
        forms = [row, {**row, "prompt": row["teacher"]}] if row["kind"] == "seed" else [row]
        for form in forms:
            ct._check_tokens(tokenizer, form, "completion",
                             ct.served_ids(tokenizer, form["prompt"], ct.row_template(template, form)),
                             ct.reply_ids(tokenizer, ct._reply(form["completion"])), template)
        done += 1
    return done


def _channel(ct: Any, cfg: dict[str, Any], model: Any, tokenizer: Any, prev: str | None, rows: list[dict[str, Any]],
             loss: Any) -> tuple[Any, Any, dict[str, Any]]:  # pragma: no cover - needs the training stack
    """The projector on the PEFT model as ``loss.Soft``, the channel module, and the channel's receipt so far."""
    cm, ch = ct.channel_module(), cfg["channel"]
    found = readings(cm, rows, ch)
    run, opened = cm.open_channel(model, tokenizer, ch, seed=cfg["seed"], prev=prev, readings=found,
                                  device=next(p for p in model.parameters() if p.requires_grad).device)
    soft = loss.Soft(run, found, tuple(run.projector_groups(projector_lr=ch["projector_lr"], anchor_lr=ch["anchor_lr"])),
                     tuple(run.params[name] for name in cm.LAYERS), ch["projector_warmup"])
    return soft, cm, {**opened, "rows_with_state": len(found) - 1}


def seed_plan(loss: Any, cfg: Mapping[str, Any], *, eot: int, pad: int, long_term_rank: int, begun: float) -> Any:
    """The one loss's ``Plan`` from the recipe: its rates and step rule, capacity's ``passes`` (0: the step
    plan), the matched control's budget and the channel's deadline."""
    ch = cfg.get("channel")
    return loss.Plan(lr=cfg["learning_rate"], min_steps=cfg["min_optimizer_steps"], beta=cfg["dpo_beta"],
                     seed=cfg["seed"], eot=eot, pad=pad, dpo_lr=cfg["dpo_learning_rate"], long_term_rank=long_term_rank,
                     dreams=cfg["terms"]["dreams"], stretch_s=cfg.get("match_minutes", 0.0) * 60.0,
                     deadline=begun + 60.0 * ch["max_minutes"] if ch else float("inf"),
                     passes=cfg["terms"].get("passes", 0))


def train(ct: Any, cfg: dict[str, Any], settings: dict[str, Any], tokenizer: Any, template: dict[str, str], kept: Kept,
          pairs: Kept, rows: list[dict[str, Any]], model: Any, prev: str | None, out: Path,
          begun: float) -> dict[str, Any]:  # pragma: no cover - needs the training stack
    """The one weighted loss over every term (the projector's too, channel on) in one loop, then the fold
    and the projector saved beside the adapter. Returns the receipt's own part."""
    loss, folding = ct.load_loss(), ct.sibling(FOLD_FILE, "sleep_fold", ct.SEED_OBJECTIVE)
    fold, ch = cfg["terms"]["fold_rank"], cfg.get("channel")
    by_kind = term_rows(ct, rows)
    batch, too_long = seed_batch(ct, loss, tokenizer, kept, pairs, by_kind, cfg, template)
    checked = _check_seed_tokens(ct, tokenizer, batch, kept, pairs, by_kind, template)
    adapter = prev if prev and (Path(prev) / "adapter_config.json").is_file() else None  # a channel prev may hold a projector only
    model, frozen = _seed_model(model, settings, adapter, fold, loss, folding)
    channel: tuple[Any, Any, dict[str, Any]] = _channel(ct, cfg, model, tokenizer, prev, rows, loss) if ch else (None, None, {})
    soft, cm, opened = channel
    guarded = protection(ct, cfg, model, batch.anchors, frozen)
    eot = tokenizer.convert_tokens_to_ids(ct.REPLY_END)
    plan = seed_plan(loss, cfg, eot=eot, pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eot,
                     long_term_rank=frozen, begun=begun)
    try:
        terms = loss.train_terms(model, batch, plan, soft=soft)
    finally:
        for hook in guarded[3] if guarded else ():
            hook.remove()
    keep: tuple[Any, ...] = ()  # the protected cut's bases and lead, protect on
    if guarded:
        protect, found, receipt, _ = guarded
        terms["protect"] = {**receipt, "slot_left_on_basis": protect.materialize(model, found, frozen)}
        keep = ({name: q.detach().float().cpu() for name, q in found.items()}, frozen)
    model.save_pretrained(str(out))
    terms |= {"dropped_truncated": too_long, "fold_rank": fold,
              "folded_to": folding.fold_folder(out, fold, *keep) if fold is not None else None}
    done = {"terms": terms, "steps": terms["steps"], "pairs": len(batch.pairs), "rows_checked": checked}
    if soft is not None:
        done["channel"] = {**cm.save_channel(soft.channel, out, ch, opened), "rows_with_state": opened["rows_with_state"],
                           "projector_only_updates": terms["projector_only_updates"]}
    return done


def main(ct: Any, cfg: dict[str, Any]) -> int:  # pragma: no cover - runs only on the GPU VM and in the CPU smoke run
    """The seeded run: ``ct`` is the training script's module (``colab_train``), ``cfg`` its recipe."""
    import numpy  # type: ignore[import-not-found, unused-ignore]
    import torch  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoTokenizer, set_seed  # type: ignore[import-not-found, unused-ignore]

    begun = time.monotonic()
    smoke, base = ct._smoke_model(), ct.base_model(os.environ)
    settings = ct.build_settings(cfg) if smoke is None else ct.smoke_settings(ct.build_settings(cfg))
    out, prev = Path(os.environ["ANJO_TRAIN_OUT"]), os.environ.get("ANJO_TRAIN_PREV") or None
    random.seed(cfg["seed"])
    numpy.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    set_seed(cfg["seed"])
    local = smoke or base
    tokenizer = AutoTokenizer.from_pretrained(local or ct.MODEL, revision=None if local else cfg["revision"])
    tokenizer.eos_token = ct.REPLY_END  # Glimmer's turn ends with <|eot|>, and each reply is closed by it
    ct.check_template(tokenizer, cfg)
    template = ct.template_kwargs(cfg)
    every = ct._rows(Path(os.environ["ANJO_TRAIN_DATASET"]))
    if cfg.get("channel") is not None and any(row.get("contrast") for row in every):
        raise ValueError(f"contrast rows train in the channel's own run, never with {ct.SEED_OBJECTIVE}")
    rows, pair_rows = ct.split_rows(every)
    surprise = ct.surprise_weights(cfg["objective"], [*rows, *pair_rows])  # checked before the model loads
    kept = ct._rendered(tokenizer, rows, cfg["max_seq_length"], template)
    ct.check_every_row_kept(cfg, len(rows), len(kept))  # before the model loads: only the boot is spent
    pairs = ct._rendered(tokenizer, pair_rows, cfg["max_seq_length"], template)
    model = ct._load_model(cfg, settings, smoke, base)
    if cfg.get("channel") is not None:  # the spliced rows go in as embeddings: checkpointing that needs no input grad
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    started = time.monotonic()
    done = train(ct, cfg, settings, tokenizer, template, kept, pairs, every, model, prev, out, begun)
    minutes = (time.monotonic() - started) / 60.0
    ct.record_rendering(out, cfg["rendering"])
    if base is not None:
        ct.record_base_model(out)
    receipt = {
        **done, "epochs": done["terms"]["epochs"], "train_minutes": minutes, "examples": len(kept),
        "dropped_truncated": len(rows) - len(kept), "gpu": torch.cuda.get_device_name(0) if smoke is None else "cpu",
        "objective": cfg["objective"], "dpo_steps": 0, "pairs_dropped_truncated": len(pair_rows) - len(pairs),
        "dpo_reference": "adapter_off+projector_start" if "channel" in done else "adapter_off",
        "rendering": cfg["rendering"], **({"surprise_weights": surprise} if surprise else {}),
        **({"match": done["terms"]["stretch"]} if "stretch" in done["terms"] else {}),
    }
    (out / ct.RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0
