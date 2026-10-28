"""The thought channel's half of the night's training job: the person's LoRA and the projector that
lets Glimmer read MIND's state, trained together in one run (one system: ``brief_one_system``).

**What it trains.** A night's rows (``train.dataset``) may carry the core state each was made under
(``core``, a ``night.core_snapshot.CoreSnapshot``: the projector's input, never the raw state). With
a recipe ``channel`` block, the training script (``colab_train.main``) hands the run to ``main`` here
instead of TRL: each row's prompt goes to Glimmer as input embeddings, with K soft tokens, the
projector's output for the row's reading, spliced in just before the ``<|eot|>`` that closes the
latest user turn, exactly as the server splices them (``serving.soft_tokens.splice``). The loss is
on the reply and ``<|eot|>`` only, as in the SFT pass; gradients reach the LoRA through the frozen
4-bit base and the projector through the soft tokens. Then, for ``sft_dpo``, a DPO pass on the pair
rows, with soft tokens on both sides; its reference is the model with the adapter switched off and
the projector as the run began (as TRL's ``adapter_off`` reference, plus the channel).

**Contrast rows** (the shared pretraining across generated lives, ``mind.core.reader_data``): a row
with ``contrast`` also carries another moment's state (``contrast.core``) and the reply that state
earns (``contrast.plain``). The run trains reply-given-state on both, and adds a state-conditional
preference: with the row's own state, the raise reply should gain on the plain one more than with
the other state (``-log sigmoid(beta * margin)``). The message is the same on all four sequences, so
only the soft tokens can move the margin: a margin near zero means the channel carries nothing.
``val_share`` of the contrast rows (by id) are held out and scored after training, with their own
state and with the empty reading (the zero-tokens arm), and the receipt reports both.

**The projector** (``forward``): one token from the core's global vector and speak probability, then
one per routed slot, in the core's own choice order (``mind.core.projector`` reads the state). Each
slot's features go through a two-layer MLP to Glimmer's width and are added, scaled by the core's
choice gate, to that rank's anchor row; the global token mixes a "speak" and a "silence" anchor by
the speak probability. Every token is RMS-normalized, as a token's own row is after Glimmer's
weightless embedding norm. ``forward`` uses only ``@``, arithmetic and the ``tanh`` and ``sqrt``
it is given, so numpy (the host, serving) and torch (here) run the same code. A fresh projector's
anchors are the rows of ``ANCHOR_WORDS``, so Glimmer reads them as words before any training.

It imports the standard library only; numpy, torch, transformers and peft load inside the functions
that need them. On Modal the image carries it beside ``colab_train.py``, which loads it by path.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

HIDDEN: Final = 6656  # Glimmer's text hidden size (serving.soft_tokens.HIDDEN; a test holds them equal)
RMS_EPS: Final = 1e-5  # Glimmer's embedding norm's epsilon (serving.soft_tokens.RMS_EPS)
EOT: Final = "<|eot|>"
PROJECTOR_FILE: Final = "projector.npz"
PROJECTOR_VERSION: Final = 1
SCALARS: Final = ("usage", "risk", "missing", "self_error", "choice", "age", "cited")
MAX_K: Final = 12
GATE_FLOOR: Final = 0.25  # a routed slot's gate never falls below this, so a low-ranked thread still speaks
INIT_OUT: Final = 0.4  # a fresh output layer's gain: soft tokens start near their anchors, already state-dependent
ANCHOR_WORDS: Final = (" concern", " nothing", " first", " second", " third", " fourth", " fifth", " sixth",
                       " seventh", " eighth", " ninth", " tenth", " eleventh")  # speak, silence, then each rank
PARAMS: Final = ("w1", "b1", "u1", "c1", "w2", "b2", "anchors", "speak", "silence",
                 "x_mean", "x_std", "g_mean", "g_std")
_GELU: Final = math.sqrt(2.0 / math.pi)
CHANNEL_KEYS: Final = ("k", "d", "width", "arm", "projector_lr", "anchor_lr", "contrast_beta",
                       "contrast_weight", "val_share", "max_minutes")
ARMS: Final = ("state", "zero")


def features(d: int) -> int:
    """A routed slot's feature count: its vector, the fast memory's answer to it, the global vector,
    and ``SCALARS``."""
    return 3 * d + len(SCALARS)


def check_channel(block: object) -> dict[str, Any]:
    """A recipe's ``channel`` block, checked: K tokens (2 to ``MAX_K``), the core's width, the
    projector's width, the arm (``state``, or ``zero``: the ablation, every reading empty), the
    projector's and anchors' learning rates, the contrast's beta and weight, the held-out share and
    the minutes the loop may train."""
    if not isinstance(block, dict) or sorted(block) != sorted(CHANNEL_KEYS):
        raise ValueError(f"the channel block holds exactly {', '.join(CHANNEL_KEYS)}")
    for key, low, high in (("k", 2, MAX_K), ("d", 1, 4096), ("width", 8, 4096)):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"channel {key} is an integer from {low} to {high}")
    if block["arm"] not in ARMS:
        raise ValueError(f"channel arm is one of {', '.join(ARMS)}")
    for key, top in (("projector_lr", 1e-2), ("anchor_lr", 1e-2), ("contrast_beta", 10.0),
                     ("contrast_weight", 10.0), ("max_minutes", 600.0)):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= top:
            raise ValueError(f"channel {key} is a number from 0 to {top:g}")
    if not 0 <= block["val_share"] < 1:
        raise ValueError("channel val_share is in [0, 1)")
    return dict(block)


# ── The projector, the same code on numpy and torch ─────────────────────────


def _gelu(x: Any, tanh: Callable[[Any], Any]) -> Any:
    return 0.5 * x * (1.0 + tanh(_GELU * (x + 0.044715 * x * x * x)))


def rms(x: Any, ones: Any, sqrt: Callable[[Any], Any]) -> Any:
    """Glimmer's weightless RMSNorm over the last axis; ``ones`` is a ``(H, 1)`` column of ones."""
    return x / sqrt(((x * x) @ ones) / ones.shape[0] + RMS_EPS)


def forward(
    p: Mapping[str, Any], slot_x: Any, gates: Any, global_x: Any, speak: Any, *,
    tanh: Callable[[Any], Any], sqrt: Callable[[Any], Any], ones: Any,
) -> tuple[Any, Any]:
    """The speak token ``(B, H)`` and the slot tokens ``(B, K-1, H)`` for readings stacked in a batch:
    ``slot_x (B, K-1, F)``, ``gates (B, K-1)``, ``global_x (B, 2d)``, ``speak (B,)``."""
    hidden = _gelu(((slot_x - p["x_mean"]) / p["x_std"]) @ p["w1"] + p["b1"], tanh)
    slots = rms(p["anchors"] + gates[..., None] * (hidden @ p["w2"] + p["b2"]), ones, sqrt)
    whole = _gelu(((global_x - p["g_mean"]) / p["g_std"]) @ p["u1"] + p["c1"], tanh) @ p["w2"] + p["b2"]
    s = speak[..., None]
    return rms(s * p["speak"] + (1.0 - s) * p["silence"] + whole, ones, sqrt), slots


def fresh_params(k: int, d: int, width: int, anchor_rows: Any, seed: int) -> dict[str, Any]:
    """A new projector (numpy float32): ``anchor_rows`` are ``k + 1`` post-norm rows (speak,
    silence, then each rank's), the first layers scaled for unit features, the output layer small."""
    import numpy as np

    rows = np.asarray(anchor_rows, dtype=np.float32)
    if rows.shape != (k + 1, HIDDEN):
        raise ValueError(f"a projector of K = {k} takes {k + 1} anchor rows of width {HIDDEN}")
    rng = np.random.default_rng(seed)
    f, g = features(d), 2 * d

    def normal(shape: tuple[int, ...], std: float) -> Any:
        return (rng.standard_normal(shape) * std).astype(np.float32)

    return {
        "w1": normal((f, width), 1 / math.sqrt(f)), "b1": np.zeros(width, np.float32),
        "u1": normal((g, width), 1 / math.sqrt(g)), "c1": np.zeros(width, np.float32),
        "w2": normal((width, HIDDEN), INIT_OUT / math.sqrt(width)), "b2": np.zeros(HIDDEN, np.float32),
        "anchors": rows[2:].copy(), "speak": rows[0].copy(), "silence": rows[1].copy(),
        "x_mean": np.zeros(f, np.float32), "x_std": np.ones(f, np.float32),
        "g_mean": np.zeros(g, np.float32), "g_std": np.ones(g, np.float32),
    }


def save_params(params: Mapping[str, Any], meta: Mapping[str, Any]) -> bytes:
    """The projector as ``.npz`` bytes (float32 arrays and a JSON ``meta``; no pickle)."""
    import io

    import numpy as np

    if sorted(params) != sorted(PARAMS):
        raise ValueError(f"a projector holds exactly {', '.join(PARAMS)}")
    out = io.BytesIO()
    arrays = {name: np.asarray(params[name], dtype=np.float32) for name in PARAMS}
    np.savez(out, meta=np.array(json.dumps({**meta, "version": PROJECTOR_VERSION}, sort_keys=True)), **arrays)
    return out.getvalue()


def load_params(blob: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """``save_params``' bytes back: the arrays and the meta; ``ValueError`` for anything else."""
    import io
    import zipfile

    import numpy as np

    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as held:
            meta = json.loads(str(held["meta"]))
            params = {name: np.array(held[name], dtype=np.float32) for name in PARAMS}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"not a saved projector: {type(error).__name__}") from None
    if meta.get("version") != PROJECTOR_VERSION:
        raise ValueError(f"projector version {meta.get('version')!r}; this build reads {PROJECTOR_VERSION}")
    return params, meta


def params_sha(params: Mapping[str, Any]) -> str:
    """A projector's id: a hash of its arrays."""
    import numpy as np

    digest = hashlib.sha256()
    for name in PARAMS:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(params[name], dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def decode_reading(obj: Mapping[str, Any]) -> tuple[Any, Any, Any, float]:
    """A row's ``core.reading`` (``mind.core.projector.Reading.to_json``) as numpy arrays:
    ``slot_x (K-1, F)``, ``gates (K-1,)``, ``global_x (2d,)`` and the speak probability."""
    import numpy as np

    k, d = int(obj["k"]), int(obj["d"])

    def arr(key: str, shape: tuple[int, ...]) -> Any:
        found: Any = np.frombuffer(base64.b64decode(obj[key]), dtype="<f4")
        if found.size != math.prod(shape) or not np.isfinite(found).all():
            raise ValueError(f"a reading's {key} isn't {shape} finite floats")
        return found.reshape(shape).astype(np.float32)

    return (arr("slot_x", (k - 1, features(d))), arr("gates", (k - 1,)), arr("global_x", (2 * d,)),
            float(obj["speak"]))


# ── The run (GPU VM only) ─────────────────────────────────────────────────────


def _row_reading(row: Mapping[str, Any], key: str = "core") -> Mapping[str, Any] | None:
    core = row.get(key)
    return None if core is None else core["reading"]


def _encode(ct: Any, tokenizer: Any, row: dict[str, Any], side: str, template: dict[str, str],
            reply: str | None = None) -> tuple[list[int], list[int]]:  # pragma: no cover - needs the training stack
    """A row's prompt ids (as the server renders them, BOS first, the reply's header last) and one
    side's reply ids (closed by ``<|eot|>``), checked against the chat template's own render."""
    shaped = {**row, side: [{"role": "assistant", "content": reply}]} if reply is not None else row
    plain = ct.plain_row(tokenizer, {k: v for k, v in shaped.items() if k in ("id", "prompt", side, "current_date")},
                         template)
    prompt_ids = list(tokenizer(text=plain["prompt"])["input_ids"])
    reply_ids = [*tokenizer(ct._reply(shaped[side]), add_special_tokens=False)["input_ids"],
                 tokenizer.convert_tokens_to_ids(EOT)]
    ct._check_tokens(tokenizer, shaped, side, prompt_ids, reply_ids, template)
    return prompt_ids, reply_ids


class _Run:  # pragma: no cover - runs only on the GPU VM
    """One run's model, projector and the sequences it scores."""

    def __init__(self, model: Any, params: dict[str, Any], ref_params: dict[str, Any], k: int, device: Any) -> None:
        import torch  # type: ignore[import-not-found, unused-ignore]

        self.torch, self.model, self.k, self.device = torch, model, k, device
        self.embed = model.get_input_embeddings()
        self.eot = -1  # <|eot|>'s id, set by main
        self.params = {n: torch.tensor(v, device=device, requires_grad=n not in ("x_mean", "x_std", "g_mean", "g_std"))
                       for n, v in params.items()}
        self.ref = {n: torch.tensor(v, device=device) for n, v in ref_params.items()}
        self.ones = torch.ones(HIDDEN, 1, device=device)

    def soft(self, reading: tuple[Any, Any, Any, float], ref: bool = False) -> Any:
        torch = self.torch
        slot_x, gates, global_x = (torch.as_tensor(a, device=self.device)[None] for a in reading[:3])
        speak = torch.tensor([reading[3]], device=self.device)
        p = self.ref if ref else self.params
        first, rest = forward(p, slot_x, gates, global_x, speak, tanh=torch.tanh, sqrt=torch.sqrt, ones=self.ones)
        return torch.cat([first[:, None], rest], dim=1)[0]

    def logps(self, seqs: Sequence[tuple[list[int], list[int], Any]], grad: bool = True) -> Any:
        """Each sequence's summed log-probability of its reply, soft tokens spliced before the
        latest user turn's ``<|eot|>``."""
        torch = self.torch
        xs, labels = [], []
        for prompt, reply, soft in seqs:
            rows = rms(self.embed(torch.tensor(prompt + reply, device=self.device)).float(), self.ones, torch.sqrt)
            cut = len(prompt) - 1 - prompt[::-1].index(self.eot)
            parts = [rows[:cut], *([soft.float()] if soft is not None else []), rows[cut:]]
            xs.append(torch.cat(parts))
            labels.append([-100] * (len(prompt) + (0 if soft is None else soft.shape[0])) + reply)
        width = max(len(lab) for lab in labels)
        x = torch.zeros(len(xs), width, HIDDEN, device=self.device)
        mask = torch.zeros(len(xs), width, dtype=torch.long, device=self.device)
        lab = torch.full((len(xs), width), -100, dtype=torch.long, device=self.device)
        for i, (seq, row) in enumerate(zip(xs, labels, strict=True)):
            x[i, :len(row)], mask[i, :len(row)] = seq, 1
            lab[i, :len(row)] = torch.tensor(row, device=self.device)
        x = x.to(torch.bfloat16)
        if grad and not x.requires_grad and torch.is_grad_enabled():
            x.requires_grad_(True)
        logits = self.model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits[:, :-1].float()
        target = lab[:, 1:]
        picked = torch.log_softmax(logits, dim=-1).gather(-1, target.clamp(min=0)[..., None])[..., 0]
        return (picked * (target != -100)).sum(-1)


def _anchor_rows(tokenizer: Any, embed: Any, k: int) -> Any:  # pragma: no cover - needs the training stack
    """Each anchor word's rows, averaged, then normalized: speak, silence, then each rank's."""
    import numpy as np
    import torch  # type: ignore[import-not-found, unused-ignore]

    rows = []
    with torch.no_grad():
        for word in ANCHOR_WORDS[: k + 1]:
            ids = torch.tensor(tokenizer(word, add_special_tokens=False)["input_ids"], device=embed.weight.device)
            rows.append(embed(ids).float().mean(0).cpu().numpy())
    raw = np.stack(rows)
    return (raw / np.sqrt((raw * raw).mean(-1, keepdims=True) + RMS_EPS)).astype(np.float32)


def _feature_stats(params: dict[str, Any], readings: Sequence[tuple[Any, Any, Any, float]]) -> None:
    """A fresh projector's feature scaling, from the run's own readings (routed slots only)."""
    import numpy as np

    slots = [r[0][r[1] > 0] for r in readings]
    stacked = np.concatenate([s for s in slots if len(s)] or [np.zeros((1, params["x_mean"].size), np.float32)])
    whole = np.stack([r[2] for r in readings]) if readings else np.zeros((1, params["g_mean"].size), np.float32)
    params["x_mean"], params["x_std"] = stacked.mean(0), stacked.std(0) + 1e-3
    params["g_mean"], params["g_std"] = whole.mean(0), whole.std(0) + 1e-3


def _check_embeds(run: _Run, tokenizer: Any) -> float:  # pragma: no cover - GPU VM only
    """Glimmer on token ids and on their normalized rows as ``inputs_embeds`` must agree: the
    channel feeds post-norm rows, as vLLM's prompt embeddings do. Returns the largest gap."""
    torch = run.torch
    ids = torch.tensor([tokenizer("The camera arrives on Friday.")["input_ids"]], device=run.device)
    with torch.no_grad():
        by_ids = run.model(input_ids=ids, use_cache=False).logits.float()
        rows = rms(run.embed(ids).float(), run.ones, torch.sqrt).to(torch.bfloat16)
        by_rows = run.model(inputs_embeds=rows, use_cache=False).logits.float()
    gap = float((by_ids - by_rows).abs().max())
    if gap > 0.5:
        raise ValueError(f"Glimmer reads normalized rows differently from its ids (gap {gap:.3f}); the channel "
                         "would train on embeddings the server never sends")
    return gap


def _check_forward(run: _Run, readings: Sequence[tuple[Any, Any, Any, float]]) -> float:  # pragma: no cover
    """The projector in torch here against the same code on numpy, as the host serves it."""
    import numpy as np

    p = {n: v.detach().cpu().numpy() for n, v in run.params.items()}
    worst = 0.0
    for reading in readings[:4]:
        first, rest = forward(p, reading[0][None], reading[1][None], reading[2][None], np.array([reading[3]], np.float32),
                              tanh=np.tanh, sqrt=np.sqrt, ones=np.ones((HIDDEN, 1), np.float32))
        here = run.soft(reading).detach().cpu().numpy()
        worst = max(worst, float(np.abs(np.concatenate([first, rest[0]]) - here).max()))
    if worst > 1e-3:
        raise ValueError(f"the projector's torch and numpy outputs differ by {worst:.2e}")
    return worst


def _held_out(row_id: str, share: float) -> bool:
    return int(hashlib.sha256(f"val:{row_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def _readings(rows: Sequence[Mapping[str, Any]], arm: str, k: int, d: int) -> dict[str, Any]:
    """Every reading the run's rows carry, by its row and role; with the ``zero`` arm every one is
    the empty reading (the zero-tokens ablation, on the same rows)."""
    import numpy as np

    empty: tuple[Any, Any, Any, float] = (np.zeros((k - 1, features(d)), np.float32), np.zeros(k - 1, np.float32),
                                         np.zeros(2 * d, np.float32), 0.0)
    found: dict[str, Any] = {}
    for row in rows:
        for role, reading in (("own", _row_reading(row)), ("other", _row_reading(row.get("contrast") or {}))):
            if reading is None:
                continue
            decoded = decode_reading(reading)
            if decoded[0].shape != empty[0].shape:
                raise ValueError(f"row {row.get('id')}'s reading is for another K or core width")
            found[f"{row['id']}/{role}"] = empty if arm == "zero" else decoded
    found["empty"] = empty
    return found


def main(ct: Any, cfg: dict[str, Any]) -> int:  # pragma: no cover - runs only on the GPU VM
    """The channel run: ``ct`` is the training script's module (``colab_train``), ``cfg`` its recipe."""
    import numpy as np
    import torch  # type: ignore[import-not-found, unused-ignore]
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoTokenizer, set_seed  # type: ignore[import-not-found, unused-ignore]

    ch = check_channel(cfg["channel"])
    smoke, base = ct._smoke_model(), ct.base_model(os.environ)
    out, prev = Path(os.environ["ANJO_TRAIN_OUT"]), os.environ.get("ANJO_TRAIN_PREV") or None
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    set_seed(cfg["seed"])
    started_all = time.monotonic()
    local = smoke or base
    tokenizer = AutoTokenizer.from_pretrained(local or ct.MODEL, revision=None if local else cfg["revision"])
    ct.check_template(tokenizer, cfg)
    template = ct.template_kwargs(cfg)
    sft_rows, pair_rows = ct.split_rows(ct._rows(Path(os.environ["ANJO_TRAIN_DATASET"])))
    if cfg["objective"] != "sft_dpo":
        pair_rows = []
    readings = _readings([*sft_rows, *pair_rows], ch["arm"], ch["k"], ch["d"])
    limit = cfg["max_seq_length"] - ch["k"]

    def fits(*parts: Sequence[int]) -> bool:
        return bool(sum(len(p) for p in parts) <= limit)

    sft: list[dict[str, Any]] = []
    contrast: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in sft_rows:
        prompt, reply = _encode(ct, tokenizer, row, "completion", template)
        item: dict[str, Any] = {"id": row["id"], "prompt": prompt, "reply": reply,
                                "own": f"{row['id']}/own" if row.get("core") else None}
        if row.get("contrast"):
            _, plain = _encode(ct, tokenizer, row, "completion", template, reply=row["contrast"]["plain"])
            item.update(plain=plain, other=f"{row['id']}/other" if row["contrast"].get("core") else "empty")
        if not fits(prompt, reply, item.get("plain", [])):
            continue
        if "plain" in item and _held_out(row["id"], ch["val_share"]):
            val.append(item)
        else:
            (contrast if "plain" in item else sft).append(item)
    pairs = []
    for row in pair_rows:
        prompt, chosen = _encode(ct, tokenizer, row, "chosen", template)
        _, rejected = _encode(ct, tokenizer, row, "rejected", template)
        if fits(prompt, chosen) and fits(prompt, rejected):
            pairs.append({"id": row["id"], "prompt": prompt, "chosen": chosen, "rejected": rejected,
                          "own": f"{row['id']}/own" if row.get("core") else None})
    ct.check_every_row_kept(cfg, len(sft_rows), len(sft) + len(contrast) + len(val))
    settings = ct.build_settings(cfg)
    model = ct._load_model(cfg, settings, smoke, base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if prev and (Path(prev) / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, prev, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(**settings["lora"]))
    device = torch.device("cpu" if smoke else "cuda:0")
    projector_prev = Path(prev) / PROJECTOR_FILE if prev else None
    if projector_prev is not None and projector_prev.is_file():
        params, meta = load_params(projector_prev.read_bytes())
        if (meta.get("k"), meta.get("d"), meta.get("width")) != (ch["k"], ch["d"], ch["width"]):
            raise ValueError("the warm start's projector has another K, core width or projector width")
        fresh = False
    else:
        params = fresh_params(ch["k"], ch["d"], ch["width"], _anchor_rows(tokenizer, model.get_input_embeddings(),
                                                                          ch["k"]), cfg["seed"])
        _feature_stats(params, [v for key, v in readings.items() if key != "empty"])
        fresh = True
    start_sha = params_sha(params)
    run = _Run(model, params, params, ch["k"], device)
    run.eot = tokenizer.convert_tokens_to_ids(EOT)
    embeds_gap = _check_embeds(run, tokenizer)
    forward_gap = _check_forward(run, [v for key, v in readings.items() if key != "empty"] or [readings["empty"]])
    trained = run.params
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg["learning_rate"]},
        {"params": [trained[n] for n in ("w1", "b1", "u1", "c1", "w2", "b2")], "lr": ch["projector_lr"]},
        {"params": [trained[n] for n in ("anchors", "speak", "silence")], "lr": ch["anchor_lr"]},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    accumulation, epochs = cfg["accumulation"], cfg["epochs"]
    planned = max(1, epochs * math.ceil((len(sft) + len(contrast)) / accumulation))
    warm = max(1, int(ct.WARMUP * planned))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, planned) / planned)))
    deadline = started_all + 60.0 * ch["max_minutes"]
    beta, weight = ch["contrast_beta"], ch["contrast_weight"]

    def soft(key: str | None) -> Any:
        return None if key is None else run.soft(readings[key])

    def item_loss(item: dict[str, Any]) -> tuple[Any, float | None]:
        if "plain" not in item:
            lp = run.logps([(item["prompt"], item["reply"], soft(item["own"]))])
            return -lp[0] / len(item["reply"]), None
        own, other = soft(item["own"]), soft(item["other"])
        lp = run.logps([(item["prompt"], item["reply"], own), (item["prompt"], item["plain"], own),
                        (item["prompt"], item["reply"], other), (item["prompt"], item["plain"], other)])
        margin = (lp[0] - lp[1]) - (lp[2] - lp[3])
        nll = -lp[0] / len(item["reply"]) - lp[3] / len(item["plain"])
        return nll - weight * torch.nn.functional.logsigmoid(beta * margin), float(margin)

    model.train()
    steps, seen, losses, margins = 0, 0, [], []
    order = [*sft, *contrast]
    for _ in range(epochs):
        random.shuffle(order)
        for n, item in enumerate(order):
            if time.monotonic() > deadline:
                break
            loss, margin = item_loss(item)
            (loss / accumulation).backward()
            losses.append(float(loss))
            if margin is not None:
                margins.append(margin)
            seen += 1
            if (n + 1) % accumulation == 0 or n + 1 == len(order):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
    dpo_steps = 0
    if pairs and time.monotonic() < deadline:
        dpo_steps = _dpo(run, pairs, readings, cfg, deadline)
    model.eval()
    held = _validate(run, val, readings) if val else {}
    model.save_pretrained(str(out))
    ct.record_rendering(out, cfg["rendering"])
    if base is not None:
        ct.record_base_model(out)
    final = {n: v.detach().cpu().numpy() for n, v in run.params.items()}
    meta = {"k": ch["k"], "d": ch["d"], "width": ch["width"], "arm": ch["arm"], "trained_from": start_sha}
    (out / PROJECTOR_FILE).write_bytes(save_params(final, meta))
    receipt = {
        "steps": steps, "epochs": epochs, "train_minutes": (time.monotonic() - started_all) / 60.0,
        "examples": len(sft) + len(contrast), "dropped_truncated": len(sft_rows) - len(sft) - len(contrast) - len(val),
        "gpu": "cpu" if smoke else torch.cuda.get_device_name(0), "objective": cfg["objective"],
        "dpo_steps": dpo_steps, "pairs": len(pairs), "pairs_dropped_truncated": len(pair_rows) - len(pairs),
        "rows_checked": seen, "dpo_reference": "adapter_off+projector_start" if pairs else "none",
        "rendering": cfg["rendering"],
        "channel": {"k": ch["k"], "arm": ch["arm"], "fresh": fresh, "projector_from": start_sha,
                    "projector": params_sha(final), "contrast_rows": len(contrast), "held_out": len(val),
                    "train_margin_mean": float(np.mean(margins)) if margins else None,
                    "loss_first": float(np.mean(losses[:20])) if losses else None,
                    "loss_last": float(np.mean(losses[-20:])) if losses else None,
                    "embeds_gap": embeds_gap, "forward_gap": forward_gap, **held},
    }
    (out / ct.RECEIPT).write_text(json.dumps(receipt), encoding="utf-8")
    return 0


def _dpo(run: _Run, pairs: list[dict[str, Any]], readings: Mapping[str, Any], cfg: dict[str, Any],
         deadline: float) -> int:  # pragma: no cover - GPU VM only
    """One DPO epoch with the channel on both sides; the reference is the adapter switched off and
    the projector as the run began, scored once before the pass."""
    torch = run.torch
    optimizer = torch.optim.AdamW([p for p in run.model.parameters() if p.requires_grad]
                                  + [run.params[n] for n in ("w1", "b1", "u1", "c1", "w2", "b2")],
                                  lr=cfg["dpo_learning_rate"], weight_decay=0.0)
    refs = []
    with torch.no_grad(), run.model.disable_adapter():
        for pair in pairs:
            ref_soft = None if pair["own"] is None else run.soft(readings[pair["own"]], ref=True)
            lp = run.logps([(pair["prompt"], pair["chosen"], ref_soft), (pair["prompt"], pair["rejected"], ref_soft)],
                           grad=False)
            refs.append(float(lp[0] - lp[1]))
    steps = 0
    for n, (pair, ref) in enumerate(zip(pairs, refs, strict=True)):
        if time.monotonic() > deadline:
            break
        own = None if pair["own"] is None else run.soft(readings[pair["own"]])
        lp = run.logps([(pair["prompt"], pair["chosen"], own), (pair["prompt"], pair["rejected"], own)])
        loss = -torch.nn.functional.logsigmoid(cfg["dpo_beta"] * ((lp[0] - lp[1]) - ref))
        (loss / cfg["accumulation"]).backward()
        if (n + 1) % cfg["accumulation"] == 0 or n + 1 == len(pairs):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    return steps


def _validate(run: _Run, val: list[dict[str, Any]], readings: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    """The held-out contrast rows' margins with their own state and with the empty reading in its
    place: if the two match, the channel carries nothing."""
    torch = run.torch
    found: dict[str, list[float]] = {"state": [], "zero": []}
    with torch.no_grad():
        for item in val:
            other = run.soft(readings[item["other"]])
            for arm, own_key in (("state", item["own"]), ("zero", "empty")):
                own = run.soft(readings[own_key or "empty"])
                lp = run.logps([(item["prompt"], item["reply"], own), (item["prompt"], item["plain"], own),
                                (item["prompt"], item["reply"], other), (item["prompt"], item["plain"], other)],
                               grad=False)
                found[arm].append(float((lp[0] - lp[1]) - (lp[2] - lp[3])))
    return {f"val_{arm}_margin": sum(m) / len(m) for arm, m in found.items()} | {
        f"val_{arm}_right": sum(x > 0 for x in m) / len(m) for arm, m in found.items()}
