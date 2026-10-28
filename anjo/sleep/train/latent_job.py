"""MIND's event latents on the Modal training app: the shared terms, and the part that runs in the
container (the encoder job of ``anjo.sleep.mind.core.latents``).

One call reads a batch of events through Glimmer's own hidden states. The events (a content key,
the speaker's role, the text, and optionally the turns before it) are uploaded to the latents volume as
``<job>/events.jsonl``; the call writes ``<job>/latents.npy`` (float32, one row per event) and
``<job>/index.jsonl`` (a header, then each row's key and token counts) beside them, and the host
fetches both back.

**The reading** is Glimmer's own chat format, so the model knows who said what. Each event is one
sequence: ``<|begin_of_text|>``, then up to ``max_context_tokens`` of the turns before it (the end of
them), each ``<|start|>{role}<|message|>{text}<|eot|>``, then the event's own turn in the same form,
its text cut to ``max_event_tokens``. The role is ``user`` (the person), ``assistant`` (Anjo,
written ``assistant to=user`` as her replies are) or ``system``. Glimmer (HF transformers, bf16,
from the weights volume at the pinned revision) runs over it with a forward hook on each named
decoder layer, and the layer's output, the residual stream, is averaged over the event's content
tokens only: not its turn's header, not its ``<|eot|>``. Several layers are concatenated, so
``D = 6656 * len(layers)``. A layer is numbered 0 to 51 by the decoder layer whose output is read;
vLLM's auxiliary hidden-state ids count one higher (``muse_glimmer.py`` adds the state after layer
``i`` as ``i + 1``). The forward stops after the deepest named layer, so a middle layer costs a
fraction of the model. The texts arrive cleaned of special tokens' text (``serving.framing``), so
only the headers written here are special tokens.

**Batches** pack sequences sorted by length so that no batch passes ``batch_tokens`` padded tokens
(``batches``); a sequence longer than that goes alone. Padding is on the right, under the attention
mask, so the causal model's real tokens never see it.

**Embedding rows** (``embedding_rows``, the ``embed_rows`` CPU function): the raw bf16 rows of
Glimmer's input embedding table for some token ids, read by offset straight out of the safetensors
file that holds the table, for ``serving.soft_tokens`` (which normalizes them as vLLM does). The
function takes no hold of its own: only a caller holding one that covers it calls it (the live
soft-token test's hold includes it, ``tools.soft_tokens_live``).

It imports the standard library, numpy, ``serving/config.py`` and ``modal_job.py`` only, and loads
torch and transformers inside ``encode_on_volume``: the training image carries these files and
nothing else of Anjo.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from anjo.sleep.serving import config
from anjo.sleep.train import modal_job

FUNCTION: Final = "encode_latents"
GPU: Final = modal_job.GPU  # one A100-80GB: Glimmer in bf16 is about 56 GB
FUNCTION_TIMEOUT_S: Final = modal_job.FUNCTION_TIMEOUT_S  # the job's hard timeout, and its hold
SCALEDOWN_S: Final = 10  # an idle container stops this soon after the call returns; the hold covers it
VOLUME: Final = "anjo-sleep-latents"
VOLUME_DIR: Final = "/latents"
EVENTS_FILE: Final = "events.jsonl"
MATRIX_FILE: Final = "latents.npy"
INDEX_FILE: Final = "index.jsonl"
CALL_FILE: Final = "call.json"  # the container's own call id, written first: a sweep finds the call by its job
JOB_FORMAT: Final = "anjo-latent-job/1"

ENCODER: Final = "glimmer-hf"
POOLING: Final = "mean_event_tokens"
HIDDEN: Final = 6656  # Glimmer's text hidden size
NUM_LAYERS: Final = 52
RENDERING: Final = "glimmer-chat/v1"  # BOS, the turns before, the event's turn; pooled over its content
ROLES: Final = ("user", "assistant", "system")
HEADERS: Final = {"user": "<|start|>user<|message|>", "assistant": "<|start|>assistant to=user<|message|>",
                  "system": "<|start|>system<|message|>"}
TURN_END: Final = "<|eot|>"
BOS_ID: Final = config.FIRST_SPECIAL_ID  # <|begin_of_text|>
DEFAULT_BATCH_TOKENS: Final = 16_384
EMBED_FUNCTION: Final = "embed_rows"
EMBED_TIMEOUT_S: Final = 300
VOCAB: Final = 202_048
MAX_EMBED_IDS: Final = 16_384  # one call's ids: about 218 MB of rows back
EMBED_KEYS: Final = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")  # canonical, then the older checkpoint layout
SAFETENSORS_INDEX: Final = "model.safetensors.index.json"
_SPEC_KEYS: Final = frozenset({
    "encoder", "model", "revision", "layers", "d", "pooling", "rendering", "max_context_tokens",
    "max_event_tokens", "batch_tokens",
})
_JOB = re.compile(r"j-[0-9a-f]{16}")
_CALL = re.compile(r"fc-[A-Za-z0-9]{1,64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class LatentSpecError(ValueError):
    """A spec, a job name or an events file the encoder job won't read."""


def check_job(job: str) -> str:
    if not isinstance(job, str) or not _JOB.fullmatch(job):
        raise LatentSpecError("a latents job is named j-<16 hex digits>")
    return job


def _count(spec: Mapping[str, Any], key: str, low: int, high: int) -> int:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise LatentSpecError(f"{key} is an integer from {low} to {high}")
    return value


def check_spec(spec_json: str, pinned: str) -> dict[str, Any]:
    """The spec the host sent, checked: Glimmer at the pinned revision, layers in order, D to match."""
    spec = json.loads(spec_json)
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS:
        raise LatentSpecError(f"a latents spec carries exactly {', '.join(sorted(_SPEC_KEYS))}")
    if spec["encoder"] != ENCODER or spec["model"] != config.MODEL:
        raise LatentSpecError(f"the encoder job reads {config.MODEL} with {ENCODER}")
    if not isinstance(spec["revision"], str) or not _COMMIT.fullmatch(spec["revision"]) or spec["revision"] != pinned:
        raise LatentSpecError("the spec's revision isn't the one this app was deployed pinned to")
    layers = spec["layers"]
    if (not isinstance(layers, list) or not layers or any(isinstance(n, bool) or not isinstance(n, int) for n in layers)
            or layers != sorted(set(layers)) or not 0 <= layers[0] <= layers[-1] < NUM_LAYERS):
        raise LatentSpecError(f"the layers are distinct decoder layers 0 to {NUM_LAYERS - 1}, in order")
    if spec["d"] != HIDDEN * len(layers):
        raise LatentSpecError(f"d is {HIDDEN} for each layer read")
    if spec["pooling"] != POOLING or spec["rendering"] != RENDERING:
        raise LatentSpecError(f"the pooling is {POOLING}, over the {RENDERING} rendering")
    _count(spec, "max_context_tokens", 0, 8192)
    _count(spec, "max_event_tokens", 1, 8192)
    _count(spec, "batch_tokens", 1024, 131_072)
    return spec


def sequence(
    *, bos: int, context: Sequence[int], header: Sequence[int], content: Sequence[int], end: Sequence[int],
    max_context: int, max_event: int,
) -> tuple[list[int], int, int, bool]:
    """One event's token ids: the BOS, the end of the turns before it, then its own turn (its header,
    the start of its content, its end). Returns the ids, where its content starts and stops, and
    whether the context or the content was cut."""
    kept_context = list(context[len(context) - max_context:]) if max_context and context else []
    kept = list(content[:max_event])
    truncated = len(kept_context) < len(context) or len(kept) < len(content)
    start = 1 + len(kept_context) + len(header)
    return [bos, *kept_context, *header, *kept, *end], start, start + len(kept), truncated


def batches(lengths: Sequence[int], batch_tokens: int) -> list[list[int]]:
    """Indices into ``lengths`` packed longest first, so that each batch's padded size (its count
    times its longest) stays within ``batch_tokens``. A sequence longer than that goes alone."""
    order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    packed: list[list[int]] = []
    for index in order:
        if packed and (len(packed[-1]) + 1) * lengths[packed[-1][0]] <= batch_tokens:
            packed[-1].append(index)
        else:
            packed.append([index])
    return packed


def pool_weights(*, starts: Sequence[int], stops: Sequence[int], width: int) -> np.ndarray:
    """Each row's averaging weights over a padded batch: 1/n on the event's n content tokens, 0 elsewhere."""
    weights: np.ndarray = np.zeros((len(starts), width), dtype=np.float32)
    for row, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if not 0 < start < stop <= width:
            raise LatentSpecError("an event has at least one content token, after the BOS and inside the batch")
        weights[row, start:stop] = 1.0 / (stop - start)
    return weights


def write_outputs(job_dir: Path, spec: Mapping[str, Any], stats: Sequence[Mapping[str, Any]], matrix: np.ndarray) -> None:
    """The job's two files: the float32 matrix, and the index (a header, then each row's key and counts)."""
    if matrix.dtype != np.float32 or matrix.ndim != 2 or matrix.shape != (len(stats), spec["d"]):
        raise LatentSpecError("the job writes one float32 row of width d for each event")
    job_dir.mkdir(parents=True, exist_ok=True)
    header = {"format": JOB_FORMAT, "count": len(stats), "d": spec["d"], "layers": spec["layers"],
              "revision": spec["revision"]}
    lines = [json.dumps(header, sort_keys=True)] + [json.dumps(dict(row), sort_keys=True) for row in stats]
    pending = job_dir / f".{MATRIX_FILE}.tmp"
    with open(pending, "wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    os.replace(pending, job_dir / MATRIX_FILE)
    (job_dir / f".{INDEX_FILE}.tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(job_dir / f".{INDEX_FILE}.tmp", job_dir / INDEX_FILE)


def read_outputs(job_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """The job's header, rows and matrix, checked against each other."""
    lines = (job_dir / INDEX_FILE).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise LatentSpecError("the job's index is empty")
    header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    matrix = np.load(job_dir / MATRIX_FILE, allow_pickle=False)
    if header.get("format") != JOB_FORMAT or header.get("count") != len(rows):
        raise LatentSpecError("the job's index doesn't count its rows")
    if matrix.dtype != np.float32 or matrix.shape != (len(rows), header.get("d")):
        raise LatentSpecError("the job's matrix doesn't match its index")
    return header, rows, matrix


def mark_call(job_dir: Path, call_id: object) -> None:
    """The container's first act: its own Modal call id in the job's folder, so a host killed between
    the spawn and its record still finds the call by the job's name (``latent_runner.sweep``)."""
    if not isinstance(call_id, str) or not _CALL.fullmatch(call_id):
        raise LatentSpecError("a Modal call id is fc-<letters and digits>")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / CALL_FILE).write_text(json.dumps({"call": call_id}), encoding="utf-8")


def read_call(data: bytes) -> str | None:
    """The call id ``mark_call`` wrote, or None if the file holds none."""
    try:
        found = json.loads(data.decode("utf-8")).get("call")
    except (ValueError, AttributeError):
        return None
    return found if isinstance(found, str) and _CALL.fullmatch(found) else None


def _turn(turn: object) -> bool:
    return (isinstance(turn, list) and len(turn) == 2 and turn[0] in ROLES and isinstance(turn[1], str)
            and bool(turn[1]))


def read_events(path: Path) -> list[dict[str, Any]]:
    """The uploaded events: each a key, a role, a non-empty text and its context ([role, text] turns)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"key", "role", "text", "context"}
                or not isinstance(row["key"], str) or row["role"] not in ROLES or not isinstance(row["text"], str)
                or not row["text"] or not isinstance(row["context"], list) or not all(map(_turn, row["context"]))):
            raise LatentSpecError("an uploaded event is a key, a role, a non-empty text and its turns before")
    return rows


def embedding_rows(base: Path, ids: Sequence[int]) -> bytes:
    """Each id's raw row of the input embedding table, little-endian bf16, ``HIDDEN`` values a row."""
    if not 0 < len(ids) <= MAX_EMBED_IDS or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < VOCAB
                                                for i in ids):
        raise LatentSpecError(f"embedding rows are asked for 1 to {MAX_EMBED_IDS} ids, each 0 to {VOCAB - 1}")
    path, key = _embedding_file(base)
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        entry = json.loads(handle.read(size)).get(key)
        if not isinstance(entry, dict) or entry.get("dtype") != "BF16" or entry.get("shape") != [VOCAB, HIDDEN]:
            raise LatentSpecError("the embedding table isn't Glimmer's bf16 table")
        start, row = 8 + size + int(entry["data_offsets"][0]), HIDDEN * 2
        parts = []
        for i in ids:
            handle.seek(start + i * row)
            parts.append(handle.read(row))
    data = b"".join(parts)
    if len(data) != len(ids) * row:
        raise LatentSpecError("the embedding table is cut short")
    return data


def _embedding_file(base: Path) -> tuple[Path, str]:
    index = base / SAFETENSORS_INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        for key in EMBED_KEYS:
            if key in weight_map:
                return base / Path(weight_map[key]).name, key
    elif (base / "model.safetensors").is_file():
        with (base / "model.safetensors").open("rb") as handle:
            header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
        for key in EMBED_KEYS:
            if key in header:
                return base / "model.safetensors", key
    raise LatentSpecError("the weights hold no input embedding table")


def encode_on_volume(
    *, job_dir: Path, spec_json: str, pinned: str, weights_root: Path, exec_s: float,
    clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print,
) -> dict[str, Any]:  # pragma: no cover - runs only in the GPU container
    """One call in the container: read ``job_dir``'s events through Glimmer and write their latents."""
    started = clock()
    spec = check_spec(spec_json, pinned)
    base = modal_job.base_folder(weights_root, spec["revision"], pinned)
    events = read_events(job_dir / EVENTS_FILE)
    tokenizer, text_model, device = _load(base)
    loaded = clock()
    sequences = [_sequence(tokenizer, row, spec) for row in events]
    matrix = np.zeros((len(events), spec["d"]), dtype=np.float32)
    deadline = started + exec_s
    for batch in batches([len(ids) for ids, _, _, _ in sequences], spec["batch_tokens"]):
        if clock() > deadline:
            raise TimeoutError("the encoder job ran out of its time")
        matrix[batch] = _encode_batch(text_model, device, [sequences[i] for i in batch], spec["layers"])
    stats = [{"key": row["key"], "tokens": len(ids), "event_tokens": stop - start, "truncated": cut}
             for row, (ids, start, stop, cut) in zip(events, sequences, strict=True)]
    write_outputs(job_dir, spec, stats, matrix)
    tokens = sum(len(ids) for ids, _, _, _ in sequences)
    done = {"events": len(events), "tokens": tokens, "event_tokens": sum(row["event_tokens"] for row in stats),
            "truncated": sum(cut for _, _, _, cut in sequences), "load_seconds": loaded - started,
            "seconds": clock() - loaded, "d": spec["d"], "layers": spec["layers"]}
    log(f"latents: {done['events']} events, {tokens} tokens, load {done['load_seconds']:.0f} s, "
        f"encode {done['seconds']:.0f} s ({tokens / max(done['seconds'], 1e-9):.0f} tokens/s)")
    return done


def _load(base: Path) -> tuple[Any, Any, Any]:  # pragma: no cover - needs the GPU stack
    import torch  # type: ignore[import-not-found, unused-ignore]
    from transformers import AutoModelForImageTextToText, AutoTokenizer  # type: ignore[import-not-found, unused-ignore]

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    model = AutoModelForImageTextToText.from_pretrained(str(base), dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    text_model = model.model.language_model
    if len(text_model.layers) != NUM_LAYERS or text_model.config.hidden_size != HIDDEN:
        raise LatentSpecError("the weights aren't Glimmer's 52-layer text decoder")
    return tokenizer, text_model, torch.device("cuda:0")


def _sequence(
    tokenizer: Any, row: Mapping[str, Any], spec: Mapping[str, Any],
) -> tuple[list[int], int, int, bool]:  # pragma: no cover - needs the tokenizer
    def ids(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])  # the headers' special tokens parse

    context = [token for role, text in row["context"] for token in ids(HEADERS[role]) + ids(text) + ids(TURN_END)]
    content = ids(row["text"])
    if not content:
        raise LatentSpecError("an event's text has no tokens")
    return sequence(bos=BOS_ID, context=context, header=ids(HEADERS[row["role"]]), content=content,
                    end=ids(TURN_END), max_context=spec["max_context_tokens"], max_event=spec["max_event_tokens"])


class _Deep(Exception):
    """Raised by the deepest hook: the layers below it are never run."""


def _encode_batch(
    text_model: Any, device: Any, rows: Sequence[tuple[list[int], int, int, bool]], layers: Sequence[int],
) -> np.ndarray:  # pragma: no cover - needs the GPU stack
    import torch  # type: ignore[import-not-found, unused-ignore]

    width = max(len(ids) for ids, _, _, _ in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for row, (tokens, _, _, _) in enumerate(rows):
        ids[row, : len(tokens)] = torch.tensor(tokens)
        mask[row, : len(tokens)] = 1
    weights = torch.from_numpy(pool_weights(starts=[start for _, start, _, _ in rows],
                                            stops=[stop for _, _, stop, _ in rows], width=width)).to(device)
    pooled: dict[int, Any] = {}

    def hook(layer: int) -> Callable[..., None]:
        def read(module: Any, args: Any, output: Any) -> None:
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            pooled[layer] = torch.bmm(weights.unsqueeze(1), hidden.float()).squeeze(1)
            if layer == layers[-1]:
                raise _Deep
        return read

    handles = [text_model.layers[layer].register_forward_hook(hook(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            text_model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
    except _Deep:
        pass  # the deepest named layer has been read; nothing below it is needed
    finally:
        for handle in handles:
            handle.remove()
    out: np.ndarray = torch.cat([pooled[layer] for layer in layers], dim=1).cpu().numpy().astype(np.float32)
    if not np.isfinite(out).all():
        raise LatentSpecError("a latent came out non-finite")
    return out
