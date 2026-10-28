"""MIND's event latents: every event read through Glimmer's own hidden states (the shared design's
part 1), so her mind lives in the same representation as the model that speaks for her.

An event (one evidence item of a life, or one turn or finding of a real person) becomes one float32
vector: Glimmer's residual stream after fixed decoder layers, mean-pooled over the event's content
tokens. The event is read as a turn in Glimmer's own chat format under its speaker's role (``user``,
``assistant`` or ``system``), so the model knows who said it, after up to ``max_context_tokens`` of
the turns before it, which are left out of the mean (``train/latent_job.py`` has the exact reading).
It is computed offline, in batch, with HF
transformers on the Modal training app's A100-80GB, and cached by content hash, so a rerun pays
nothing for an event it has seen.

**The layer.** ``DEFAULT_LAYERS`` is decoder layer 35: one of Glimmer's 13 global layers (full
attention, no RoPE: 3, 7, ..., 51; the other 39 see a 2,048-token window), about two thirds of the
way down. A global layer has read the whole context, not only its last 2,048 tokens. Mid-to-late
layers carry what a text is about better than the last one, which leans toward the next token. And
the forward stops after layer 35, about 30% less compute than the full model. ``--layers`` reads
any others (concatenated, ``D = 6656 * len(layers)``); the index header records the layers, the
model revision, the pooling, the rendering and ``D``.

**Files**, one folder per life or person, so a delete finds them by its id: ``<out>/<life>/latents.npy``
is the ``(n, D)`` float32 matrix; ``<out>/<life>/index.jsonl`` is a header line, then one line per row
(event id, life or person, time, kind, role, content key, and whether the context or the text was
cut to fit). ``load_latents`` reads one life's folder back as ``EventLatent``s, ``load_run`` all of
them. Everything stays in the sleep data folder, ``$ANJO_DATA_DIR/sleep/latents/``: ``--out`` is a
folder in its ``runs/``, and the content-hash cache is ``cache/<life>/<spec>/``. A delete removes
the person's folders in both (``erase_paths``, ``app.ownership``).

**The CLI**, plan first:

    python -m anjo.sleep.mind.core.latents encode --events <jsonl> --out <dir> [--layers 35 ...] [--plan]

``--plan`` prints the hold, the token and time estimates and the cost, and touches nothing: no
ledger, no volume, no call. Without it, the events not in the cache go up in one call, under a hold
on ``modal_usd`` taken before anything touches Modal (``train/latent_runner.py``), under MIND's lock,
after a sweep that clears what a killed encode left. ``--fake --d N`` writes deterministic stand-in
latents locally, for tests and for the core before the real run.

Each line of the events file is an object with ``event_id`` (or ``id``), ``life_or_person`` (or
``person_id``; a plain name, since it names a folder), ``t`` (an ISO time, a naive one read as UTC,
or a date read as its UTC midnight), ``kind``, ``text`` and, optionally, ``role`` (who said it;
``user`` by default, as for a life's evidence) and ``context`` (the turns before it, oldest first,
each ``{"role": ..., "content": ...}``); other fields are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from anjo.sleep.budget import DATA_DIR_ENV, Bound, BudgetError, Ledger, default_ledger
from anjo.sleep.mind.sweep import lock_path, mind_lock
from anjo.sleep.serving import config, framing
from anjo.sleep.train import latent_job, latent_runner, modal_job

Matrix: TypeAlias = NDArray[np.float32]
FORMAT: Final = "anjo-latents/1"
INDEX_FILE: Final = "index.jsonl"
MATRIX_FILE: Final = "latents.npy"
HIDDEN: Final = latent_job.HIDDEN
NUM_LAYERS: Final = latent_job.NUM_LAYERS
GLOBAL_LAYERS: Final = tuple(range(3, NUM_LAYERS, 4))  # config.json: full_attention, layer_rope_theta 0
DEFAULT_LAYERS: Final = (35,)
DEFAULT_MAX_CONTEXT_TOKENS: Final = 1024
DEFAULT_MAX_EVENT_TOKENS: Final = 2048
LAYER_INDEXING: Final = "decoder layer output, 0-based (vLLM aux hidden-state id = layer + 1)"
# Estimates for the plan, not measurements (the job prints its own): about 3.5 characters a token,
# and an A100-80GB in bf16 reading about 2,500 tokens a second through all 52 layers, faster when the
# forward stops early. Half that rate is the floor one call must still fit its tokens under.
CHARS_PER_TOKEN: Final = 3.5
FULL_DEPTH_TOKENS_PER_S: Final = 2500.0
LOAD_S: Final = 360.0  # the container's start and Glimmer's 56 GB off the weights volume
LATENTS_DIR: Final = "latents"  # under <data dir>/sleep/
CACHE_DIR: Final = "cache"
RUNS_DIR: Final = "runs"
ROLES: Final = latent_job.ROLES
DEFAULT_ROLE: Final = "user"
_LIFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
Turns: TypeAlias = tuple[tuple[str, str], ...]  # (role, text), oldest first


class LatentError(ValueError):
    """An events file, a spec, a matrix or a latents folder that can't be used."""


@dataclass(frozen=True)
class EventInput:
    """One event to read: its id, whose life it is, when, its kind, its text, who said it, and the
    turns before it."""

    event_id: str
    life_or_person: str
    t: datetime
    kind: str
    text: str
    context: Turns = ()
    role: str = DEFAULT_ROLE


@dataclass(frozen=True, eq=False)
class EventLatent:
    """The design's interface: one event and its latent ``vec`` (float32, shape ``(D,)``)."""

    event_id: str
    life_or_person: str
    t: datetime
    kind: str
    vec: Matrix


@dataclass(frozen=True)
class EncodeItem:
    """What an encoder reads for one event: its content key, its speaker's role, and its cleaned text
    and turns before it."""

    key: str
    role: str
    text: str
    context: Turns = ()


@dataclass(frozen=True, eq=False)
class Encoded:
    """An encoder's rows, and whether each event's context or text was cut to fit."""

    matrix: Matrix
    truncated: tuple[bool, ...]


@dataclass(frozen=True)
class LatentSpec:
    """How a latent is read. Every field is in each event's content key and in the index header."""

    encoder: str
    model: str
    revision: str
    layers: tuple[int, ...]
    d: int
    pooling: str = latent_job.POOLING
    rendering: str = latent_job.RENDERING
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_event_tokens: int = DEFAULT_MAX_EVENT_TOKENS
    sanitizer: str = framing.SANITIZER

    def identity(self) -> dict[str, Any]:
        return {**asdict(self), "layers": list(self.layers)}

    def spec_id(self) -> str:
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()[:16]

    def key(self, role: str, text: str, context: Turns) -> str:
        """The content key: what is read, and how. Two events with the same speaker, text and context share it."""
        blob = json.dumps({"spec": self.identity(), "role": role, "text": text, "context": [list(turn) for turn in context]},
                          sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def as_json(self, batch_tokens: int = latent_job.DEFAULT_BATCH_TOKENS) -> str:
        """The spec the encoder job reads (``latent_job.check_spec``)."""
        sent = {key: value for key, value in self.identity().items() if key != "sanitizer"}
        return json.dumps({**sent, "batch_tokens": batch_tokens}, sort_keys=True)

    def header(self, count: int) -> dict[str, Any]:
        return {"format": FORMAT, **self.identity(), "dtype": "float32", "count": count,
                "layer_indexing": LAYER_INDEXING}


def glimmer_spec(
    revision: str, layers: Sequence[int] = DEFAULT_LAYERS, *, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_event_tokens: int = DEFAULT_MAX_EVENT_TOKENS,
) -> LatentSpec:
    """Glimmer at ``revision``, read at ``layers`` (decoder layers 0 to 51, kept in order)."""
    chosen = tuple(sorted(set(layers)))
    if not chosen or any(isinstance(n, bool) or not isinstance(n, int) for n in chosen) or not (
            0 <= chosen[0] and chosen[-1] < NUM_LAYERS):
        raise LatentError(f"a layer is a decoder layer from 0 to {NUM_LAYERS - 1}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise LatentError("the revision is Glimmer's pinned 40-character commit")
    return LatentSpec(latent_job.ENCODER, config.MODEL, revision, chosen, HIDDEN * len(chosen),
                      max_context_tokens=max_context_tokens, max_event_tokens=max_event_tokens)


def fake_spec(d: int, seed: int = 0) -> LatentSpec:
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise LatentError("a fake latent has a positive width")
    return LatentSpec("fake", "fake", f"seed-{seed}", (0,), d, pooling="hashed_words", rendering="role+text+context/5")


class Encoder(Protocol):
    spec: LatentSpec

    def encode(self, items: Sequence[EncodeItem]) -> Matrix | Encoded:
        """One float32 row of width ``spec.d`` for each item, in order (``Encoded`` also says which
        were cut to fit; a bare matrix cut none)."""
        ...


class FakeEncoder:
    """Deterministic stand-in latents of any width: each word has a fixed random direction, and an
    event's latent is the unit mean of its words' (its context's at a quarter weight) plus a smaller
    direction of its own content and speaker, so every text has its own latent, the same words said by
    another speaker another one, and shared words pull together."""

    def __init__(self, d: int, seed: int = 0) -> None:
        self.spec = fake_spec(d, seed)
        self._seed = seed
        self._words: dict[str, Matrix] = {}

    def _direction(self, name: str) -> Matrix:
        found = self._words.get(name)
        if found is None:
            digest = hashlib.sha256(f"{self._seed}:{name}".encode()).digest()
            raw = np.random.default_rng(int.from_bytes(digest[:8], "little")).standard_normal(self.spec.d)
            found = (raw / np.linalg.norm(raw)).astype(np.float32)
            self._words[name] = found
        return found

    def _one(self, item: EncodeItem) -> Matrix:
        total: Matrix = np.zeros(self.spec.d, dtype=np.float32)
        for weight, text in ((1.0, item.text), *((0.25, turn) for _, turn in item.context)):
            for word in re.findall(r"\w+", text.lower()):
                total += weight * self._direction(f"w:{word}")
        norm = float(np.linalg.norm(total))
        content = json.dumps([item.role, item.text, [list(turn) for turn in item.context]])
        mixed = (total / norm if norm > 0 else total) + 0.3 * self._direction(f"c:{content}")
        return (mixed / np.linalg.norm(mixed)).astype(np.float32)

    def encode(self, items: Sequence[EncodeItem]) -> Matrix:
        if not items:
            return np.zeros((0, self.spec.d), dtype=np.float32)
        return np.stack([self._one(item) for item in items]).astype(np.float32)


Cached: TypeAlias = tuple[Matrix, bool]  # a latent, and whether its event was cut to fit


class LatentCache:
    """Latents by content key, one folder per life and spec (``<root>/<life>/<spec>/``, so a delete
    finds a life's), appended in shards and never rewritten."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _folder(self, spec: LatentSpec, life: str) -> Path:
        return self.root / check_life(life) / spec.spec_id()

    def get(self, spec: LatentSpec, life: str, keys: Sequence[str]) -> dict[str, Cached]:
        wanted, found = set(keys), dict[str, Cached]()
        folder = self._folder(spec, life)
        for index in sorted(folder.glob("shard-*.jsonl")) if folder.is_dir() else []:
            stored = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]
            matrix = np.load(index.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
            if matrix.shape != (len(stored), spec.d) or matrix.dtype != np.float32:
                raise LatentError(f"cache shard {index.name} doesn't match its keys or D")
            for row, entry in enumerate(stored):
                if entry["key"] in wanted and entry["key"] not in found:
                    found[entry["key"]] = (np.array(matrix[row], dtype=np.float32), bool(entry["truncated"]))
        return found

    def put(self, spec: LatentSpec, life: str, keys: Sequence[str], rows: Sequence[Cached]) -> None:
        folder = self._folder(spec, life)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "spec.json").write_text(json.dumps(spec.identity(), sort_keys=True), encoding="utf-8")
        name = "shard-" + hashlib.sha256("\n".join(keys).encode()).hexdigest()[:16]
        _save(folder / f"{name}.npy", np.stack([vec for vec, _ in rows]).astype(np.float32))
        lines = [json.dumps({"key": key, "truncated": cut}) for key, (_, cut) in zip(keys, rows, strict=True)]
        _write(folder / f"{name}.jsonl", "\n".join(lines) + "\n")  # the index lands last: a torn shard is unseen


@dataclass(frozen=True)
class EncodeResult:
    """The events' latents in order, their keys, whether each was cut to fit, and how many events
    were encoded or found cached."""

    matrix: Matrix
    keys: tuple[str, ...]
    encoded: int
    cached: int
    truncated: tuple[bool, ...] = ()


def items_for(events: Sequence[EventInput], spec: LatentSpec) -> list[EncodeItem]:
    """Each event's role, cleaned text and turns (special tokens' text broken) and its content key."""
    items = []
    for event in events:
        text = framing.clean_text(event.text)
        context = tuple((role, framing.clean_text(turn)) for role, turn in event.context)
        items.append(EncodeItem(spec.key(event.role, text, context), event.role, text, context))
    return items


def _by_life(events: Sequence[EventInput], items: Sequence[EncodeItem]) -> dict[str, list[str]]:
    lives: dict[str, list[str]] = {}
    for event, item in zip(events, items, strict=True):
        keys = lives.setdefault(event.life_or_person, [])
        if item.key not in keys:
            keys.append(item.key)
    return lives


def _stored(spec: LatentSpec, lives: Mapping[str, Sequence[str]], cache: LatentCache | None) -> dict[str, dict[str, Cached]]:
    return {life: cache.get(spec, life, keys) for life, keys in lives.items()} if cache is not None else {}


def _encoded(fresh: Matrix | Encoded, rows: int, d: int) -> tuple[Matrix, tuple[bool, ...]]:
    matrix, cut = (fresh.matrix, fresh.truncated) if isinstance(fresh, Encoded) else (fresh, (False,) * rows)
    _check_matrix(matrix, rows, d)
    if len(cut) != rows:
        raise LatentError("an encoder says whether each event was cut, and only those")
    return matrix, cut


def encode_events(events: Sequence[EventInput], encoder: Encoder, cache: LatentCache | None = None) -> EncodeResult:
    """Every event's latent: from the cache when its content was read before (in any life of this
    run), else from ``encoder`` (once per content key), then cached in each life's own folder."""
    spec = encoder.spec
    items = items_for(events, spec)
    lives = _by_life(events, items)
    stored = _stored(spec, lives, cache)
    found = {key: row for rows in stored.values() for key, row in rows.items()}
    todo = [item for item in {item.key: item for item in items}.values() if item.key not in found]
    if todo:
        matrix, cut = _encoded(encoder.encode(todo), len(todo), spec.d)
        found = {**found, **{item.key: (matrix[row], cut[row]) for row, item in enumerate(todo)}}
    for life, keys in lives.items() if cache is not None else ():
        missing = [key for key in keys if key not in stored[life]]
        if missing and cache is not None:
            cache.put(spec, life, missing, [found[key] for key in missing])
    new = {item.key for item in todo}
    matrix = np.stack([found[item.key][0] for item in items]).astype(np.float32) if items else (
        np.zeros((0, spec.d), dtype=np.float32))
    encoded = sum(item.key in new for item in items)
    return EncodeResult(matrix, tuple(item.key for item in items), encoded, len(items) - encoded,
                        tuple(found[item.key][1] for item in items))


def _check_matrix(matrix: Matrix, rows: int, d: int) -> None:
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32 or matrix.shape != (rows, d):
        raise LatentError(f"expected a float32 matrix of {rows} rows and D = {d}")
    if not np.isfinite(matrix).all():
        raise LatentError("a latent is not finite")


def _save(path: Path, matrix: Matrix) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(pending, "wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    os.replace(pending, path)


def _write(path: Path, text: str) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pending.write_text(text, encoding="utf-8")
    os.replace(pending, path)


def write_latents(
    out: Path, spec: LatentSpec, events: Sequence[EventInput], matrix: Matrix,
    truncated: Sequence[bool] | None = None,
) -> list[Path]:
    """Each life's matrix and index in ``<out>/<life>/``, the header declaring ``D``, the layers, the
    revision and the pooling, each row whether its event was cut to fit. Returns the folders."""
    try:
        _check_matrix(matrix, len(events), spec.d)
    except LatentError as error:
        raise LatentError(f"the matrix doesn't fit the spec's D or the events: {error}") from None
    cut = tuple(truncated) if truncated is not None else (False,) * len(events)
    if len(cut) != len(events):
        raise LatentError("say whether each event was cut, and only those")
    items, folders = items_for(events, spec), []
    for life in dict.fromkeys(event.life_or_person for event in events):
        rows = [n for n, event in enumerate(events) if event.life_or_person == life]
        lines = [json.dumps({"row": row, "event_id": events[n].event_id, "life_or_person": life,
                             "t": events[n].t.isoformat(), "kind": events[n].kind, "role": events[n].role,
                             "key": items[n].key, "truncated": cut[n]}, sort_keys=True)
                 for row, n in enumerate(rows)]
        folder = out / check_life(life)
        folder.mkdir(parents=True, exist_ok=True)
        _save(folder / MATRIX_FILE, matrix[rows])
        header = {**spec.header(len(rows)), "truncated": sum(cut[n] for n in rows)}
        _write(folder / INDEX_FILE, "\n".join([json.dumps(header, sort_keys=True), *lines]) + "\n")
        folders.append(folder)
    return folders


def load_run(out: Path) -> dict[str, tuple[dict[str, Any], tuple[EventLatent, ...]]]:
    """Every life's header and latents in a run's ``out`` folder, by life or person."""
    return {folder.name: load_latents(folder) for folder in sorted(out.iterdir()) if (folder / INDEX_FILE).is_file()}


def load_latents(folder: Path) -> tuple[dict[str, Any], tuple[EventLatent, ...]]:
    """One life's latents folder: its header and its events' latents, checked against each other."""
    lines = (folder / INDEX_FILE).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise LatentError("the index is empty")
    header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    if header.get("format") != FORMAT or header.get("count") != len(rows):
        raise LatentError(f"the index isn't {FORMAT}, or doesn't count its rows")
    matrix = np.load(folder / MATRIX_FILE, allow_pickle=False)
    _check_matrix(matrix, len(rows), int(header["d"]))
    return header, tuple(
        EventLatent(row["event_id"], row["life_or_person"], datetime.fromisoformat(row["t"]), row["kind"], matrix[n])
        for n, row in enumerate(rows))


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise LatentError("an event's t is an ISO time or date")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            day = date.fromisoformat(value)
            return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise LatentError(f"an event's t can't be read: {value!r}") from None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def check_life(life: str) -> str:
    """A life or person id, which names a folder: a plain name."""
    if not isinstance(life, str) or not _LIFE.fullmatch(life):
        raise LatentError("a life or person id is a plain name (letters, digits, '.', '_', '-')")
    return life


def _role(value: object) -> str:
    if value not in ROLES:
        raise LatentError(f"a role is one of {', '.join(ROLES)}")
    return str(value)


def _turns(value: object) -> Turns:
    if not isinstance(value, list):
        raise LatentError("an event's context is a list of turns, each {role, content}")
    turns = []
    for turn in value:
        if not isinstance(turn, dict) or not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise LatentError("a context turn is {role, content}, its content non-empty text")
        turns.append((_role(turn.get("role")), turn["content"]))
    return tuple(turns)


def parse_event(row: Mapping[str, Any]) -> EventInput:
    fields = {"event_id": row.get("event_id", row.get("id")), "life_or_person": row.get("life_or_person",
              row.get("person_id")), "kind": row.get("kind"), "text": row.get("text")}
    for name, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            raise LatentError(f"an event needs a non-empty {name}")
    return EventInput(fields["event_id"], check_life(fields["life_or_person"]), _time(row.get("t")), fields["kind"],
                      fields["text"], _turns(row.get("context", [])), _role(row.get("role", DEFAULT_ROLE)))


def read_events(path: Path) -> list[EventInput]:
    """The events file, every line checked; an event id appears once."""
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise LatentError(f"line {number} isn't JSON") from None
        if not isinstance(row, dict):
            raise LatentError(f"line {number} isn't an object")
        try:
            events.append(parse_event(row))
        except LatentError as error:
            raise LatentError(f"line {number}: {error}") from None
    ids = [event.event_id for event in events]
    if len(set(ids)) != len(ids):
        raise LatentError("an event id appears twice")
    return events


class ModalEncoder:
    """The real encoder: one call of the Modal training app under a hold (``latent_runner.run_job``).
    ``lives`` names whose events the call reads, for the record a delete reads."""

    def __init__(
        self, spec: LatentSpec, *, calls: latent_runner.LatentCalls, ledger: Ledger, clock: Callable[[], float],
        scratch: Path, log: Callable[[str], None], lives: Sequence[str] = (),
        batch_tokens: int = latent_job.DEFAULT_BATCH_TOKENS, records: latent_runner.CallRecords | None = None,
    ) -> None:
        self.spec = spec
        self._calls, self._ledger, self._clock, self._scratch, self._log = calls, ledger, clock, scratch, log
        self._lives, self._batch_tokens = tuple(lives), batch_tokens
        self._records = records or latent_runner.CallRecords(ledger.path.parent / latent_runner.CALLS_DIR)

    def encode(self, items: Sequence[EncodeItem]) -> Encoded:
        data = b"".join(json.dumps({"key": i.key, "role": i.role, "text": i.text,
                                    "context": [list(turn) for turn in i.context]}).encode() + b"\n" for i in items)
        job = latent_runner.JobInput(tuple(item.key for item in items), data, self.spec.as_json(self._batch_tokens),
                                     self._lives)
        outcome = latent_runner.run_job(job, calls=self._calls, ledger=self._ledger, records=self._records,
                                        clock=self._clock, scratch=self._scratch, log=self._log)
        return Encoded(outcome.matrix, outcome.truncated)


@dataclass(frozen=True)
class Plan:
    """What an encode would do: events, distinct contents, cached and to encode, and its estimates."""

    events: int
    unique: int
    cached: int
    todo: int
    tokens: int
    seconds: float
    usd: float
    hold: Bound
    out_bytes: int


def tokens_per_second(spec: LatentSpec) -> float:
    """The plan's estimated rate: the forward stops after the deepest layer read."""
    return FULL_DEPTH_TOKENS_PER_S * NUM_LAYERS / (max(spec.layers) + 1)


TURN_TOKENS: Final = 8  # a turn's header and end, about


def estimate_tokens(item: EncodeItem, spec: LatentSpec) -> int:
    context = sum(math.ceil(len(text) / CHARS_PER_TOKEN) + TURN_TOKENS for _, text in item.context)
    return 1 + min(context, spec.max_context_tokens) + TURN_TOKENS + min(
        math.ceil(len(item.text) / CHARS_PER_TOKEN), spec.max_event_tokens)


def plan_encode(events: Sequence[EventInput], spec: LatentSpec, cache: LatentCache) -> Plan:
    items = items_for(events, spec)
    unique = list({item.key: item for item in items}.values())
    cached = {key for rows in _stored(spec, _by_life(events, items), cache).values() for key in rows}
    todo = [item for item in unique if item.key not in cached]
    tokens = sum(estimate_tokens(item, spec) for item in todo)
    seconds = (LOAD_S + tokens / tokens_per_second(spec)) if todo else 0.0
    bound = latent_runner.hold_bound()
    return Plan(len(events), len(unique), len(unique) - len(todo), len(todo), tokens, seconds,
                seconds / 3600.0 * bound.rate_per_hour, bound, len(events) * spec.d * 4)


def check_fits(plan: Plan, spec: LatentSpec) -> None:
    """One call must fit its tokens in its time at half the estimated rate."""
    room = (modal_job.exec_seconds(latent_job.FUNCTION_TIMEOUT_S) - LOAD_S) * tokens_per_second(spec) / 2
    if plan.tokens > room:
        raise LatentError(f"about {plan.tokens:,} tokens won't surely fit one call ({room:,.0f}); "
                          "split the events file and encode each part")


def plan_lines(plan: Plan, spec: LatentSpec, remaining: float | None) -> list[str]:
    lines = [f"latents plan: {plan.events} events, {plan.unique} distinct, {plan.cached} cached, "
             f"{plan.todo} to encode",
             f"  spec: {spec.encoder} {spec.model}@{spec.revision[:12]} layers {list(spec.layers)} "
             f"pooling {spec.pooling} D={spec.d}",
             f"  writes about {plan.out_bytes / 1e6:,.0f} MB of float32 latents (and as much to the cache)"]
    if spec.encoder == "fake" or not plan.todo:
        return [*lines, "  nothing to hold: " + ("the fake encoder runs here" if plan.todo else "every event is cached")]
    left = "" if remaining is None else f"; {latent_runner.POOL} has ${remaining:.2f} left"
    return [*lines,
            f"  tokens (estimated at {CHARS_PER_TOKEN} characters a token): {plan.tokens:,}",
            f"  time (estimated): {LOAD_S / 60:.0f} min to load, then {plan.tokens / tokens_per_second(spec) / 60:.0f} "
            f"min at about {tokens_per_second(spec):,.0f} tokens/s",
            f"  cost (estimated): ${plan.usd:.2f} on {latent_runner.POOL}; hold ${plan.hold.amount:.2f} "
            f"({latent_job.GPU} for its {latent_job.FUNCTION_TIMEOUT_S / 3600:.1f} h hard timeout at "
            f"${plan.hold.rate_per_hour:.2f}/h){left}"]


class _CachedOnly:
    """The encoder of a run the plan found fully cached: it is never asked for anything."""

    def __init__(self, spec: LatentSpec) -> None:
        self.spec = spec

    def encode(self, items: Sequence[EncodeItem]) -> Matrix:
        raise LatentError("the plan found every event cached; nothing should need encoding")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.mind.core.latents")
    commands = parser.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="read events through Glimmer's hidden states")
    encode.add_argument("--events", type=Path, required=True, help="the events file (JSON lines)")
    encode.add_argument("--out", type=Path, required=True,
                        help=f"a folder in $ANJO_DATA_DIR/sleep/{LATENTS_DIR}/{RUNS_DIR}/; each life's files go in <out>/<life>/")
    encode.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS), help="decoder layers, 0-51")
    encode.add_argument("--plan", action="store_true", help="print the hold and the estimates; touch nothing")
    encode.add_argument("--revision", help=f"Glimmer's pinned commit (default: ${config.REVISION_ENV})")
    encode.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    encode.add_argument("--max-event-tokens", type=int, default=DEFAULT_MAX_EVENT_TOKENS)
    encode.add_argument("--batch-tokens", type=int, default=latent_job.DEFAULT_BATCH_TOKENS)
    encode.add_argument("--fake", action="store_true", help="deterministic stand-in latents, computed here")
    encode.add_argument("--d", type=int, default=64, help="with --fake: the latents' width")
    return parser


def _spec(args: argparse.Namespace) -> LatentSpec:
    if args.fake:
        return fake_spec(args.d)
    revision = args.revision or config.ServeConfig.from_env().revision
    if revision is None:
        raise LatentError(f"name Glimmer's pinned commit with --revision or {config.REVISION_ENV}")
    return glimmer_spec(revision, args.layers, max_context_tokens=args.max_context_tokens,
                        max_event_tokens=args.max_event_tokens)


def latents_root(environ: Mapping[str, str]) -> Path:
    """Where latents live: ``$ANJO_DATA_DIR/sleep/latents``."""
    data = environ.get(DATA_DIR_ENV, "").strip()
    if not data:
        raise LatentError(f"{DATA_DIR_ENV} is not set: latents live in the sleep data folder")
    return Path(data) / "sleep" / LATENTS_DIR


def check_out(out: Path, root: Path) -> Path:
    """``out`` must be a folder directly in ``<root>/runs/``, where a delete finds each life's files."""
    runs = (root / RUNS_DIR).resolve()
    if out.resolve().parent != runs or out.name.startswith("."):
        raise LatentError(f"--out is a folder in {runs}: latents stay in the sleep data folder, where a delete finds them")
    return out


def erase_paths(root: Path, life: str) -> list[Path]:
    """Everything kept of ``life`` under ``root`` (``latents_root``): its cache and its folder in every run."""
    runs = root / RUNS_DIR
    found = [root / CACHE_DIR / check_life(life)]
    return found + ([run / life for run in sorted(runs.iterdir()) if run.is_dir()] if runs.is_dir() else [])


def _remaining(ledger: Ledger | None) -> float | None:
    try:
        return (ledger or default_ledger()).remaining(latent_runner.POOL)
    except BudgetError:
        return None


def _encode_on_modal(
    args: argparse.Namespace, spec: LatentSpec, events: Sequence[EventInput], cache: LatentCache, root: Path,
    calls: latent_runner.LatentCalls, ledger: Ledger, clock: Callable[[], float], out: Callable[[str], None],
) -> EncodeResult:
    """Under MIND's lock, after clearing what a killed encode left, one call under its own hold."""
    records = latent_runner.CallRecords(ledger.path.parent / latent_runner.CALLS_DIR)
    with mind_lock(lock_path(ledger, root)):
        swept = latent_runner.sweep(records, calls, ledger)
        if swept.folders_removed or swept.holds_settled:
            out(f"latents sweep: cancelled {swept.calls_cancelled} leftover call(s), removed {swept.folders_removed} "
                f"job folder(s), settled {swept.holds_settled} stale hold(s) at their full bound")
        encoder = ModalEncoder(spec, calls=calls, ledger=ledger, clock=clock, scratch=root / ".fetch", log=out,
                               lives=sorted({event.life_or_person for event in events}),
                               batch_tokens=args.batch_tokens, records=records)
        return encode_events(events, encoder, cache)


def main(
    argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
    calls: latent_runner.LatentCalls | None = None, ledger: Ledger | None = None,
    clock: Callable[[], float] = time.monotonic, environ: Mapping[str, str] | None = None,
) -> int:
    args = _parser().parse_args(argv)
    root = latents_root(os.environ if environ is None else environ)
    target = check_out(args.out, root)
    spec = _spec(args)
    events = read_events(args.events)
    cache = LatentCache(root / CACHE_DIR)
    plan = plan_encode(events, spec, cache)
    if spec.encoder != "fake":
        check_fits(plan, spec)
    for line in plan_lines(plan, spec, _remaining(ledger) if spec.encoder != "fake" and plan.todo else None):
        out(line)
    if args.plan:
        out("nothing was held, uploaded or called (--plan)")
        return 0
    if plan.todo and not args.fake:
        result = _encode_on_modal(args, spec, events, cache, root, calls or latent_runner.SdkLatentCalls(),
                                  ledger or default_ledger(), clock, out)
    else:
        result = encode_events(events, FakeEncoder(args.d) if plan.todo else _CachedOnly(spec), cache)
    write_latents(target, spec, events, result.matrix, result.truncated)
    out(f"latents: {len(events)} events ({result.encoded} encoded, {result.cached} from the cache, "
        f"{sum(result.truncated)} cut to fit), D={spec.d}, written to {target}/<life>/")
    return 0


if __name__ == "__main__":  # pragma: no cover - the CLI
    sys.exit(main())
