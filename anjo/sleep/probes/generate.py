"""The probe generator and sealing (Task 2).

One procedure derives every person's tests from what is held out of their training. The rule:
`taste_ab` and `new_work` come only from held-out items. `tone_hold`, `vocab` and `pushback` are
grounded in what is visible. The model client is injected (Spark Open on Baseten in development,
Muse Spark on the Meta API for the final test) and always runs at temperature 0.

`generate_probes` makes one model call per written kind (tone_hold, vocab, pushback, new_work).
`taste_ab` for a persona is copied from its held-out items and needs no call.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from anjo.sleep.model_api import ChatResult
from anjo.sleep.personas.model import PersonFile, to_visible_dict
from anjo.sleep.types import Evidence, Probe, ProbeKind

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "prompts" / "probe_v1.md"
TEMPERATURE = 0.0
TIMEOUT_S = 180.0
TEST_ITEM_PREFIX = "test-item:"

# Registered counts per life (thresholds.json counts_per_life).
COUNTS: dict[ProbeKind, int] = {
    "new_work": 25,
    "vocab": 15,
    "tone_hold": 15,
    "taste_ab": 15,
    "pushback": 10,
}
# Kevin's nightly probes: small, because one day yields little.
DAY_COUNTS: dict[ProbeKind, int] = {"tone_hold": 2, "vocab": 2, "pushback": 1, "taste_ab": 2}
WRITTEN_KINDS: tuple[ProbeKind, ...] = ("tone_hold", "vocab", "pushback")


class ProbeGenError(RuntimeError):
    """The model's output broke a rule, or the input could not support the probes."""


class SealError(RuntimeError):
    """A sealed file already exists with different content."""


class ChatFn(Protocol):
    """`baseten_api.chat` and `meta_api.chat` both fit this."""

    def __call__(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        timeout_s: float,
        *,
        temperature: float | None = None,
    ) -> ChatResult: ...


def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _rank(seed: int, label: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{key}".encode()).hexdigest()


def _call(chat: ChatFn, model: Any, user: str) -> dict[str, Any]:
    messages = [{"role": "system", "content": prompt_text()}, {"role": "user", "content": user}]
    text = chat(model, messages, None, TIMEOUT_S, temperature=TEMPERATURE).text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ProbeGenError("model output is not JSON") from exc
    if not isinstance(data, dict) or not isinstance(data.get("probes"), list):
        raise ProbeGenError("model output has no 'probes' list")
    return data


def _request(kind: str, count: int, visible: Any, sources: Any) -> str:
    return (
        f"KIND: {kind}\nCOUNT: {count}\n"
        f"VISIBLE: {json.dumps(visible, ensure_ascii=False)}\n"
        f"SOURCES: {json.dumps(sources, ensure_ascii=False)}"
    )


def _check_cue_free(prompt: str, banned: list[str]) -> None:
    low = prompt.lower()
    for text in banned:
        if text.strip().lower() in low:
            raise ProbeGenError("a probe quotes a preference or tone request (cue)")


def _text(row: Any, key: str) -> str:
    v = row.get(key) if isinstance(row, dict) else None
    if not isinstance(v, str) or not v.strip():
        raise ProbeGenError(f"probe is missing '{key}'")
    return v.strip()


def _written(
    person_id: str,
    kind: ProbeKind,
    rows: list[Any],
    count: int,
    allowed_sources: set[str],
    banned: list[str],
) -> list[Probe]:
    if len(rows) != count:
        raise ProbeGenError(f"{kind}: expected {count} probes, got {len(rows)}")
    out: list[Probe] = []
    used: set[str] = set()
    for i, row in enumerate(rows):
        prompt = _text(row, "prompt")
        _check_cue_free(prompt, banned)
        sources = tuple(str(s) for s in (row.get("source_ids") or []))
        if kind == "new_work":
            if len(sources) != 1 or sources[0] not in allowed_sources or sources[0] in used:
                raise ProbeGenError("new_work: each probe needs one distinct held-out source")
            used.add(sources[0])
        elif sources:
            raise ProbeGenError(f"{kind}: probes carry no source ids")
        expected = row.get("expected")
        if kind == "pushback":
            expected = _text(row, "expected")
        elif expected is not None:
            raise ProbeGenError(f"{kind}: only pushback has 'expected'")
        out.append(
            Probe(f"{person_id}-{kind}-{i + 1:02d}", person_id, kind, prompt, None, expected, sources)
        )
    return out


def generate_probes(
    person: PersonFile, seed: int, *, chat: ChatFn, model: Any
) -> list[Probe]:
    """Synthetic people: sealed once. Only this and `judge_brief` read `held_out`."""
    v, h = person.visible, person.held_out
    pid = v.person_id
    visible = to_visible_dict(v)
    banned = [s.text for s in (*v.preferences, *v.tone_requests)]
    banned += [s.text for s in h.preferences]
    probes: list[Probe] = []

    for kind in WRITTEN_KINDS:
        data = _call(chat, model, _request(kind, COUNTS[kind], visible, []))
        probes += _written(pid, kind, data["probes"], COUNTS[kind], set(), banned)

    work = sorted(h.new_work, key=lambda w: _rank(seed, "new_work", w.id))[: COUNTS["new_work"]]
    if len(work) < COUNTS["new_work"]:
        raise ProbeGenError("not enough held-out new_work items")
    sources = [{"id": w.id, "text": w.text} for w in work]
    data = _call(chat, model, _request("new_work", len(work), visible, sources))
    new_work = _written(pid, "new_work", data["probes"], len(work), {w.id for w in work}, banned)
    probes += new_work

    items = sorted(h.taste_ab, key=lambda t: _rank(seed, "taste_ab", t.id))[: COUNTS["taste_ab"]]
    if len(items) < COUNTS["taste_ab"]:
        raise ProbeGenError("not enough held-out taste_ab items")
    for i, t in enumerate(items):
        probes.append(
            Probe(f"{pid}-taste_ab-{i + 1:02d}", pid, "taste_ab", t.prompt, t.options,
                  t.options[t.pick], (t.id,))
        )
    return probes


def generate_day_probes(
    trainable_day: list[Evidence],
    held_out_day: list[Evidence],
    seed: int,
    *,
    chat: ChatFn,
    model: Any,
    skips: list[str] | None = None,
) -> list[Probe]:
    """Kevin, nightly, every kind except `new_work`.

    tone_hold, vocab and pushback are grounded in the trainable day. taste_ab comes only from
    held-out preferences; with none, it is skipped and the skip is recorded (log, and `skips`).
    """
    people = {e.person_id for e in (*trainable_day, *held_out_day)}
    if len(people) != 1:
        raise ProbeGenError("a day's evidence must belong to exactly one person")
    pid = next(iter(people))
    statements = [{"id": e.id, "kind": e.kind, "text": e.text} for e in trainable_day]
    banned = [e.text for e in (*trainable_day, *held_out_day) if e.kind in ("preference", "tone_request")]
    probes: list[Probe] = []
    for kind in WRITTEN_KINDS:
        n = DAY_COUNTS[kind]
        data = _call(chat, model, _request(kind, n, statements, []))
        probes += _written(pid, kind, data["probes"], n, set(), banned)

    prefs = sorted(
        (e for e in held_out_day if e.kind == "preference"),
        key=lambda e: _rank(seed, "day_taste", e.id),
    )[: DAY_COUNTS["taste_ab"]]
    if not prefs:
        logger.info("taste_ab skipped for %s: no held-out preference today", pid)
        if skips is not None:
            skips.append("taste_ab: no held-out preference")
        return probes
    sources = [{"id": e.id, "text": e.text} for e in prefs]
    data = _call(chat, model, _request("taste_ab", len(prefs), [], sources))
    rows = data["probes"]
    if len(rows) != len(prefs):
        raise ProbeGenError("taste_ab: wrong number of probes")
    for i, (e, row) in enumerate(zip(prefs, rows, strict=True)):
        opts, pick = row.get("options"), row.get("pick")
        if not (isinstance(opts, list) and len(opts) == 2 and pick in (0, 1)):
            raise ProbeGenError("taste_ab: needs two options and a pick")
        a, b = str(opts[0]), str(opts[1])
        _check_cue_free(_text(row, "prompt") + " " + a + " " + b, banned)
        probes.append(
            Probe(f"{pid}-taste_ab-{i + 1:02d}", pid, "taste_ab", _text(row, "prompt"),
                  (a, b), (a, b)[pick], (e.id,))
        )
    return probes


def test_item(user_id: str, text: str) -> Probe:
    """Seal Kevin's morning `new_work` item. The source id marks it so extraction skips it."""
    if not text.strip():
        raise ProbeGenError("a test item needs text")
    digest = hashlib.sha256(f"{user_id}\n{text}".encode()).hexdigest()[:16]
    return Probe(
        f"{user_id}-new_work-{digest}", user_id, "new_work", text.strip(), None, None,
        (f"{TEST_ITEM_PREFIX}{digest}",),
    )


test_item.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def is_test_item(probe: Probe) -> bool:
    return any(s.startswith(TEST_ITEM_PREFIX) for s in probe.source_ids)


def _line(p: Probe) -> str:
    d = asdict(p)
    d["options"] = list(p.options) if p.options else None
    d["source_ids"] = list(p.source_ids)
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def seal_probes(probes: list[Probe], out_dir: str | Path) -> str:
    """Write one person's probes as sorted-key JSONL in id order; return the file's sha256.

    The file is `<out_dir>/<person_id>.jsonl`. An existing file with other content is refused.
    """
    people = {p.person_id for p in probes}
    if len(people) != 1:
        raise SealError("seal one person's probes at a time")
    ids = [p.id for p in probes]
    if len(set(ids)) != len(ids):
        raise SealError("probe ids must be unique")
    body = "".join(_line(p) + "\n" for p in sorted(probes, key=lambda p: p.id)).encode()
    path = Path(out_dir) / f"{next(iter(people))}.jsonl"
    if path.exists():
        if path.read_bytes() != body:
            raise SealError("a different sealed file already exists")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def load_sealed(path: str | Path) -> list[Probe]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        opts = d["options"]
        out.append(
            Probe(d["id"], d["person_id"], d["kind"], d["prompt"],
                  (opts[0], opts[1]) if opts else None, d["expected"], tuple(d["source_ids"]))
        )
    return out


