# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.model_api import ChatResult
from anjo.sleep.personas import model as pm
from anjo.sleep.probes import generate as g
from anjo.sleep.types import Evidence

PERSONAS = Path(pm.__file__).parent


def person(name="film_student"):
    return pm.load_person_file(PERSONAS / f"{name}.json")


class FakeChat:
    """Answers each request from its KIND, COUNT and SOURCES lines. Records every call."""

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None):
        user = messages[-1]["content"]
        self.calls.append({"model": model, "system": messages[0]["content"], "user": user,
                           "temperature": temperature})
        kind = user.split("KIND: ")[1].split("\n")[0]
        count = int(user.split("COUNT: ")[1].split("\n")[0])
        sources = json.loads(user.split("SOURCES: ")[1])
        rows: list[dict[str, Any]] = []
        for i in range(count):
            row: dict[str, Any] = {"prompt": f"{kind} message number {i} about my project",
                   "source_ids": [], "expected": None, "options": None, "pick": None}
            if kind == "new_work":
                row["source_ids"] = [sources[i]["id"]]
                row["prompt"] = f"new thing: {sources[i]['text']}"
            if kind == "pushback":
                row["expected"] = "push back on the wrong claim"
            if kind == "taste_ab":
                row.update(options=[f"option x{i}", f"option y{i}"], pick=i % 2)
                row["source_ids"] = [sources[i]["id"]]
            rows.append(row)
        if self.mutate:
            rows = self.mutate(kind, rows)
        text = json.dumps({"probes": rows})
        return ChatResult(text, (), "fake", 1, 1, "stop")


def evidence(eid, kind, text, held_out=False):
    return Evidence(eid, "kevin", date(2026, 10, 28), kind, text, "chat", held_out,
                    datetime(2026, 10, 28, tzinfo=UTC))


def test_probe_generator_emits_all_five_kinds():
    probes = g.generate_probes(person(), 7, chat=FakeChat(), model="m")
    by: dict[str, list[Any]] = {}
    for p in probes:
        by.setdefault(p.kind, []).append(p)
    assert {k: len(v) for k, v in by.items()} == g.COUNTS
    assert len({p.id for p in probes}) == len(probes)
    assert all(p.person_id == "film_student" for p in probes)
    assert all(p.options is not None and p.expected in p.options for p in by["taste_ab"])
    assert all(p.expected for p in by["pushback"])


def test_every_call_runs_at_temperature_zero_with_the_registered_prompt():
    chat = FakeChat()
    g.generate_probes(person(), 1, chat=chat, model="m")
    assert len(chat.calls) == 4
    assert all(c["temperature"] == 0.0 and c["system"] == g.prompt_text() for c in chat.calls)


def test_tone_vocab_pushback_grounded_in_visible_part():
    pf = person()
    chat = FakeChat()
    g.generate_probes(pf, 3, chat=chat, model="m")
    held = [s.text for s in pf.held_out.preferences] + [w.text for w in pf.held_out.new_work]
    for c in chat.calls:
        kind = c["user"].split("KIND: ")[1].split("\n")[0]
        if kind in ("tone_hold", "vocab", "pushback"):
            assert pf.visible.project.title in c["user"]
            assert pf.visible.tone_requests[0].text in c["user"]
            assert not [h for h in held if h in c["user"]]
            assert "SOURCES: []" in c["user"]


def test_taste_probes_use_only_held_out_preferences():
    pf = person()
    probes = g.generate_probes(pf, 3, chat=FakeChat(), model="m")
    held = {t.id: t for t in pf.held_out.taste_ab}
    taste = [p for p in probes if p.kind == "taste_ab"]
    assert len(taste) == 15
    for p in taste:
        t = held[p.source_ids[0]]
        assert (p.prompt, p.options, p.expected) == (t.prompt, t.options, t.options[t.pick])
    visible_blob = json.dumps(pm.to_visible_dict(pf.visible))
    assert not [p for p in taste if p.prompt in visible_blob]


def test_new_work_probes_come_only_from_held_out_items_never_visible():
    pf = person()
    probes = g.generate_probes(pf, 3, chat=FakeChat(), model="m")
    ids = {w.id for w in pf.held_out.new_work}
    nw = [p for p in probes if p.kind == "new_work"]
    assert len(nw) == 25 and len({p.source_ids for p in nw}) == 25
    assert all(p.source_ids[0] in ids for p in nw)


def test_seed_changes_the_drawn_items_and_is_reproducible():
    a = g.generate_probes(person(), 1, chat=FakeChat(), model="m")
    assert a == g.generate_probes(person(), 1, chat=FakeChat(), model="m")
    assert a != g.generate_probes(person(), 2, chat=FakeChat(), model="m")


def test_probe_quoting_a_preference_is_rejected_as_a_cue():
    pf = person()
    quote = pf.visible.preferences[0].text

    def leak(kind, rows):
        if kind == "vocab":
            rows[0]["prompt"] = f"Thinking about this: {quote}"
        return rows

    with pytest.raises(g.ProbeGenError):
        g.generate_probes(pf, 1, chat=FakeChat(leak), model="m")


def test_wrong_count_or_bad_source_is_rejected():
    with pytest.raises(g.ProbeGenError):
        g.generate_probes(person(), 1, chat=FakeChat(lambda k, r: r[:-1]), model="m")

    def swap(kind, rows):
        if kind == "new_work":
            rows[0]["source_ids"] = ["not-a-held-out-id"]
        return rows

    with pytest.raises(g.ProbeGenError):
        g.generate_probes(person(), 1, chat=FakeChat(swap), model="m")


def test_non_json_output_is_rejected():
    def bad(*a, **k):
        return ChatResult("sorry", (), "m", 1, 1, "stop")

    with pytest.raises(g.ProbeGenError):
        g.generate_probes(person(), 1, chat=bad, model="m")


def test_day_probes_skip_taste_without_held_out_preference():
    day = [evidence("e1", "lesson", "Talked about my essay."),
           evidence("e2", "tone_request", "Keep it short.")]
    held = [evidence("e3", "lesson", "an unrelated held-out note", True)]
    skips: list[str] = []
    chat = FakeChat()
    probes = g.generate_day_probes(day, held, 1, chat=chat, model="m", skips=skips)
    assert {p.kind for p in probes} == {"tone_hold", "vocab", "pushback"}
    assert skips == ["taste_ab: no held-out preference"]
    assert len(chat.calls) == 3


def test_day_probes_make_taste_only_from_held_out_preferences():
    day = [evidence("e1", "lesson", "Talked about my essay.")]
    held = [evidence("h1", "preference", "I like short openings.", True)]
    chat = FakeChat()
    probes = g.generate_day_probes(day, held, 1, chat=chat, model="m")
    taste = [p for p in probes if p.kind == "taste_ab"]
    assert len(taste) == 1 and taste[0].source_ids == ("h1",)
    assert taste[0].expected in taste[0].options  # type: ignore[operator]
    assert not any(p.kind == "new_work" for p in probes)
    grounded = [c["user"] for c in chat.calls if "KIND: taste_ab" not in c["user"]]
    assert all("I like short openings." not in u for u in grounded)


def test_test_item_is_marked_and_stable():
    p = g.test_item("kevin", "A new thing from my morning.")
    assert p.kind == "new_work" and g.is_test_item(p)
    assert p == g.test_item("kevin", "A new thing from my morning.")
    assert p.id != g.test_item("kevin", "Another thing.").id


def test_seal_is_deterministic_and_hash_matches_file(tmp_path):
    import hashlib

    probes = g.generate_probes(person(), 1, chat=FakeChat(), model="m")
    h1 = g.seal_probes(probes, tmp_path / "a")
    h2 = g.seal_probes(list(reversed(probes)), tmp_path / "b")
    assert h1 == h2
    f = tmp_path / "a" / "film_student.jsonl"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == h1
    assert g.load_sealed(f) == sorted(probes, key=lambda p: p.id)
    assert g.seal_probes(probes, tmp_path / "a") == h1  # idempotent


def test_seal_refuses_to_overwrite_with_different_content(tmp_path):
    probes = g.generate_probes(person(), 1, chat=FakeChat(), model="m")
    g.seal_probes(probes, tmp_path)
    with pytest.raises(g.SealError):
        g.seal_probes(probes[:-1], tmp_path)


def test_seal_rejects_mixed_people(tmp_path):
    a = g.test_item("kevin", "x")
    b = g.test_item("sam", "y")
    with pytest.raises(g.SealError):
        g.seal_probes([a, b], tmp_path)


def test_sealed_file_hashes_through_the_registration_script(tmp_path):
    from anjo.sleep.prereg import register_night1 as r

    probes = g.generate_probes(person(), 1, chat=FakeChat(), model="m")
    digest = g.seal_probes(probes, tmp_path)
    items = r.compute_items(sealed=(tmp_path / "film_student.jsonl",))
    assert [i.sha256 for i in items if i.kind == "sealed_probes"] == [digest]


def test_day_probes_apply_the_cue_free_check():
    day = [evidence("e1", "tone_request", "Keep it very short please."),
           evidence("e2", "lesson", "Talked about my essay.")]

    def leak(kind, rows):
        if kind == "tone_hold":
            rows[0]["prompt"] = "Please, Keep it very short please. Thanks"
        return rows

    with pytest.raises(g.ProbeGenError):
        g.generate_day_probes(day, [], 1, chat=FakeChat(leak), model="m")
