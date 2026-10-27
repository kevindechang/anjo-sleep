# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The six arms (Task 2, spec 5.1): who gets which weights and which context."""
from __future__ import annotations

import ast
import inspect
from typing import Any, cast
from pathlib import Path

import pytest

from anjo.sleep.context.recall_bm25 import Bm25Recall
from anjo.sleep.probes import arms
from anjo.sleep.probes.arms import REPLY_RESERVE_TOKENS, Arm, ArmDeps, ArmReply, DonorError, reply
from anjo.sleep.tests.sleep_fakes import (
    DAY, FakeContexts, FakeLogprobs, FakeModel, FakeRegistry, LeakyStore, StoreSpy, ckpt, ev, probe, taste,
)
from anjo.sleep.tests.test_context_builder import Briefs, Memory, Prefs
from anjo.sleep.context.builder import ComposedContextBuilder, RecallSettings

DONORS = {"p1": "p2", "p2": "p1"}


def deps(*, store=None, contexts=None, model=None, registry=None, logprobs=None, window=131_072):
    store = store or StoreSpy([ev("e1", "first lesson", minutes=1), ev("e2", "second lesson", minutes=2),
                               ev("hw", "homework found X", kind="homework", minutes=3)])
    return ArmDeps(
        model=model or FakeModel(),
        adapters=registry or FakeRegistry({"p1": ckpt("p1"), "p2": ckpt("p2")}),
        contexts=contexts or FakeContexts(),
        evidence=lambda user_id: store,
        logprobs=logprobs or FakeLogprobs(lambda adapter, messages: "A"),
        donors=DONORS, until=DAY, window_tokens=window,
    )


def calls(d: ArmDeps) -> list[dict[str, Any]]:
    assert isinstance(d.model, FakeModel)
    return d.model.calls


def lp_calls(d: ArmDeps) -> list[dict[str, Any]]:
    assert isinstance(d.logprobs, FakeLogprobs)
    return d.logprobs.calls


ALL_ARMS: tuple[Arm, ...] = ("1", "2C", "2F", "3", "3x", "4")


def test_arm1_is_base_with_no_context():
    d = deps()
    out = reply("1", "p1", probe(), deps=d)
    call = calls(d)[0]
    assert call["adapter"] is None
    assert [m["role"] for m in call["messages"]] == ["user"]
    assert isinstance(out, ArmReply) and out.dropped_ids == ()


def test_arm3_is_the_personal_adapter_with_no_context():
    d = deps()
    reply("3", "p1", probe(), deps=d)
    call = calls(d)[0]
    assert call["adapter"].id == "p1-n1" and [m["role"] for m in call["messages"]] == ["user"]


def test_arm3_falls_back_to_base_when_no_night_was_promoted():
    d = deps(registry=FakeRegistry({}))
    reply("3", "p1", probe(), deps=d)
    assert calls(d)[0]["adapter"] is None


def test_arm2c_and_arm4_context_identical():
    contexts = FakeContexts()
    d = deps(contexts=contexts)
    reply("2C", "p1", probe(), deps=d)
    reply("4", "p1", probe(), deps=d)
    first, second = (c["messages"] for c in calls(d))
    assert first == second and first[0]["role"] == "system"
    assert contexts.builds == 1
    adapters = [c["adapter"] for c in calls(d)]
    assert adapters[0] is None and adapters[1].id == "p1-n1"


def test_recall_identical_for_2c_and_4():
    store = StoreSpy([ev("a", "long take on the ferry deck", minutes=1), ev("b", "sound mix", minutes=2)])
    builder = ComposedContextBuilder(
        memory=Memory(), briefs=Briefs(), preferences=Prefs(),
        recall=Bm25Recall(lambda u: store, until=DAY), settings=RecallSettings(k=5, budget_tokens=2000),
    )
    d = deps(contexts=builder)
    q = probe(prompt="long take on the ferry deck?")
    reply("2C", "p1", q, deps=d)
    reply("4", "p1", q, deps=d)
    first, second = (c["messages"][0]["content"] for c in calls(d))
    assert first == second and "[a] long take" in first


def test_same_context_object_survives_a_builder_that_changes_between_calls():
    contexts = FakeContexts()
    d = deps(contexts=contexts)
    reply("4", "p1", probe("q1"), deps=d)
    reply("2C", "p1", probe("q2"), deps=d)
    assert contexts.builds == 2  # one build per probe, not one per arm


def test_arm2f_holds_arm2c_plus_every_stored_item():
    d = deps()
    reply("2C", "p1", probe(), deps=d)
    out = reply("2F", "p1", probe(), deps=d)
    c_ctx = calls(d)[0]["messages"][0]["content"]
    f_ctx = calls(d)[1]["messages"][0]["content"]
    assert f_ctx.startswith(c_ctx)
    for text in ("first lesson", "second lesson", "homework found X"):
        assert text in f_ctx
    assert out.dropped_ids == ()
    assert calls(d)[1]["adapter"] is None


def test_arm2f_never_holds_held_out_items():
    store = LeakyStore([ev("e1", "visible"), ev("h", "HELD OUT TEXT", held_out=True)])
    d = deps(store=store)
    reply("2F", "p1", probe(), deps=d)
    assert "HELD OUT TEXT" not in calls(d)[0]["messages"][0]["content"]


def test_arm2f_truncation_keeps_newest_and_records_dropped():
    items = [ev(f"e{n}", "lesson " + "word " * 50, minutes=n) for n in range(10)]
    d = deps(store=StoreSpy(items), window=REPLY_RESERVE_TOKENS + 500)
    out = reply("2F", "p1", probe(), deps=d)
    ctx = calls(d)[0]["messages"][0]["content"]
    kept = [e.id for e in items if f"[{e.id}]" in ctx]
    assert kept and out.dropped_ids
    assert set(kept) | set(out.dropped_ids) == {e.id for e in items}
    assert not set(kept) & set(out.dropped_ids)
    assert max(int(i[1:]) for i in out.dropped_ids) < min(int(i[1:]) for i in kept)  # newest kept
    assert list(out.dropped_ids) == sorted(out.dropped_ids, key=lambda i: int(i[1:]))


def test_arm3x_uses_only_the_registered_donor():
    reg = FakeRegistry({"p1": ckpt("p1"), "p2": ckpt("p2"), "p9": ckpt("p9")})
    d = deps(registry=reg)
    reply("3x", "p1", probe(), deps=d)
    assert calls(d)[0]["adapter"].person_id == "p2"
    assert [m["role"] for m in calls(d)[0]["messages"]] == ["user"]
    with pytest.raises(DonorError, match="registered"):
        reply("3x", "p1", probe(), deps=d, donor="p9")
    assert "p9" not in reg.asked
    assert len(calls(d)) == 1


def test_arm3x_refuses_a_person_with_no_registered_donor():
    d = deps()
    with pytest.raises(DonorError, match="no registered donor"):
        reply("3x", "p5", probe(person="p5"), deps=d)


def test_arm3x_refuses_a_registry_that_hands_back_another_persons_adapter():
    d = deps(registry=FakeRegistry({"p2": ckpt("p9")}))
    with pytest.raises(DonorError, match="registered"):
        reply("3x", "p1", probe(), deps=d)


def test_arm3x_refuses_an_unpromoted_donor():
    d = deps(registry=FakeRegistry({"p1": ckpt("p1")}))
    with pytest.raises(DonorError, match="never promoted"):
        reply("3x", "p1", probe(), deps=d)


def test_a_probe_for_another_person_is_refused():
    with pytest.raises(ValueError, match="another person"):
        reply("1", "p1", probe(person="p2"), deps=deps())


def test_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="arm"):
        reply(cast(Any, "9"), "p1", probe(), deps=deps())


def test_arms_reply_saves_nothing(tmp_path: Path):
    from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
    from anjo.sleep.evidence.store import JsonlEvidenceStore

    path = tmp_path / "evidence.jsonl"
    real = JsonlEvidenceStore(path, "p1", codec=PLAIN, guard=NO_GUARD)
    real.add([ev("e1", "a lesson"), ev("e2", "held", held_out=True)])
    before = path.read_bytes()
    d = ArmDeps(
        model=FakeModel(), adapters=FakeRegistry({"p1": ckpt("p1"), "p2": ckpt("p2")}),
        contexts=FakeContexts(), evidence=lambda u: real,
        logprobs=FakeLogprobs(lambda a, m: "A"), donors=DONORS, until=DAY,
    )
    for arm in ALL_ARMS:
        reply(arm, "p1", probe(), deps=d)
        reply(arm, "p1", taste(), deps=d)
    assert path.read_bytes() == before
    store = StoreSpy([ev("e1")])
    for arm in ("2F", "4"):
        reply(arm, "p1", probe(), deps=deps(store=store))
    assert store.writes == 0


def test_judge_brief_never_reaches_any_arm():
    from anjo.sleep.probes.judge_brief import judge_brief
    from anjo.sleep.personas.model import load_person_file

    secret = "HELD-OUT-SECRET-PREFERENCE"
    pf = load_person_file(Path(inspect.getfile(arms)).parents[1] / "personas" / "film_student.json")
    held = pf.held_out.preferences[0]
    import dataclasses

    swapped = dataclasses.replace(
        pf, held_out=dataclasses.replace(
            pf.held_out, preferences=(dataclasses.replace(held, text=secret), *pf.held_out.preferences[1:]),
        ),
    )
    assert secret in judge_brief("film_student", {"film_student": swapped})

    d = deps(logprobs=FakeLogprobs(lambda a, m: "A"))
    for arm in ALL_ARMS:
        reply(arm, "p1", probe(), deps=d)
        reply(arm, "p1", taste(), deps=d)
    seen = repr(calls(d)) + repr(lp_calls(d))
    assert secret not in seen
    tree = ast.parse(Path(inspect.getfile(arms)).read_text())
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("judge_brief" in m or m.endswith("score") for m in imported)


def test_taste_probe_is_answered_by_forced_choice_not_sampling():
    d = deps(logprobs=FakeLogprobs(lambda adapter, messages: "A"))
    out = reply("4", "p1", taste(), deps=d)
    assert calls(d) == []
    assert out.choice is not None
    assert len(lp_calls(d)) == 4  # two orders, two labels


def test_reply_seed_is_stable_per_probe_arm_and_run_and_differs_across_them():
    d = deps()
    reply("1", "p1", probe("q1"), deps=d)
    reply("1", "p1", probe("q1"), deps=d)
    reply("3", "p1", probe("q1"), deps=d)
    reply("1", "p1", probe("q2"), deps=d)
    seeds = [c["seed"] for c in calls(d)]
    assert seeds[0] == seeds[1] and len(set(seeds)) == 3 and all(isinstance(s, int) for s in seeds)
    other = deps()
    other = ArmDeps(**{**other.__dict__, "run_seed": 9, "context_cache": {}})
    reply("1", "p1", probe("q1"), deps=other)
    assert calls(other)[0]["seed"] != seeds[0]
