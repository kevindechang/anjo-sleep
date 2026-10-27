# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""What MIND's findings may become (Task 17): a share result only on an unconsumed anchor,
homework only from a resolution with a verified source; plus her workspace log on disk and the
evidence store's ``get``."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from anjo.sleep.evidence import targets as task3_targets
from anjo.sleep.evidence.jsonl import NO_GUARD, PLAIN
from anjo.sleep.evidence.store import JsonlEvidenceStore
from anjo.sleep.mind import snapshot
from anjo.sleep.mind.inquiry import findings, kernel
from anjo.sleep.mind.inquiry import types as it
from anjo.sleep.mind.inquiry.ops import ConclusionKind
from anjo.sleep.mind.inquiry.log import EMPTY_DIGEST, JsonlWorkspaceLog, WorkspaceLogError
from anjo.sleep.provenance import GLIMMER, may_train
from anjo.sleep.tests.inquiry_fakes import PERSON, Ws, commitment, day, ev, hypothesize, night, notice_op, noon, thread
from anjo.sleep.types import Evidence, Result

SCENE = "t-scene4"
E1 = ev("e-d2-1", "Scene 4 wants an 85 mm medium shot.", n=2, target=SCENE)
E2 = ev("e-d5-1", "Scene 4 is in their 3 m kitchen.", n=5, target=SCENE)
E3 = ev("e-d3-1", "They might shoot scene 4 from the courtyard.", n=3, target=SCENE)
SOURCE = ev("e-web-1", "An 85 mm medium shot needs about 3 m.", n=5, kind="homework", model=GLIMMER,
            ancestry=frozenset({GLIMMER}), source="https://example.org/lens-distance")
DUE = commitment(SCENE, "Scene 4 shoot", due=noon(8), n=2)
AS_OF = night(6)


def _resolved(*, cites: tuple[str, ...], kind: ConclusionKind = "resolution", research: bool = False) -> Ws:
    """q1 on scene 4, its leader at 0.96 on a hit found after hypothesizing, then concluded."""
    w = Ws()
    w.start(5)
    w.arrive(E1, E2)
    w.ok(notice_op([E1.id, E2.id], targets=(SCENE,)))
    w.ok(hypothesize("q1", 0.6, 0.2))
    hit = SOURCE if research else E3
    w.found("q1", hit, tool="research" if research else "evidence_search")
    w.ok(it.AssessOp("q1", hit.id, likelihoods=dict(zip(w.ids("q1"), (4.0, 0.25, 0.25), strict=True))))
    w.ok(it.ConcludeOp("q1", w.ids("q1")[0], kind, "The kitchen can't hold an 85 mm medium shot", cites))
    return w


def test_person_kinds_match_task3() -> None:
    assert findings.PERSON_KINDS == task3_targets.PERSON_KINDS


# ── Share anchors ─────────────────────────────────────────────────────────────


def test_finding_anchor_rules_for_share() -> None:
    cited = (E1, E2, E3)
    w = _resolved(cites=(E1.id, E2.id, E3.id))
    result = findings.share_result(w.ws, "q1", cited=cited, targets=(DUE,), results=(), as_of=AS_OF)
    assert result == Result(id="f4.resolution", target_id=SCENE, homework_id=None, verified_sources=0, created_at=AS_OF,
                            consumed=False, basis="evidence", user_cites=3)
    built = snapshot.build([DUE], [result], AS_OF, None)
    assert built.results == (result,)

    one_cite = _resolved(cites=(E1.id,))
    assert findings.share_result(one_cite.ws, "q1", cited=(E1,), targets=(DUE,), results=(), as_of=AS_OF) is None
    held = replace(E2, held_out=True)
    two = _resolved(cites=(E1.id, E2.id))
    assert findings.share_result(two.ws, "q1", cited=(E1, held), targets=(DUE,), results=(), as_of=AS_OF) is None
    for target in (replace(DUE, due_noted=True), replace(DUE, check_after=noon(20)), thread(SCENE, "scene 4"),
                   replace(DUE, status="closed")):
        assert findings.share_result(w.ws, "q1", cited=cited, targets=(target,), results=(), as_of=AS_OF) is None
    assert findings.share_result(w.ws, "q1", cited=cited, targets=(DUE,), results=(result,), as_of=AS_OF) is None


def test_research_source_is_the_first_anchor_until_consumed() -> None:
    w = _resolved(cites=(SOURCE.id, E1.id), research=True)
    open_thread = thread(SCENE, "scene 4")
    result = findings.share_result(w.ws, "q1", cited=(SOURCE, E1), targets=(open_thread,), results=(), as_of=AS_OF)
    assert result is not None and (result.basis, result.homework_id, result.verified_sources) == ("research", SOURCE.id, 1)
    assert snapshot.build([open_thread], [result], AS_OF, None).results == (result,)

    consumed = Result(id="r-old", target_id=SCENE, homework_id=SOURCE.id, verified_sources=1, created_at=night(4),
                      consumed=True)
    assert findings.share_result(w.ws, "q1", cited=(SOURCE, E1), targets=(open_thread,), results=(consumed,),
                                 as_of=AS_OF) is None


def test_only_suspicions_and_resolutions_raise_share() -> None:
    suspicion = _resolved(cites=(E1.id, E2.id), kind="suspicion")
    assert findings.share_result(suspicion.ws, "q1", cited=(E1, E2), targets=(DUE,), results=(), as_of=AS_OF) is not None
    suspicion.ok(it.RetractOp("q1", "the courtyard plan settles it", (E3.id,)))
    assert suspicion.ws.finding_of("q1").kind == "retraction"  # type: ignore[union-attr]
    assert findings.share_result(suspicion.ws, "q1", cited=(E1, E2), targets=(DUE,), results=(), as_of=AS_OF) is None
    assert findings.share_result(Ws().ws, "q9", cited=(), targets=(DUE,), results=(), as_of=AS_OF) is None


# ── Homework ──────────────────────────────────────────────────────────────────


def test_sourceless_finding_never_trainable() -> None:
    def made(w: Ws, cited: tuple[Evidence, ...]) -> Evidence | None:
        return findings.artifact(w.ws, "q1", cited=cited, person_id=PERSON, night=day(6), at=AS_OF, model=GLIMMER)

    assert made(_resolved(cites=(E1.id, E2.id, E3.id)), (E1, E2, E3)) is None
    assert made(_resolved(cites=(SOURCE.id,), kind="suspicion", research=True), (SOURCE,)) is None

    item = made(_resolved(cites=(SOURCE.id, E1.id), research=True), (SOURCE, E1))
    assert item is not None and (item.kind, item.target_id, item.source, item.held_out) == (
        "homework", SCENE, SOURCE.source, False)
    assert item.ancestry == frozenset({GLIMMER}) and SOURCE.source in item.text
    assert may_train(item.ancestry, hosted_in_training=False)

    unknown = replace(E1, ancestry=None)
    blocked = made(_resolved(cites=(SOURCE.id, E1.id), research=True), (SOURCE, unknown))
    assert blocked is not None and blocked.ancestry is None and not may_train(blocked.ancestry, hosted_in_training=True)
    hosted = made(_resolved(cites=(SOURCE.id,), research=True), (replace(SOURCE, ancestry=frozenset({"wonder-x"})),))
    assert hosted is not None and not may_train(hosted.ancestry, hosted_in_training=False)


# ── The log on disk, and the store's get ──────────────────────────────────────


def test_jsonl_workspace_log_round_trips_and_replays(tmp_path: Path) -> None:
    w = _resolved(cites=(E1.id, E2.id, E3.id))
    log = JsonlWorkspaceLog(tmp_path / "workspace.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    assert log.events() == () and log.digest() == EMPTY_DIGEST == log.replay_digest()
    for event in w.events:
        log.append(event)
    assert log.events() == tuple(w.events) and log.workspace() == w.ws
    assert log.digest() == log.replay_digest() == kernel.digest(w.ws)
    with pytest.raises(WorkspaceLogError):
        log.append(w.events[-1])
    with pytest.raises(WorkspaceLogError):
        log.append(replace(w.events[-1], step=replace(w.events[-1].step, seq=len(w.events))))
    exported = log.export()["workspace"]
    assert isinstance(exported, dict) and len(exported["events"]) == len(w.events)
    assert exported["state"]["findings"][0]["id"] == "f4"


def test_store_get_is_time_masked_and_keeps_the_order_asked(tmp_path: Path) -> None:
    store = JsonlEvidenceStore(tmp_path / "evidence.jsonl", PERSON, codec=PLAIN, guard=NO_GUARD)
    held = ev("e-held", "a sealed probe", n=3, held_out=True)
    store.add([E1, E3, E2, held])
    assert store.get([E2.id, E1.id, "e-missing", held.id], night(5)) == (E2, E1)
    assert store.get([E2.id, E1.id, E3.id], night(4)) == (E1, E3)
