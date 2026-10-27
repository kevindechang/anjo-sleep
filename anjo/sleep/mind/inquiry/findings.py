"""What a finding may become outside the workspace: a ``share`` result, or homework (Task 17,
night step 10). Pure: the night glue passes the cited items, targets, results and ``as_of``.

**Share needs an unconsumed anchor.** A suspicion or resolution raises ``share`` only with one of:

1. a verified research source it cites, one research returned for its question that no consumed
   result has used: it enters as ``Result(basis="research")``;
2. an open commitment among its question's targets, due within ``EVIDENCE_WINDOW`` of ``as_of`` and
   not yet noted, while it cites at least ``MIN_USER_CITES`` of the person's own items: it enters
   as ``Result(basis="evidence")``, which ``snapshot.build`` admits only under this same rule.

**A finding without a source never trains.** Only a resolution citing a verified research source
becomes the night's homework artifact (``artifact``), an evidence item whose ancestry carries the
thinker and every cited item's ancestry, so ``provenance.may_train`` still decides. A suspicion,
a retraction and a resolution resting on the person's items alone give nothing to train on.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Final

from anjo.sleep.mind.inquiry.kernel import research_sources
from anjo.sleep.mind.inquiry.types import Finding, Workspace
from anjo.sleep.mind.snapshot import EVIDENCE_WINDOW, MIN_USER_CITES
from anjo.sleep.types import Evidence, EvidenceKind, Result, Target

# Task 3's ``PERSON_KINDS`` (``anjo.sleep.evidence.targets``), restated so this pure module loads no
# file code; a test pins the two together.
PERSON_KINDS: Final[frozenset[EvidenceKind]] = frozenset({"lesson", "correction", "preference", "tone_request"})
SHARED_KINDS: Final = frozenset({"suspicion", "resolution"})


def _finding(ws: Workspace, question_id: str) -> Finding | None:
    return ws.finding_of(question_id)


def _due_commitment(question_targets: Sequence[str], targets: Sequence[Target], as_of: datetime) -> Target | None:
    held = frozenset(question_targets)
    for target in targets:
        if target.id not in held or target.kind != "commitment" or target.status != "open":
            continue
        if target.check_after is None or target.due_noted:
            continue
        if timedelta(0) <= target.check_after - as_of <= EVIDENCE_WINDOW:
            return target
    return None


def _open_target(question_targets: Sequence[str], targets: Sequence[Target]) -> Target | None:
    by_id = {t.id: t for t in targets}
    return next((by_id[t] for t in question_targets if t in by_id and by_id[t].status == "open"), None)


def share_result(
    ws: Workspace,
    question_id: str,
    *,
    cited: Sequence[Evidence],
    targets: Sequence[Target],
    results: Sequence[Result],
    as_of: datetime,
) -> Result | None:
    """The result the question's finding raises ``share`` with, or ``None`` without an unconsumed
    anchor. ``cited`` is the finding's cited items, fetched by id; ``results`` the target log's."""
    finding = _finding(ws, question_id)
    question = ws.question(question_id)
    if finding is None or question is None or finding.kind not in SHARED_KINDS:
        return None
    result_id = f"{finding.id}.{finding.kind}"
    if any(r.id == result_id for r in results):
        return None
    consumed = frozenset(r.homework_id for r in results if r.consumed and r.homework_id is not None)
    sources = [i for i in finding.cites if i in research_sources(ws, question_id) and i not in consumed]
    target = _open_target(question.target_ids, targets)
    if sources and target is not None:
        return Result(
            id=result_id, target_id=target.id, homework_id=sources[0], verified_sources=len(sources),
            created_at=as_of, consumed=False, basis="research",
        )
    commitment = _due_commitment(question.target_ids, targets, as_of)
    cites = frozenset(finding.cites)
    own = sum(1 for item in cited if item.id in cites and item.kind in PERSON_KINDS and not item.held_out)
    if commitment is None or own < MIN_USER_CITES:
        return None
    return Result(
        id=result_id, target_id=commitment.id, homework_id=None, verified_sources=0, created_at=as_of,
        consumed=False, basis="evidence", user_cites=own,
    )


def artifact(
    ws: Workspace,
    question_id: str,
    *,
    cited: Sequence[Evidence],
    person_id: str,
    night: date,
    at: datetime,
    model: str,
) -> Evidence | None:
    """The homework artifact a resolution with a verified research source becomes, or ``None``.
    ``model`` is the thinker that wrote the finding."""
    finding = _finding(ws, question_id)
    question = ws.question(question_id)
    if finding is None or question is None or finding.kind != "resolution":
        return None
    research = research_sources(ws, question_id)
    sources = [item for item in cited if item.id in finding.cites and item.id in research]
    if not sources:
        return None
    ancestry: frozenset[str] | None = frozenset({model})
    for item in cited:
        if item.id in finding.cites:
            ancestry = None if ancestry is None or item.ancestry is None else ancestry | item.ancestry
    links = "\n".join(f"- {item.source}" for item in sources)
    return Evidence(
        id=f"mind-{finding.id}", person_id=person_id, day=night, kind="homework",
        text=f"{finding.statement}\n\nSources:\n{links}", source=sources[0].source, held_out=False,
        created_at=at, model=model, ancestry=ancestry, target_id=question.target_ids[0],
    )
