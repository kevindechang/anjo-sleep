# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Homework (Task 7): Muse Wonder studies the target her chooser picked, with two tools (search the
web, fetch a public page), and builds an artifact where every claim links a source it fetched and
quoted. Each Wonder call holds its worst case on ``meta_usd`` and is debited from the night's
Wonder allowance, the rolling $3-a-month cap included. Wonder, the web and the search are fakes."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.homework.agent import (
    MAX_INPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    MetaWonder,
    WonderHomework,
    call_worst_usd,
    do_homework,
)
from anjo.sleep.meta_api import PINNED_MODELS
from anjo.sleep.model_api import ModelApiError
from anjo.sleep.night.meter import WonderMeter, WonderSpent
from anjo.sleep.provenance import may_train
from anjo.sleep.tests.homework_fakes import (
    GOOD_CLAIMS,
    GUIDE_QUOTE,
    GUIDE_URL,
    STUDY_QUOTE,
    STUDY_URL,
    FakeSearch,
    FakeWonder,
    final,
    research_script,
    study_web,
    tool,
)
from anjo.sleep.tests.sleep_fakes import ev
from anjo.sleep.types import Evidence, Target

NIGHT = date(2026, 10, 28)
AT = datetime(2026, 10, 29, 0, 0, tzinfo=timezone.utc)
WONDER_ID = PINNED_MODELS["muse-wonder"]
_LINK = re.compile(r"\]\((https?://[^)\s]+)\)")


def _thread(label: str = "sorting film plastics on the new line") -> Target:
    return Target(id="t-sort", person_id="p1", kind="thread", label=label, check_after=None, status="open",
                  revision=2, acted_revision=0, confidence=0.2, last_mentioned=NIGHT)


def _evidence() -> list[Evidence]:
    return [replace(ev("e1", "I can't get the sorter to see black film at all."), target_id="t-sort")]


def _run(wonder: FakeWonder, *, meter: WonderMeter | None = None, action: str = "revisit_thread",
         evidence: list[Evidence] | None = None, target: Target | None = None, **kw: Any) -> Any:
    web = study_web()
    return do_homework(
        "p1", NIGHT, target or _thread(), action, _evidence() if evidence is None else evidence,
        wonder=wonder, search=FakeSearch(), fetcher=web.fetcher(), meter=meter or WonderMeter(12, 3.0), **kw,
    )


def _claim_lines(artifact: str) -> list[str]:
    return [line for line in artifact.splitlines() if line.startswith("- ")]


def test_homework_claims_have_sources() -> None:
    wonder = FakeWonder(research_script())
    homework = _run(wonder)
    lines = _claim_lines(homework.artifact_md)
    assert len(lines) == 2
    verified = {source.url for source in homework.sources}
    assert verified == {STUDY_URL, GUIDE_URL}
    for line in lines:
        links = _LINK.findall(line)
        assert links and set(links) <= verified  # every claim links a fetched, quoted source
    quotes = {source.url: source.quotes for source in homework.sources}
    assert quotes[STUDY_URL] == (STUDY_QUOTE,) and quotes[GUIDE_URL] == (GUIDE_QUOTE,)
    assert homework.calls_used == 3 and homework.target_id == "t-sort"
    assert "## What the trials show" in homework.artifact_md


def test_unverified_source_is_dropped() -> None:
    claims = [
        {"text": "Black film needs light from below.", "sources": [
            {"url": STUDY_URL, "quote": STUDY_QUOTE},
            {"url": "https://example.org/never-fetched", "quote": "a sentence nobody fetched at all"}]},
        {"text": "The guide says belts must stop hourly.", "sources": [
            {"url": GUIDE_URL, "quote": "stop the belt every single hour without fail"}]},
        {"text": "A search snippet said so.", "sources": [
            {"url": "https://example.org/only-in-search", "quote": "A trial of optical sorters on film."}]},
        {"text": "Too short a quote proves nothing.", "sources": [{"url": STUDY_URL, "quote": "film"}]},
        {"text": "No source at all.", "sources": []},
    ]
    homework = _run(FakeWonder(research_script(claims)))
    lines = _claim_lines(homework.artifact_md)
    assert len(lines) == 1 and "Black film needs light from below." in lines[0]
    assert "never-fetched" not in homework.artifact_md and "only-in-search" not in homework.artifact_md
    assert "hourly" not in homework.artifact_md and "No source at all" not in homework.artifact_md
    assert [source.url for source in homework.sources] == [STUDY_URL]
    assert homework.dropped_claims == 4 and homework.dropped_sources == 4


def test_a_quote_matches_across_case_spacing_and_typographic_quotes() -> None:
    web = study_web()
    curly = "https://example.org/curly"
    web.html(curly, "<p>The belt’s speed matters more than the camera — always.</p>")
    claims = [
        {"text": "Lit from below.", "sources": [
            {"url": STUDY_URL, "quote": "optical SORTERS miss\nblack film   unless it is lit"}]},
        {"text": "Speed first.", "sources": [
            {"url": curly, "quote": "the belt's speed matters more than the camera - always"}]},
    ]
    script = [(tool("fetch_page", "c1", url=STUDY_URL), tool("fetch_page", "c2", url=curly)), final(claims)]
    homework = do_homework("p1", NIGHT, _thread(), "revisit_thread", _evidence(), wonder=FakeWonder(script),
                           search=FakeSearch(), fetcher=web.fetcher(), meter=WonderMeter(12, 3.0))
    assert [source.url for source in homework.sources] == [STUDY_URL, curly]


def test_a_redirected_page_verifies_under_both_its_urls() -> None:
    web = study_web()
    web.redirect("https://example.org/old-study", STUDY_URL)
    claims = [{"text": "Lit from below.", "sources": [{"url": "https://example.org/old-study", "quote": STUDY_QUOTE}]}]
    script = [(tool("fetch_page", "c1", url="https://example.org/old-study"),), final(claims)]
    homework = do_homework("p1", NIGHT, _thread(), "revisit_thread", _evidence(), wonder=FakeWonder(script),
                           search=FakeSearch(), fetcher=web.fetcher(), meter=WonderMeter(12, 3.0))
    assert [source.url for source in homework.sources] == [STUDY_URL]


def test_homework_studies_the_chosen_target() -> None:
    wonder = FakeWonder(research_script())
    target = replace(_thread(), id="t-pilot", kind="commitment", label="the film-plastics pilot on Friday",
                     confidence=None)
    item = replace(_evidence()[0], target_id="t-pilot")
    homework = _run(wonder, action="prepare_commitment", target=target, evidence=[item])
    assert homework.target_id == "t-pilot"
    first = json.dumps(wonder.calls[0]["messages"])
    assert "the film-plastics pilot on Friday" in first and "black film at all" in first
    assert "prepare" in first.lower()
    with pytest.raises(ValueError, match="study action"):
        _run(FakeWonder(research_script()), action="hold_note")
    with pytest.raises(ValueError, match="another person"):
        _run(FakeWonder(research_script()), evidence=[replace(item, person_id="p2")])


def test_the_only_tools_are_search_and_fetch() -> None:
    script = [(tool("run_shell", "c1", command="cat /etc/passwd"), tool("fetch_page", "c2", url="file:///etc/passwd")),
              *research_script()]
    wonder = FakeWonder(script)
    homework = _run(wonder)
    offered = {spec["function"]["name"] for spec in wonder.calls[0]["tools"]}
    assert offered == {"web_search", "fetch_page"}
    replies = {m["tool_call_id"]: m["content"] for m in wonder.calls[1]["messages"] if m["role"] == "tool"}
    assert "unknown tool" in replies["c1"] and "refused" in replies["c2"]
    assert homework.refused == 1


def test_every_wonder_call_holds_its_worst_case_on_meta_usd(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    meter = WonderMeter(12, 3.0)
    homework = _run(FakeWonder(research_script()), meter=meter, ledger=ledger)
    assert homework.calls_used == meter.calls == 3
    assert ledger.open_holds() == ()
    assert abs(ledger.used("meta_usd") - meter.usd) < 1e-9 and meter.usd > 0
    reserves = [json.loads(line) for line in (tmp_path / "budget.jsonl").read_text().splitlines()
                if json.loads(line)["op"] == "reserve"]
    assert len(reserves) == 3 and all(r["pools"] == ["meta_usd"] for r in reserves)
    assert all(abs(r["amount"] - call_worst_usd()) < 1e-9 for r in reserves)


def test_a_failed_wonder_call_is_settled_at_its_bound_and_debited(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    meter = WonderMeter(12, 3.0)
    with pytest.raises(ModelApiError):
        _run(FakeWonder(research_script(), fail_at=2), meter=meter, ledger=ledger)
    assert meter.calls == 2 and ledger.open_holds() == ()
    assert abs(ledger.used("meta_usd") - meter.usd) < 1e-9
    assert meter.usd > call_worst_usd()  # the failed call counts at its worst case


def test_a_study_never_spends_past_the_monthly_cap() -> None:
    worst = call_worst_usd()
    with pytest.raises(WonderSpent):  # above the cap: no Wonder call at all
        _run(wonder := FakeWonder(research_script()), meter=WonderMeter(12, 0.0))
    assert wonder.calls == []
    with pytest.raises(WonderSpent):  # less than one call's worst case left
        _run(wonder := FakeWonder(research_script()), meter=WonderMeter(12, worst * 0.9))
    assert wonder.calls == []
    meter = WonderMeter(12, worst * 2.5)
    wonder = FakeWonder(research_script())
    homework = _run(wonder, meter=meter)
    assert len(wonder.calls) == 2 and wonder.calls[-1]["tools"] is None  # the last affordable call writes it
    assert meter.usd <= worst * 2.5 and homework.calls_used == 2


def test_the_call_cap_ends_the_study_with_a_written_artifact() -> None:
    wonder = FakeWonder([(tool("web_search", f"s{i}", query="film sorting"),) for i in range(20)] + [final(GOOD_CLAIMS)])
    homework = _run(wonder, meter=WonderMeter(4, 3.0))
    assert homework.calls_used == 4 and wonder.calls[-1]["tools"] is None


def test_wonder_requests_stay_inside_their_bound() -> None:
    web = study_web()
    big = "<html><head><title>Long</title></head><body><p>" + "sorting words " * 3000 + "</p></body></html>"
    urls = [f"https://example.org/long-{i}" for i in range(12)]
    for url in urls:
        web.html(url, big)
    script: list[Any] = [(tool("fetch_page", f"c{i}", url=url),) for i, url in enumerate(urls)] + [final(GOOD_CLAIMS)]
    wonder = FakeWonder(script)
    do_homework("p1", NIGHT, _thread(), "revisit_thread", _evidence(), wonder=wonder, search=FakeSearch(),
                fetcher=web.fetcher(), meter=WonderMeter(14, 30.0))
    for call in wonder.calls:
        size = len(json.dumps({"messages": call["messages"], "tools": call["tools"]}).encode())
        assert size + 16 * len(call["messages"]) <= MAX_INPUT_TOKENS
        assert call["max_tokens"] == MAX_OUTPUT_TOKENS


def test_meta_wonder_sends_the_pinned_model_and_an_output_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANJO_META_API_KEY", "test-key")
    sent: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": WONDER_ID, "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})

    reply = MetaWonder(transport=httpx.MockTransport(handle)).chat(
        [{"role": "user", "content": "hi"}], None, max_tokens=MAX_OUTPUT_TOKENS, timeout_s=5.0)
    assert reply.model == WONDER_ID
    assert sent[0]["model"] == WONDER_ID and sent[0]["max_completion_tokens"] == MAX_OUTPUT_TOKENS


class _Store:
    def __init__(self, items: list[Evidence]) -> None:
        self.items = items

    def stored(self, until: date) -> list[Evidence]:
        return [item for item in self.items if item.day <= until and not item.held_out]


def test_the_homework_port_stores_the_artifact_as_hosted_evidence() -> None:
    items = [
        *_evidence(),
        replace(ev("e2", "A different topic: my sister's wedding toast."), target_id="t-other"),
        replace(ev("e3", "A held-out line about the sorter.", held_out=True), target_id="t-sort"),
        replace(ev("hw-old", "Old homework text about sorters.", kind="homework",
                   ancestry=frozenset({WONDER_ID})), target_id="t-sort"),
    ]
    web, wonder = study_web(), FakeWonder(research_script())
    port = WonderHomework(wonder=wonder, search=FakeSearch(), fetcher=web.fetcher(), evidence=lambda uid: _Store(items))
    outcome = port.study("p1", _thread(), NIGHT, WonderMeter(12, 3.0), action="revisit_thread", at=AT)
    prompt = json.dumps(wonder.calls[0]["messages"])
    assert "black film at all" in prompt
    assert "wedding" not in prompt and "held-out line" not in prompt and "Old homework" not in prompt
    [item] = outcome.evidence
    assert item.kind == "homework" and item.target_id == "t-sort" and item.model == WONDER_ID
    assert WONDER_ID in (item.ancestry or ()) and not may_train(item.ancestry, hosted_in_training=False)
    assert item.created_at == AT and item.day == NIGHT and item.source == STUDY_URL
    assert outcome.result is not None
    assert outcome.result.basis == "research" and outcome.result.verified_sources == 2
    assert outcome.result.homework_id == item.id and outcome.result.created_at == AT
    assert outcome.receipt["verified_sources"] == 2 and outcome.receipt["wonder_calls"] == 3


def test_a_study_with_nothing_verified_records_a_sourceless_result() -> None:
    claims = [{"text": "Unsupported.", "sources": [{"url": STUDY_URL, "quote": "words that are not on the page"}]}]
    web, wonder = study_web(), FakeWonder(research_script(claims))
    port = WonderHomework(wonder=wonder, search=FakeSearch(), fetcher=web.fetcher(), evidence=lambda uid: _Store(_evidence()))
    outcome = port.study("p1", _thread(), NIGHT, WonderMeter(12, 3.0), action="revisit_thread", at=AT)
    assert outcome.evidence == ()
    assert outcome.result is not None and outcome.result.verified_sources == 0


def test_unreadable_final_answer_is_a_study_with_no_claims() -> None:
    homework = _run(FakeWonder([(tool("fetch_page", "c1", url=STUDY_URL),), "I could not decide."]))
    assert homework.sources == () and _claim_lines(homework.artifact_md) == []
