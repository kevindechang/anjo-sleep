# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The live page's engine (Task 13a, minimal step): limits, visits, blind pairs, starter pairs and
the starter scenes. The routes and the crisis wiring are tested in ``tests/app/test_demo_page.py``."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.demo import words
from anjo.sleep.demo.limits import CounterError, JsonCounterStore, Limiter, Limits, Refused
from anjo.sleep.demo.page import COPY_SLOT, page_html, render
from anjo.sleep.demo.pairs import CHOICES, PairRegistry, SideFacts, Vote
from anjo.sleep.demo.service import Answer, Asleep, DemoConfig, DemoService, Fixed, PairView
from anjo.sleep.demo.starters import LABEL, Side, StarterSet, from_json, generate, load, to_json
from anjo.sleep.demo.visits import NON_GLOBAL, VISIT_TTL_S, client_key, mint_visit, visit_nonce
from anjo.sleep.probes.decontam import OverlapDecontaminator
from anjo.sleep.sets.loader import load_collapse, load_general
from anjo.sleep.types import CheckpointRef, Probe

NOW = datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)
KEY = b"k" * 32
NONCE = "n" * 22


class MemoryStore:
    """A ``CounterStore`` in memory that can be told to fail."""

    def __init__(self) -> None:
        self.saved: dict[str, Any] | None = None
        self.fail_load = False
        self.fail_save = False

    def load(self) -> Mapping[str, Any] | None:
        if self.fail_load:
            raise CounterError("unreadable")
        return self.saved

    def save(self, counters: Mapping[str, Any]) -> None:
        if self.fail_save:
            raise CounterError("unsaved")
        self.saved = json.loads(json.dumps(dict(counters)))


def _refused(call: Any, *args: Any) -> Refused:
    with pytest.raises(Refused) as caught:
        call(*args)
    refused: Refused = caught.value
    return refused


# ── the limits ────────────────────────────────────────────────────────────────


def test_limiter_caps_calls_per_client_per_minute_and_counts_every_side() -> None:
    limiter = Limiter(MemoryStore(), Limits(calls_per_client_per_minute=4))
    for n in range(2):
        hold = limiter.admit("a", f"visit-{n}", 2, NOW)
        limiter.settle(hold, 100, NOW)
    refused = _refused(limiter.admit, "a", "visit-9", 2, NOW)
    assert (refused.status, refused.reason) == (429, "slow_down")
    assert limiter.admit("b", "visit-b", 2, NOW).calls == 2  # another client has its own minute
    assert limiter.admit("a", "visit-9", 2, NOW + timedelta(seconds=61)).calls == 2  # the minute passed


def test_limiter_caps_visits_per_client_and_per_day() -> None:
    limiter = Limiter(MemoryStore(), Limits(visits_per_client_per_day=2, visits_per_day=3))
    limiter.visit("a", NOW)
    limiter.visit("a", NOW)
    assert _refused(limiter.visit, "a", NOW).reason == "visits_spent"
    limiter.visit("b", NOW)
    assert _refused(limiter.visit, "c", NOW).reason == "resting_today"
    limiter.visit("a", NOW + timedelta(days=1))  # a new UTC day


def test_limiter_caps_requests_per_minute_and_asks_per_visit() -> None:
    limiter = Limiter(MemoryStore(), Limits(requests_per_client_per_minute=3, asks_per_visit=2))
    for _ in range(3):
        limiter.request("a", NOW)
    assert _refused(limiter.request, "a", NOW).reason == "slow_down"
    for _ in range(2):
        limiter.settle(limiter.admit("b", "visit", 1, NOW), 10, NOW)
    assert _refused(limiter.admit, "b", "visit", 1, NOW).reason == "visit_spent"
    asleep = limiter.admit("c", "other", 1, NOW)
    limiter.release(asleep, NOW)  # no call ran: the question goes back to the visit
    limiter.settle(limiter.admit("c", "other", 1, NOW), 10, NOW)
    limiter.settle(limiter.admit("c", "other", 1, NOW), 10, NOW)


def test_limiter_caps_calls_in_flight() -> None:
    limiter = Limiter(MemoryStore(), Limits(max_inflight_calls=4))
    first = limiter.admit("a", "v1", 2, NOW)
    limiter.admit("b", "v2", 2, NOW)
    refused = _refused(limiter.admit, "c", "v3", 2, NOW)
    assert (refused.status, refused.reason) == (503, "busy")
    limiter.settle(first, 50, NOW)
    assert limiter.admit("c", "v3", 2, NOW).calls == 2


def test_limiter_admits_on_the_worst_case_and_settles_to_use() -> None:
    store = MemoryStore()
    limiter = Limiter(store, Limits(tokens_per_call=1_000, tokens_per_day=3_000, calls_per_day=50))
    hold = limiter.admit("a", "v1", 2, NOW)
    assert store.saved is not None and store.saved["tokens"] == 2_000
    assert _refused(limiter.admit, "b", "v2", 2, NOW).reason == "resting_today"  # 4,000 > 3,000
    limiter.settle(hold, 300, NOW)
    assert store.saved["tokens"] == 300 and store.saved["calls"] == 2
    asleep = limiter.admit("b", "v2", 2, NOW)
    limiter.release(asleep, NOW)
    assert store.saved["tokens"] == 300 and store.saved["calls"] == 2


def test_limiter_caps_calls_per_day() -> None:
    limiter = Limiter(MemoryStore(), Limits(calls_per_day=4, calls_per_client_per_minute=100))
    for n in range(2):
        limiter.settle(limiter.admit(f"c{n}", f"v{n}", 2, NOW), 10, NOW)
    assert _refused(limiter.admit, "c9", "v9", 2, NOW).reason == "resting_today"


def test_limiter_fails_closed() -> None:
    store = MemoryStore()
    limiter = Limiter(store)
    limiter.visit("a", NOW)
    saved = dict(store.saved or {})

    store.fail_save = True
    refused = _refused(limiter.visit, "a", NOW)
    assert (refused.status, refused.reason) == (503, "resting")
    assert _refused(limiter.admit, "a", "v", 2, NOW).reason == "resting"
    assert store.saved == saved  # memory never ran ahead of what a restart would load
    store.fail_save = False
    hold = limiter.admit("a", "v", 2, NOW)
    store.fail_save = True
    limiter.settle(hold, 10, NOW)  # a refund that can't be saved is dropped, never raised
    assert store.saved is not None and store.saved["tokens"] == 2 * Limits().tokens_per_call

    restarted = Limiter(store)
    store.fail_load = True
    assert _refused(restarted.visit, "b", NOW).status == 503
    assert _refused(restarted.admit, "b", "v", 1, NOW).status == 503
    assert _refused(restarted.claim_wake, NOW).status == 503
    store.fail_load = False
    store.fail_save = False
    restarted.visit("b", NOW)
    assert store.saved is not None and store.saved["visits"] == 2  # the day carried over


def test_limiter_refuses_unreadable_counter_files(tmp_path: Path) -> None:
    path = tmp_path / "counters.json"
    limiter = Limiter(JsonCounterStore(path))
    limiter.visit("a", NOW)
    assert json.loads(path.read_text())["visits"] == 1
    path.write_text("{not json")
    assert _refused(Limiter(JsonCounterStore(path)).visit, "a", NOW).status == 503
    path.write_text(json.dumps({"day": NOW.date().isoformat(), "visits": -4}))
    assert _refused(Limiter(JsonCounterStore(path)).visit, "a", NOW).status == 503


def test_wakes_are_capped_per_day() -> None:
    limiter = Limiter(MemoryStore(), Limits(wakes_per_day=2))
    assert [limiter.claim_wake(NOW) for _ in range(3)] == [True, True, False]


# ── visits and client buckets ─────────────────────────────────────────────────


def test_client_buckets_are_keyed_and_private_ranges_share_one() -> None:
    key = client_key("8.8.8.8", KEY)
    assert "8.8.8.8" not in key and key == client_key(" 8.8.8.8 ", KEY)
    assert key != client_key("8.8.8.8", b"x" * 32)
    shared = client_key(NON_GLOBAL, KEY)
    for address in ("10.0.0.1", "192.168.1.9", "127.0.0.1", "::1", "169.254.1.1", "192.0.2.7",
                    "198.51.100.3", "203.0.113.9", "2001:db8::1", "fe80::1", "100.64.0.1", "testclient",
                    "::ffff:10.0.0.1"):
        assert client_key(address, KEY) == shared, address
    assert client_key("2606:4700::1", KEY) == client_key("2606:4700::ffff:1", KEY)  # one /64
    assert client_key("2606:4700::1", KEY) != client_key("2606:4700:0:1::1", KEY)
    assert client_key("::ffff:8.8.8.8", KEY) == key


def test_the_saved_counters_hold_no_address(tmp_path: Path) -> None:
    path = tmp_path / "counters.json"
    Limiter(JsonCounterStore(path)).visit(client_key("8.8.4.4", KEY), NOW)
    assert "8.8.4.4" not in path.read_text()


def test_visit_tokens_are_signed_and_expire() -> None:
    token = mint_visit(KEY, NONCE, NOW)
    assert visit_nonce(token, KEY, NOW) == NONCE
    assert visit_nonce(token, KEY, NOW + timedelta(seconds=VISIT_TTL_S + 1)) is None
    assert visit_nonce(token, b"y" * 32, NOW) is None
    forged = token.replace(token.split(".")[2], str(int(NOW.timestamp()) + 999))
    assert visit_nonce(forged, KEY, NOW) is None
    for junk in ("", "v1", "v1.a.b.c", token + ".x"):
        assert visit_nonce(junk, KEY, NOW) is None


# ── blind pairs ───────────────────────────────────────────────────────────────


@dataclass
class Votes:
    stored: list[dict[str, str]] = field(default_factory=list)
    fail: bool = False

    def __call__(self, vote: Vote) -> None:
        if self.fail:
            raise Refused(503, "resting")
        self.stored.append(vote.record())


def test_vote_offers_tie_and_neither() -> None:
    assert CHOICES == ("left", "right", "tie", "neither")
    for key in ("vote_left", "vote_right", "vote_tie", "vote_neither", "vote_question"):
        assert words.COPY[key]
    assert words.COPY["vote_question"].startswith("Which reply is more useful to her, given her brief")
    registry, votes = PairRegistry(), Votes()
    product, baseline = SideFacts("4", 900, 0.01), SideFacts("2C", 880, 0.009)
    for n, choice in enumerate(CHOICES):
        registry.open(f"p{n}", left=product, right=baseline, votable=True, now=NOW)
        assert registry.vote(f"p{n}", choice, now=NOW, record=votes).vote.choice == choice
    registry.open("p9", left=product, right=baseline, votable=True, now=NOW)
    with pytest.raises(Refused) as caught:
        registry.vote("p9", "both", now=NOW, record=votes)
    assert caught.value.status == 422
    assert [row["choice"] for row in votes.stored] == list(CHOICES)


def test_a_pair_takes_one_stored_vote_then_reveals() -> None:
    registry, votes = PairRegistry(ttl_s=60), Votes()
    registry.open("p", left=SideFacts("2F", 9000, 0.05), right=SideFacts("4", 800, 0.01), votable=True, now=NOW)
    votes.fail = True
    with pytest.raises(Refused):
        registry.vote("p", "left", now=NOW, record=votes)
    votes.fail = False
    reveal = registry.vote("p", "left", now=NOW, record=votes)  # the failed write left it open
    assert (reveal.left.arm, reveal.right.arm, reveal.left.prompt_tokens) == ("2F", "4", 9000)
    assert votes.stored == [{"choice": "left", "left": "2F", "right": "4", "at": NOW.isoformat(timespec="seconds")}]
    with pytest.raises(Refused) as again:
        registry.vote("p", "right", now=NOW, record=votes)
    assert again.value.reason == "already_voted"
    registry.open("q", left=SideFacts("4", 1, None), right=SideFacts("2C", 1, None), votable=True, now=NOW)
    with pytest.raises(Refused) as expired:
        registry.vote("q", "tie", now=NOW + timedelta(seconds=61), record=votes)
    assert expired.value.reason == "pair_gone"
    registry.open("w", left=SideFacts("4", 1, None), right=SideFacts("2C", 1, None), votable=False, now=NOW)
    with pytest.raises(Refused) as withheld:
        registry.vote("w", "tie", now=NOW, record=votes)
    assert withheld.value.reason == "not_votable" and len(votes.stored) == 1


# ── the service, on fakes ─────────────────────────────────────────────────────


class PassFloor:
    def normalize(self, text: str) -> str:
        return text

    def fixed_reply(self, text: str) -> str | None:
        return "FIXED" if "crisis-marker" in text else None


class PassCheck:
    def replacement(self, reply: str) -> str | None:
        return "PROTOCOL" if "trip-marker" in reply else None


@dataclass
class FakeArms:
    up: bool = True
    calls: int = 0
    pings: int = 0
    product_text: str = "product reply"
    baseline_text: str = "baseline reply"

    def awake(self) -> bool:
        self.pings += 1
        return self.up

    def answer_pair(self, question: str, baseline: str) -> tuple[Answer, Answer]:
        self.calls += 2
        return (Answer(self.product_text, 1200, 2224, 6.0), Answer(self.baseline_text, 1150, 2174, 5.0))


@dataclass
class FakeWaker:
    started: int = 0

    def running(self) -> bool:
        return False

    def start(self) -> None:
        self.started += 1


def _service(arms: FakeArms, *, coin: Iterator[bool] | None = None, ids: Iterator[str] | None = None,
             starters: StarterSet | None = None, waker: FakeWaker | None = None) -> tuple[DemoService, Votes]:
    votes = Votes()
    flips = coin or iter(lambda: True, None)
    names = ids or (f"id-{n:020d}" for n in range(10_000))
    service = DemoService(
        config=DemoConfig("2C", KEY, 3.6), limiter=Limiter(MemoryStore()), pairs=PairRegistry(),
        floor=PassFloor(), reply_check=PassCheck(), arms=arms, starters=lambda: starters, votes=votes,
        waker=waker, coin=lambda: next(flips), new_id=lambda: next(names),
    )
    return service, votes


def test_the_mapping_is_not_recoverable_before_the_vote() -> None:
    sent: list[PairView] = []
    for flip in (True, False):
        service, _ = _service(FakeArms(), coin=iter([flip]), ids=iter(["visit-nonce-0000000", "same-pair-id"]))
        token = service.start_visit("c", True, NOW)
        pair = service.ask("c", token, "How do I light the pier?", NOW)
        assert isinstance(pair, PairView)
        sent.append(pair)
    first, second = sent
    assert first.pair_id == second.pair_id  # the id carries nothing about the order
    assert (first.left, first.right) == (second.right, second.left)
    for pair in sent:
        shown = json.dumps(pair.__dict__, default=lambda o: o.__dict__)
        for leak in ('"4"', "2C", "2F", "1200", "1150", "0.006", "arm", "product\""):
            assert leak not in shown, leak


def test_side_order_follows_the_coin_and_the_reveal_matches() -> None:
    service, votes = _service(FakeArms(), coin=iter([False]))
    token = service.start_visit("c", True, NOW)
    pair = service.ask("c", token, "Should I keep the flare?", NOW)
    assert isinstance(pair, PairView) and pair.left.text == "baseline reply"
    reveal = service.vote("c", token, pair.pair_id, "right", NOW)
    assert (reveal.left.arm, reveal.right.arm) == ("2C", "4")
    assert reveal.right.cost_usd == pytest.approx(6.0 * 3.6 / 3600)
    assert votes.stored[0]["left"] == "2C"


def test_crisis_text_is_answered_before_any_model_or_limit() -> None:
    arms = FakeArms()
    service, _ = _service(arms)
    token = service.start_visit("c", True, NOW)
    assert service.ask("c", token, "crisis-marker", NOW) == Fixed("FIXED")
    assert (arms.calls, arms.pings) == (0, 0)


def test_a_tripped_reply_is_withheld_and_takes_no_vote() -> None:
    service, votes = _service(FakeArms(baseline_text="a trip-marker reply"))
    token = service.start_visit("c", True, NOW)
    pair = service.ask("c", token, "What should I cut?", NOW)
    assert isinstance(pair, PairView) and not pair.votable
    assert pair.right.withheld and pair.right.text == "PROTOCOL" and "trip-marker" not in pair.right.text
    with pytest.raises(Refused):
        service.vote("c", token, pair.pair_id, "left", NOW)
    assert votes.stored == []


def test_asleep_never_wakes_without_a_waker_and_wakes_are_capped() -> None:
    arms = FakeArms(up=False)
    service, _ = _service(arms)
    token = service.start_visit("c", True, NOW)
    assert service.ask("c", token, "Is she there?", NOW) == Asleep(waking=False)
    assert arms.calls == 0
    waker = FakeWaker()
    waking, _ = _service(arms, waker=waker)
    token = waking.start_visit("c", True, NOW)
    asked = [waking.ask("c", token, f"q{n}", NOW + timedelta(minutes=n)) for n in range(4)]
    assert asked == [Asleep(True)] * 3 + [Asleep(False)]
    assert waker.started == Limits().wakes_per_day


def test_chat_needs_an_adult_visit() -> None:
    service, _ = _service(FakeArms())
    with pytest.raises(Refused) as refused:
        service.start_visit("c", False, NOW)
    assert (refused.value.status, refused.value.reason) == (403, "adult_required")
    for call in (lambda: service.ask("c", "", "hi", NOW), lambda: service.starter("c", "forged", "credit", NOW),
                 lambda: service.vote("c", "v1.x.y.z", "p", "tie", NOW)):
        with pytest.raises(Refused) as no_visit:
            call()
        assert no_visit.value.status == 401


# ── starter pairs ─────────────────────────────────────────────────────────────


def _starters(baseline: str = "2C") -> StarterSet:
    ckpt = CheckpointRef("p-abc-n3", "person", date(2026, 10, 30), "uri://x", None)
    return generate(words.SCENES, lambda s: (Side(f"P:{s.id}", 1000, 0.002), Side(f"B:{s.id}", 990, None)),
                    person="film_student", checkpoint=ckpt, baseline=baseline, now=NOW)


def test_starter_pairs_are_labeled_and_round_trip(tmp_path: Path) -> None:
    starters = _starters()
    path = tmp_path / "starter_pairs.json"
    path.write_text(to_json(starters))
    loaded = load(path)
    assert loaded == starters
    assert (loaded.label, loaded.checkpoint, loaded.night) == (LABEL, "p-abc-n3", "2026-10-30")
    assert load(tmp_path / "missing.json") is None
    changed = words.Scene(words.SCENES[0].id, "x", words.SCENES[0].prompt + " Also, music?")
    assert loaded.pair(words.SCENES[0]) is not None and loaded.pair(changed) is None
    unlabeled = json.loads(to_json(starters)) | {"label": "live"}
    with pytest.raises(ValueError):
        from_json(json.dumps(unlabeled))


def test_starter_pairs_serve_with_no_model_and_only_for_their_baseline() -> None:
    arms = FakeArms(up=False)
    service, _ = _service(arms, starters=_starters())
    token = service.start_visit("c", True, NOW)
    pair = service.starter("c", token, "credit", NOW)
    assert pair.source.kind == "pre-generated" and pair.source.checkpoint == "p-abc-n3"
    assert {pair.left.text, pair.right.text} == {"P:credit", "B:credit"}
    assert (arms.calls, arms.pings) == (0, 0)
    stale, _ = _service(arms, starters=_starters("2F"))
    views, source = stale.scenes("c", NOW)
    assert source is None and not any(view.ready for view in views)


def test_starter_scenes_not_in_training_or_sealed_sets() -> None:
    """The scenes against every text a training input or a sealed probe can come from: each persona
    and donor file in full (the visible part trains; the held-out part is what the sealed probes are
    written from), every expanded life, and both gate sets. Both directions, with the engine's own
    decontamination rules. The sealed probe files themselves are never opened."""
    sleep = Path(__file__).parents[1]
    texts = [text for path in sorted((sleep / "personas").glob("*.json")) for text in _strings(json.loads(path.read_text()))]
    texts += _strings(json.loads((sleep / "lives" / "lives.json").read_text()))
    texts += [c.prompt for c in load_collapse()]
    texts += [part for item in load_general() for part in (item.question, *item.choices)]
    assert len(texts) > 500
    scenes = [text for scene in words.SCENES for text in (scene.title, scene.prompt)]
    check = OverlapDecontaminator().check

    def probes(items: list[str]) -> list[Probe]:
        return [Probe(f"x{n}", "p", "new_work", text, None, None, ()) for n, text in enumerate(items)]

    assert 4 <= len(words.SCENES) <= 6
    assert not any(check(scenes, probes(texts)))
    assert not any(check(texts, probes(scenes)))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if len(value.split()) >= 3 else []
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


# ── the page's words ──────────────────────────────────────────────────────────


def test_every_slot_in_the_page_has_its_words_and_the_script_data_is_inert() -> None:
    page = page_html()
    assert "{{" not in page
    assert words.COPY["vote_question"] in page.replace("&#x27;", "'")
    hostile = {"a": "</script><script>alert(1)</script>"}
    rendered = render(f"<p>{{{{a}}}}</p><script type=\"application/json\">{COPY_SLOT}</script>", hostile)
    assert "<script>alert" not in rendered and rendered.count("</script>") == 1
