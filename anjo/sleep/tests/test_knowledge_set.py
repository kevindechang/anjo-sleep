# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The post-hoc knowledge measure: a frozen forced-choice set built from a persona's visible statements
(``sets/knowledge.py``) and the scorer that reads each arm's nightly adapters
(``python -m anjo.sleep.tools.knowledge_score``)."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import pytest

from anjo.sleep.gates import EVAL_DAY
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.probes.decontam import NGRAM
from anjo.sleep.serving.client import WakeResult
from anjo.sleep.sets import knowledge
from anjo.sleep.sets.loader import SetError
from anjo.sleep.tools import knowledge_score

PERSONA = "film_founder"
START = date(2026, 11, 1)


def _persona_path() -> Path:
    return knowledge.persona_path(PERSONA)


def _items() -> tuple[knowledge.KnowledgeItem, ...]:
    return knowledge.load(PERSONA).items


def _grams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _imports(module: Any) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found |= {node.module} | {f"{node.module}.{alias.name}" for alias in node.names}
    return found


# ── the set ──────────────────────────────────────────────────────────────────────────────────────────


def test_set_is_frozen_and_deterministic(tmp_path: Path) -> None:
    built = knowledge.build_file(_persona_path())
    assert built == knowledge.build_file(_persona_path())  # the same items, choices and order every time
    path = knowledge.set_path(PERSONA)
    frozen = knowledge.load(PERSONA)
    assert frozen.items == built and frozen.version == knowledge.VERSION
    assert knowledge.dumps(built) == path.read_text(encoding="utf-8")
    assert frozen.sha == knowledge.FROZEN[PERSONA] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(built) >= knowledge.MIN_ITEMS
    assert len({item.id for item in built}) == len(built)
    visible = load_person_file(_persona_path()).visible
    statements = {text for _, text in knowledge.statements(visible)}
    for item in built:
        assert item.prompt == knowledge.PROMPT
        assert 2 <= len(item.choices) <= knowledge.MAX_CHOICES and len(set(item.choices)) == len(item.choices)
        assert item.choices[item.answer] in statements  # the true choice is a visible statement, verbatim
        assert not any(choice in statements for n, choice in enumerate(item.choices) if n != item.answer)
    changed = tmp_path / "knowledge.jsonl"
    changed.write_text(path.read_text(encoding="utf-8").replace("eighty", "ninety", 1), encoding="utf-8")
    with pytest.raises(SetError):
        knowledge.load(PERSONA, path=changed)  # one changed word and the sha no longer matches
    with pytest.raises(SetError):
        knowledge.load("researcher")  # a persona never frozen has no set


def test_no_item_comes_from_sealed_probes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from anjo.sleep.probes import generate

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the knowledge set never reads, writes or makes a probe")

    for name in ("load_sealed", "seal_probes", "generate_probes", "generate_day_probes"):
        monkeypatch.setattr(generate, name, refuse)
    original = knowledge.build_file(_persona_path())
    held = load_person_file(_persona_path()).held_out
    texts = ([s.text for s in held.preferences] + [w.text for w in held.new_work]
             + [t.prompt for t in held.taste_ab] + [o for t in held.taste_ab for o in t.options])
    # The rule C is held to: no run of NGRAM words of a held-out text in any choice, and no whole one, unless the
    # held-out text only quotes a visible statement (a taste option naming a visible plan's "two strangers").
    banned = {gram for text in texts for gram in _grams(text, NGRAM)}
    choices = [choice for item in original for choice in item.choices]
    assert not any(_grams(choice, NGRAM) & banned for choice in choices)
    visible = [text.lower() for _, text in knowledge.statements(load_person_file(_persona_path()).visible)]
    quoted = [text.lower() for text in texts if not any(text.lower() in v for v in visible)]
    assert not any(text in choice.lower() for text in quoted for choice in choices)
    data = json.loads(_persona_path().read_text(encoding="utf-8"))

    def scramble(value: Any) -> Any:  # every held-out text reversed word by word: still a valid file
        if isinstance(value, str):
            return " ".join(reversed(value.split()))
        if isinstance(value, list):
            return [scramble(v) for v in value]
        if isinstance(value, dict):
            return {k: v if k in {"id", "pref_id", "day", "pick", "parent_project"} else scramble(v)
                    for k, v in value.items()}
        return value

    data["held_out"] = scramble(data["held_out"])
    other = tmp_path / f"{PERSONA}.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    assert load_person_file(other).held_out != held
    assert knowledge.build_file(other) == original  # the set is a function of the visible statements alone
    for module in (knowledge, knowledge_score):
        assert not any(name.startswith(("anjo.sleep.probes", "anjo.sleep.prereg")) for name in _imports(module))


# Counterfeits that give themselves away by their grammar or their scale, not by what the persona said.
BAD: Final = (
    r"\ba (?:[aei]|o(?!ne)|8|1[18]\b|1[18],)",  # "a" before a vowel sound: "a 11 dollar grant"
    r"\ban (?![aeiou]|8|1[18]\b|1[18],|hour|honest|heir)",  # "an" before a consonant sound
    r"\b(?:the|a|whole|each) (?:midnight|noon|dawn)s?\b",  # a point in time used as a stretch of time
    r"\b(?:years|minutes|hours) of runway\b",  # a unit far off the line's own scale (weeks)
    r"\bfinal (?:minute|hour|day)\b",
    r"\bat the same (?:day|week|month|year)\b", r"\blate-night (?:minutes|days)\b",  # a unit that doesn't read there
    r"\ba [0-9] dollar grant\b", r"\bcosts [0-9,]{4,} dollars per hour\b",  # an amount far off the true one
    r"\bDay [0-9]+\.[0-9]",  # a fraction where a whole number stood
)


def test_no_counterfeit_gives_itself_away() -> None:
    items = knowledge.build_file(_persona_path())
    for item in items:
        for choice in item.choices:
            assert not any(re.search(bad, choice, flags=re.IGNORECASE) for bad in BAD), (item.id, choice)
    scale = list(knowledge.UNITS)
    for item in (i for i in items if i.kind == "unit"):
        true = scale.index(knowledge.unit_of(item.key))
        for n, choice in enumerate(item.choices):
            if n != item.answer:  # a unit swaps only with its neighbours on the scale: days with hours or weeks
                changed = [word for word, was in zip(choice.lower().split(), item.statement.lower().split(), strict=True)
                           if word != was and knowledge.unit_of(word.strip(".,;")) in scale]
                assert changed and all(abs(scale.index(knowledge.unit_of(w.strip(".,;"))) - true) == 1 for w in changed), (
                    item.id, choice)
    assert [knowledge.indefinite(word) for word in ("11", "18", "8", "80", "800", "8,000", "11,000", "8.5", "eighty",
                                                    "hour", "evening")] == ["an"] * 11
    assert [knowledge.indefinite(word) for word in ("14,000", "1,100", "110", "2.10", "year", "one", "week",
                                                    "1,300", "night")] == ["a"] * 9


# ── the scorer ───────────────────────────────────────────────────────────────────────────────────────


def _adapter(arm: str, night: int) -> str:
    return f"p-{hashlib.sha256(arm.encode()).hexdigest()[:12]}-n{night}"


def _life(root: Path, account: str, arm: str, nights: int, *, untrained: Sequence[int] = (),
          first: date = START) -> None:
    for number in range(1, nights + 1):
        night = first + timedelta(days=number - 1)
        folder = root / account / "nights" / night.isoformat()
        folder.mkdir(parents=True)
        start = datetime(2026, 10, 29, tzinfo=timezone.utc) + timedelta(hours=2 * number)
        trained = number not in untrained
        adapter = _adapter(arm, number) if trained else None
        steps = [{"step": "close_day", "status": "ok", "window": [start.isoformat(), (start + timedelta(hours=1)).isoformat()],
                  "night": night.isoformat()},
                 {"step": "train", "status": "ok", "adapter": adapter} if trained else {"step": "train", "status": "skipped"}]
        (folder / "log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps), encoding="utf-8")
        (folder / "summary.json").write_text(json.dumps({
            "status": "promoted" if trained else "nothing_new", "trained_adapter": adapter, "cost_usd": 0.4,
        }), encoding="utf-8")


class FakeClient:
    """Glimmer's log-probabilities: an adapter "knows" the statements it was given, which win their item;
    a denied statement loses its item; otherwise every choice scores by its length alone."""

    def __init__(self, knows: dict[str | None, set[str]] | None = None, denied: frozenset[str] = frozenset()) -> None:
        self.knows = knows or {}
        self.denied = denied  # statements every adapter that doesn't know them finds least likely
        self.rendering = "r1"
        self.sent: list[tuple[str | None, list[dict[str, str]], str]] = []
        self.days: list[date] = []

    def on(self, day: date) -> FakeClient:
        self.days.append(day)
        return self

    def continuation_logprobs(self, adapter: Any, messages: list[dict[str, str]], continuation: str) -> Sequence[float]:
        name = None if adapter is None else adapter.id
        self.sent.append((name, [dict(m) for m in messages], continuation))
        known = continuation in self.knows.get(name, set())
        return (-1.0, 2.0 if known else -5.0 if continuation in self.denied else -0.001 * len(continuation))


class _Never:
    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the plan sends nothing, holds nothing and reads no day")


def _run(root: Path, client: FakeClient, accounts: dict[str, str], day_text: Any, capsys: pytest.CaptureFixture[str],
         *extra: str) -> list[dict[str, Any]]:
    argv = [arg for arm, account in accounts.items() for arg in ("--account", f"{arm}={account}")]
    code = knowledge_score.main([*argv, "--go", *extra], environ={}, client=client, wake=lambda: WakeResult(1.0, True),
                                ledger=None, root=root, day_text=day_text)
    assert code == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_scoring_is_weights_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _life(tmp_path, "acct-seed", "seed", 2)
    _life(tmp_path, "acct-today", "today", 2, untrained=(2,))
    client = FakeClient()
    lines = _run(tmp_path, client, {"seed": "acct-seed", "today": "acct-today"}, lambda account, window: [], capsys)
    items = _items()
    assert client.days and set(client.days) == {EVAL_DAY}  # one fixed date on every side
    assert all(messages == [{"role": "user", "content": knowledge.PROMPT}] for _, messages, _ in client.sent)
    assert knowledge.PROMPT.strip() and not any(item.choices[item.answer] in knowledge.PROMPT for item in items)
    scored = {name for name, _, _ in client.sent}
    assert scored == {None, _adapter("seed", 1), _adapter("seed", 2), _adapter("today", 1)}  # base and each trained night
    choices = sum(len(item.choices) for item in items)
    assert len(client.sent) == 4 * choices  # every choice of every item, once an adapter
    assert {(name, continuation) for name, _, continuation in client.sent} == {
        (name, choice) for name in scored for item in items for choice in item.choices}
    out = json.dumps(lines)
    assert not any(item.choices[0] in out for item in items)  # ids and numbers only, never a statement
    nights = [line for line in lines if line.get("line") == "arm_night"]
    assert [(n["arm"], n["night"], n["trained"]) for n in nights] == [
        ("seed", 1, True), ("seed", 2, True), ("today", 1, True), ("today", 2, False)]
    for module_name in _imports(knowledge_score):
        assert not module_name.startswith(("anjo.sleep.context", "anjo.sleep.mind", "anjo.sleep.evidence",
                                           "anjo.sleep.night.core_snapshot", "anjo.sleep.homework"))


def test_paired_ci_uses_the_same_items() -> None:
    b = {f"k{n}": float(n % 7) for n in range(40)}
    a = {key: value + 1.0 for key, value in reversed(list(b.items()))}  # every item exactly 1 above, listed backwards
    b["only-b"] = 50.0  # an item one side lacks never enters the comparison
    sources = {key: f"s{n // 5}" for n, key in enumerate(b)}  # 8 statements, 5 items each (and the extra)
    found = knowledge_score.paired(a, b, clusters=sources, resamples=400, seed=3)
    assert found.n == 40 and found.clusters == 8 and found.diff == pytest.approx(1.0)
    assert found.low == pytest.approx(1.0) and found.high == pytest.approx(1.0)  # each resample draws the same items on both sides
    noisy = {key: value + (1.0 if n % 2 else -1.0) for n, (key, value) in enumerate(b.items()) if key != "only-b"}
    spread = knowledge_score.paired(noisy, b, resamples=400, seed=3)
    assert spread.low is not None and spread.diff is not None and spread.high is not None
    assert spread.low < spread.diff < spread.high
    assert knowledge_score.paired(noisy, b, resamples=400, seed=3) == spread  # a fixed seed: the same interval
    assert knowledge_score.paired({}, b, resamples=10, seed=3).n == 0
    together = {key: (1.0 if n // 5 % 2 else -1.0) for n, key in enumerate(a)}  # a statement's items move together
    zero = dict.fromkeys(a, 0.0)
    by_item = knowledge_score.paired(together, zero, resamples=400, seed=3)
    by_source = knowledge_score.paired(together, zero, clusters=sources, resamples=400, seed=3)
    assert by_item.clusters == 40 and by_source.clusters == 8
    assert by_item.low is not None and by_item.high is not None and by_source.low is not None and by_source.high is not None
    assert by_source.high - by_source.low > by_item.high - by_item.low  # the statement is what is resampled


def test_unseen_items_are_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = _items()
    eighty = [item for item in items if item.key.lower() == "eighty"]
    assert eighty
    day = ["Keep it to 80 words, and give me the answer first."]  # digits count as the word, with one other word of the statement
    seen = knowledge_score.seen(items, day)
    assert {item.id for item in eighty} <= seen
    assert not knowledge_score.seen(items, ["eighty"])  # the key alone, with nothing else of the statement, is not a mention
    assert not knowledge_score.seen(items, [])
    waitlist = [item for item in items if item.key == "1,400"]
    assert waitlist and not {item.id for item in waitlist} & seen
    _life(tmp_path, "acct-seed", "seed", 1)
    _life(tmp_path, "acct-today", "today", 1)
    known = {item.statement for item in eighty} | {waitlist[0].statement}  # each known line wins all its items
    every_truth = frozenset(item.statement for item in items)  # the base and today's arm get every item wrong
    client = FakeClient({_adapter("seed", 1): known}, every_truth)
    days: dict[str, list[str]] = {"acct-seed": day, "acct-today": day}
    asked: list[tuple[str, Any]] = []

    def day_text(account: str, window: Any) -> list[str]:
        asked.append((account, window))
        return days[account]

    lines = _run(tmp_path, client, {"seed": "acct-seed", "today": "acct-today"}, day_text, capsys)
    assert sorted(account for account, _ in asked) == ["acct-seed", "acct-today"]
    seed = next(line for line in lines if line.get("line") == "arm_night" and line["arm"] == "seed")
    assert seed["seen_items"] == len(seen) and seed["unseen"] == sorted(set(i.id for i in items) - seen)
    right = sorted(item.id for item in items if item.statement in known)
    assert seed["newly_right"] == right and waitlist[0].id in right
    assert seed["seen_accuracy"] == pytest.approx(1.0)  # the main number: the items the days mentioned
    assert seed["accuracy"] == pytest.approx(len(right) / len(items), abs=1e-4)  # the unseen ones count here only
    pair = next(line for line in lines if line.get("line") == "pair" and line["a"] == "seed" and line["b"] == "today")
    assert pair["seen_items"] == len(seen) and pair["date"] == START.isoformat()
    assert pair["seen_clusters"] == len({item.source for item in items if item.id in seen})
    assert pair["clusters"] == len({item.source for item in items}) and pair["items"] == len(items)
    assert pair["seen_accuracy_diff"] == pytest.approx(1.0)  # seed right on both seen items, today on neither


def test_pairs_match_nights_by_date(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _life(tmp_path, "acct-seed", "seed", 2)
    _life(tmp_path, "acct-today", "today", 3, first=START - timedelta(days=1))  # one night earlier than seed's first
    lines = _run(tmp_path, FakeClient(), {"seed": "acct-seed", "today": "acct-today"}, lambda account, window: [], capsys)
    pairs = [line for line in lines if line.get("line") == "pair"]
    assert [(p["date"], p["night"], p["b_night"]) for p in pairs] == [
        (START.isoformat(), 1, 2), ((START + timedelta(days=1)).isoformat(), 2, 3)]
    base = next(line for line in lines if line.get("line") == "base")
    chance = sum(1 / len(item.choices) for item in _items()) / len(_items())
    assert base["chance"] == pytest.approx(chance, abs=1e-4) and base["accuracy"] is not None


def test_the_plan_sends_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _life(tmp_path, "acct-seed", "seed", 3)
    _life(tmp_path, "acct-fold", "fold", 3)
    never = _Never()
    code = knowledge_score.main(["--account", "seed=acct-seed", "--account", "fold=acct-fold"], environ={},
                                client=never, wake=never, ledger=never, root=tmp_path, day_text=never)
    assert code == 0
    out = capsys.readouterr().out
    assert "plan" in out and "--go" in out and "modal_usd" in out
    assert f"{len(_items())} items" in out and "7 adapters" in out  # base and 3 nights for each of 2 arms
    chance = sum(1 / len(item.choices) for item in _items()) / len(_items())
    assert f"chance {100 * chance:.0f}%" in out and "floor" in out
    with pytest.raises(SystemExit):  # the days are the app's own reader's, never a module named on the command line
        knowledge_score.main(["--account", "seed=acct-seed", "--days", "os:system"], environ={}, client=never,
                             wake=never, ledger=never, root=tmp_path, day_text=never)
    with pytest.raises(SystemExit):
        knowledge_score.main(["--account", "seed=../x"], environ={}, client=never, wake=never, ledger=never,
                             root=tmp_path, day_text=never)


def test_a_ceiling_is_scored_beside_the_base_and_labelled_never_a_result(tmp_path: Path,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    items, ceiling, never = _items(), _adapter("ceiling", 1), _Never()
    assert knowledge_score.main(["--ceiling", f"recipe={ceiling}"], environ={}, client=never, wake=never,
                                ledger=never, root=tmp_path, day_text=never) == 0
    out = capsys.readouterr().out
    assert "2 adapters" in out and "CEILING" in out and "never a result" in out and ceiling in out
    client = FakeClient({ceiling: {item.statement for item in items}}, frozenset(item.statement for item in items))
    lines = _run(tmp_path, client, {}, never, capsys, "--ceiling", f"recipe={ceiling}")  # no arm: the days are never read
    assert {name for name, _, _ in client.sent} == {None, ceiling} and set(client.days) == {EVAL_DAY}
    [line] = [line for line in lines if line.get("line") == "ceiling"]
    assert line["label"] == "recipe" and line["adapter"] == ceiling and "never a result" in line["CEILING"]
    assert line["accuracy"] == pytest.approx(1.0) and line["seen_items"] == len(items) and line["unseen"] == []
    assert line["vs_base"]["accuracy_diff"] == pytest.approx(1.0) and line["vs_base"]["items"] == len(items)
    assert line["vs_base"]["clusters"] == len({item.source for item in items})
    assert not [line for line in lines if line.get("line") in ("arm_night", "pair")]
    assert not any(item.statement in json.dumps(lines) for item in items)
    for argv in ([], ["--ceiling", "recipe=not-an-adapter"], ["--ceiling", f"a b={ceiling}"],
                 ["--ceiling", f"x={ceiling}", "--ceiling", f"x={_adapter('other', 1)}"]):
        with pytest.raises(SystemExit):  # an account or a ceiling, each named once, an adapter by its volume name
            knowledge_score.main(argv, environ={}, client=never, wake=never, ledger=never, root=tmp_path, day_text=never)
