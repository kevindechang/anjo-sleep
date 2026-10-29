# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""A day written again that comes back as it was (MIND's data; ``--repair``): on 2026-10-29 a
day's line survived two repair rounds word for word, and each round paid for the gates' reads
again only to flag it again. A rewrite that comes back unchanged is asked once more inside its
round, hotter, with a code-side line that it gave the day back as it was and the round's reason
stands; still unchanged, the round fails at once, before anything reads it. Each write that came
back unchanged is marked in the call log, never its text. Every model is a fake: no network."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.mind.apd import calls, gen_batch, generate, repair
from anjo.sleep.mind.apd.calls import CallLog
from anjo.sleep.mind.apd.ceiling import RunLedger
from anjo.sleep.model_api import ChatResult
from anjo.sleep.tests.test_apd_repair import LEAKY_DAY, MARK, SPEC, FlagsTheMark, LeakyDay, _backend


@dataclasses.dataclass
class GivesItBack(LeakyDay):
    """``LeakyDay`` whose rewrite of the leaky day gives it back as it was, ``MARK`` and all, until
    asked again (``generate.UNCHANGED``) and ``mends`` is set; each rewrite's system message and
    temperature kept."""

    mends: bool = True
    asks: list[tuple[str, float | None]] = dataclasses.field(default_factory=list)

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None) -> ChatResult:
        result: ChatResult = self.inner(model, messages, tools, timeout_s, temperature=temperature)
        if calls.kind_of(messages) != "write":
            return result
        body = json.loads(messages[-1]["content"])
        self.days.append(body["day"])
        if body["day"] != self.day:
            return result
        if "reason" in body:
            self.repairs.append(body)
            self.asks.append((str(messages[0]["content"]), temperature))
            if self.mends and generate.UNCHANGED in str(messages[0]["content"]):
                return result
        day = json.loads(result.text)
        return dataclasses.replace(result, text=json.dumps({**day, "messages": [*day["messages"], MARK]}))


def test_a_rewrite_given_back_unchanged_is_asked_once_more_hotter_and_the_life_passes(tmp_path: Path) -> None:
    # Arrange
    writer, log = GivesItBack(), CallLog()
    backend = _backend(tmp_path, life_chat=writer, reader=FlagsTheMark(), repair=True)

    # Act
    made = gen_batch.generate_life(SPEC, backend, log)

    # Assert: one round, its day asked twice, the second hotter and told it came back as it was
    assert made.attempts == 1 and made.repaired == 1 and all(MARK not in s.text for s in made.life.sessions)
    assert writer.days == [*range(1, 11), LEAKY_DAY, LEAKY_DAY]
    (first, first_heat), (again, heat) = writer.asks
    assert generate.UNCHANGED not in first and again.endswith(generate.UNCHANGED)
    assert first_heat == generate.WRITE_TEMPERATURE < generate.REWRITE_TEMPERATURE == heat
    assert writer.repairs[0]["reason"] == writer.repairs[1]["reason"]  # the round's own reason, no gate text
    # the log keeps which write came back unchanged, never its text
    rewrites = [c for c in log.calls if c.kind == "write" and c.repair == 1]
    assert [c.unchanged for c in rewrites] == [True, False]
    shown = "\n".join(calls.breakdown({made.life.id: log.calls}))
    assert "round 1: 1 day(s) written again, 1 came back unchanged and was asked again, 0 still unchanged" in shown


def test_a_rewrite_still_unchanged_fails_the_round_before_anything_reads_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange: one attempt; the leaky day comes back as it was however it is asked
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)
    writer, log = GivesItBack(mends=False), CallLog()
    backend = _backend(tmp_path, life_chat=writer, reader=FlagsTheMark(), repair=True)

    # Act
    with pytest.raises(generate.GenerationError):
        gen_batch.generate_life(SPEC, backend, log)

    # Assert: the round failed at the writer on the unchanged rule; no gate read in it, no second round
    assert writer.days == [*range(1, 11), LEAKY_DAY, LEAKY_DAY]
    (failed,) = log.failures
    assert (failed.step, failed.day, failed.rule) == ("writer", LEAKY_DAY, generate.UNCHANGED_RULE)
    assert {c.kind for c in log.calls if c.repair == 1} == {"write"}
    assert [c.unchanged for c in log.calls if c.repair == 1] == [True, True]
    assert RunLedger(tmp_path / "budget.jsonl").open_holds() == ()


# A day of seven messages, about 1,350 characters, whose fourth carries a leaky clause of about 50.
DAY = [
    "Up at six to sand the oak legs for the client's table before the shop got warm; the dust settles faster "
    "when the air is still cold, and the grain shows every pass of the paper.",
    "The client rang mid-morning to ask whether the top could be a shade darker than the sample board, which "
    "means another coat of the walnut oil and a full day's wait before the wax.",
    "Lunch on the bench outside, watching the delivery van come and go from the timber yard across the road "
    "while the first coat of oil soaked into the legs.",
    "Fitted the drawer runners this afternoon, and they ran smooth on the first try, so the damp is clearly not "
    "a problem any more, and the second drawer went in just as easily.",
    "The serving board idea came up again when a neighbour asked whether I sell anything small, which I don't "
    "yet, though the offcuts pile is getting tall enough to fill a shelf.",
    "Spent an hour at the end of the day sorting the offcuts by species and length, walnut on the top shelf, "
    "oak below it, the ash and cherry scraps in the crate by the stove, and the long thin strips that might "
    "make handles tied in a bundle on the wall, because the floor had gone from cluttered to dangerous.",
    "Swept up, oiled the plane, and wrote tomorrow's list on the board by the door: wax the legs, check the "
    "drawer fronts, and call the yard about the next delivery of oak boards.",
]
MENDED = [*DAY[:3], DAY[3].replace(", so the damp is clearly not a problem any more", ""), *DAY[4:]]


@pytest.mark.parametrize(("after", "same"), [
    (DAY, True),  # given back verbatim, the case seen on 2026-10-29
    ([f"  {text.upper()} " for text in DAY], True),  # case and spacing aside
    (MENDED, False),  # repair_v1's minimal fix: the one leaky clause dropped, the rest as it was
    ([*DAY[:6], DAY[6].replace("oak boards.", "oak boards!")], False),  # any other change in any message
    (DAY[:6], False),  # a message dropped
    ([*DAY, "One more line."], False),  # a message added
])
def test_a_rewrite_is_unchanged_only_when_every_message_is_its_original(after: list[str], same: bool) -> None:
    assert 1_300 < len(" ".join(DAY)) < 1_400 and 45 < len(DAY[3]) - len(MENDED[3]) < 55
    assert generate.unchanged(after, DAY) is same


def test_a_round_holds_each_written_days_ask_again_with_its_day_retries() -> None:
    # days 2 and 4 written, each ask and its ask again with 2 day retries each, and days 2 to 10 extracted
    assert repair.round_calls(4, {1, 3}, day_retries=2).writes == 2 * 2 * 3 + 9


def test_the_plan_says_a_day_given_back_unchanged_is_asked_again(tmp_path: Path) -> None:
    from anjo.sleep.mind.apd import __main__ as cli
    lines: list[str] = []
    assert cli.main(["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75",
                     "--data", str(tmp_path), "--only", "2", "generate", "--split", "core-train", "--repair"],
                    out=lines.append) == 0
    text = "\n".join(lines)
    assert "each writing at most 3 days again (a day that comes back as it was asked once more, held)" in text
