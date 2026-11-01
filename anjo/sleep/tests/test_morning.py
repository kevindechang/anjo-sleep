# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""The morning probe run (Task 12a): every arm answers the sealed probes on the checkpoint that is
serving after the night, the pairs are judged in both orders, and one content-free score file is
written. Nothing reaches history or evidence (RF5), a scored night is never rerun, and every paid call
is held first. Fakes only."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.arms import ArmDeps
from anjo.sleep.probes.morning import COMMITTED_ARMS, Morning, MorningError, PAIRS, _rows, night_accelerated, run_morning
from anjo.sleep.probes.score import ProbeOutcome, Verdict
from anjo.sleep.reports import results
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tests.sleep_fakes import (
    FakeContexts,
    FakeLogprobs,
    FakeModel,
    FakeRegistry,
    StoreSpy,
    ckpt,
    probe,
    taste,
)

NIGHT = date(2026, 11, 3)
PRICES = Prices(spark_in_per_m=0.2, spark_out_per_m=0.6, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95)
# A full run without ANJO_SLEEP_ARMS_EXTRA asks the six committed arms, so only their pairs answer.
COMMITTED_PAIRS = frozenset(name for name, (a, b) in PAIRS.items() if a in COMMITTED_ARMS and b in COMMITTED_ARMS)


class Judge:
    """Deterministic: the longer reply wins; calls are counted."""

    def __init__(self) -> None:
        self.calls = 0

    def verdict(self, p: Any, first: str, second: str) -> Verdict:
        self.calls += 1
        return "A" if len(first) > len(second) else "B" if len(second) > len(first) else "tie"


class Rig:
    def __init__(self, tmp_path: Path, *, donor: bool = True, ledger: Ledger | None = None) -> None:
        self.model, self.judge, self.store = FakeModel(), Judge(), StoreSpy([])
        self.registry = FakeRegistry({"p1": ckpt("p1", 1), **({"donor": ckpt("donor", 1)} if donor else {})})
        self.probes = [probe("q1", "new_work"), probe("q2", "vocab"), probe("q3", "tone_hold"), taste("t1")]
        deps = ArmDeps(model=self.model, adapters=self.registry, contexts=FakeContexts(),
                       evidence=lambda uid: self.store, logprobs=FakeLogprobs(lambda adapter, messages: "A"),
                       donors={"p1": "donor"}, until=NIGHT)
        self.morning = Morning(arms=deps, sealed=lambda uid: self.probes, judge=self.judge, prices=PRICES,
                               ledger=ledger, root=tmp_path / "results")

    def run(self, night: date = NIGHT, label: str = "development", serving: Any = "registry",
            promoted: bool = True) -> Any:
        found = self.registry.adapters["p1"] if serving == "registry" else serving
        return run_morning("p1", night, found, morning=self.morning, label=label, promoted=promoted)

    @property
    def scores(self) -> Path:
        return self.morning.root / "p1" / "scores"  # type: ignore[operator]


def test_morning_run_never_saved_to_history(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    done = rig.run()
    assert rig.store.writes == 0  # no evidence, no history (RF5)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["results/p1/scores/night-01.json"]
    body = done.path.read_text(encoding="utf-8")
    assert "reply from" not in body and all(p.prompt not in body for p in rig.probes)  # no reply or probe text
    assert set(json.loads(body)) == {"night", "label", "checkpoint", "checkpoint_basis", "probes", "arms", "pairs", "unavailable",
                               "outcomes", "seconds", "judge_model", "judge", "synthetic",
                               "accelerated"}  # its wall time (the arms and the judge), a number only
    assert json.loads(body)["judge_model"] == "unknown"  # the fake has no model ID
    assert (json.loads(body)["synthetic"], json.loads(body)["accelerated"]) == (False, None)  # no flag, no receipt
    assert json.loads(body)["judge"] == "muse"  # and no other judge than the morning's own


def test_morning_rerun_refused_once_scored(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    asked, judged = len(rig.model.calls), rig.judge.calls
    with pytest.raises(MorningError, match="already scored"):
        rig.run()
    assert (len(rig.model.calls), rig.judge.calls) == (asked, judged)  # refused before a single call
    assert [p.name for p in rig.scores.iterdir()] == ["night-01.json"]
    rig.run(night=date(2026, 11, 4))  # the next night is scored as its own file
    assert sorted(p.name for p in rig.scores.iterdir()) == ["night-01.json", "night-02.json"]


def test_morning_scores_serving_checkpoint(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    serving = rig.registry.adapters["p1"]
    done = rig.run()
    personal = {c["adapter"].id for c in rig.model.calls if c["adapter"] is not None and c["adapter"].person_id == "p1"}
    assert personal == {serving.id}  # arms 3 and 4 ran on the serving checkpoint
    assert json.loads(done.path.read_text(encoding="utf-8"))["checkpoint"] == serving.id
    stale = replace(serving, id="p1-n0")
    other = Rig(tmp_path / "other")
    with pytest.raises(MorningError, match="serving"):
        other.run(serving=stale)
    assert other.model.calls == [] and other.judge.calls == 0


def test_every_arm_is_scored_on_both_dimensions(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run(label="final").path.read_text(encoding="utf-8"))
    assert record["label"] == "final" and record["night"] == NIGHT.isoformat() and record["probes"] == 4
    assert set(record["arms"]) == set(COMMITTED_ARMS)  # 2A and 4M need ANJO_SLEEP_ARMS_EXTRA=1
    assert {d: v["score"] for d, v in record["arms"]["1"].items()} == {"knowledge": 0.5, "interaction": 0.5, "overall": 0.5}
    for arm in set(COMMITTED_ARMS) - {"1"}:
        assert set(record["arms"][arm]) == {"knowledge", "interaction", "overall"}
    assert set(record["pairs"]) == COMMITTED_PAIRS
    assert rig.judge.calls > 0 and record["unavailable"] == {} and record["checkpoint_basis"] == "promoted"


def test_the_score_file_is_what_the_results_table_reads(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    night = rig.morning.root / "p1" / "nights" / NIGHT.isoformat()  # type: ignore[operator]
    night.mkdir(parents=True)
    (night / "summary.json").write_text(json.dumps({"status": "promoted", "night": NIGHT.isoformat(),
                                                    "label": "development", "gates": {}, "cost_usd": 0.0}))
    text = results.table(["p1"], root=rig.morning.root)
    assert "arm 4" in text and "pair 4_vs_2C" in text and "/" in next(row for row in text.splitlines() if row.startswith(NIGHT.isoformat() + " | promoted"))


def test_a_donor_that_never_promoted_leaves_3x_unavailable_not_invented(tmp_path: Path) -> None:
    rig = Rig(tmp_path, donor=False)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    assert "3x" not in record["arms"] and record["unavailable"] == {"3x": "donor_unavailable"}  # a code, no text
    assert "3_vs_3x" not in record["pairs"] and {"4_vs_2C", "4_vs_2F"} <= set(record["pairs"])


def test_paid_calls_are_held_first_and_settled(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl")
    Rig(tmp_path, ledger=ledger).run()
    events = read_records(ledger.path, PLAIN)
    reserves = [e for e in events if e["op"] == "reserve"]
    assert [e["pools"] for e in reserves] == [["meta_usd", "judge_usd"], ["modal_usd", "baseten_sleep_usd"]]  # both held before a call
    assert [e["op"] for e in events].count("settle") == 2 and not ledger.open_holds()
    assert all(e["job"].startswith("morning:2026-11-03:") for e in reserves)


def test_no_room_in_a_pool_stops_before_any_call(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps={**CAPS, "meta_usd": 0.01})
    rig = Rig(tmp_path, ledger=ledger)
    with pytest.raises(BudgetStop):
        rig.run()
    assert rig.judge.calls == 0 and not ledger.open_holds() and not rig.scores.exists()


def test_a_probe_of_another_person_is_refused(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.probes = [probe("q9", "new_work", person="someone-else")]
    with pytest.raises(MorningError, match="another person"):
        rig.run()
    assert rig.model.calls == []


STAT_KEYS = {"n", "wins", "ties", "losses", "score", "decisive_share", "ci"}


def test_score_files_carry_the_statistics_the_claim_rules_need(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    for name, stats in [*((f"{a}_vs_1", record["arms"][a]) for a in set(COMMITTED_ARMS) - {"1"}), *record["pairs"].items()]:
        assert set(stats) == {"knowledge", "interaction", "overall"}, name
        for stat in stats.values():
            assert set(stat) == STAT_KEYS and stat["n"] == stat["wins"] + stat["ties"] + stat["losses"]
            low, high = stat["ci"]
            assert 0.0 <= low <= stat["score"] <= high <= 1.0
            assert stat["score"] == pytest.approx((stat["wins"] + 0.5 * stat["ties"]) / stat["n"])


def test_per_probe_outcomes_recompute_the_counts(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    ids = {p.id for p in rig.probes}
    for row in record["outcomes"]:
        margins = {"margin_a", "margin_b"} if row["kind"] == "taste_ab" else set()
        assert set(row) == {"probe", "pair", "kind", "outcome", "flip"} | margins
        assert row["probe"] in ids and row["outcome"] in {"win", "tie", "loss"} and isinstance(row["flip"], bool)
    taste_rows = [row for row in record["outcomes"] if row["kind"] == "taste_ab"]
    assert taste_rows and all(isinstance(row["margin_a"], float) and isinstance(row["margin_b"], float)
                              for row in taste_rows)
    assert {row["pair"] for row in record["outcomes"]} == {f"{a}_vs_1" for a in set(COMMITTED_ARMS) - {"1"}} | set(record["pairs"])
    mine = [r for r in record["outcomes"] if r["pair"] == "4_vs_2C" and r["kind"] != "pushback"]
    stat = record["pairs"]["4_vs_2C"]["overall"]
    assert [sum(r["outcome"] == o for r in mine) for o in ("win", "tie", "loss")] == [stat["wins"], stat["ties"], stat["losses"]]
    text = json.dumps(record)
    assert not any(p.prompt in text for p in rig.probes) and "reply from" not in text


def test_taste_rows_carry_both_margins_and_other_rows_do_not() -> None:
    p, q = taste("t1"), probe("q1")
    rows = _rows("3_vs_1", [p, q], [
        ProbeOutcome("t1", "taste_ab", "tie", True, 1.23456789, None),
        ProbeOutcome("q1", "new_work", "win", False),
    ])
    assert rows[0] == {"probe": "t1", "pair": "3_vs_1", "kind": "taste_ab", "outcome": "tie", "flip": True,
                       "margin_a": 1.2346, "margin_b": None}
    assert rows[1] == {"probe": "q1", "pair": "3_vs_1", "kind": "new_work", "outcome": "win", "flip": False}


def test_the_results_table_reproduces_the_ci_from_the_file_alone(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    night = rig.morning.root / "p1" / "nights" / NIGHT.isoformat()  # type: ignore[operator]
    night.mkdir(parents=True)
    (night / "summary.json").write_text(json.dumps({"status": "promoted", "night": NIGHT.isoformat(),
                                                    "label": "development", "gates": {}, "cost_usd": 0.0}))
    text = results.table(["p1"], root=rig.morning.root)
    low, high = record["arms"]["4"]["knowledge"]["ci"]
    assert f"{record['arms']['4']['knowledge']['score']:.2f} [{low:.2f},{high:.2f}]" in text


def test_a_night_that_promoted_nothing_scores_what_was_serving_and_says_so(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    record = json.loads(rig.run(serving=None, promoted=False).path.read_text(encoding="utf-8"))
    assert record["checkpoint_basis"] == "not_promoted" and record["checkpoint"] == rig.registry.adapters["p1"].id


def test_a_promoted_night_must_give_its_own_checkpoint(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(MorningError, match="promoted"):
        rig.run(serving=None, promoted=True)
    assert rig.model.calls == []


class UnreadableChat:
    """Muse Spark that answers prose (no JSON) on PROMPT-NEVER always and on PROMPT-ONCE on its first call
    only; every other answer is a JSON verdict. Every call's cap is recorded, in order (the run is serial)."""

    def __init__(self) -> None:
        self.caps: list[tuple[str, int | None]] = []

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None, max_tokens: int | None = None) -> Any:
        from anjo.sleep.model_api import ChatResult

        content = messages[-1]["content"]
        asked = next(word for word in ("NEVER", "ONCE", "PLAIN") if f"PROMPT-{word}" in content)
        first_once = asked == "ONCE" and all(seen != "ONCE" for seen, _ in self.caps)
        self.caps.append((asked, max_tokens))
        unreadable = asked == "NEVER" or first_once
        text = "Reply A reads better here." if unreadable else json.dumps({"verdict": "A", "reason": "x"})
        return ChatResult(text, (), "m", 2900, 2200, "stop")


def test_judge_non_json_retries_then_records_no_verdict(tmp_path: Path) -> None:
    from anjo.sleep.probes.morning import JUDGE_RETRY_SHARE
    from anjo.sleep.probes.score import JUDGE_RETRY_MAX_TOKENS, MuseJudge

    ledger = Ledger(tmp_path / "budget.jsonl")
    rig = Rig(tmp_path, ledger=ledger)
    rig.probes = [probe("q1", "new_work", prompt="PROMPT-ONCE"), probe("q2", "vocab", prompt="PROMPT-NEVER"),
                  probe("q3", "tone_hold", prompt="PROMPT-PLAIN"), probe("q4", "new_work", prompt="PROMPT-PLAIN"),
                  probe("q5", "tone_hold", prompt="PROMPT-PLAIN"), taste("t1")]  # 1 of 5 judged unreadable: under a quarter
    chat = UnreadableChat()
    rig.morning = replace(rig.morning, judge=MuseJudge(chat=chat, brief_for=lambda pid: "brief"))
    done = rig.run()  # the run ends and writes its file: an unreadable verdict never crashes the morning
    record = json.loads(done.path.read_text(encoding="utf-8"))
    comparisons = {f"{a}_vs_1" for a in set(COMMITTED_ARMS) - {"1"}} | COMMITTED_PAIRS
    # q1's first answer was unreadable, and its one retry, with more room, gave the verdict.
    assert chat.caps[:2] == [("ONCE", None), ("ONCE", JUDGE_RETRY_MAX_TOKENS)]
    assert {r["outcome"] for r in record["outcomes"] if r["probe"] == "q1"} <= {"win", "tie", "loss"}
    # q2 never gave one: no verdict in every comparison, counted, shown, and left out of the score.
    assert record["no_verdict"] == dict.fromkeys(comparisons, 1) and done.no_verdict == len(comparisons)
    assert [r["outcome"] for r in record["outcomes"] if r["probe"] == "q2"] == ["no_verdict"] * len(comparisons)
    assert record["pairs"]["4_vs_2C"]["overall"]["n"] == 5  # every probe but q2
    # Retries are a share of the run's judge calls (80 here: 8): q1's, then q2's until they ran out.
    retried = [asked for asked, cap in chat.caps if cap == JUDGE_RETRY_MAX_TOKENS]
    assert retried == ["ONCE"] + ["NEVER"] * 7 and math.ceil(80 * JUDGE_RETRY_SHARE) == 8
    assert {cap for asked, cap in chat.caps if asked == "PLAIN"} == {None}  # a readable answer: one call, the provider's cap
    assert JUDGE_RETRY_MAX_TOKENS >= 8_944  # the longest judge answer seen (Teo's night 1, 257 calls)
    # Every call, retries included, is counted, and the judge's hold covers them all: no overage.
    assert done.judge_calls == len(chat.caps)
    events = read_records(ledger.path, PLAIN)
    held = next(e for e in events if e["op"] == "reserve" and e["pools"] == ["meta_usd", "judge_usd"])
    settled = next(e for e in events if e["op"] == "settle" and e["job"] == held["job"])
    assert settled["amount"] <= held["amount"] and not ledger.open_holds()


class MeteredChat:
    """Muse Spark answering a JSON verdict; its usage is reported unless ``usage`` is off (a reply with
    0 tokens in and out)."""

    def __init__(self, *, usage: bool) -> None:
        self.usage, self.calls = usage, 0

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float, *,
                 temperature: float | None = None, max_tokens: int | None = None) -> Any:
        from anjo.sleep.model_api import ChatResult

        self.calls += 1
        tokens = (3_000, 500) if self.usage else (0, 0)
        return ChatResult(json.dumps({"verdict": "A", "reason": "x"}), (), "m", *tokens, "stop")


def _metered_run(tmp_path: Path, *, usage: bool) -> tuple[Any, MeteredChat, dict[str, Any]]:
    from anjo.sleep.probes.score import JudgeTokens, MuseJudge

    ledger = Ledger(tmp_path / "budget.jsonl")
    rig, chat, tokens = Rig(tmp_path, ledger=ledger), MeteredChat(usage=usage), JudgeTokens()
    rig.morning = replace(rig.morning, judge=MuseJudge(chat=chat, brief_for=lambda pid: "brief", tokens=tokens),
                          judge_tokens=tokens)
    done = rig.run()
    events = read_records(ledger.path, PLAIN)
    held = next(e for e in events if e["op"] == "reserve" and e["pools"] == ["meta_usd", "judge_usd"])
    settled = next(e for e in events if e["op"] == "settle" and e["job"] == held["job"])
    return done, chat, settled


def test_a_judge_call_with_usage_settles_at_its_tokens(tmp_path: Path) -> None:
    done, chat, settled = _metered_run(tmp_path, usage=True)
    usd = chat.calls * (3_000 * PRICES.spark_in_per_m + 500 * PRICES.spark_out_per_m) / 1e6
    assert chat.calls > 0 and done.judge_calls == chat.calls and done.judge_estimated == 0
    assert done.judge_usd == pytest.approx(usd) and settled["amount"] == pytest.approx(usd)


def test_a_judge_call_without_usage_settles_at_the_bound_marked_estimated(tmp_path: Path) -> None:
    from anjo.sleep.probes.morning import JUDGE_USD_PER_CALL

    done, chat, settled = _metered_run(tmp_path, usage=False)
    assert chat.calls > 0 and done.judge_estimated == done.judge_calls == chat.calls
    assert done.judge_usd == pytest.approx(chat.calls * JUDGE_USD_PER_CALL) == settled["amount"]


def test_all_no_verdict_is_missing_not_a_tie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.demo.record import chart_view
    from anjo.sleep.probes import morning as morning_module
    from anjo.sleep.probes.morning import NO_VERDICT_MAX_SHARE, TooManyNoVerdicts
    from anjo.sleep.probes.score import MuseJudge
    from anjo.sleep.reports.results import load, table

    def run(folder: Path) -> Any:
        ledger = Ledger(folder / "budget.jsonl")
        rig = Rig(folder, ledger=ledger)
        rig.probes = [probe("q1", "new_work", prompt="PROMPT-NEVER"), probe("q2", "vocab", prompt="PROMPT-NEVER"),
                      probe("q3", "tone_hold", prompt="PROMPT-NEVER")]  # no taste_ab: the judge decides every probe
        rig.morning = replace(rig.morning, judge=MuseJudge(chat=UnreadableChat(), brief_for=lambda pid: "brief"))
        return rig, ledger

    # A judge whose format changed fails loudly: past a quarter of a comparison's probes the morning stops,
    # writes no score (a rerun may score the night) and settles what it spent.
    rig, ledger = run(tmp_path / "loud")
    with pytest.raises(TooManyNoVerdicts, match=r"no readable verdict on 3 of 3 probes in 2C_vs_1") as stopped:
        rig.run()
    assert NO_VERDICT_MAX_SHARE == 0.25 and "PROMPT" not in str(stopped.value)
    assert not rig.scores.exists() and not ledger.open_holds()
    # Even let through, a comparison with no verdict at all is written as missing, never as a 0.50 tie.
    monkeypatch.setattr(morning_module, "NO_VERDICT_MAX_SHARE", 1.0)
    rig, _ = run(tmp_path / "quiet")
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    assert record["arms"] == {} and record["pairs"] == {} and set(record["no_verdict"].values()) == {3}
    # Readers show a stat with no probe behind it as missing too, never as an exact tie.
    path = rig.scores / "night-01.json"
    tie = {"n": 0, "wins": 0, "ties": 0, "losses": 0, "score": 0.5, "decisive_share": None, "ci": [0.5, 0.5]}
    path.write_text(json.dumps({**record, "arms": {"3": {"overall": tie, "knowledge": tie}}}), encoding="utf-8")
    nights_dir = tmp_path / "quiet" / "results" / "p1" / "nights" / NIGHT.isoformat()
    nights_dir.mkdir(parents=True)
    (nights_dir / "summary.json").write_text(json.dumps({"night": NIGHT.isoformat(), "label": "development",
                                                         "status": "promoted"}), encoding="utf-8")
    night = load("p1", tmp_path / "quiet" / "results")[0]
    assert night.arms == {} and night.no_verdict["3_vs_1"] == 3
    assert "0.50" not in table(["p1"], root=tmp_path / "quiet" / "results")
    assert chart_view([night], [], name="Teo")["development"]["lives"][0]["nights"][0]["scores"] == {}


def test_no_verdict_counts_show_beside_their_scores(tmp_path: Path) -> None:
    from anjo.sleep.demo.dailies import DailySources, NightFiles, dailies_view
    from anjo.sleep.demo.heldout import HeldOutText
    from anjo.sleep.demo.record import chart_view
    from anjo.sleep.reports.results import Night, Stat, _section

    stat = Stat(0.6, n=3, wins=2, ties=0, losses=1, decisive_share=1.0, ci=(0.2, 0.9))
    night = Night(person="p1", night="2026-11-03", label="development", status="promoted", cost_usd=0.0, gates={},
                  arms={"3": {"overall": stat, "knowledge": stat}}, pairs={"4_vs_2C": {"overall": stat}},
                  no_verdict={"3_vs_1": 1, "4_vs_2C": 2})
    lines = _section("development", ["p1"], {"p1": [night]})  # the rows table() prints
    assert any("0.60 [0.20,0.90] (1 no verdict)" in line for line in lines)  # the arm's cell
    assert any("pair 4_vs_2C" in line and "2 no verdict" in line for line in lines)
    assert chart_view([night], [], name="Teo")["development"]["lives"][0]["nights"][0]["no_verdict"] == {"3": 1}
    summary = {"night": "2026-11-03", "status": "promoted", "label": "development", "gates": {}}
    files = NightFiles(summary, [{"step": "morning", "status": "ok", "judge_calls": 22, "no_verdict": 2}])
    card = dailies_view(DailySources(person="p1", nights=[files]), HeldOutText([]), passes=lambda text: True)
    assert card["no_verdict"] == 2
    quiet = dailies_view(DailySources(person="p1", nights=[NightFiles(summary, [])]), HeldOutText([]),
                         passes=lambda text: True)
    assert quiet["no_verdict"] == 0


def test_a_score_file_carries_provenance_from_the_night_receipts(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.morning = replace(rig.morning, synthetic=True)
    receipts = tmp_path / "results" / "p1" / "nights"
    (receipts / NIGHT.isoformat()).mkdir(parents=True)
    (receipts / NIGHT.isoformat() / "summary.json").write_text(json.dumps({"night": NIGHT.isoformat(), "status": "promoted"}))
    (receipts / f"{NIGHT.isoformat()}.2").mkdir()
    (receipts / f"{NIGHT.isoformat()}.2" / "summary.json").write_text(
        json.dumps({"night": NIGHT.isoformat(), "accelerated": True}))
    record = json.loads(rig.run().path.read_text(encoding="utf-8"))
    assert (record["synthetic"], record["accelerated"]) == (True, True)
    assert night_accelerated(tmp_path / "results" / "p1", date(2026, 1, 1)) is None
    (receipts / f"{NIGHT.isoformat()}.2" / "summary.json").write_text(json.dumps({"night": NIGHT.isoformat()}))
    assert night_accelerated(tmp_path / "results" / "p1", NIGHT) is False


def test_the_app_morning_marks_its_account_synthetic() -> None:
    text = (Path(__file__).parents[1] / "app" / "morning.py").read_text(encoding="utf-8")
    assert "if not is_synthetic(user_id):" in text and "synthetic=True)" in text
