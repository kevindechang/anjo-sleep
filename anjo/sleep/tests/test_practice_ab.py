# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The practice A/B (``tools.practice_ab``): plan-first; on the same scenes and seeds; judged by a
family neither arm uses (Gemini), blind, in both orders, a flip a tie and a dropped scene its arm's
loss; and the default changes only when arm B is non-inferior at the bar fixed before any run. Critic
agreement with Spark Open is reported, never a bar."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.budget import CAPS, Ledger
from anjo.sleep.evidence.jsonl import PLAIN, read_records
from anjo.sleep.night.deps import NightOptions, Prices
from anjo.sleep.practice.full import FullPractice
from anjo.sleep.practice.roles import TODAY, Roles, ROLES_ENV, practice_roles
from anjo.sleep.tests.test_full_practice import ADAPTER, Glimmer, Spark, _evidence, _scenes, scored
from anjo.sleep.tests.test_practice_roles import DEEPSEEK, DeepSeek, _critic
from anjo.sleep.tools import practice_ab

UID = "synthetic-p1"
NIGHTS = [date(2026, 10, 18) + timedelta(days=i) for i in range(10)]
CASES = [arg for night in NIGHTS for arg in ("--case", f"{UID}:film_student:{night.isoformat()}")]
ARGS = [*CASES, "--arm-b", "critic=deepseek"]
PRICES = Prices(
    spark_in_per_m=1.0, spark_out_per_m=2.0, train_gpu_per_hour=2.5, serve_gpu_per_hour=1.95
)
LABELS = (
    "arm a",
    "arm b",
    "arm_a",
    "arm_b",
    "deepseek",
    "spark",
    "critic",
    "gemini",
    "person=",
    "teacher=",
    "roles",
)


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the plan sends nothing")


def test_practice_ab_plan_prints_the_bar_its_scenes_and_holds_and_sends_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        practice_ab.SCENES_NEEDED == 155
    )  # 95% one-sided, margin 0.10, power 80% at the widest spread
    assert (
        practice_ab.main(ARGS, wiring=_refuse, critic_port=_refuse, judge=_refuse, ledger=_refuse)
        == 0
    )
    out = capsys.readouterr().out
    for word in (
        "160 scenes (the bar needs 155)",
        "baseten_sleep_usd",
        "fireworks_sleep_usd",
        "vertex_usd",
        "modal_usd",
        "both orders",
        "non-inferior",
        "never a bar",
        "expected",
    ):
        assert word in out
    with pytest.raises(SystemExit):  # too few scenes for the bar: refused before anything is built
        practice_ab.main(
            [*CASES[:4], "--arm-b", "critic=deepseek", "--out", "x", "--go"], wiring=_refuse
        )
    with pytest.raises(SystemExit):
        practice_ab.main([*CASES, "--arm-b", "person=spark"], wiring=_refuse)  # arm B is today's
    assert practice_ab.JUDGE_FAMILY not in practice_ab.families(TODAY) | practice_ab.families(
        Roles(critic="deepseek")
    )


class Judge:
    """Gemini as the blind judge: prefers the example whose reply to learn is the teacher's."""

    def __init__(self, always: str | None = None) -> None:
        self.always = always
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> tuple[str, int, int]:
        self.prompts.append(prompt)
        first = prompt.split("## Example 1", 1)[1].split("## Example 2", 1)[0]
        verdict = self.always or ("1" if "Reply to learn: TEACHER" in first else "2")
        return json.dumps({"verdict": verdict}), 1_000, 50


def _wiring(spark: Spark, deepseek: DeepSeek) -> Any:
    def build(trainer: Any, options: NightOptions) -> Any:
        assert trainer is None and options.practice_roles is not None
        roles: Roles = options.practice_roles
        critic = _critic(deepseek) if roles.critic == "deepseek" else None

        def practice(chat: Any, night: Any) -> FullPractice:
            return FullPractice(
                chat=chat,
                client=Glimmer(),
                adapter=lambda u: ADAPTER,
                seed=7,
                snapshot=lambda u, m: (None, frozenset({SPARK_OPEN})),
                critic=critic,
                third_party=lambda u: True,
                lazy_teacher=roles.teacher == "lazy",
            )

        store = SimpleNamespace(trainable=lambda night: _evidence())
        return SimpleNamespace(
            practice=practice,
            evidence=lambda uid: store,
            spark_chat=spark,
            prices=PRICES,
            wake=None,
            ledger=None,
        )

    return build


def _go(
    tmp_path: Path,
    judge: Judge,
    capsys: pytest.CaptureFixture[str],
    *,
    cut: Any = lambda n: False,
    args: list[str] = ARGS,
    ledger: Any = None,
) -> tuple[Spark, DeepSeek, list[Any]]:
    spark = Spark(
        _scenes(), scored(5, (6, 7, 8), lambda text: False)
    )  # arm A: Glimmer's third, nothing flagged
    deepseek = DeepSeek(
        scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER")), cut=cut
    )  # every Glimmer one misses
    code = practice_ab.main(
        [*args, "--out", str(tmp_path), "--go"],
        wiring=_wiring(spark, deepseek),
        critic_port=lambda night, ledger: _critic(deepseek),
        judge=judge,
        ledger=ledger,
    )
    assert code == 0
    lines = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    return spark, deepseek, lines


def test_practice_ab_is_blind_and_order_swapped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    judge = Judge()
    spark, deepseek, lines = _go(tmp_path, judge, capsys)
    arms, (retest, new), judged, decision = lines[:20], lines[20:22], lines[22], lines[23]
    # Same scenes: arm A's are handed to arm B, so the person is asked once a case.
    assert sum(c["messages"][-1]["content"].startswith("# scenes_v1") for c in spark.calls) == 10
    assert {a["roles"]["critic"] for a in arms[0::2]} == {"spark"} and {
        a["roles"]["critic"] for a in arms[1::2]
    } == {DEEPSEEK}
    # Critic agreement on the very same prompts, reported against Spark Open, never a bar.
    asked, judged_by_b = (
        spark.critic_prompts(),
        deepseek.prompts,
    )  # a case: A's 16, then their retest / B's 16, then A's
    assert len(asked) == len(judged_by_b) == 10 * 32
    for case in range(10):
        arm_a = asked[32 * case : 32 * case + 16]
        assert (
            asked[32 * case + 16 : 32 * case + 32]
            == arm_a
            == judged_by_b[32 * case + 16 : 32 * case + 32]
        )
    assert retest["critic"] == "spark-retest" and retest["agree_best_with_spark"] == 1.0
    assert (
        new["critic"] == "deepseek"
        and new["flag_rate"] == 0.75
        and new["agree_best_with_spark"] == 0.0
    )
    # Blind: no arm, role or model named; both orders, so 320 judge calls for 160 scenes.
    assert len(judge.prompts) == 320
    assert not any(label in prompt.lower() for prompt in judge.prompts for label in LABELS)
    for first, second in zip(judge.prompts[::2], judge.prompts[1::2], strict=True):
        one, two = (p.split("## Example 2")[0].split("## Example 1")[1] for p in (first, second))
        assert one != two  # the second order shows the other arm's example first
    assert (
        judged["judge"]["win"] == 160
        and judged["score_b"] == 1.0
        and judged["non_inferior"] is True
    )
    assert decision == {
        "switch_default": True
    }  # the downstream judgement decides, not agreement with Spark
    saved = (tmp_path / "arm_b.jsonl").read_text().splitlines()
    assert len(saved) == 160 and (tmp_path / "arm_a.jsonl").exists()


def test_practice_ab_counts_a_dropped_scene_as_its_arms_loss(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arm B's critic cuts every 4th verdict, so B drops 4 of each case's 16 scenes; the judge always picks
    # what it sees first, so every scene both kept flips to a tie.
    _, _, lines = _go(tmp_path, Judge(always="1"), capsys, cut=lambda n: n % 4 == 0)
    judged = lines[22]
    assert (
        judged["judge"]["dropped_b"] == 40
        and judged["judge"]["flips"] == 120
        and judged["judge"]["tie"] == 120
    )
    assert judged["score_b"] == pytest.approx(
        (120 * 0.5) / 160
    )  # the 40 dropped scenes are B's losses
    assert judged["lower_bound"] < judged["bar"] == 0.4 and judged["non_inferior"] is False
    assert lines[23] == {"switch_default": False}


PILOT = [*CASES[:2], "--arm-b", "critic=deepseek", "--pilot"]
VERDICT_KEYS = (
    "non_inferior",
    "switch_default",
    "score_b",
    "lower_bound",
    "bar",
    "win",
    "tie",
    "loss",
    "judge",
)


def test_the_pilot_plan_is_one_case_with_its_own_small_holds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        practice_ab.main(PILOT, wiring=_refuse, critic_port=_refuse, judge=_refuse, ledger=_refuse)
        == 0
    )
    out = capsys.readouterr().out
    assert "1 synthetic person-night of 16 scenes" in out and "never a verdict" in out
    assert "32 calls (both orders)" in out and "fireworks_sleep_usd at $0.010" in out
    gpu = (
        practice_ab._LIST.serve_gpu_per_hour
        * (practice_ab.CASE_MAX_S + practice_ab.WAKE_S + 600)
        / 3600
    )
    assert f"held on modal_usd: ${gpu:.2f}" in out  # one case's GPU, not ten
    with pytest.raises(SystemExit):  # a pilot is one case, so it never stands in for the full run
        practice_ab.main([*CASES[:4], "--arm-b", "critic=deepseek", "--pilot"], wiring=_refuse)


def test_the_pilot_reports_cost_a_scene_and_judge_agreement_and_never_a_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = Ledger(tmp_path / "budget.jsonl", caps=CAPS)
    judge = Judge()
    _, _, lines = _go(tmp_path / "out", judge, capsys, args=PILOT, ledger=ledger)
    arms, agreement, pilot = lines[:2], lines[2:4], lines[4]
    assert [a["arm"] for a in arms] == ["A", "B"] and len(lines) == 5
    assert len(judge.prompts) == 32 and [a["critic"] for a in agreement] == [
        "spark-retest",
        "deepseek",
    ]
    assert pilot["pilot"] is True and pilot["scenes"] == 16
    assert pilot["dropped_scenes"] == pilot["unreadable_scenes"] == 0
    assert pilot["judge_order_agreement"] == 1.0  # this judge never flips with the order
    assert pilot["usd"]["judge"] > 0 and pilot["usd"]["critic"] > 0 and pilot["usd"]["spark"] > 0
    assert pilot["usd_per_scene"] == pytest.approx(pilot["usd"]["total"] / 16, abs=1e-4)
    assert set(pilot) == {
        "pilot",
        "scenes",
        "judge_order_agreement",
        "dropped_scenes",
        "unreadable_scenes",
        "usd",
        "usd_per_scene",
    }
    assert not any(
        key in line for line in lines for key in VERDICT_KEYS
    )  # no tally, score, bound or switch
    reserved = [e for e in read_records(ledger.path, PLAIN) if e["op"] == "reserve"]
    assert reserved and all(e["job"].startswith("practice_ab:pilot:") for e in reserved)
    (gpu,) = [e for e in reserved if e["pools"] == ["modal_usd"]]
    assert gpu["amount"] < 1.5 and ledger.open_holds() == ()  # one case's hold, all settled


def test_the_pilot_counts_a_flip_against_the_judges_agreement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, lines = _go(
        tmp_path, Judge(always="1"), capsys, args=PILOT
    )  # picks whatever it sees first
    assert lines[-1]["judge_order_agreement"] == 0.0


def test_the_pilot_counts_scenes_dropped_or_left_unreadable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arm B's critic cuts every 4th verdict (4 scenes dropped); the judge answers nothing readable
    _, _, lines = _go(tmp_path, Judge(always="?"), capsys, args=PILOT, cut=lambda n: n % 4 == 0)
    assert lines[-1]["dropped_scenes"] == 4 and lines[-1]["unreadable_scenes"] == 12


def _seed_cases(tmp_path: Path, count: int, capsys: pytest.CaptureFixture[str]) -> None:
    import time

    spark = Spark(_scenes(), scored(5, (6, 7, 8), lambda text: False))
    deepseek = DeepSeek(scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER")))
    roles_b = practice_roles({ROLES_ENV: "critic=deepseek"})
    cases = [practice_ab._case(f"{UID}:film_student:{NIGHTS[i].isoformat()}") for i in range(count)]
    args = argparse.Namespace(
        case=cases,
        scenes=16,
        pilot=False,
        out=str(tmp_path),
        resume=False,
        wiring=practice_ab.WIRING,
        critic=practice_ab.CRITIC,
        judge=practice_ab.JUDGE,
    )
    build = _wiring(spark, deepseek)
    deps = [
        (
            build(None, practice_ab._options(c, 16, TODAY)),
            build(None, practice_ab._options(c, 16, roles_b)),
        )
        for c in cases
    ]
    assert (
        practice_ab._go(
            args,
            roles_b,
            deps,
            None,
            lambda night, ledger: _critic(deepseek),
            Judge(),
            time.monotonic,
            {},
        )
        == 0
    )
    capsys.readouterr()


def _write_run(tmp_path: Path) -> None:
    roles_b = practice_roles({ROLES_ENV: "critic=deepseek"})
    settings = practice_ab._run_settings(
        argparse.Namespace(
            wiring=practice_ab.WIRING,
            critic=practice_ab.CRITIC,
            judge=practice_ab.JUDGE,
            scenes=16,
            pilot=False,
            resume=False,
        ),
        roles_b,
    )
    (tmp_path / practice_ab.RUN_NAME).write_text(
        json.dumps(settings, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_arm_a_only(out: Path, night: date) -> None:
    lines = out.joinpath("arm_a.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines[-16:]:
        row = json.loads(line)
        row["night"] = night.isoformat()
        row["id"] = f"{row['id']}-partial"
        out.joinpath("arm_a.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")


def test_resume_reuses_complete_cases_in_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, lines = _go(tmp_path, Judge(), capsys)
    first_decision = lines[22]
    spark = Spark(_scenes(), scored(5, (6, 7, 8), lambda text: False))
    deepseek = DeepSeek(scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER")))
    assert (
        practice_ab.main(
            [*ARGS, "--out", str(tmp_path), "--resume", "--go"],
            wiring=_wiring(spark, deepseek),
            critic_port=lambda night, ledger: _critic(deepseek),
            judge=Judge(),
        )
        == 0
    )
    assert sum(c["messages"][-1]["content"].startswith("# scenes_v1") for c in spark.calls) == 0
    resumed = [
        json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")
    ]
    assert resumed[-2]["score_b"] == first_decision["score_b"] and resumed[-2]["n"] == 160


def test_resume_reruns_a_case_with_only_one_arm_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_cases(tmp_path, 6, capsys)
    _append_arm_a_only(tmp_path, NIGHTS[6])
    spark = Spark(_scenes(), scored(5, (6, 7, 8), lambda text: False))
    practice_ab.main(
        [*ARGS, "--out", str(tmp_path), "--resume", "--go"],
        wiring=_wiring(
            spark, DeepSeek(scored(9, (5, 6, 7), lambda text: text.startswith("GLIMMER")))
        ),
        critic_port=lambda night, ledger: _critic(DeepSeek(scored(9, (5, 6, 7), lambda t: False))),
        judge=Judge(),
    )
    assert sum(c["messages"][-1]["content"].startswith("# scenes_v1") for c in spark.calls) == 4


def test_resume_refuses_a_settings_mismatch_before_any_call(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _write_run(tmp_path)
    saved = json.loads((tmp_path / practice_ab.RUN_NAME).read_text(encoding="utf-8"))
    saved["scenes"] = 8
    (tmp_path / practice_ab.RUN_NAME).write_text(
        json.dumps(saved, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit):
        practice_ab.main(
            [*ARGS, "--out", str(tmp_path), "--resume"],
            wiring=_refuse,
            critic_port=_refuse,
            judge=_refuse,
            ledger=_refuse,
        )


def test_resume_plan_counts_reused_scenes_toward_the_bar(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _seed_cases(tmp_path, 6, capsys)
    assert (
        practice_ab.main(
            [*ARGS, "--out", str(tmp_path), "--resume"],
            wiring=_refuse,
            critic_port=_refuse,
            judge=_refuse,
            ledger=_refuse,
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "reuse 6 case(s), run 4 case(s)" in out and "160 scenes (the bar needs 155)" in out


def test_without_resume_an_existing_out_folder_is_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _go(tmp_path, Judge(), capsys)
    before = (tmp_path / "arm_a.jsonl").read_text(encoding="utf-8")
    assert (
        practice_ab.main(
            [*ARGS, "--out", str(tmp_path), "--go"],
            wiring=_wiring(
                Spark(_scenes(), scored(5, (6, 7, 8), lambda t: False)),
                DeepSeek(scored(9, (5, 6, 7), lambda t: False)),
            ),
            critic_port=lambda night, ledger: _critic(
                DeepSeek(scored(9, (5, 6, 7), lambda t: False))
            ),
            judge=Judge(),
        )
        == 0
    )
    assert (tmp_path / "arm_a.jsonl").read_text(encoding="utf-8") != before
    assert "reuse" not in capsys.readouterr().out
