"""Today's practice roles against another setting (``ANJO_SLEEP_PRACTICE_ROLES``) on one synthetic
person's night (``python -m anjo.sleep.tools.practice_ab``). Without ``--go`` it prints the plan, its
holds and the decision bars, and sends nothing.

    python -m anjo.sleep.tools.practice_ab --user U --persona P --night 2026-10-28 --arm-b critic=gemini
    ANJO_SLEEP_HOST=1 python -m anjo.sleep.tools.practice_ab ... --out DIR --go

With ``--go``, on the night's wiring (``--wiring``, loaded by name; it refuses a person not registered
as synthetic):

1. **Arm A** (today's roles) runs full practice; every Spark Open call is kept.
2. **Arm B** runs on the same night, seed and evidence. While Spark Open plays its person it is handed
   arm A's scenes, so both arms answer the same messages, and Glimmer's candidates are asked with the
   same seeds.
3. **Critics on the same replies.** Each of arm A's critic prompts goes again to Spark Open (the
   test-retest floor) and, when arm B's critic is Gemini, to Gemini (``--critic``, loaded by name).
   Reported: the share of candidates each flags as missing a constraint, and how often its best, its
   worst and whether it keeps a pair agree with arm A's verdict.
4. **A blind judge** (Muse Spark, ``ab_judge_v1``) compares each scene's teacher material, the target
   and the rejected reply, from the two arms in both orders. It is never told which arm, role or model
   wrote which; a verdict that flips with the order is a tie.
5. Each arm's corrections (flags, pairs, teacher targets) and its dollars a scene.

The decision bars are set here, before any result: a new critic flags 0.8 to 1.25 times as often as
Spark Open on the same prompts, agrees with arm A's best within ``AGREE_SLACK`` of Spark Open's own
retest, and the judge scores arm B at least ``JUDGE_FLOOR`` (ties half). A default changes only when
every bar holds. It prints JSON lines of counts and dollars, never a scene's words; both arms'
transcripts go to ``--out`` (a synthetic person's only) and never to the person's sleep folder, so
nothing here trains.

**Spend**, each held before its first call and settled at what it used: Spark Open's worst for both
arms and the retest on ``baseten_sleep_usd``; the judge's on ``meta_usd``; each Gemini call on
``vertex_usd`` (the app's critic holds it); the serving GPU's ``AB_MAX_S``, the wake and Modal's idle
window on ``modal_usd``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Literal

from anjo.sleep import meta_api
from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.template import render
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import NightOptions, Prices
from anjo.sleep.night.holds import held, spark_usd, spark_worst_usd, usd
from anjo.sleep.practice.full import CRITIC_TEMPERATURE, SPARK_CAPS, spark_calls
from anjo.sleep.practice.roles import TODAY, Roles, RolesError, practice_roles, ROLES_ENV
from anjo.sleep.practice.run import SCENE_TIMEOUT_S, _known
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.probes.morning import JUDGE_USD_PER_CALL
from anjo.sleep.probes.score import JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_TIMEOUT_S
from anjo.sleep.serving import config

WIRING: Final = "anjo.sleep.app.night:build"
CRITIC: Final = "anjo.sleep.app.night:gemini_practice_critic"
JUDGE_PATH: Final = Path(__file__).parents[1] / "practice" / "prompts" / "ab_judge_v1.md"
SCENES_HEAD: Final = "# scenes_v1"
CRITIC_HEAD: Final = "# critic_v1"
AB_MAX_S: Final = 60 * 60.0  # both arms' practice on the serving GPU, at its worst
WAKE_S: Final = 600.0  # a cold start's worst
FLAG_RATIO: Final = (0.8, 1.25)
AGREE_SLACK: Final = 0.10
JUDGE_FLOOR: Final = 0.45
Outcome = Literal["win", "tie", "loss"]


class _Recorded:
    """The night's Spark Open chat with every call kept; given ``scenes``, it hands them back for the
    scenes prompt instead of asking again."""

    def __init__(self, chat: Callable[..., ChatResult], scenes: ChatResult | None = None) -> None:
        self._chat, self._scenes = chat, scenes
        self.calls: list[tuple[str, ChatResult]] = []
        self._lock = threading.Lock()

    def __call__(self, model: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                 timeout_s: float, *, temperature: float | None = None, max_tokens: int | None = None) -> ChatResult:
        asked = str(messages[-1]["content"])
        if self._scenes is not None and asked.startswith(SCENES_HEAD):
            return self._scenes
        result = self._chat(model, messages, tools, timeout_s, temperature=temperature, max_tokens=max_tokens)
        with self._lock:
            self.calls.append((asked, result))
        return result

    def scenes(self) -> ChatResult | None:
        return next((result for asked, result in self.calls if asked.startswith(SCENES_HEAD)), None)

    def critic(self) -> list[tuple[str, str]]:
        return [(asked, result.text) for asked, result in self.calls if asked.startswith(CRITIC_HEAD)]

    def usd(self, prices: Prices) -> float:
        return sum(spark_usd(prices, r.input_tokens, r.output_tokens) for _, r in self.calls)


@dataclass(frozen=True)
class Read:
    """A critic's verdict by letter: its best, its worst (None: no pair) and the letters it flagged."""

    best: str
    worst: str | None
    flagged: frozenset[str]
    letters: int


def read_verdict(text: str) -> Read | None:
    try:
        data: Any = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
        scores, misses = data["scores"], data.get("misses_constraint", [])
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(scores, dict) or not scores or not isinstance(misses, list):
        return None
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in scores.values()):
        return None
    order = sorted(scores)
    best = max(order, key=lambda k: (scores[k], -order.index(k)))
    worst = min(order, key=lambda k: (scores[k], order.index(k)))
    return Read(best, worst if scores[best] > scores[worst] else None,
                frozenset(m for m in misses if m in scores), len(scores))


def agreement(name: str, base: Sequence[Read | None], other: Sequence[Read | None]) -> dict[str, Any]:
    """``other``'s verdicts on the same prompts as ``base`` (arm A's): its flag rate and agreement."""
    both = [(a, b) for a, b in zip(base, other, strict=True) if a is not None and b is not None]
    paired = [(a, b) for a, b in both if a.worst is not None and b.worst is not None]
    read = [b for b in other if b is not None]
    letters = sum(b.letters for b in read)
    return {
        "critic": name, "prompts": len(other), "unreadable": len(other) - len(read),
        "flag_rate": round(sum(len(b.flagged) for b in read) / letters, 4) if letters else None,
        "agree_best": round(sum(a.best == b.best for a, b in both) / len(both), 4) if both else None,
        "agree_pair": round(sum((a.worst is None) == (b.worst is None) for a, b in both) / len(both), 4) if both else None,
        "agree_worst": round(sum(a.worst == b.worst for a, b in paired) / len(paired), 4) if paired else None,
    }


def judge_prompt(known: str, first: PracticeTranscript, second: PracticeTranscript) -> str:
    """Two scenes' teacher material, numbered by place only: no arm, role or model is named."""
    return render(JUDGE_PATH.read_text(encoding="utf-8"), {
        "KNOWN": known, "MESSAGE_FIRST": first.message, "TARGET_FIRST": first.reply,
        "REJECTED_FIRST": first.rejected or "(none)", "MESSAGE_SECOND": second.message, "TARGET_SECOND": second.reply,
        "REJECTED_SECOND": second.rejected or "(none)",
    })


def _verdict(text: str) -> str | None:
    try:
        found: Any = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
    except ValueError:
        return None
    verdict = found.get("verdict") if isinstance(found, dict) else None
    return verdict if verdict in ("1", "2", "tie") else None


def judge_arms(ask: Callable[[str], str], known: Callable[[PracticeTranscript], str],
               arm_a: Sequence[PracticeTranscript], arm_b: Sequence[PracticeTranscript]) -> dict[str, Any]:
    """Arm B's outcome on each scene both arms kept: judged with A shown first, then B first. A flip
    or an unreadable verdict is a tie; ties count half."""
    by_scene = {t.scene: t for t in arm_b}
    tally: dict[str, int] = {"win": 0, "tie": 0, "loss": 0, "flips": 0, "unreadable": 0, "unpaired": 0}
    for a in arm_a:
        b = by_scene.get(a.scene)
        if b is None:
            tally["unpaired"] += 1
            continue
        first, second = _verdict(ask(judge_prompt(known(a), a, b))), _verdict(ask(judge_prompt(known(a), b, a)))
        tally["unreadable"] += (first is None) + (second is None)
        b_first: Outcome = {"2": "win", "1": "loss"}.get(first or "tie", "tie")  # type: ignore[assignment]
        b_second: Outcome = {"1": "win", "2": "loss"}.get(second or "tie", "tie")  # type: ignore[assignment]
        tally["flips"] += b_first != b_second
        tally[b_first if b_first == b_second else "tie"] += 1
    judged = tally["win"] + tally["tie"] + tally["loss"]
    return {"judge": tally, "score_b": round((tally["win"] + 0.5 * tally["tie"]) / judged, 4) if judged else None}


def decide(roles_b: Roles, base: Mapping[str, Any], retest: Mapping[str, Any], other: Mapping[str, Any] | None,
           judged: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-set bars: a new critic's flag rate and agreement on the same prompts, then the judge."""
    bars: dict[str, bool] = {}
    if roles_b.critic != TODAY.critic:
        if other is None or other["flag_rate"] is None or base["flag_rate"] is None:
            bars["critic_read"] = False
        else:
            low, high = FLAG_RATIO
            bars["flag_ratio"] = low * base["flag_rate"] <= other["flag_rate"] <= high * base["flag_rate"]
            bars["agree_best"] = (other["agree_best"] or 0.0) >= (retest["agree_best"] or 0.0) - AGREE_SLACK
    bars["judge"] = judged["score_b"] is not None and judged["score_b"] >= JUDGE_FLOOR
    return {"bars": bars, "switch_default": all(bars.values())}


def _save(path: Path, transcripts: Sequence[PracticeTranscript]) -> None:
    def plain(value: object) -> object:
        return sorted(value) if isinstance(value, frozenset) else str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(t), default=plain) + "\n" for t in transcripts), encoding="utf-8")


def _load(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _arm_line(arm: str, run: Any, chat: _Recorded, prices: Prices, gpu_usd: float) -> dict[str, Any]:
    spark = chat.usd(prices)
    scenes = len(run.transcripts)
    total = spark + run.critic_usd + gpu_usd
    return {"arm": arm, "roles": dict(run.roles), "scenes": scenes, "flagged": run.flagged, "pairs": run.pairs,
            "teacher_targets": run.teacher_targets, "critic_unreadable": run.critic_unreadable,
            "spark_calls": {role: c["calls"] for role, c in run.spark.items()},
            "usd": {"spark": round(spark, 4), "critic": round(run.critic_usd, 4), "gpu": round(gpu_usd, 4)},
            "usd_per_scene": round(total / scenes, 4) if scenes else None}


def plan(args: argparse.Namespace, roles_b: Roles, prices: Prices | None) -> str:
    calls = sum(spark_calls(args.scenes, glimmer_person=r.person == "glimmer", lazy_teacher=r.teacher == "lazy")
                for r in (TODAY, roles_b))
    retest = args.scenes
    judge = 2 * args.scenes
    rate = prices.serve_gpu_per_hour if prices is not None else None
    lines = [
        f"practice A/B, night {args.night}, {args.scenes} scenes, one synthetic person",
        f"  arm A     {asdict(TODAY)} (today)",
        f"  arm B     {asdict(roles_b)}",
        f"  spark     at most {calls} calls for both arms and {retest} retest calls, held on baseten_sleep_usd",
        f"  gemini    {args.scenes * (2 if roles_b.critic == 'gemini' else 0)} calls at most beyond arm B's own, "
        "each held on vertex_usd",
        f"  judge     {judge} Muse Spark calls (both orders), held on meta_usd at ${JUDGE_USD_PER_CALL:.2f} each",
        f"  gpu       {AB_MAX_S / 60:.0f} min, the wake and the {config.SCALEDOWN_S / 60:.0f}-min idle window "
        f"on modal_usd" + (f" at ${rate:.2f}/h" if rate is not None else ""),
        f"  bars      flag ratio {FLAG_RATIO[0]}-{FLAG_RATIO[1]}, best agreement >= retest - {AGREE_SLACK}, "
        f"judge score B >= {JUDGE_FLOOR}; the default changes only when all hold",
        "  prints    counts and dollars per arm, critic and judge; never a scene's words",
        f"  saves     both arms' transcripts under --out ({args.out or 'required with --go'})",
    ]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None, *, wiring: Callable[[Any, NightOptions], Any] | None = None,
    critic_port: Callable[[date, Any], Any] | None = None, judge_chat: Callable[..., ChatResult] = meta_api.chat,
    ledger: Any = ..., clock: Callable[[], float] = time.monotonic,
) -> int:
    parser = argparse.ArgumentParser(description="Today's practice roles against another setting, plan-first.")
    parser.add_argument("--user", required=True)
    parser.add_argument("--persona", required=True, help="the synthetic person's persona (a real person is refused)")
    parser.add_argument("--night", type=date.fromisoformat, required=True)
    parser.add_argument("--arm-b", required=True, help=f"arm B's {ROLES_ENV}, such as critic=gemini")
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--wiring", default=WIRING, help="the night's NightDeps builder, <module>:<function>")
    parser.add_argument("--critic", default=CRITIC, help="the third-party critic, <module>:<function>")
    parser.add_argument("--out", help="where both arms' transcripts go (required with --go)")
    parser.add_argument("--go", action="store_true", help="run it; without it, print the plan only")
    args = parser.parse_args(argv)
    try:
        roles_b = practice_roles({ROLES_ENV: args.arm_b})
    except RolesError as error:
        parser.error(str(error))
    if roles_b == TODAY:
        parser.error("arm B is today's roles: there is nothing to compare")
    if not args.go:
        print(plan(args, roles_b, None))
        return 0
    if not args.out:
        parser.error("--go needs --out")
    build = wiring or _load(args.wiring)
    options = NightOptions(night=args.night, personas={args.user: args.persona}, practice_scenes=args.scenes)
    deps_a = build(None, replace(options, practice_roles=TODAY))  # no trainer: nothing here trains
    deps_b = build(None, replace(options, practice_roles=roles_b))
    book = deps_a.ledger if ledger is ... else ledger
    print(plan(args, roles_b, deps_a.prices))
    return _go(args, roles_b, deps_a, deps_b, book, critic_port or _load(args.critic), judge_chat, clock)


def _arm(job: str, arm: str, deps: Any, chat: _Recorded, args: argparse.Namespace, evidence: list[Any],
         ledger: Any, clock: Callable[[], float]) -> Any:
    runner = deps.practice(chat, args.night)
    prices = deps.prices
    with held(ledger, f"{job}:spark:{arm}", ("baseten_sleep_usd",),
              usd(spark_worst_usd(prices, runner.max_spark_calls)), lambda: chat.usd(prices)):
        begun = clock()
        run = runner.run(args.user, args.night, evidence, None, at=datetime.now(UTC))
    _save(Path(args.out) / f"arm_{arm.lower()}.jsonl", run.transcripts)
    print(json.dumps(_arm_line(arm, run, chat, prices, prices.serve_gpu_per_hour * (clock() - begun) / 3600)))
    return run


def _critics(job: str, roles_b: Roles, deps: Any, prompts: Sequence[str], base: Sequence[Read | None],
             critic_port: Callable[[date, Any], Any], night: date, ledger: Any) -> list[dict[str, Any]]:
    """Arm A's critic prompts asked again: of Spark Open (its retest) and of a new critic."""
    retest = _Recorded(deps.spark_chat)
    with held(ledger, f"{job}:spark:retest", ("baseten_sleep_usd",), usd(spark_worst_usd(deps.prices, len(prompts))),
              lambda: retest.usd(deps.prices)):
        again = [read_verdict(retest(SPARK_OPEN, [{"role": "user", "content": p}], None, SCENE_TIMEOUT_S,
                                     temperature=CRITIC_TEMPERATURE, max_tokens=SPARK_CAPS["critic"]).text)
                 for p in prompts]
    lines = [agreement("spark-arm-a", base, base), agreement("spark-retest", base, again)]
    if roles_b.critic == "gemini":
        third = critic_port(night, ledger)  # each call held on vertex_usd
        lines.append(agreement(third.model, base, [read_verdict(third.ask(p)[0]) for p in prompts]))
    return lines


def _go(args: argparse.Namespace, roles_b: Roles, deps_a: Any, deps_b: Any, ledger: Any,
        critic_port: Callable[[date, Any], Any], judge_chat: Callable[..., ChatResult], clock: Callable[[], float]) -> int:
    job, rate = f"practice_ab:{args.night.isoformat()}:{args.user}", deps_a.prices.serve_gpu_per_hour
    evidence = list(deps_a.evidence(args.user).trainable(args.night))
    started = clock()
    with held(ledger, f"{job}:gpu", ("modal_usd",), usd(rate * (AB_MAX_S + WAKE_S + config.SCALEDOWN_S) / 3600),
              lambda: rate * (clock() - started + config.SCALEDOWN_S) / 3600):
        if deps_a.wake is not None and not deps_a.wake().ok:
            print(json.dumps({"line": "wake", "ok": False}))
            return 1
        chat_a = _Recorded(deps_a.spark_chat)
        run_a = _arm(job, "A", deps_a, chat_a, args, evidence, ledger, clock)
        replay = chat_a.scenes() if roles_b.person == "spark" else None  # the same messages for both arms
        run_b = _arm(job, "B", deps_b, _Recorded(deps_b.spark_chat, replay), args, evidence, ledger, clock)
    prompts = [prompt for prompt, _ in chat_a.critic()]
    base = [read_verdict(text) for _, text in chat_a.critic()]
    lines = _critics(job, roles_b, deps_a, prompts, base, critic_port, args.night, ledger)
    for line in lines:
        print(json.dumps(line))
    known, asked = {e.id: e for e in evidence}, [0]

    def ask(prompt: str) -> str:
        asked[0] += 1
        return judge_chat(JUDGE_MODEL, [{"role": "user", "content": prompt}], None, JUDGE_TIMEOUT_S,
                          temperature=JUDGE_TEMPERATURE).text

    with held(ledger, f"{job}:judge", ("meta_usd",), usd(JUDGE_USD_PER_CALL * 2 * len(run_a.transcripts)),
              lambda: JUDGE_USD_PER_CALL * asked[0]):
        judged = judge_arms(ask, lambda t: _known([known[i] for i in t.f_ids if i in known]),
                            run_a.transcripts, run_b.transcripts)
    print(json.dumps(judged))
    print(json.dumps(decide(roles_b, lines[0], lines[1], lines[2] if len(lines) > 2 else None, judged)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
