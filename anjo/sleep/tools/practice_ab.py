"""Today's practice roles against another setting (``ANJO_SLEEP_PRACTICE_ROLES``) on synthetic people's
nights (``python -m anjo.sleep.tools.practice_ab``). Without ``--go`` it prints the plan, the scenes the
bar needs, its holds and its expected cost, and sends nothing.

    python -m anjo.sleep.tools.practice_ab --arm-b critic=deepseek --case U:P:2026-10-20 ... (10 cases)
    ANJO_SLEEP_HOST=1 python -m anjo.sleep.tools.practice_ab ... --out DIR --go
    ANJO_SLEEP_HOST=1 python -m anjo.sleep.tools.practice_ab --pilot --arm-b critic=deepseek --case U:P:D --out DIR --go

A case is one synthetic person's night, ``user:persona:night``; the night's wiring (``--wiring``, loaded by
name) refuses a person not registered as synthetic. With ``--go``, case by case:

1. **Arm A** (today's roles) runs full practice; every Spark Open call is kept.
2. **Arm B** runs on the same night, seed and evidence. While Spark Open plays its person it is handed
   arm A's scenes, so both arms answer the same messages, and Glimmer's candidates are asked with the
   same seeds.
3. **The judge**, Gemini 3.8 Flash (``--judge``, the APD judge's Vertex call): a family neither arm uses
   (Spark Open teaches in both and judges in A; DeepSeek may judge in B; Glimmer answers), so it is
   refused if an arm ever does. It compares each scene's teacher material, the target and the rejected
   reply, from the two arms in both orders (``ab_judge_v1``), never told which arm, role or model wrote
   which. A verdict that flips with the order is a tie; a scene one arm dropped is that arm's loss.
4. **Critic agreement**, reported, never a bar: when arm B changes the critic, arm A's critic prompts go
   again to Spark Open (its own retest) and to arm B's critic. Spark Open's verdicts are not the truth.

**The bar**, fixed here before any run: arm B's tie-inclusive score against arm A (a win 1, a tie a half)
is non-inferior when its one-sided ``CONFIDENCE`` lower bound (normal, from the scenes' own spread) is
above 0.5 minus ``MARGIN``. ``SCENES_NEEDED`` gives that test ``POWER`` when the arms are equal, at the
widest spread a scene's score can have; ``--go`` refuses fewer. Scenes are counted as independent,
though a night's share a person. The default changes only when the bar holds. It prints JSON lines of
counts and dollars, never a scene's words; both arms' transcripts go to ``--out`` and never to a
person's sleep folder, so nothing here trains.

**Spend**, each held before its first call and settled at what it used: Spark Open's worst, an arm at a
time, on ``baseten_sleep_usd``; DeepSeek's calls on its critic's own pool (``practice.critics.POOL``,
each call held);
each judge call on ``vertex_usd``; the serving GPU's ``CASE_MAX_S`` a case, the wake and Modal's idle
window on ``modal_usd``.

**The pilot** (``--pilot``) proves the machinery end to end before the full run: one case (one
synthetic person-night of ``--scenes``), the same arms, the same blind judge in both orders and the
same critic agreement, with holds sized for that one case and its own job names. It prints what each
scene cost and how often the judge's verdict held when the order was swapped, and never a verdict:
it has no score, no bound and no switch, whatever its scenes say.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

from anjo.sleep.baseten_api import SPARK_OPEN
from anjo.sleep.context.template import render
from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.deps import NightOptions, Prices
from anjo.sleep.night.holds import SPARK_MAX_IN_TOKENS, held, spark_usd, spark_worst_usd, usd
from anjo.sleep.practice import critics
from anjo.sleep.practice.full import CRITIC_TEMPERATURE, SPARK_CAPS, spark_calls
from anjo.sleep.practice.roles import ROLES_ENV, TODAY, Roles, RolesError, practice_roles
from anjo.sleep.practice.run import SCENE_TIMEOUT_S, _known
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.probes.second_judge import MODEL_ID as JUDGE_MODEL
from anjo.sleep.serving import config

WIRING: Final = "anjo.sleep.app.night:build"
CRITIC: Final = "anjo.sleep.app.night:deepseek_practice_critic"
JUDGE: Final = "anjo.sleep.mind.apd.backend:_gemini"  # Vertex, thinking low, its reply cap (no second client)
JUDGE_FAMILY: Final = "gemini"
JUDGE_IN_PER_M, JUDGE_OUT_PER_M = 0.75, 3.75  # Gemini 3.8 Flash on Vertex, list price (2026-10-28)
JUDGE_MAX_OUT: Final = 400 + 512  # the APD path's reply cap and the thinking it reserves
JUDGE_PATH: Final = Path(__file__).parents[1] / "practice" / "prompts" / "ab_judge_v1.md"
SCENES_HEAD: Final = "# scenes_v1"
CRITIC_HEAD: Final = "# critic_v1"
CASE_MAX_S: Final = 20 * 60.0  # both arms' practice of one case on the serving GPU, at its worst
WAKE_S: Final = 600.0  # a cold start's worst
# The bar, fixed before any run (2026-10-29).
MARGIN: Final = 0.10  # B may score at most this far below an even 0.5
CONFIDENCE: Final = 0.95  # one-sided
POWER: Final = 0.80  # to pass when the arms are equal
WIDEST_SD: Final = 0.5  # a scene's score is 0, 0.5 or 1
_Z = statistics.NormalDist().inv_cdf
SCENES_NEEDED: Final = math.ceil(((_Z(CONFIDENCE) + _Z(POWER)) * WIDEST_SD / MARGIN) ** 2)  # 155
# Expected prices a call, for the plan's estimate only (holds use the worst case): night.cost.UNITS.
SPARK_CALL_USD: Final = 0.83 / 33  # the measured practice night: 33 Spark Open calls
DEEPSEEK_CALL_USD: Final = (6_000 * critics.DEEPSEEK_IN_PER_M + 300 * critics.DEEPSEEK_OUT_PER_M) / 1e6
JUDGE_CALL_USD: Final = (5_000 * JUDGE_IN_PER_M + 600 * JUDGE_OUT_PER_M) / 1e6
ARM_GPU_S: Final = 1174.0 / 16 * 2  # measured serial practice, at 8 at once


@dataclass(frozen=True)
class Case:
    user: str
    persona: str
    night: date


def _case(raw: str) -> Case:
    user, persona, night = (raw.split(":") + ["", ""])[:3]
    try:
        return Case(user, persona, date.fromisoformat(night))
    except ValueError:
        raise argparse.ArgumentTypeError("a case is user:persona:YYYY-MM-DD") from None


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
    """How often ``other`` agrees with Spark Open's verdicts on the same prompts (agreement, not accuracy)."""
    both = [(a, b) for a, b in zip(base, other, strict=True) if a is not None and b is not None]
    paired = [(a, b) for a, b in both if a.worst is not None and b.worst is not None]
    read = [b for b in other if b is not None]
    letters = sum(b.letters for b in read)
    return {
        "critic": name, "prompts": len(other), "unreadable": len(other) - len(read),
        "flag_rate": round(sum(len(b.flagged) for b in read) / letters, 4) if letters else None,
        "agree_best_with_spark": round(sum(a.best == b.best for a, b in both) / len(both), 4) if both else None,
        "agree_pair_with_spark": round(sum((a.worst is None) == (b.worst is None) for a, b in both) / len(both), 4)
        if both else None,
        "agree_worst_with_spark": round(sum(a.worst == b.worst for a, b in paired) / len(paired), 4) if paired else None,
    }


def families(roles: Roles) -> frozenset[str]:
    """The model families an arm's practice uses: Spark Open (Muse) teaches, and writes the scenes or
    judges unless another plays them; Glimmer answers; DeepSeek may judge."""
    return frozenset({"muse", "glimmer"} | ({"deepseek"} if roles.critic == "deepseek" else set()))


def _spark_calls(roles: Roles, scenes: int) -> int:
    """An arm's Spark Open calls at most, on a synthetic person (whose critic may be DeepSeek)."""
    lazy = roles.teacher == "lazy"
    critic = scenes * (2 if lazy else 1) if roles.critic == "deepseek" else 0
    return spark_calls(scenes, glimmer_person=roles.person == "glimmer", lazy_teacher=lazy) - critic


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


def judge_case(ask: Callable[[str], str], known: Callable[[PracticeTranscript], str], scenes: int,
               arm_a: Sequence[PracticeTranscript], arm_b: Sequence[PracticeTranscript], tally: dict[str, int]) -> list[float]:
    """Arm B's point on each of a case's scenes: judged with A shown first, then B first (a flip or an
    unreadable verdict a tie); a scene one arm dropped is that arm's loss, and one both dropped a tie."""
    by_a, by_b = {t.scene: t for t in arm_a}, {t.scene: t for t in arm_b}
    points: list[float] = []
    for index in range(scenes):
        a, b = by_a.get(index), by_b.get(index)
        if a is None or b is None:
            tally["dropped_a" if b is not None else "dropped_b" if a is not None else "dropped_both"] += 1
            points.append(1.0 if b is not None else 0.0 if a is not None else 0.5)
            continue
        first, second = _verdict(ask(judge_prompt(known(a), a, b))), _verdict(ask(judge_prompt(known(a), b, a)))
        tally["unreadable"] += (first is None) + (second is None)
        tally["unreadable_scenes"] += first is None or second is None
        b_first = {"2": 1.0, "1": 0.0}.get(first or "tie", 0.5)
        b_second = {"1": 1.0, "2": 0.0}.get(second or "tie", 0.5)
        tally["flips"] += b_first != b_second
        point = b_first if b_first == b_second else 0.5
        tally[{1.0: "win", 0.5: "tie", 0.0: "loss"}[point]] += 1
        points.append(point)
    return points


def non_inferior(points: Sequence[float]) -> dict[str, Any]:
    """The bar: arm B's score, its one-sided lower bound, and whether it clears 0.5 minus the margin."""
    n = len(points)
    mean = math.fsum(points) / n if n else 0.5
    spread = statistics.stdev(points) if n > 1 else WIDEST_SD
    lower = mean - _Z(CONFIDENCE) * spread / math.sqrt(n) if n else 0.0
    return {"n": n, "score_b": round(mean, 4), "lower_bound": round(lower, 4), "bar": 0.5 - MARGIN,
            "confidence": CONFIDENCE, "non_inferior": n >= SCENES_NEEDED and lower > 0.5 - MARGIN}


def _save(path: Path, transcripts: Sequence[PracticeTranscript]) -> None:
    def plain(value: object) -> object:
        return sorted(value) if isinstance(value, frozenset) else str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write("".join(json.dumps(asdict(t), default=plain) + "\n" for t in transcripts))


def _load(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


_DEEPSEEK_WORST: Final = critics.ThirdPartyCritic(
    "deepseek", lambda prompt: ChatResult("", (), "", 0, 0, "stop"), critics.DEEPSEEK_IN_PER_M,
    critics.DEEPSEEK_OUT_PER_M).worst_usd
_LIST: Final = Prices(spark_in_per_m=1.25, spark_out_per_m=4.25, train_gpu_per_hour=2.50, serve_gpu_per_hour=1.96)


def plan(args: argparse.Namespace, roles_b: Roles, rate: float) -> str:
    cases, scenes = len(args.case), len(args.case) * args.scenes
    arms = sum(_spark_calls(r, args.scenes) for r in (TODAY, roles_b))
    new_critic = roles_b.critic != TODAY.critic
    spark = cases * (arms + (args.scenes if new_critic else 0))  # both arms, and the retest of A's critic
    deepseek = cases * args.scenes * (2 if new_critic else 0)
    gpu_s = cases * 2 * ARM_GPU_S + config.SCALEDOWN_S + 180.0
    expected = (spark * SPARK_CALL_USD + deepseek * DEEPSEEK_CALL_USD + 2 * scenes * JUDGE_CALL_USD
                + gpu_s * rate / 3600)
    worst_judge = (SPARK_MAX_IN_TOKENS * JUDGE_IN_PER_M + JUDGE_MAX_OUT * JUDGE_OUT_PER_M) / 1e6
    head = (f"practice A/B pilot, 1 synthetic person-night of {args.scenes} scenes: proves the machinery; prints "
            "the cost a scene and the judge's agreement across orders, never a verdict") if args.pilot else (
        f"practice A/B, {cases} synthetic person-nights of {args.scenes} scenes: {scenes} scenes "
        f"(the bar needs {SCENES_NEEDED})")
    bar = (f"  bar       none: a pilot prints no score, bound or switch (the full run needs {SCENES_NEEDED} "
           "scenes)") if args.pilot else (
        f"  bar       B non-inferior: its {CONFIDENCE:.0%} one-sided lower bound above {0.5 - MARGIN:.2f} "
        f"(margin {MARGIN}, power {POWER:.0%} at equal arms); a dropped scene is its arm's loss")
    lines = [
        head,
        f"  arm A     {asdict(TODAY)} (today)",
        f"  arm B     {asdict(roles_b)}",
        bar,
        f"  judge     {JUDGE_MODEL} ({JUDGE_FAMILY}, in neither arm), {2 * scenes} calls (both orders), each held "
        f"on vertex_usd at ${worst_judge:.3f}",
        f"  spark     {spark} calls at most, an arm at a time on baseten_sleep_usd (peak hold "
        f"${spark_worst_usd(_LIST, max(_spark_calls(r, args.scenes) for r in (TODAY, roles_b))):.2f})",
        f"  deepseek  {deepseek} calls at most, each held on {critics.POOL} at ${_DEEPSEEK_WORST:.3f}",
        f"  gpu       {CASE_MAX_S / 60:.0f} min a case, the wake and the {config.SCALEDOWN_S / 60:.0f}-min idle window "
        f"held on modal_usd: ${rate * (cases * CASE_MAX_S + WAKE_S + config.SCALEDOWN_S) / 3600:.2f} at ${rate:.2f}/h",
        f"  cost      about ${expected:.2f} expected (estimate: ${SPARK_CALL_USD:.3f} a Spark Open call, "
        f"${JUDGE_CALL_USD:.3f} a judge call, {ARM_GPU_S:.0f} GPU-s an arm)",
        "  agreement critic agreement with Spark Open is reported, never a bar",
        "  prints    counts and dollars; never a scene's words",
        f"  saves     both arms' transcripts under --out ({args.out or 'required with --go'})",
    ]
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None, *, wiring: Callable[[Any, NightOptions], Any] | None = None,
    critic_port: Callable[[date, Any], Any] | None = None, judge: Callable[[str], tuple[str, int, int]] | None = None,
    ledger: Any = ..., clock: Callable[[], float] = time.monotonic,
) -> int:
    parser = argparse.ArgumentParser(description="Today's practice roles against another setting, plan-first.")
    parser.add_argument("--case", type=_case, action="append", required=True,
                        help="a synthetic person's night, user:persona:YYYY-MM-DD (repeat)")
    parser.add_argument("--arm-b", required=True, help=f"arm B's {ROLES_ENV}, such as critic=deepseek")
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--wiring", default=WIRING, help="the night's NightDeps builder, <module>:<function>")
    parser.add_argument("--critic", default=CRITIC, help="arm B's third-party critic, <module>:<function>")
    parser.add_argument("--judge", default=JUDGE, help="the judge's text call, <module>:<function>")
    parser.add_argument("--out", help="where both arms' transcripts go (required with --go)")
    parser.add_argument("--go", action="store_true", help="run it; without it, print the plan only")
    parser.add_argument("--pilot", action="store_true",
                        help="one case to prove the machinery: cost a scene and judge agreement, never a verdict")
    args = parser.parse_args(argv)
    try:
        roles_b = practice_roles({ROLES_ENV: args.arm_b})
    except RolesError as error:
        parser.error(str(error))
    if roles_b == TODAY:
        parser.error("arm B is today's roles: there is nothing to compare")
    if JUDGE_FAMILY in families(TODAY) | families(roles_b):
        parser.error(f"the judge's family ({JUDGE_FAMILY}) plays in an arm")
    if args.pilot and len(args.case) != 1:
        parser.error("a pilot is one case; the full run takes the rest")
    if not args.go:
        print(plan(args, roles_b, _LIST.serve_gpu_per_hour))
        return 0
    if not args.pilot and len(args.case) * args.scenes < SCENES_NEEDED:
        parser.error(f"the bar needs {SCENES_NEEDED} scenes: add cases (or --pilot for one)")
    if not args.out:
        parser.error("--go needs --out")
    build = wiring or _load(args.wiring)
    deps = [(build(None, _options(case, args.scenes, TODAY)), build(None, _options(case, args.scenes, roles_b)))
            for case in args.case]  # no trainer: nothing here trains
    book = deps[0][0].ledger if ledger is ... else ledger
    print(plan(args, roles_b, deps[0][0].prices.serve_gpu_per_hour))
    return _go(args, roles_b, deps, book, critic_port or _load(args.critic), judge or _load(args.judge), clock)


def _options(case: Case, scenes: int, roles: Roles) -> NightOptions:
    return NightOptions(night=case.night, personas={case.user: case.persona}, practice_scenes=scenes,
                        practice_roles=roles)


def _arm(job: str, arm: str, deps: Any, chat: _Recorded, case: Case, evidence: list[Any], out: Path,
         ledger: Any) -> Any:
    runner = deps.practice(chat, case.night)
    with held(ledger, f"{job}:spark:{arm}", ("baseten_sleep_usd",),
              usd(spark_worst_usd(deps.prices, runner.max_spark_calls)), lambda: chat.usd(deps.prices)):
        run = runner.run(case.user, case.night, evidence, None, at=datetime.now(UTC))
    _save(out / f"arm_{arm.lower()}.jsonl", run.transcripts)
    print(json.dumps({"case": job, "arm": arm, "roles": dict(run.roles), "scenes": len(run.transcripts),
                      "flagged": run.flagged, "pairs": run.pairs, "teacher_targets": run.teacher_targets,
                      "usd": {"spark": round(chat.usd(deps.prices), 4), "critic": round(run.critic_usd, 4)}}))
    return run


def _critics(job: str, deps: Any, prompts: Sequence[str], third: Any, ledger: Any,
             spend: dict[str, float]) -> tuple[list[Read | None], list[Read | None]]:
    """Arm A's critic prompts asked again: of Spark Open (its retest) and of arm B's critic."""
    retest = _Recorded(deps.spark_chat)
    with held(ledger, f"{job}:spark:retest", ("baseten_sleep_usd",), usd(spark_worst_usd(deps.prices, len(prompts))),
              lambda: retest.usd(deps.prices)):
        again = [read_verdict(retest(SPARK_OPEN, [{"role": "user", "content": p}], None, SCENE_TIMEOUT_S,
                                     temperature=CRITIC_TEMPERATURE, max_tokens=SPARK_CAPS["critic"]).text)
                 for p in prompts]
    replies = [third.ask(p) for p in prompts]  # each call held on its own pool
    spend["spark"] += retest.usd(deps.prices)
    spend["critic"] += sum(third.usd(r.input_tokens, r.output_tokens) for r in replies)
    return again, [None if r.finish_reason == "length" else read_verdict(r.text) for r in replies]


def _held_judge(judge: Callable[[str], tuple[str, int, int]], ledger: Any, job: str,
                spend: dict[str, float]) -> Callable[[str], str]:
    numbers = itertools.count(1)
    worst = (SPARK_MAX_IN_TOKENS * JUDGE_IN_PER_M + JUDGE_MAX_OUT * JUDGE_OUT_PER_M) / 1e6

    def ask(prompt: str) -> str:
        spent = [0.0]
        with held(ledger, f"{job}:judge:{next(numbers)}", ("vertex_usd",), usd(worst), lambda: spent[0]):
            text, tokens_in, tokens_out = judge(prompt)
            spent[0] = (tokens_in * JUDGE_IN_PER_M + tokens_out * JUDGE_OUT_PER_M) / 1e6
        spend["judge"] += spent[0]
        return text

    return ask


def _go(args: argparse.Namespace, roles_b: Roles, deps: Sequence[tuple[Any, Any]], ledger: Any,
        critic_port: Callable[[date, Any], Any], judge: Callable[[str], tuple[str, int, int]],
        clock: Callable[[], float]) -> int:
    first = deps[0][0]
    rate, out = first.prices.serve_gpu_per_hour, Path(args.out)
    run_job = f"practice_ab:{'pilot:' if args.pilot else ''}{int(time.time())}"
    tally = dict.fromkeys(("win", "tie", "loss", "flips", "unreadable", "unreadable_scenes", "dropped_a", "dropped_b",
                           "dropped_both"), 0)
    spend = dict.fromkeys(("spark", "critic", "judge", "gpu"), 0.0)
    points: list[float] = []
    base: list[Read | None] = []
    again: list[Read | None] = []
    other: list[Read | None] = []
    ask = _held_judge(judge, ledger, run_job, spend)
    started = clock()
    bound = rate * (len(deps) * CASE_MAX_S + WAKE_S + config.SCALEDOWN_S) / 3600
    with held(ledger, f"{run_job}:gpu", ("modal_usd",), usd(bound),
              lambda: rate * (clock() - started + config.SCALEDOWN_S) / 3600):
        if first.wake is not None and not first.wake().ok:
            print(json.dumps({"line": "wake", "ok": False}))
            return 1
        for case, (deps_a, deps_b) in zip(args.case, deps, strict=True):
            job = f"{run_job}:{case.night.isoformat()}:{case.user}"
            evidence = list(deps_a.evidence(case.user).trainable(case.night))
            chat_a = _Recorded(deps_a.spark_chat)
            run_a = _arm(job, "A", deps_a, chat_a, case, evidence, out, ledger)
            replay = chat_a.scenes() if roles_b.person == "spark" else None  # the same messages for both arms
            chat_b = _Recorded(deps_b.spark_chat, replay)
            run_b = _arm(job, "B", deps_b, chat_b, case, evidence, out, ledger)
            spend["spark"] += chat_a.usd(deps_a.prices) + chat_b.usd(deps_b.prices)
            spend["critic"] += run_a.critic_usd + run_b.critic_usd
            known = {e.id: e for e in evidence}
            points += judge_case(ask, lambda t: _known([known[i] for i in t.f_ids if i in known]), args.scenes,
                                 run_a.transcripts, run_b.transcripts, tally)
            if roles_b.critic != TODAY.critic:
                read = [read_verdict(text) for _, text in chat_a.critic()]
                retest, new = _critics(job, deps_a, [p for p, _ in chat_a.critic()],
                                       critic_port(case.night, ledger), ledger, spend)
                base, again, other = base + read, again + retest, other + new
        spend["gpu"] = rate * (clock() - started + config.SCALEDOWN_S) / 3600
    if roles_b.critic != TODAY.critic:
        for line in (agreement("spark-retest", base, again), agreement(roles_b.critic, base, other)):
            print(json.dumps(line))
    if args.pilot:
        print(json.dumps(pilot_line(tally, len(points), spend)))
        return 0
    decision = non_inferior(points)
    print(json.dumps({"judge": tally, **decision}))
    print(json.dumps({"switch_default": decision["non_inferior"]}))
    return 0


def pilot_line(tally: dict[str, int], scenes: int, spend: dict[str, float]) -> dict[str, Any]:
    """What a pilot reports: how often the judge held its verdict across the two orders, the scenes
    dropped or left unreadable, and its spend a scene. No win, tie or loss count, score, bound or
    switch: a pilot's scenes are too few to decide anything, so nothing here reads as one."""
    dropped = tally["dropped_a"] + tally["dropped_b"] + tally["dropped_both"]
    judged = scenes - dropped
    total = math.fsum(spend.values())
    return {"pilot": True, "scenes": scenes,
            "judge_order_agreement": round(1 - tally["flips"] / judged, 4) if judged else None,
            "dropped_scenes": dropped, "unreadable_scenes": tally["unreadable_scenes"],
            "usd": {step: round(value, 4) for step, value in spend.items()} | {"total": round(total, 4)},
            "usd_per_scene": round(total / scenes, 4) if scenes else None}


if __name__ == "__main__":
    raise SystemExit(main())
