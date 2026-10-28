"""The thought channel's evaluation, live (``mind.core.reader_eval``): plan first, then, with ``--yes``,
the replies on the served Glimmer under a hold on ``modal_usd``, then the blind judge under a hold on
``meta_usd``, each settled at what it used. Both run under MIND's lock after the thought channel's own
sweep (``thought_channel.swept_channel``), their holds of its kind (``thought_channel:eval_serve:``,
``thought_channel:eval_judge:``).

    python -m anjo.sleep.tools.thought_channel_eval --cases FILE --readings FILE --projector FILE
        --adapter NAME --out DIR [--export TEST] [--pairs DIR] [--data DIR] [--judge-prices IN,OUT]
        [--hours 1] [--concurrency 4] [--max-tokens 400] [--seed 0] [--yes]

1. **Replies.** Every case (``thought_channel cases``) in four arms (``reader_eval.arm_prompts``): the
   reader adapter with the case's own state as soft tokens, with the empty reading, with another
   person's state, and Glimmer as today with the cited events pasted as text. Same message, system
   turn and seed in every arm. Each reply is kept in ``DIR/replies.jsonl`` as it comes, so a rerun
   asks only for what is missing. The hold is ``--hours`` of the serving GPU (the fourth of
   ``ANJO_SLEEP_PRICES``) and the embedding call, then the idle window; no request starts that could
   pass ``--hours``.
2. **Scores**, only after every reply is written: the problems are read then, and only then, from the
   MIND data dir (``--data``: core-test's sealed problem files and the pairs), never by the replies.
   A deterministic pre-score (the transplant's flips) and Muse Spark as the blind judge, asked twice
   per reply, the problems in opposite orders, counted only when both agree; its answers are kept by
   prompt in ``DIR/judge.jsonl``. The hold is every call at ``JUDGE_IN_TOKENS`` in and
   ``JUDGE_MAX_TOKENS`` out at ``--judge-prices`` (dollars per million).

``DIR/report.json``: per arm, raise hits on the days a problem is open, false raises per 100
person-days, the transplant's flip rate (pre-scored and judged) and how often it left the unrelated
item as it was (raised on both sides or on neither, where the judge agreed on both), the judge's disagreements and mean
reply tokens with the arm's share of the serving spend. If ``zero`` or ``shuffled`` does as well as
``state``, it says the channel carries nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from anjo.sleep import meta_api
from anjo.sleep.budget import Bound, Ledger, default_ledger
from anjo.sleep.mind.apd import labels
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.core import reader_data, reader_eval
from anjo.sleep.mind.core.projector import Projector, Reading
from anjo.sleep.mind.sweep import MindBusy
from anjo.sleep.night.holds import hold, usd
from anjo.sleep.serving import backend, config, soft_tokens
from anjo.sleep.serving.client import REQUEST_TIMEOUT_S, ServingError, VllmClient
from anjo.sleep.serving.soft_tokens import EmbeddingRows, SoftTokens
from anjo.sleep.stops import BudgetStop
from anjo.sleep.tools import soft_tokens_live
from anjo.sleep.tools.thought_channel import HOLD_KIND, SHARED_PERSON, records_root, stamp, swept_channel
from anjo.sleep.train import lora
from anjo.sleep.train.runners.modal import SdkCalls
from anjo.sleep.types import CheckpointRef

SERVE_POOL: Final = "modal_usd"
JUDGE_POOL: Final = "meta_usd"
TEMPERATURE: Final = 0.6
JUDGE_IN_TOKENS: Final = 4096  # a judge prompt is cut to this many characters, so at most this many tokens
JUDGE_MAX_TOKENS: Final = 200
JUDGE_TIMEOUT_S: Final = 120.0
STEP_MAX_S: Final = 2 * REQUEST_TIMEOUT_S + soft_tokens_live.EMBED_CALL_MAX_S  # one reply's worst case
CARRIES_MARGIN: Final = 0.05  # the state arm's lead over zero and shuffled that counts as carrying something
Judge = Callable[[str], tuple[str, int, int]]  # a prompt -> the answer, its tokens in and out


def _muse(prompt: str) -> tuple[str, int, int]:  # pragma: no cover - a paid call
    found = meta_api.chat("muse-spark", [{"role": "user", "content": prompt}], None, JUDGE_TIMEOUT_S,
                          temperature=0.0, max_tokens=JUDGE_MAX_TOKENS)
    return found.text, found.input_tokens, found.output_tokens


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _judge_prices(raw: str) -> tuple[float, float]:
    try:
        found = tuple(float(p) for p in raw.split(","))
    except ValueError:
        found = ()
    if len(found) != 2 or any(p <= 0 for p in found):
        raise ValueError("--judge-prices is the judge's dollars per million tokens in and out, as IN,OUT")
    return found


# ── The replies ───────────────────────────────────────────────────────────────


def _life_events(args: argparse.Namespace, life: str) -> dict[str, reader_data.Event]:
    for folder in (args.export, args.pairs):
        if folder is not None and (path := folder / "events" / f"{life}.jsonl").is_file():
            return {e.event_id: e for e in (reader_data.Event.from_record(r) for r in _jsonl(path))}
    raise ValueError(f"no events for {life} in --export or --pairs")


def _notes(cases: Sequence[reader_eval.Case], readings: Mapping[str, tuple[Any, tuple[tuple[str, ...], ...]]],
           args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    """The text arm's memory: the events each case's own reading cites."""
    found: dict[str, tuple[str, ...]] = {}
    for case in cases:
        snapshot, cites = readings[case.reading_key]
        found[case.case_id] = reader_eval.notes_for(replace(snapshot.reading, cites=cites),
                                                    _life_events(args, case.reading_key.split(":", 1)[0]))
    return found


@dataclass(frozen=True)
class _Deadline:
    clock: Callable[[], float]
    ends: float

    def room(self) -> bool:
        return self.clock() + STEP_MAX_S <= self.ends


def _replies(args: argparse.Namespace, prompts: Sequence[reader_eval.ArmPrompt], live: VllmClient,
             soft: Mapping[str, SoftTokens], deadline: _Deadline, out: Callable[[str], None]) -> None:
    path = args.out / "replies.jsonl"
    reader = CheckpointRef(args.adapter, SHARED_PERSON, reader_data.READER_DAY, f"{lora.VOLUME_URI}/{args.adapter}",
                           None, live.rendering)

    def one(p: reader_eval.ArmPrompt) -> str:
        if not deadline.room():
            return "skipped"
        try:
            found = live.complete(reader if p.reader else None, p.messages, temperature=TEMPERATURE, seed=p.seed,
                                  max_tokens=args.max_tokens, soft=None if p.reading is None else soft[p.reading])
        except ServingError:
            return "failed"
        _append(path, {"case_id": p.case_id, "arm": p.arm, "text": found.text, "tokens_out": found.output_tokens})
        return "answered"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, prompts))
    out(f"replies: {results.count('answered')} answered, {results.count('failed')} failed, "
        f"{results.count('skipped')} skipped at the {args.hours:g} h ceiling")


def _serve(args: argparse.Namespace, readings: Mapping[str, tuple[Any, Any]], todo: Sequence[reader_eval.ArmPrompt],
           bound: Bound, led: Ledger, make: Callable[[], VllmClient], wake: Callable[[], bool],
           rows: EmbeddingRows | None, clock: Callable[[], float], out: Callable[[str], None]) -> float | None:
    """The replies asked for, under the hold; what was settled, or None when the server didn't wake."""
    projector = Projector.load(args.projector.read_bytes())
    keys = {p.reading for p in todo if p.reading is not None}
    soft = {k: projector.tokens(Reading.empty(projector.k, projector.d) if k == reader_eval.EMPTY else readings[k][0].reading)
            for k in keys}
    reservation = hold(led, f"{HOLD_KIND}:eval_serve:{stamp()}", (SERVE_POOL,), bound)
    started = clock()
    try:
        if not wake():
            out("not run: the server didn't wake")
            return None
        live = make().once().with_load_wait(0).with_soft_tokens(rows or soft_tokens.ModalEmbeddingRows())
        _replies(args, todo, live.on(reader_data.READER_DAY), soft, _Deadline(clock, started + args.hours * 3600.0), out)
    finally:
        elapsed = clock() - started
        spent = min((elapsed + soft_tokens_live.IDLE_S) * bound.rate_per_hour / 3600.0, bound.amount)
        reservation.settle(spent)
        out(f"serving: ${spent:.2f} settled on {SERVE_POOL} for {elapsed:.0f} s and the idle window")
    return spent


def _serving_total(folder: Path, spent: float) -> float:
    """Every run's serving spend on these replies, this one's added (``DIR/spend.json``)."""
    path = folder / "spend.json"
    total = (json.loads(path.read_text(encoding="utf-8"))["serving_usd"] if path.is_file() else 0.0) + spent
    path.write_text(json.dumps({"serving_usd": total}), encoding="utf-8")
    return float(total)


# ── The scores ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Problems:
    topics: tuple[str, ...]  # the judge's list
    open: frozenset[int]  # the listed problems the reply should raise


def _life_problems(data: DataDir, cases: Sequence[reader_eval.Case]) -> dict[str, _Problems]:
    """Each person-day's problems, from the sealed files: those knowable by then, and which are open."""
    found = {}
    for life_id in sorted({c.life_id for c in cases if c.kind == "life"}):
        life = data.life(life_id)
        stream = [reader_data.Event.from_record(r) for r in labels.events(life)]
        items = tuple(reader_data.Item.from_record(r) for r in labels.item_labels(life, data.sealed(life_id)))
        known = reader_data.Life(life_id, life.field, tuple(stream), items)
        where = {e.event_id: n for n, e in enumerate(stream)}
        for case in (c for c in cases if c.life_id == life_id):
            at = where[case.event_id]
            listed = [i for i in items if where.get(i.knowable_event_id, len(stream)) < at]
            live = {i.item_id for i in reader_data.open_items(known, at)}
            found[case.case_id] = _Problems(tuple(i.topic for i in listed),
                                            frozenset(n for n, i in enumerate(listed) if i.item_id in live))
    return found


def _pair_problems(data: DataDir, cases: Sequence[reader_eval.Case]) -> tuple[dict[str, _Problems], dict[str, frozenset[str]]]:
    """Each history's list (the commitment first, then the unrelated item) and the commitment's words."""
    found, words = {}, {}
    for pair_id in sorted({str(c.pair_id) for c in cases if c.pair_id is not None}):
        pair = data.pair(pair_id)
        slot = next(i for i in pair.failed_problems.items if i.id != pair.unrelated_item)
        other = next(i for i in pair.kept_problems.items if i.id == pair.unrelated_item)
        texts = {e.id: e.text for e in pair.failed.evidence}
        cited = [texts[i] for i in pair.slot_ids if i in texts]
        words[pair_id] = reader_eval.keywords(cited, [t for i, t in texts.items() if i not in pair.slot_ids])
        for case in (c for c in cases if c.pair_id == pair_id):
            found[case.case_id] = _Problems((slot.topic, other.topic), frozenset({0}) if case.kind == "failed" else frozenset())
    return found, words


class _CachedJudge:
    """The judge's answers by prompt, kept as they come; the spend counted from what each call used."""

    def __init__(self, ask: Judge, path: Path, prices: tuple[float, float]) -> None:
        self._ask, self._path, self._prices = ask, path, prices
        self._seen = {str(r["id"]): str(r["answer"]) for r in _jsonl(path)}
        self.calls, self.usd = 0, 0.0

    def __call__(self, prompt: str) -> str:
        prompt = prompt[:JUDGE_IN_TOKENS]
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if key not in self._seen:
            answer, tokens_in, tokens_out = self._ask(prompt)
            self.calls += 1
            self.usd += (tokens_in * self._prices[0] + tokens_out * self._prices[1]) / 1e6
            self._seen[key] = answer
            _append(self._path, {"id": key, "answer": answer})
        return self._seen[key]


def _arm_scores(arm: str, cases: Sequence[reader_eval.Case], replies: Mapping[tuple[str, str], Mapping[str, Any]],
                problems: Mapping[str, _Problems], words: Mapping[str, frozenset[str]],
                ask: Callable[[str], str]) -> dict[str, Any]:
    verdicts = {c.case_id: reader_eval.judged(ask, c.message, str(replies[(c.case_id, arm)]["text"]),
                                              problems[c.case_id].topics)
                for c in cases if (c.case_id, arm) in replies}
    days = [c for c in cases if c.kind == "life" and c.case_id in verdicts]
    raise_days = [c for c in days if problems[c.case_id].open]
    hits = sum(verdicts[c.case_id].raised in problems[c.case_id].open for c in raise_days)
    false = sum(v.agree and ((v.raised is not None and v.raised not in problems[c.case_id].open)
                             or (v.other and not problems[c.case_id].open))
                for c in days for v in (verdicts[c.case_id],))
    pairs = [(a, b) for a in cases for b in cases if a.kind == "failed" and b.kind == "kept" and a.pair_id == b.pair_id]
    texts = {c.case_id: str(replies[(c.case_id, arm)]["text"]) for c in cases if (c.case_id, arm) in replies}
    flipped, scored = reader_eval.flips(pairs, texts, keywords=words)
    judged = sum(verdicts[a.case_id].raised == 0 and verdicts[b.case_id].agree and verdicts[b.case_id].raised != 0
                 for a, b in pairs if a.case_id in verdicts and b.case_id in verdicts)
    both = [(verdicts[a.case_id], verdicts[b.case_id]) for a, b in pairs
            if a.case_id in verdicts and b.case_id in verdicts and verdicts[a.case_id].agree and verdicts[b.case_id].agree]
    stable = sum((va.raised == 1) == (vb.raised == 1) for va, vb in both)  # the unrelated item, listed second
    tokens = [int(replies[(c.case_id, arm)]["tokens_out"]) for c in cases if (c.case_id, arm) in replies]
    return {"replies": len(verdicts), "raise_days": len(raise_days), "raise_hit": hits / max(len(raise_days), 1),
            "false_per_100_days": 100.0 * false / max(len(days), 1), "pairs": scored,
            "flip_prescore": flipped / max(scored, 1), "flip_judged": judged / max(scored, 1),
            "unrelated_stable": stable / max(len(both), 1), "unrelated_pairs": len(both),
            "disagree": sum(not v.agree for v in verdicts.values()) / max(len(verdicts), 1),
            "mean_tokens": sum(tokens) / max(len(tokens), 1), "tokens": sum(tokens)}


def _verdict(arms: Mapping[str, Mapping[str, Any]]) -> str:
    state = arms["state"]
    best = max((arms[a] for a in ("zero", "shuffled")), key=lambda s: (s["raise_hit"], s["flip_judged"]))
    if state["raise_hit"] > best["raise_hit"] + CARRIES_MARGIN or state["flip_judged"] > best["flip_judged"] + CARRIES_MARGIN:
        return "the channel carries the state: the state arm leads zero and shuffled"
    return "the channel carries nothing here: zero or shuffled does as well as the state"


def _score(args: argparse.Namespace, cases: Sequence[reader_eval.Case], judge: Judge, worst: float, led: Ledger,
           serve_usd: float, out: Callable[[str], None]) -> dict[str, Any]:
    replies = {(str(r["case_id"]), str(r["arm"])): r for r in _jsonl(args.out / "replies.jsonl")}
    data = DataDir(args.data)  # the problems are read here, after every reply is written
    problems = _life_problems(data, cases)
    pair_lists, words = _pair_problems(data, cases)
    problems |= pair_lists
    cached = _CachedJudge(judge, args.out / "judge.jsonl", _judge_prices(args.judge_prices))
    reservation = hold(led, f"{HOLD_KIND}:eval_judge:{stamp()}", (JUDGE_POOL,), usd(max(worst, 0.01)))
    try:
        arms = {arm: _arm_scores(arm, cases, replies, problems, words, cached) for arm in reader_eval.ARMS}
    finally:
        reservation.settle(min(cached.usd, reservation.bound.amount))
        out(f"judge: {cached.calls} calls, ${cached.usd:.2f} settled on {JUDGE_POOL}")
    total = sum(a["tokens"] for a in arms.values()) or 1
    for arm in arms.values():
        arm["serving_usd_share"] = serve_usd * arm["tokens"] / total
    return {"arms": arms, "verdict": _verdict(arms), "cases": len(cases), "seed": args.seed}


# ── The command ───────────────────────────────────────────────────────────────


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="thought_channel_eval", description=(__doc__ or "").split("\n\n")[0])
    for flag in ("--cases", "--readings", "--projector", "--out"):
        parser.add_argument(flag, type=Path, required=True)
    parser.add_argument("--adapter", required=True, help="the reader adapter, as thought_channel train named it")
    parser.add_argument("--export", type=Path, default=None, help="the core-test export the cases came from (events)")
    parser.add_argument("--pairs", type=Path, default=None, help="the pairs export the cases came from (events)")
    parser.add_argument("--data", type=Path, default=None, help="the MIND data dir: the problems, read only to score")
    parser.add_argument("--judge-prices", default="", help="the judge's dollars per million tokens, IN,OUT")
    parser.add_argument("--hours", type=float, default=1.0, help="the serving ceiling, and its hold")
    parser.add_argument("--concurrency", type=int, default=4, choices=range(1, config.CONCURRENT_REQUESTS + 1),
                        metavar="C")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes", action="store_true", help="hold, then send")
    args = parser.parse_args(list(argv))
    config.check_adapter_name(args.adapter)
    return args


def _plan(args: argparse.Namespace, todo: int, cases: int, bound: Bound, judge_worst: float,
          out: Callable[[str], None]) -> None:
    out(f"thought channel eval: {cases} cases in {len(reader_eval.ARMS)} arms ({', '.join(reader_eval.ARMS)}), "
        f"{todo} replies to ask for, at most {args.max_tokens} tokens each, concurrency {args.concurrency}")
    out(f"hold ${bound.amount:.2f} on {SERVE_POOL} (${bound.rate_per_hour:.2f}/h for {args.hours:g} h of serving and "
        f"{soft_tokens_live.IDLE_S / 60:.0f} min idle); no request starts that could pass {args.hours:g} h")
    out(f"then the judge: {2 * cases * len(reader_eval.ARMS)} Muse Spark calls at most, hold ${judge_worst:.2f} on "
        f"{JUDGE_POOL} (each at {JUDGE_IN_TOKENS} tokens in and {JUDGE_MAX_TOKENS} out); the problems are read "
        "from --data only after every reply is written")
    out("run again with --yes to send")


def main(
    argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print, ledger: Ledger | None = None,
    environ: Mapping[str, str] | None = None, client: Callable[[], VllmClient] | None = None,
    wake: Callable[[], bool] | None = None, rows: EmbeddingRows | None = None, judge: Judge | None = None,
    clock: Callable[[], float] = time.monotonic, records: Path | None = None,
) -> int:
    args = _parse([] if argv is None else argv)
    env = os.environ if environ is None else environ
    try:
        rate = soft_tokens_live.serve_rate(env) + soft_tokens_live.EMBED_USD_PER_HOUR
        prices = _judge_prices(args.judge_prices)
    except ValueError as error:
        out(f"not run: {error}")
        return 2
    bound = Bound(rate, args.hours * 3600.0 + soft_tokens_live.IDLE_S)
    cases = [reader_eval.Case.from_json(r) for r in _jsonl(args.cases)]
    done = {(str(r["case_id"]), str(r["arm"])) for r in _jsonl(args.out / "replies.jsonl")}
    judge_worst = 2 * len(cases) * len(reader_eval.ARMS) * (JUDGE_IN_TOKENS * prices[0] + JUDGE_MAX_TOKENS * prices[1]) / 1e6
    todo_count = len(cases) * len(reader_eval.ARMS) - len(done)
    if not args.yes:
        _plan(args, todo_count, len(cases), bound, judge_worst, out)
        return 0
    if args.data is None:
        out("not run: --data names the MIND data dir the problems are scored from")
        return 2
    if client is None or wake is None:
        url, token = env.get(backend.MODAL_URL_ENV, ""), env.get(backend.MODAL_TOKEN_ENV, "")
        if not url or not token:
            out(f"not run: {backend.MODAL_URL_ENV} and {backend.MODAL_TOKEN_ENV} are set to the serving app")
            return 2
        serving = backend.ModalServing(url, token)
        client = client or serving.client
        wake = wake or (lambda: serving.wake().ok)
    led = ledger or default_ledger()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        readings = reader_data.read_readings(args.readings)
        missing = sorted({c.reading_key for c in cases} - readings.keys())
        if missing:
            raise ValueError(f"{len(missing)} cases have no reading (first {missing[0]}): run thought_channel readings")
        prompts = reader_eval.arm_prompts(cases, _notes(cases, readings, args), seed=args.seed)
        todo = [p for p in prompts if (p.case_id, p.arm) not in done]
        with swept_channel(led, args.out, records_root(env, records), SdkCalls):
            serve_usd = _serve(args, readings, todo, bound, led, client, wake, rows, clock, out) if todo else 0.0
            if serve_usd is None:
                return 1
            report = _score(args, cases, judge or _muse, judge_worst, led, _serving_total(args.out, serve_usd), out)
    except (BudgetStop, MindBusy, ValueError) as stop:
        out(f"not run: {stop}")
        return 2
    (args.out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    for arm, s in report["arms"].items():
        out(f"{arm}: raise hit {s['raise_hit']:.2f} on {s['raise_days']} days, {s['false_per_100_days']:.1f} false "
            f"raises per 100 days, flips {s['flip_prescore']:.2f} pre-scored / {s['flip_judged']:.2f} judged of "
            f"{s['pairs']} pairs, {s['mean_tokens']:.0f} tokens a reply")
    out(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
