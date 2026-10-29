"""Prepare mapped synthetic development lives for MIND collect/fit, offline by default.

The plan reads stores and recorded development points, but never calls a model or writes.
All snapshot targets (plus expectation targets) are covered, crossed with all held-out work:
this conservative superset covers D1 even when it is replayed later, without paying for D1
in a plan or exposing a decider's identity to the labeller. Existing labels are reused only
when their prompt, input fingerprint, coverage and registered digest still match.

The host's morning price configuration (ANJO_SLEEP_PRICES, input/output USD per million)
is used for Muse Spark too, as in app.morning. Estimates assume 3,000 output tokens;
the hold caps each completion at 16,384, with a conservative UTF-8 input bound. Calls
that fail without usage retain their full per-call charge; successful calls settle at tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from anjo.sleep import meta_api
from anjo.sleep.budget import Ledger, default_ledger
from anjo.sleep.context.brief import ChatFn
from anjo.sleep.context.template import render
from anjo.sleep.evidence.jsonl import canonical, read_records
from anjo.sleep.evidence.store import evidence_from_record
from anjo.sleep.evidence.targets import fold
from anjo.sleep.mind import collect, points
from anjo.sleep.mind.expect_register import development_accounts
from anjo.sleep.model_api import ChatResult, ModelApiError
from anjo.sleep.night.holds import held, usd
from anjo.sleep.night.nightlog import NightLog
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.prereg import ledger as prereg
from anjo.sleep.probes import relevance
from anjo.sleep.tools.fit_estimator import _named, host_check

MAX_OUTPUT = 16_384
EXPECTED_OUTPUT = 3_000
STORAGE = "anjo.sleep.app.storage:"


class Refused(ValueError):
    """An unsafe account, changed frozen artifact, or unusable development life."""


@dataclass(frozen=True)
class Records:
    targets_path: Path
    evidence_path: Path
    targets: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]


def host_records(user: str) -> Records:
    """Decode target events intact; evidence uses the store's existing export port."""
    targets = _named(STORAGE + "target_store")(user)
    evidence = _named(STORAGE + "evidence_store")(user)
    # Store constructors and exports are read-only. Do not enter writer_guard here:
    # its account lock creates a file even on a plan-only invocation.
    events = read_records(targets.path, _named(STORAGE + "CODEC"))
    items = evidence.export()["evidence"]
    return Records(targets.path, evidence.path, tuple(events), tuple(items))


@dataclass(frozen=True)
class Plan:
    stem: str
    user: str
    root: Path
    records: Records
    pairs: tuple[relevance.Pair, ...]
    fingerprint: str
    frozen: relevance.FrozenLabels | None
    expected_usd: float
    hold_usd: float

    @property
    def kind(self) -> str:
        return f"mind_relevance:{self.stem}:{self.user}"


def _input_bytes(pair: relevance.Pair) -> int:
    prompt = render(relevance.RELEVANCE_PROMPT.read_text(encoding="utf-8"), {
        "TARGET": pair.target_label, "PROJECT": pair.item.parent_project, "ITEM": pair.item.text,
    })
    return len(prompt.encode("utf-8")) + 256  # token count upper bound, including chat framing


def plan(
    results: Path, lives: Sequence[tuple[str, str]], *,
    is_synthetic: Callable[[str], bool], records: Callable[[str], Records],
    prices: tuple[float, float], personas: Path = collect.PERSONAS_DIR,
) -> tuple[Plan, ...]:
    accounts = development_accounts(lives)
    # Validate every account before any person's text is read.
    for stem, user in accounts:
        _named("anjo.core.ids:validate_user_id")(user)
        collect._refuse_ids(stem, user)
        if not is_synthetic(user):
            raise Refused(f"not registered synthetic: {user}")
    if any(not math.isfinite(p) or p <= 0 for p in prices):
        raise Refused("input/output prices must be finite and positive")
    planned = []
    for stem, user in accounts:
        source = records(user)
        state = fold(source.targets)
        if any(t.person_id != user for t in state.targets.values()):
            raise Refused("targets belong to another person")
        if any(evidence_from_record(row).person_id != user for row in source.evidence):
            raise Refused("evidence belongs to another person")
        labels = {t.id: t.label for t in state.targets.values()}
        root = results / user
        folders = collect._collectible_folders(root / "nights", "development")
        picked = {e.target_id for e in state.expectations}
        for folder in folders:
            for record in NightLog(folder).points():
                point = points.point_from_record(record)
                # points.py:124 writes snapshot targets with person_id and label blanked.
                if any(t.person_id not in ("", user) for t in point.raw.targets):
                    raise Refused("snapshot targets belong to another person")
                picked.update(t.id for t in point.raw.targets)
        persona = load_person_file(personas / f"{stem}.json")
        collect._refuse_ids(persona.visible.person_id, user)
        if persona.visible.person_id != stem:
            raise Refused("persona does not match registered stem")
        pairs = relevance.pairs_to_label(picked, labels, persona.held_out.new_work)
        fingerprint = hashlib.sha256(json.dumps([
            [p.target_id, p.target_label, p.item.id, p.item.parent_project, p.item.text]
            for p in pairs
        ], ensure_ascii=True).encode()).hexdigest()
        path = root / collect.LABELS_FILE
        frozen = None
        if path.exists():
            frozen = collect.load_frozen_labels(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            if (frozen.prompt_sha256 != relevance.prompt_sha256()
                    or set(frozen.labels) != {p.key for p in pairs}
                    or document.get("input_sha256") != fingerprint):
                raise Refused(f"{path}: frozen relevance differs from current inputs")
        inputs = sum(_input_bytes(p) for p in pairs)
        expected = (inputs / 4 * prices[0] + len(pairs) * EXPECTED_OUTPUT * prices[1]) / 1e6
        bound = (inputs * prices[0] + len(pairs) * MAX_OUTPUT * prices[1]) / 1e6
        planned.append(Plan(stem, user, root, source, pairs, fingerprint, frozen,
                            0.0 if frozen else expected, 0.0 if frozen else bound))
    return tuple(planned)


def _check_registered(chosen: Sequence[Plan], path: Path) -> None:
    entries = prereg.read_entries(path)
    prereg.verify_entries(entries)
    for life in chosen:
        old = [e.sha256 for e in entries if e.kind == life.kind]
        if old and (life.frozen is None or any(d != life.frozen.digest for d in old)):
            raise Refused(f"{life.stem}: relevance differs from registered digest")


def _label(life: Plan, *, chat: ChatFn, budget: Ledger,
           prices: tuple[float, float]) -> relevance.FrozenLabels:
    if life.frozen is not None:
        return life.frozen
    if not life.pairs:
        return relevance.freeze(relevance.prompt_sha256(), {}, frozenset())
    spent = 0.0

    def metered(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float,
                *, temperature: float | None = None) -> ChatResult:
        nonlocal spent
        # Charge unknown usage at the bound, including a failed paid request.
        input_bound = len(str(messages[0]["content"]).encode("utf-8")) + 256
        worst = (input_bound * prices[0] + MAX_OUTPUT * prices[1]) / 1e6
        spent += worst
        reply = chat(model, messages, tools, timeout_s, temperature=temperature)
        spent += (reply.input_tokens * prices[0] + reply.output_tokens * prices[1]) / 1e6 - worst
        return reply

    with held(budget, f"mind-dev-prepare:{life.user}:{life.fingerprint}:{uuid4().hex}",
              ("meta_usd", "judge_usd"), usd(life.hold_usd), lambda: spent):
        return relevance.label_pairs(life.pairs, metered)


def prepare(chosen: Sequence[Plan], *, chat: ChatFn, budget: Ledger,
            prices: tuple[float, float], registration: Path) -> None:
    _check_registered(chosen, registration)
    for life in chosen:
        frozen = _label(life, chat=chat, budget=budget, prices=prices)
        document = {
            "prompt_sha256": frozen.prompt_sha256, "digest": frozen.digest,
            "input_sha256": life.fingerprint,
            "labels": [[t, i, value] for (t, i), value in sorted(frozen.labels.items())],
            "unreadable": sorted([t, i] for t, i in frozen.unreadable),
        }
        path = life.root / collect.LABELS_FILE
        # Exclusive creation: never truncate a frozen file, including a concurrent writer's.
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(document, sort_keys=True) + "\n")
        elif json.loads(path.read_text(encoding="utf-8")) != document:
            raise Refused(f"{path}: frozen relevance differs")
        for name, rows in ((collect.TARGETS_FILE, life.records.targets),
                           (collect.EVIDENCE_FILE, life.records.evidence)):
            (life.root / name).write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
        if not any(e.kind == life.kind and e.sha256 == frozen.digest
                   for e in prereg.read_entries(registration)):
            prereg.register(life.kind, frozen.digest, "development relevance; frozen before scoring",
                            path=registration)


def _chat(model: Any, messages: list[dict[str, Any]], tools: Any, timeout_s: float,
          *, temperature: float | None = None) -> ChatResult:
    return meta_api.chat(model, messages, tools, timeout_s, temperature=temperature,
                         max_tokens=MAX_OUTPUT)


def main(argv: Sequence[str] | None = None, *, out: Callable[[str], None] = print,
         is_synthetic: Callable[[str], bool] | None = None,
         records: Callable[[str], Records] = host_records, chat: ChatFn = _chat,
         prices: tuple[float, float] | None = None, budget: Ledger | None = None,
         registration: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--life", action="append", required=True, metavar="STEM=UUID")
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args(argv)
    try:
        lives = []
        for value in args.life:
            stem, separator, user = value.partition("=")
            if not separator:
                raise Refused("--life needs STEM=UUID")
            lives.append((stem, user))
        development_accounts(lives)
        if prices is None:
            configured = _named("anjo.sleep.app.night:prices")()
            prices = (configured.spark_in_per_m, configured.spark_out_per_m)
        chosen = plan(args.results, lives, is_synthetic=is_synthetic or host_check(),
                      records=records, prices=prices)
        path = registration if registration is not None else prereg.ledger_path()
        _check_registered(chosen, path)
        out(f"labeller: {relevance.RELEVANCE_MODEL} ({meta_api.PINNED_MODELS['muse-spark']}), temperature 0")
        out(f"prices USD/million input/output: {prices[0]:g}/{prices[1]:g}; "
            f"estimated output {EXPECTED_OUTPUT}/pair, cap {MAX_OUTPUT}/pair")
        for life in chosen:
            out(f"{life.stem} ({life.user}): {life.root}")
            out(f"  targets: {life.records.targets_path}: {len(life.records.targets)} rows (events)")
            out(f"  evidence: {life.records.evidence_path}: {len(life.records.evidence)} rows")
            out(f"  relevance: {0 if life.frozen else len(life.pairs)} pairs need labels "
                f"({len(life.pairs)} total; all recorded targets x held-out work)")
            out(f"  expected ${life.expected_usd:.6f}; budget hold ${life.hold_usd:.6f} "
                "on meta_usd + judge_usd; settle at reported tokens")
        out(f"register label digests locally: {path}")
        if not args.go:
            out("plan only: nothing written, called or held")
            return 0
        prepare(chosen, chat=chat, budget=budget if budget is not None else default_ledger(),
                prices=prices, registration=path)
        out("prepared development lives; relevance frozen and registered")
        return 0
    except (ValueError, RuntimeError, PermissionError, OSError, ModelApiError) as error:
        out(f"refused: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
