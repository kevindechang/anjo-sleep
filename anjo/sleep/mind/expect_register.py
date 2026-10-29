"""Register Task 16's ``p_needed`` estimator: fit on development expectations, save/load ``estimator.json``."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from anjo.sleep.mind import expect, points
from anjo.sleep.mind.collect import DEVELOPMENT_LIVES, collect_life
from anjo.sleep.mind.expect import Coefficients, FitRow, Split
from anjo.sleep.night.nightlog import SUMMARY, NightLog
from anjo.sleep.reports import mind
from anjo.sleep.types import Expectation, Target, WantKind

ESTIMATOR: Final = Path(__file__).parents[1] / "prereg" / "estimator.json"
PERSONAS_DIR: Final = Path(__file__).parents[1] / "personas"
_VERSION: Final = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COEFF_KEYS: Final = ("intercept", "understand", "unacted", "days", "undated_days")


class EstimatorError(ValueError):
    """The registered estimator file is missing fields or its hash does not verify."""


def development_accounts(
    lives: Sequence[tuple[str, str]] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Resolve an explicit development mapping without widening the persona allowlist."""
    if lives is None:
        return DEVELOPMENT_LIVES
    stems = {stem for stem, _ in DEVELOPMENT_LIVES}
    seen_stems: set[str] = set()
    seen_ids: set[str] = set()
    for stem, user_id in lives:
        if stem not in stems:
            raise EstimatorError(f"unregistered development stem {stem!r}")
        if stem in seen_stems or user_id in seen_ids:
            raise EstimatorError(f"duplicate development stem or account: {stem!r}={user_id!r}")
        seen_stems.add(stem)
        seen_ids.add(user_id)
    return tuple(lives)


def fit_from_logs(
    log_roots: Sequence[Path],
    *,
    is_synthetic: Callable[[str], bool],
    personas_dir: Path = PERSONAS_DIR,
    split: Split = "development",
    lives: Sequence[tuple[str, str]] | None = None,
    report_counts: Callable[[int, int], None] | None = None,
) -> Coefficients:
    """Fit development expectations, bootstrapping missing ones from D0 study choices in ``log_roots``.

    ``is_synthetic`` is the host's check (``anjo.sleep.app.cleared_sessions.is_synthetic``), passed in
    because the engine core imports no host module. The report_counts callback receives derived
    and recorded counts of resolved rows passed to the fit; censored rows are excluded.
    """
    if split != "development":
        raise ValueError("the estimator is fitted on development lives only")
    accounts = development_accounts(lives)
    rows: list[FitRow] = []
    sources: list[bool] = []
    for root in log_roots:
        rows.extend(_fit_rows_from_root(
            root.resolve(), personas_dir=personas_dir, split=split, is_synthetic=is_synthetic,
            lives=accounts, on_row=sources.append,
        ))
    fitted = expect.fit(rows)
    if report_counts is not None:
        report_counts(sum(sources), len(sources) - sum(sources))
    return fitted


def load_estimator(path: Path | None = None) -> Coefficients | None:
    """Load verified coefficients, or ``None`` when the file is a structure-only placeholder."""
    file = ESTIMATOR if path is None else path
    if not file.is_file():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    if data.get("version") != _VERSION:
        raise EstimatorError("estimator version is not supported")
    digest = data.get("sha256")
    if any(data.get(key) is None for key in (*_COEFF_KEYS, "base_rate", "n")):
        if digest is None:
            return None
        raise EstimatorError("estimator is missing required coefficients")
    payload = {key: data[key] for key in ("version", *_COEFF_KEYS, "base_rate", "n")}
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise EstimatorError("estimator sha256 must be 64 lowercase hex characters")
    if digest != _digest(payload):
        raise EstimatorError("estimator sha256 does not match its coefficients")
    _require_numeric_payload(payload)
    return Coefficients(
        float(payload["intercept"]),
        float(payload["understand"]),
        float(payload["unacted"]),
        float(payload["days"]),
        base_rate=float(payload["base_rate"]),
        n=payload["n"],
        undated_days=float(payload["undated_days"]),
    )


def save_estimator(coefficients: Coefficients, path: Path | None = None) -> None:
    """Write ``coefficients`` and its content hash to ``path``."""
    file = ESTIMATOR if path is None else path
    payload: dict[str, Any] = {
        "version": _VERSION,
        "intercept": coefficients.intercept,
        "understand": coefficients.understand,
        "unacted": coefficients.unacted,
        "days": coefficients.days,
        "undated_days": coefficients.undated_days,
        "base_rate": coefficients.base_rate,
        "n": coefficients.n,
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    doc = {**payload, "sha256": _digest(payload)}
    file.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def registered_estimator() -> Coefficients | None:
    """The estimator the sleep host loads for synthetic nights, when the file verifies."""
    return load_estimator(ESTIMATOR)


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _persona_for_user(
    user_id: str, personas_dir: Path, lives: Sequence[tuple[str, str]]
) -> Path:
    for stem, uid in lives:
        if uid == user_id:
            return personas_dir / f"{stem}.json"
    raise EstimatorError(f"no registered development persona for account {user_id!r}")


def _fit_rows_from_root(
    root: Path, *, personas_dir: Path, split: Split, is_synthetic: Callable[[str], bool],
    lives: Sequence[tuple[str, str]] | None = None,
    on_row: Callable[[bool], None] | None = None,
) -> list[FitRow]:
    accounts = development_accounts(lives)
    nights_dir, life_root, user_id = _layout(root)
    if user_id not in {uid for _, uid in accounts}:
        raise EstimatorError(f"estimator fit refuses account {user_id!r}")
    if not is_synthetic(user_id):
        raise EstimatorError(f"estimator fit refuses non-synthetic account {user_id!r}")
    persona = _persona_for_user(user_id, personas_dir, accounts)
    life = collect_life(nights_dir, persona, split=split)
    return _fit_rows(life, nights_dir, on_row=on_row)


def _layout(root: Path) -> tuple[Path, Path, str]:
    resolved = root.resolve()
    if resolved.name == "nights":
        nights_dir = resolved
        life_root = resolved.parent
    elif (resolved / "nights").is_dir():
        nights_dir = resolved / "nights"
        life_root = resolved
    else:
        nights_dir = resolved
        life_root = resolved.parent
    user_id = life_root.name
    return nights_dir, life_root, user_id


def _fit_rows(
    life: mind.LifeRecord, nights_dir: Path, *,
    on_row: Callable[[bool], None] | None = None,
) -> list[FitRow]:
    """Development-only bootstrap: recorded expectations take precedence for each choice."""
    if life.split != "development":
        raise ValueError("the estimator is fitted on development lives only")
    end = _virtual(life.last_day + timedelta(days=1))
    recorded = {(made.target_id, made.made_at): made for made in life.expectations}
    rows: list[FitRow] = []
    seen: set[tuple[str, datetime]] = set()
    for made_at, kind, target in _studies(nights_dir, life):
        key = (target.id, made_at)
        if key in seen:
            continue
        seen.add(key)
        made = recorded.get(key)
        derived = made is None
        if made is None:
            # The forecast is unused by resolution and fitting; no estimator is needed here.
            made = Expectation(
                id=expect.expectation_id(target.id, made_at), target_id=target.id,
                made_at=made_at, horizon=expect.horizon(target, made_at), p_needed=0.5,
            )
        outcome = made.outcome
        if outcome is None:
            arrivals = [
                _virtual(a.day)
                for a in life.arrivals
                if life.labels.relevant(made.target_id, a.item_id)
            ]
            returns = [_virtual(day) for day in life.returns.get(made.target_id, ())]
            outcome = expect.resolve(
                made, returns=returns, arrivals=arrivals, life_end=end, as_of=end
            )
        row = expect.fit_row(target, kind, made.made_at, outcome, life.split)
        if row is not None:
            rows.append(row)
            if on_row is not None:
                on_row(derived)
    return rows


def _studies(
    nights_dir: Path, life: mind.LifeRecord,
) -> Iterator[tuple[datetime, WantKind, Target]]:
    for night_pick in life.nights:
        folder = _folder_for_night(nights_dir, night_pick.night, split=life.split)
        if folder is None:
            continue
        for record in NightLog(folder).points():
            point = points.point_from_record(record)
            if point.phase != "pre_study":
                continue
            choice = point.choice
            if choice is None or choice.want is None:
                continue
            target_id = points.study_target(choice, point.raw)
            if target_id is None or night_pick.study.get("D0") != target_id:
                continue
            target = next((t for t in point.raw.targets if t.id == target_id), None)
            if target is not None:
                yield point.raw.as_of, choice.want.kind, target


def _virtual(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _folder_for_night(nights_dir: Path, night: date, *, split: Split) -> Path | None:
    want = "final" if split == "held_out" else "development"
    prefix = night.isoformat()
    candidates: list[Path] = []
    for folder in nights_dir.iterdir():
        if (
            not folder.is_dir()
            or not folder.name.startswith(prefix)
            or not (folder / SUMMARY).is_file()
        ):
            continue
        summary = json.loads((folder / SUMMARY).read_text(encoding="utf-8"))
        if summary.get("label") != want or not NightLog(folder).points():
            continue
        candidates.append(folder)
    if not candidates:
        return None
    return max(candidates, key=_folder_key)


def _require_numeric_payload(payload: Mapping[str, Any]) -> None:
    for key in _COEFF_KEYS:
        value = payload[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise EstimatorError(f"estimator {key} must be a number")
    base_rate = payload["base_rate"]
    if not isinstance(base_rate, (int, float)) or isinstance(base_rate, bool):
        raise EstimatorError("estimator base_rate must be a number")
    n = payload["n"]
    if not isinstance(n, int) or isinstance(n, bool):
        raise EstimatorError("estimator n must be an integer")


def _folder_key(folder: Path) -> tuple[str, int]:
    day, _, attempt = folder.name.partition(".")
    return day, int(attempt) if attempt.isdigit() else 1
