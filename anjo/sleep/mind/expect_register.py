"""Register Task 16's ``p_needed`` estimator: fit on development expectations, save/load ``estimator.json``."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
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


def fit_from_logs(
    log_roots: Sequence[Path],
    *,
    personas_dir: Path = PERSONAS_DIR,
    split: Split = "development",
) -> Coefficients:
    """Fit the registered coefficients from development lives' resolved expectations in ``log_roots``."""
    if split != "development":
        raise ValueError("the estimator is fitted on development lives only")
    rows: list[FitRow] = []
    for root in log_roots:
        rows.extend(_fit_rows_from_root(root.resolve(), personas_dir=personas_dir, split=split))
    return expect.fit(rows)


def load_estimator(path: Path | None = None) -> Coefficients | None:
    """Load verified coefficients, or ``None`` when the file is a structure-only placeholder."""
    file = ESTIMATOR if path is None else path
    if not file.is_file():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    if data.get("version") != _VERSION:
        raise EstimatorError("estimator version is not supported")
    if any(data.get(key) is None for key in (*_COEFF_KEYS, "base_rate", "n")):
        return None
    payload = {key: data[key] for key in ("version", *_COEFF_KEYS, "base_rate", "n")}
    digest = data.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise EstimatorError("estimator sha256 must be 64 lowercase hex characters")
    if digest != _digest(payload):
        raise EstimatorError("estimator sha256 does not match its coefficients")
    return Coefficients(
        float(payload["intercept"]),
        float(payload["understand"]),
        float(payload["unacted"]),
        float(payload["days"]),
        base_rate=float(payload["base_rate"]),
        n=int(payload["n"]),
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


def _persona_for_user(user_id: str, personas_dir: Path) -> Path:
    for stem, uid in DEVELOPMENT_LIVES:
        if uid == user_id:
            return personas_dir / f"{stem}.json"
    raise EstimatorError(f"no registered development persona for account {user_id!r}")


def _fit_rows_from_root(root: Path, *, personas_dir: Path, split: Split) -> list[FitRow]:
    nights_dir, life_root, user_id = _layout(root)
    persona = _persona_for_user(user_id, personas_dir)
    life = collect_life(nights_dir, persona, split=split)
    return _fit_rows(life, nights_dir)


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


def _fit_rows(life: mind.LifeRecord, nights_dir: Path) -> list[FitRow]:
    if life.split != "development":
        raise ValueError("the estimator is fitted on development lives only")
    end = _virtual(life.last_day + timedelta(days=1))
    rows: list[FitRow] = []
    for made in life.expectations:
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
        kind_target = _study(nights_dir, life, made)
        if kind_target is None:
            continue
        kind, target = kind_target
        row = expect.fit_row(target, kind, made.made_at, outcome, "development")
        if row is not None:
            rows.append(row)
    return rows


def _study(
    nights_dir: Path, life: mind.LifeRecord, made: Expectation
) -> tuple[WantKind, Target] | None:
    for night_pick in life.nights:
        if _night_end(night_pick.night) != made.made_at:
            continue
        if night_pick.study.get("D0") != made.target_id:
            continue
        folder = _latest_folder(nights_dir, night_pick.night)
        if folder is None:
            continue
        log = NightLog(folder)
        for record in log.points():
            point = points.point_from_record(record)
            if point.phase != "pre_study" or point.raw.as_of != made.made_at:
                continue
            choice = point.choice
            if choice is None or choice.want is None:
                continue
            if points.study_target(choice, point.raw) != made.target_id:
                continue
            target = next((t for t in point.raw.targets if t.id == made.target_id), None)
            if target is None:
                continue
            return choice.want.kind, target
    return None


def _virtual(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _night_end(night: date) -> datetime:
    return datetime.combine(night + timedelta(days=1), time.min, tzinfo=timezone.utc)


def _latest_folder(nights_dir: Path, night: date) -> Path | None:
    prefix = night.isoformat()
    candidates = [
        folder
        for folder in nights_dir.iterdir()
        if folder.is_dir() and folder.name.startswith(prefix) and (folder / SUMMARY).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=_folder_key)


def _folder_key(folder: Path) -> tuple[str, int]:
    day, _, attempt = folder.name.partition(".")
    return day, int(attempt) if attempt.isdigit() else 1
