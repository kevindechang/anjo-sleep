"""Fit and register Task 16's ``p_needed`` estimator from the development lives.

    python -m anjo.sleep.tools.fit_estimator --results <dir>
    python -m anjo.sleep.tools.fit_estimator --results <dir> --go [--replace "<reason>"]

Without ``--go`` it prints a plan only: every registered development account
(``mind.collect.DEVELOPMENT_LIVES``) mapped to ``<results>/<user_id>``, whether that folder
exists, and how many night folders it holds. It reads no night and writes nothing, and no
life's text is printed.

With ``--go`` it checks every account is registered synthetic, fits
``expect_register.fit_from_logs`` on the development lives, saves the coefficients to
``prereg/estimator.json``, loads them back and prints them with ``n`` and the sha256. It
refuses before writing when the fit saw fewer rows than ``expect.fit`` needs, when an account
isn't synthetic, or when the file already holds fitted coefficients and ``--replace`` gives
no reason.

The engine never imports the app: the host's ``is_synthetic`` is named here and imported when
the command runs, as ``morning_backfill`` names its ports.
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, cast

from anjo.sleep.mind.collect import DEVELOPMENT_LIVES, CollectRefused
from anjo.sleep.mind.expect import Coefficients
from anjo.sleep.mind.expect_register import (
    ESTIMATOR,
    EstimatorError,
    development_accounts,
    fit_from_logs,
    load_estimator,
    save_estimator,
)
from anjo.sleep.night.nightlog import SUMMARY
from anjo.sleep.reports.results import ResultsError, results_root

SYNTHETIC: Final = "anjo.sleep.app.cleared_sessions:is_synthetic"
VALIDATE_USER_ID: Final = "anjo.core.ids:validate_user_id"
MIN_ROWS: Final = 1  # expect.fit refuses an empty set: one resolved expectation is its minimum
FitFn = Callable[[Sequence[Path], Callable[[str], bool]], Coefficients]


class Refused(RuntimeError):
    """The fit or the registration may not run; the message names the rule."""


@dataclass(frozen=True)
class Account:
    persona: str
    user_id: str
    root: Path
    exists: bool
    nights: int


@dataclass(frozen=True)
class Plan:
    results: Path
    accounts: tuple[Account, ...]
    lives: tuple[tuple[str, str], ...] = DEVELOPMENT_LIVES

    @property
    def log_roots(self) -> tuple[Path, ...]:
        selected = {stem for stem, _ in self.lives}
        return tuple(account.root for account in self.accounts if account.persona in selected)


def plan(results: Path, lives: Sequence[tuple[str, str]] | None = None) -> Plan:
    """The four registered development accounts under ``results``, without reading a night."""
    accounts = development_accounts(lives)
    if lives is not None:
        validate = _named(VALIDATE_USER_ID)
        for _, user_id in accounts:
            try:
                validate(user_id)
            except ValueError as error:
                raise Refused(str(error)) from None
    mapped = dict(accounts)
    return Plan(
        results,
        tuple(
            _account(results, stem, mapped[stem]) if stem in mapped
            else Account(stem, user_id, results / user_id, False, 0)
            for stem, user_id in DEVELOPMENT_LIVES
        ),
        accounts,
    )


def _account(results: Path, persona: str, user_id: str) -> Account:
    root = results / user_id
    nights = root / "nights"
    count = (
        sum(1 for folder in nights.iterdir() if folder.is_dir() and (folder / SUMMARY).is_file())
        if nights.is_dir()
        else 0
    )
    return Account(persona, user_id, root, root.is_dir(), count)


def describe(chosen: Plan) -> list[str]:
    """The plan's lines: account ids, folders and night counts only, never a night's text."""
    lines = [f"estimator fit: {chosen.results}"]
    for account in chosen.accounts:
        state = f"{account.nights} night folders" if account.exists else "missing"
        lines.append(f"  {account.persona} ({account.user_id}): {account.root} - {state}")
    lines.append(
        "fit rows: derived=unknown, recorded=unknown (plan reads no nights; counts after fit)"
    )
    return lines


def _fitted(chosen: Plan, *, is_synthetic: Callable[[str], bool], fit: FitFn) -> Coefficients:
    selected = {stem for stem, _ in chosen.lives}
    accounts = [account for account in chosen.accounts if account.persona in selected]
    missing = [account.user_id for account in accounts if not account.exists]
    if missing:
        raise Refused(f"missing development account(s) under {chosen.results}: {', '.join(missing)}")
    real = [account.user_id for account in accounts if not is_synthetic(account.user_id)]
    if real:
        raise Refused(f"not registered synthetic: {', '.join(real)}")
    try:
        coefficients = fit(chosen.log_roots, is_synthetic)
    except (ValueError, CollectRefused) as error:
        raise Refused(str(error)) from None
    if coefficients.n < MIN_ROWS:
        raise Refused(
            f"the fit saw {coefficients.n} resolved rows; expect.fit needs at least {MIN_ROWS}"
        )
    return coefficients


def register(
    chosen: Plan,
    *,
    is_synthetic: Callable[[str], bool],
    path: Path,
    replace: str | None,
    out: Callable[[str], None],
    fit: FitFn,
) -> Coefficients:
    """Fit, save and load back; every refusal happens before anything is written."""
    existing = load_estimator(path)
    if existing is not None:
        if replace is None:
            raise Refused(
                f"{path} already holds fitted coefficients; pass --replace \"<reason>\" to overwrite"
            )
        out(f"replacing fitted coefficients: {replace}")
    fitted = _fitted(chosen, is_synthetic=is_synthetic, fit=fit)
    save_estimator(fitted, path)
    loaded = load_estimator(path)
    if loaded != fitted:
        raise Refused(f"{path} did not load back as the fitted coefficients")
    return loaded


def _sha256(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.get("sha256")
    if not isinstance(digest, str):
        raise Refused(f"{path} has no sha256 after saving")
    return digest


def _named(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def host_check() -> Callable[[str], bool]:
    """The sleep host's synthetic check, imported when the command runs."""
    return cast(Callable[[str], bool], _named(SYNTHETIC))


def _fit(
    log_roots: Sequence[Path], is_synthetic: Callable[[str], bool], *,
    lives: Sequence[tuple[str, str]] | None = None,
    out: Callable[[str], None] | None = None,
) -> Coefficients:
    kwargs: dict[str, Any] = {}
    if out is not None:
        kwargs["report_counts"] = lambda derived, recorded: out(
            f"fit rows: derived={derived}, recorded={recorded} (resolved rows only)"
        )
    return fit_from_logs(log_roots, is_synthetic=is_synthetic, lives=lives, **kwargs)


def _report(loaded: Coefficients, path: Path, out: Callable[[str], None]) -> None:
    out(
        f"coefficients: intercept={loaded.intercept:.6f} understand={loaded.understand:.6f} "
        f"unacted={loaded.unacted:.6f} days={loaded.days:.6f} "
        f"undated_days={loaded.undated_days:g}"
    )
    out(f"n: {loaded.n}, base_rate: {loaded.base_rate:.6f}")
    out(f"sha256: {_sha256(path)}")
    out(f"registered: {path}")


def main(
    argv: Sequence[str] | None = None,
    *,
    out: Callable[[str], None] = print,
    is_synthetic: Callable[[str], bool] | None = None,
    estimator: Path | None = None,
    fit: FitFn | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Fit and register the p_needed estimator from development lives."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="results tree (default: ANJO_DATA_DIR/sleep/results)",
    )
    parser.add_argument("--go", action="store_true", help="fit and register (without: the plan only)")
    parser.add_argument(
        "--life", action="append", metavar="STEM=UUID",
        help="map a registered development stem to an account (repeatable)",
    )
    parser.add_argument(
        "--replace",
        default=None,
        metavar="REASON",
        help="overwrite fitted coefficients, saying why",
    )
    args = parser.parse_args(argv)
    path = ESTIMATOR if estimator is None else estimator
    try:
        lives = None
        if args.life is not None:
            lives = []
            for value in args.life:
                stem, separator, user_id = value.partition("=")
                if not separator:
                    raise Refused("--life needs STEM=UUID")
                lives.append((stem, user_id))
        chosen = plan(results_root(args.results), lives)
        for line in describe(chosen):
            out(line)
        if not args.go:
            out("plan only: nothing written; run again with --go")
            return 0
        reason = args.replace
        if reason is not None and not reason.strip():
            raise Refused("--replace needs a reason")
        loaded = register(
            chosen,
            is_synthetic=is_synthetic if is_synthetic is not None else host_check(),
            path=path,
            replace=reason,
            out=out,
            fit=fit if fit is not None else partial(_fit, lives=chosen.lives, out=out),
        )
    except (Refused, EstimatorError, ResultsError) as refused:
        out(f"refused: {refused}")
        return 1
    _report(loaded, path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
