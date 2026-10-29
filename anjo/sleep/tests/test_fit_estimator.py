"""The estimator fit CLI (Task 16): plan, fit, register, and every refusal. Fakes only."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from anjo.sleep.mind import expect
from anjo.sleep.mind.collect import DEVELOPMENT_LIVES
from anjo.sleep.mind.expect_register import load_estimator, save_estimator
from anjo.sleep.tools import fit_estimator

COEFFICIENTS = expect.Coefficients(
    intercept=-0.5, understand=0.4, unacted=0.3, days=-0.05, base_rate=0.3, n=12
)
FOUNDER_ID = "6f8063cb-7e48-4dc8-8d4e-26665047418e"
RESEARCHER_ID = "8a701d9d-0730-4df8-914f-4b8f627a6888"


def _tree(root: Path, counts: dict[str, int]) -> Path:
    """A results tree with the four registered accounts and ``counts`` night folders each."""
    for _, user_id in DEVELOPMENT_LIVES:
        account = root / user_id
        account.mkdir(parents=True)
        for day in range(counts.get(user_id, 0)):
            night = f"2026-10-{day + 1:02d}"
            folder = account / "nights" / night
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text(
                json.dumps({"night": night, "label": "development"}), encoding="utf-8"
            )
    return root


def _fit_returning(
    coefficients: expect.Coefficients,
    seen: list[tuple[tuple[Path, ...], bool]] | None = None,
) -> fit_estimator.FitFn:
    def fit(roots: Sequence[Path], is_synthetic: Callable[[str], bool]) -> expect.Coefficients:
        if seen is not None:
            seen.append((tuple(roots), is_synthetic("founder")))
        return coefficients

    return fit


def test_plan_lists_development_accounts_and_writes_nothing(tmp_path: Path) -> None:
    results = _tree(tmp_path / "results", {"founder": 3, "researcher": 1})
    (results / "film_student").rmdir()
    path = tmp_path / "estimator.json"
    lines: list[str] = []

    code = fit_estimator.main(["--results", str(results)], out=lines.append, estimator=path)

    text = "\n".join(lines)
    assert code == 0
    assert f"estimator fit: {results}" in text
    for _, user_id in DEVELOPMENT_LIVES:
        assert user_id in text
    assert f"{results / 'founder'} - 3 night folders" in text
    assert f"{results / 'researcher'} - 1 night folders" in text
    assert f"{results / 'film_founder'} - 0 night folders" in text
    assert f"{results / 'film_student'} - missing" in text
    assert "plan only: nothing written" in text
    assert not path.exists()


def test_results_defaults_to_the_sleep_results_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv("ANJO_DATA_DIR", str(data))
    lines: list[str] = []

    code = fit_estimator.main([], out=lines.append, estimator=tmp_path / "estimator.json")

    assert code == 0
    assert f"estimator fit: {data / 'sleep' / 'results'}" in lines[0]


def test_mapped_plan_and_fit_use_only_supplied_accounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _tree(tmp_path / "results", {"film_founder": 2})
    for user_id in (FOUNDER_ID, RESEARCHER_ID):
        folder = results / user_id / "nights" / "2026-10-01"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text("{}", encoding="utf-8")
    args = [
        "--results", str(results), "--life", f"founder={FOUNDER_ID}",
        "--life", f"researcher={RESEARCHER_ID}",
    ]
    lines: list[str] = []
    assert fit_estimator.main(args, out=lines.append) == 0
    text = "\n".join(lines)
    assert f"founder ({FOUNDER_ID}): {results / FOUNDER_ID} - 1 night folders" in text
    assert f"researcher ({RESEARCHER_ID}): {results / RESEARCHER_ID} - 1 night folders" in text
    assert f"{results / 'film_founder'} - missing" in text
    assert f"{results / 'film_student'} - missing" in text

    def fit(roots, *, is_synthetic, lives):
        assert tuple(roots) == (results / FOUNDER_ID, results / RESEARCHER_ID)
        assert lives == (("founder", FOUNDER_ID), ("researcher", RESEARCHER_ID))
        assert all(is_synthetic(uid) for _, uid in lives)
        return COEFFICIENTS

    monkeypatch.setattr(fit_estimator, "fit_from_logs", fit)
    chosen = fit_estimator.plan(results, [("founder", FOUNDER_ID), ("researcher", RESEARCHER_ID)])
    fitted = fit_estimator.register(
        chosen, is_synthetic=lambda uid: uid in {FOUNDER_ID, RESEARCHER_ID},
        path=tmp_path / "estimator.json", replace=None, out=lines.append,
        fit=lambda roots, synthetic: fit_estimator._fit(roots, synthetic, lives=chosen.lives),
    )
    assert fitted == COEFFICIENTS


@pytest.mark.parametrize("lives, message", [
    ([f"stranger={FOUNDER_ID}"], "unregistered"),
    ([f"founder={FOUNDER_ID}", f"founder={RESEARCHER_ID}"], "duplicate"),
    ([f"founder={FOUNDER_ID}", f"researcher={FOUNDER_ID}"], "duplicate"),
    (["founder=../escape"], "invalid user_id"),
    (["founder="], "invalid user_id"),
    (["founder=" + "a" * 65], "invalid user_id"),
    (["founder"], "STEM=UUID"),
])
def test_life_mapping_refused_before_planning(tmp_path: Path, lives: list[str], message: str) -> None:
    args = ["--results", str(tmp_path)]
    for life in lives:
        args.extend(["--life", life])
    lines: list[str] = []
    assert fit_estimator.main(args, out=lines.append) == 1
    assert len(lines) == 1 and message in lines[0]


def test_go_fits_saves_and_loads_the_registered_file(tmp_path: Path) -> None:
    results = _tree(tmp_path / "results", {user_id: 1 for _, user_id in DEVELOPMENT_LIVES})
    path = tmp_path / "estimator.json"
    seen: list[tuple[tuple[Path, ...], bool]] = []
    lines: list[str] = []

    code = fit_estimator.main(
        ["--results", str(results), "--go"],
        out=lines.append,
        is_synthetic=lambda _user_id: True,
        estimator=path,
        fit=_fit_returning(COEFFICIENTS, seen),
    )

    text = "\n".join(lines)
    assert code == 0, text
    assert load_estimator(path) == COEFFICIENTS
    assert seen == [(tuple(results / user_id for _, user_id in DEVELOPMENT_LIVES), True)]
    assert "intercept=-0.500000" in text
    assert "n: 12" in text
    digest = json.loads(path.read_text(encoding="utf-8"))["sha256"]
    assert f"sha256: {digest}" in text
    assert f"registered: {path}" in text


def test_go_refuses_a_fit_with_too_few_rows(tmp_path: Path) -> None:
    results = _tree(tmp_path / "results", {user_id: 1 for _, user_id in DEVELOPMENT_LIVES})
    path = tmp_path / "estimator.json"
    empty = expect.Coefficients(
        intercept=0.0, understand=0.0, unacted=0.0, days=0.0, base_rate=0.5, n=0
    )
    lines: list[str] = []

    code = fit_estimator.main(
        ["--results", str(results), "--go"],
        out=lines.append,
        is_synthetic=lambda _user_id: True,
        estimator=path,
        fit=_fit_returning(empty),
    )

    text = "\n".join(lines)
    assert code == 1 and "resolved rows" in text and "expect.fit" in text
    assert not path.exists()


def test_go_refuses_a_non_synthetic_account_before_fitting(tmp_path: Path) -> None:
    results = _tree(tmp_path / "results", {user_id: 1 for _, user_id in DEVELOPMENT_LIVES})
    path = tmp_path / "estimator.json"
    calls: list[int] = []

    def fit(roots: Sequence[Path], is_synthetic: Callable[[str], bool]) -> expect.Coefficients:
        calls.append(1)
        return COEFFICIENTS

    lines: list[str] = []
    code = fit_estimator.main(
        ["--results", str(results), "--go"],
        out=lines.append,
        is_synthetic=lambda user_id: user_id != "researcher",
        estimator=path,
        fit=fit,
    )

    text = "\n".join(lines)
    assert code == 1 and "not registered synthetic" in text and "researcher" in text
    assert calls == []
    assert not path.exists()


def test_go_refuses_an_existing_fit_without_replace_and_replaces_with_a_reason(
    tmp_path: Path,
) -> None:
    results = _tree(tmp_path / "results", {user_id: 1 for _, user_id in DEVELOPMENT_LIVES})
    path = tmp_path / "estimator.json"
    save_estimator(COEFFICIENTS, path)

    lines: list[str] = []
    code = fit_estimator.main(
        ["--results", str(results), "--go"],
        out=lines.append,
        is_synthetic=lambda _user_id: True,
        estimator=path,
        fit=_fit_returning(COEFFICIENTS),
    )

    assert code == 1 and "already holds fitted coefficients" in "\n".join(lines)
    assert load_estimator(path) == COEFFICIENTS

    replacement = expect.Coefficients(
        intercept=0.1, understand=-0.2, unacted=0.05, days=-0.01, base_rate=0.4, n=9
    )
    lines = []
    code = fit_estimator.main(
        ["--results", str(results), "--go", "--replace", "four lives resolved more rows"],
        out=lines.append,
        is_synthetic=lambda _user_id: True,
        estimator=path,
        fit=_fit_returning(replacement),
    )

    text = "\n".join(lines)
    assert code == 0, text
    assert "four lives resolved more rows" in text
    assert load_estimator(path) == replacement
