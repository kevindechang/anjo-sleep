# mypy: disallow-untyped-decorators=False
"""The accelerated scene-count lever reaches every night on the offline rig."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any

import pytest

from anjo.sleep.night.deps import NightDeps
from anjo.sleep.sim import accelerated
from anjo.sleep.types import NightResult


@pytest.mark.parametrize(("flags", "expected"), [([], 16), (["--practice-scenes", "16"], 16),
                                               (["--practice-scenes", "20"], 20)])
def test_scene_count_reaches_each_night_and_report(
    flags: list[str], expected: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    original = accelerated.run_night_batch_with
    seen: list[int] = []

    def night(deps: NightDeps, nights: Mapping[str, date], **kwargs: Any) -> dict[str, NightResult]:
        seen.append(deps.settings.practice_scenes)
        return original(deps, nights, **kwargs)

    monkeypatch.setattr(accelerated, "run_night_batch_with", night)
    assert accelerated.main(["--fake", "--days", "2", *flags]) == 0
    assert seen == [expected, expected]
    assert json.loads(capsys.readouterr().out)["practice_scenes"] == expected


@pytest.mark.parametrize("value", ["15", "21", "1.5"])
def test_invalid_scene_count_is_refused_before_starting(
    value: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> int:
        pytest.fail("an invalid scene count must not start the fake rig")

    monkeypatch.setattr(accelerated, "_fake", unexpected)
    with pytest.raises(SystemExit) as refused:
        accelerated.main(["--fake", "--days", "2", "--practice-scenes", value])
    assert refused.value.code == 2
    assert "--practice-scenes" in capsys.readouterr().err


def test_omitted_scene_count_matches_explicit_default(capsys: pytest.CaptureFixture[str]) -> None:
    assert accelerated.main(["--fake", "--days", "2"]) == 0
    default = capsys.readouterr().out
    assert accelerated.main(["--fake", "--days", "2", "--practice-scenes", "16"]) == 0
    assert capsys.readouterr().out == default
