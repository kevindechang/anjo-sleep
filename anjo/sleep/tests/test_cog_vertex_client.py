"""The paid Gemini client's spend cap and accounting, with a fake model (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from anjo.sleep.mind.cog import vertex_client as vc


class Transient(Exception):
    code = 429


class Fatal(Exception):
    code = 400


def _settings(tmp_path: Path, cap: float) -> vc.Settings:
    return vc.Settings(tmp_path / "spend.json", cap)


def test_a_call_settles_what_it_really_cost_and_frees_its_hold(tmp_path: Path) -> None:
    s = _settings(tmp_path, 1.0)
    complete = vc.make_complete(s, generate=lambda model, prompt: ("{}", 1000, 200, "STOP"))
    assert complete("hello") == "{}"
    totals = json.loads(s.spend.read_text())
    assert totals["held"] == pytest.approx(0.0)
    assert totals["usd"] == pytest.approx(vc.cost_usd(1000, 200))
    assert totals["calls"] == 1
    log = [json.loads(line) for line in s.spend.with_suffix(".json.calls.jsonl").read_text().splitlines()]
    assert log == [{"in": 1000, "out": 200, "finish": "STOP", "s": log[0]["s"]}]


def test_a_call_whose_worst_case_passes_the_cap_is_refused_before_it_is_made(tmp_path: Path) -> None:
    made: list[str] = []
    s = _settings(tmp_path, vc.worst_case_usd("x" * 300) * 0.5)
    complete = vc.make_complete(s, generate=lambda model, prompt: (made.append(prompt), "{}", 1, 1, "STOP")[1:])
    with pytest.raises(vc.SpendCapReached):
        complete("x" * 300)
    assert made == []


def test_spend_accumulates_until_the_cap_refuses(tmp_path: Path) -> None:
    worst = vc.worst_case_usd("p")
    s = _settings(tmp_path, worst * 2.5)
    # each call really costs its worst case, so two fit and the third would pass the cap
    tokens_out = vc.MAX_OUTPUT_TOKENS
    complete = vc.make_complete(s, generate=lambda model, prompt: ("{}", 0, tokens_out, "STOP"))
    complete("p")
    complete("p")
    with pytest.raises(vc.SpendCapReached):
        complete("p")
    assert json.loads(s.spend.read_text())["calls"] == 2


def test_transient_errors_are_retried_and_fatal_ones_raise_with_the_hold_freed(tmp_path: Path) -> None:
    s = _settings(tmp_path, 1.0)
    tries = iter([Transient(), Transient(), ("{\"q\": 0.7}", 10, 5, "STOP")])

    def flaky(model: str, prompt: str) -> tuple[str, int, int, str]:
        step = next(tries)
        if isinstance(step, Exception):
            raise step
        return step

    slept: list[float] = []
    assert vc.make_complete(s, generate=flaky, sleep=slept.append)("p") == "{\"q\": 0.7}"
    assert slept == [vc.BACKOFF_S[0], vc.BACKOFF_S[1]]

    def broken(model: str, prompt: str) -> tuple[str, int, int, str]:
        raise Fatal("bad request")

    with pytest.raises(Fatal):
        vc.make_complete(s, generate=broken, sleep=slept.append)("p")
    assert json.loads(s.spend.read_text())["held"] == pytest.approx(0.0)


def test_settings_need_the_spend_file_and_a_positive_cap() -> None:
    with pytest.raises(RuntimeError):
        vc.Settings.from_env({})
    with pytest.raises(RuntimeError):
        vc.Settings.from_env({"COG_VERTEX_SPEND": "/x/spend.json", "COG_VERTEX_CAP_USD": "0"})
    got = vc.Settings.from_env({"COG_VERTEX_SPEND": "/x/spend.json", "COG_VERTEX_CAP_USD": "5"})
    assert got.cap_usd == 5.0 and got.model == vc.MODEL_ID
