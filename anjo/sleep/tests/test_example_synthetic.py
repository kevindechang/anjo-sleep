# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The standalone synthetic example (Task 15): it runs end to end on fakes, prints her chooser's
receipt, extracts and folds before it chooses, and holds no private Anjo code.
"""
from __future__ import annotations

import ast
import json
import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import anjo.sleep
from anjo.sleep.examples.synthetic import fakes, persona, run
from anjo.sleep.examples.synthetic.world import open_world
from anjo.sleep.mind.choose import StateChooser
from anjo.sleep.types import ChoiceContext, Want, WantSnapshot

SLEEP = Path(anjo.sleep.__file__).parent
EXAMPLE = SLEEP / "examples" / "synthetic"
NO_GATE = "no crisis gate"
FIVE_ARMS = ("1", "2C", "2F", "3", "4")


def _run(tmp_path: Path, *extra: str, chooser: Any = None) -> tuple[int, list[str], run.ExampleReport | None]:
    lines: list[str] = []
    services = fakes.fake_services() if chooser is None else fakes.fake_services(chooser=chooser)
    try:
        report = run.execute(
            ["--fake", "--workdir", str(tmp_path / "w"), "--no-chat", *extra], services, say=lines.append,
        )
    except run.ExampleError as error:
        lines.append(str(error))
        return 1, lines, None
    return 0, lines, report


# ── The core imports no private Anjo ──────────────────────────────────────────


def _anjo_imports(source: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
            if node.module == "anjo":
                found |= {f"anjo.{alias.name}" for alias in node.names}
    return {name for name in found if name == "anjo" or name.startswith("anjo.")}


def test_core_imports_no_private_anjo() -> None:
    offenders: dict[str, set[str]] = {}
    for path in sorted(SLEEP.rglob("*.py")):
        relative = path.relative_to(SLEEP)
        if relative.parts[0] in {"app", "tests"}:
            continue
        private = {
            name for name in _anjo_imports(path.read_text(encoding="utf-8"))
            if name != "anjo.sleep" and not name.startswith("anjo.sleep.")
        }
        if private:
            offenders[relative.as_posix()] = private
    assert offenders == {}


# ── The example, end to end, on fakes ─────────────────────────────────────────


def test_example_runs_end_to_end_with_fakes(tmp_path: Path) -> None:
    rc, lines, report = _run(tmp_path)

    text = "\n".join(lines)
    assert rc == 0 and report is not None, text
    assert report.stages == (
        "load", "probes", "extract", "fold", "choose", "practice", "dataset", "train", "arms", "chat",
    )
    assert NO_GATE in text.lower() and "not promoted" in text.lower()
    assert report.checkpoint is not None and report.checkpoint.person_id == "example-calla"
    assert report.dataset_examples > 0
    assert {row.arm_a for row in report.table} >= {"3", "4"}
    assert all(row.n > 0 for row in report.table)
    assert "fakes" in text.lower()  # the run says that its scores mean nothing
    world = tmp_path / "w"
    assert not list(world.rglob("promoted.jsonl")) and not list(world.rglob("current.json"))
    assert (world / "probes" / "example-calla.jsonl").is_file()
    assert len(report.probe_sha256) == 64


def test_example_runs_all_five_arms_and_3x_with_a_donor(tmp_path: Path) -> None:
    _, _, report = _run(tmp_path, "--donor", fakes.FAKE_DONOR)

    assert report is not None
    assert set(report.arms) == {*FIVE_ARMS, "3x"}
    _, _, without = _run(tmp_path / "again")
    assert without is not None and set(without.arms) == set(FIVE_ARMS)


def test_example_says_there_is_no_crisis_gate_on_every_run(tmp_path: Path) -> None:
    _, lines, _ = _run(tmp_path)

    gate_lines = [line for line in lines if NO_GATE in line.lower()]
    assert gate_lines and "private" in gate_lines[0].lower()
    assert any("not promoted" in line.lower() for line in lines)


def test_example_fails_fast_when_a_key_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "zq-secret-value-0000"
    rc = run.main(["--workdir", str(tmp_path / "w")], environ={"MODAL_TOKEN_ID": secret})

    err = capsys.readouterr().err
    assert rc == 2
    for name in ("MODAL_TOKEN_SECRET", "ANJO_META_API_KEY", "BASETEN_API_KEY"):
        assert name in err
    assert "--fake" in err
    assert not (tmp_path / "w").exists()  # nothing ran before the check
    assert secret not in err  # a name is reported, never a value


def test_example_chat_answers_from_base_and_from_the_adapter(tmp_path: Path) -> None:
    lines: list[str] = []
    typed = iter(["How do I bend a rib?", "/quit"])

    report = run.execute(
        ["--fake", "--workdir", str(tmp_path / "w")], fakes.fake_services(), say=lines.append,
        input_fn=lambda _prompt: next(typed),
    )

    assert report.checkpoint is not None
    base = [line for line in lines if line.startswith("[base] ")]
    adapter = [line for line in lines if line.startswith("[base + adapter] ")]
    assert len(base) == len(adapter) == 1
    assert report.checkpoint.id in adapter[0] and report.checkpoint.id not in base[0]


def test_example_refuses_a_donor_that_is_not_an_adapter_name(tmp_path: Path) -> None:
    rc, lines, report = _run(tmp_path, "--donor", "not-an-adapter")

    assert rc == 1 and report is None
    assert "--donor" in lines[-1]


def test_example_refuses_a_workdir_that_already_holds_files(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "old.txt").write_text("x")

    rc, lines, _ = _run(tmp_path)

    assert rc == 1 and "new folder" in lines[-1]


# ── Her chooser ───────────────────────────────────────────────────────────────


def test_example_prints_a_chooser_receipt(tmp_path: Path) -> None:
    _, lines, report = _run(tmp_path)

    assert report is not None
    receipt = report.receipt
    assert receipt["opportunity"] == "away" and receipt["candidates"]
    text = "\n".join(lines)
    assert "chooser receipt" in text.lower()
    assert str(receipt["winner"]) in text  # the winning want is named
    for candidate in receipt["candidates"]:
        assert candidate["want_id"] in text and candidate["status"] in text
    assert json.dumps(receipt, sort_keys=True) in text  # the receipt itself, exactly as recorded


class _Spy(StateChooser):
    """Records what her stores held at the moment the chooser first read them."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.seen: list[tuple[bool, int, int]] = []

    def derive(self, snapshot: WantSnapshot) -> tuple[Want, ...]:
        world = open_world(self.workdir / "w", "example-calla", date(2026, 3, 2))
        brief = world.briefs.latest()
        folded = len(brief.folded) if brief is not None else 0
        self.seen.append((brief is not None, folded, len(snapshot.targets)))
        return super().derive(snapshot)

    def choose(self, opportunity: Any, wants: Sequence[Want], ctx: ChoiceContext) -> Any:
        assert self.seen, "choose ran before derive read her state"
        return super().choose(opportunity, wants, ctx)


def test_example_extracts_and_folds_before_choosing(tmp_path: Path) -> None:
    spy = _Spy(tmp_path)

    _, _, report = _run(tmp_path, chooser=spy)

    assert report is not None
    stages = report.stages
    assert stages.index("extract") < stages.index("fold") < stages.index("choose")
    assert spy.seen, "the chooser never ran"
    has_brief, folded, targets = spy.seen[0]
    assert has_brief and folded > 0 and targets > 0  # her state already held the folded day
    assert report.extracted_items >= folded > 0


# ── The inputs are invented ───────────────────────────────────────────────────


def test_persona_and_day_are_entirely_invented() -> None:
    person = persona.load_persona(EXAMPLE / "persona.json")
    day = persona.load_day(EXAMPLE / "day1.jsonl", person.day)

    registered = {
        (d["visible"]["person_id"], d["visible"]["name"])
        for d in (json.loads(p.read_text()) for p in (SLEEP / "personas").glob("*.json"))
    }
    assert person.person_id not in {pid for pid, _ in registered}
    assert person.name not in {name for _, name in registered}
    blob = ((EXAMPLE / "persona.json").read_text() + (EXAMPLE / "day1.jsonl").read_text()).lower()
    for term in ("tessa", "marlowe", "kevin", "anjo.ai"):
        assert term not in blob
    said = " ".join(turn.text.lower() for turn in day)
    for held in person.held_out_preferences:
        assert held.text.lower() not in said  # nothing held out is in the day
    assert len(person.new_work) >= 3 and all(turn.role in {"user", "assistant"} for turn in day)


def test_example_files_name_only_what_the_example_defines() -> None:
    assert (EXAMPLE / "run.py").is_file() and (EXAMPLE / "__main__.py").is_file()
    assert persona.resolve("persona.json") == EXAMPLE / "persona.json"  # a bare name finds the bundled file
    assert persona.resolve("day1.jsonl") == EXAMPLE / "day1.jsonl"


# ── Packaging and the README ──────────────────────────────────────────────────

THIRD_PARTY_TO_PACKAGE = {"httpx": "httpx", "modal": "modal", "boto3": "boto3"}


def _third_party_imports(root: Path) -> set[str]:
    import sys

    found: set[str] = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
            )
            found |= {n.split(".")[0] for n in names if n}
    return {n for n in found if n not in sys.stdlib_module_names and n not in {"anjo", "__future__"}}


def test_public_pyproject_covers_the_example() -> None:
    meta = tomllib.loads((SLEEP / "pyproject.public.toml").read_text(encoding="utf-8"))["project"]
    declared = {
        spec.split("[")[0].split(">")[0].split("=")[0].split("<")[0].split(";")[0].strip().lower()
        for spec in [*meta["dependencies"], *(s for extra in meta["optional-dependencies"].values() for s in extra)]
    }
    needed = _third_party_imports(EXAMPLE)
    assert needed <= declared, sorted(needed - declared)


def test_readme_runs_the_fake_example_before_the_live_one() -> None:
    readme = (SLEEP / "README.md").read_text(encoding="utf-8")

    assert "## Run it yourself" in readme
    section = readme.split("## Run it yourself", 1)[1].split("\n## ", 1)[0]
    fake = section.index("python -m anjo.sleep.examples.synthetic --fake")
    live = section.index("python -m anjo.sleep.examples.synthetic --persona persona.json --day day1.jsonl")
    assert fake < live
    for name in ("MODAL_TOKEN_ID", "ANJO_META_API_KEY", "BASETEN_API_KEY"):
        assert name in section
