# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The standalone synthetic example (Task 15): it runs end to end on fakes, prints her chooser's
receipt, extracts and folds before it chooses, and holds no private Anjo code.
"""
from __future__ import annotations

import ast
import json
import re
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
from anjo.sleep.serving import config
from anjo.sleep.types import ChoiceContext, Want, WantSnapshot

SLEEP = Path(anjo.sleep.__file__).parent
EXAMPLE = SLEEP / "examples" / "synthetic"
NO_GATE = "no crisis gate"


def _published_file(name: str, root_name: str) -> Path:
    """A file that publishing renames to the repo root (README, pyproject): where it sits in either tree."""
    inside = SLEEP / name
    return inside if inside.is_file() else SLEEP.parents[1] / root_name
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


DOTTED = re.compile(r"anjo(\.\w+)+")


def _anjo_imports(source: str, package: tuple[str, ...]) -> set[str]:
    """Every ``anjo`` module the source reaches: imports (relative ones resolved against
    ``package``, the module's own package) and dotted names written as strings, which is how
    ``mock.patch`` and ``importlib.import_module`` name a module."""
    found: set[str] = set()
    tree = ast.parse(source)
    probes = {  # find_spec("anjo.core") asks whether private Anjo is there; it imports nothing
        id(arg) for node in ast.walk(tree) if isinstance(node, ast.Call)
        and getattr(node.func, "attr", getattr(node.func, "id", "")) == "find_spec" for arg in node.args
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            keep = len(package) - (node.level - 1) if node.level else 0
            base = [*package[:keep]] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            found.add(module)
            found |= {f"{module}.{alias.name}" for alias in node.names}
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in probes
            and DOTTED.fullmatch(node.value) and not node.value.startswith("anjo.sleep")
        ):  # a string naming the adapters is a guard's text, not an import
            found.add(node.value)
    return {name for name in found if name == "anjo" or name.startswith("anjo.")}


def test_core_imports_no_private_anjo() -> None:
    offenders: dict[str, set[str]] = {}
    for path in sorted(SLEEP.rglob("*.py")):
        relative = path.relative_to(SLEEP)
        if relative.parts[0] == "app" or relative.parts[:2] == ("tests", "app"):
            continue  # the adapters, and their tests: the published tree leaves both out
        package = ("anjo", "sleep", *relative.parts[:-1])
        private = {
            name for name in _anjo_imports(path.read_text(encoding="utf-8"), package)
            if name != "anjo" and name != "anjo.sleep"
            and (not name.startswith("anjo.sleep.") or name.startswith("anjo.sleep.app"))
        }  # function-level imports count too; so does the adapters package, which needs private Anjo
        if private:
            offenders[relative.as_posix()] = private
    assert offenders == {}


def test_the_import_check_sees_relative_imports_and_imports_by_name() -> None:
    here = ("anjo", "sleep", "tests")
    anjo = "anjo"  # the examples below are built, so this file holds no private module's name

    assert _anjo_imports("from ...core import paths", here) == {f"{anjo}.core", f"{anjo}.core.paths"}
    assert f"{anjo}.core" in _anjo_imports("from ... import core", here)
    assert _anjo_imports("from . import test_x", here) == {"anjo.sleep.tests", "anjo.sleep.tests.test_x"}
    assert _anjo_imports(f'mock.patch("{anjo}.dashboard.auth.x")', here) == {f"{anjo}.dashboard.auth.x"}
    assert _anjo_imports(f'importlib.import_module("{anjo}.core.db")', here) == {f"{anjo}.core.db"}
    assert _anjo_imports('print("anjo is a word, not a module")', here) == set()
    assert _anjo_imports(f'importlib.util.find_spec("{anjo}.core")', here) == set()


# ── The example, end to end, on fakes ─────────────────────────────────────────


def test_example_runs_end_to_end_with_fakes(tmp_path: Path) -> None:
    rc, lines, report = _run(tmp_path)

    text = "\n".join(lines)
    assert rc == 0 and report is not None, text
    assert report.stages == (
        "load", "probes", "extract", "fold", "choose", "practice", "dataset", "train", "arms", "chat",
    )
    assert NO_GATE in text.lower() and "not promoted" in text.lower()
    assert report.checkpoint is not None and report.checkpoint.person_id == "example-teo"
    assert report.dataset_examples > 0
    assert {row.arm_a for row in report.table} >= {"3", "4"}
    assert all(row.n > 0 for row in report.table)
    assert "fakes" in text.lower()  # the run says that its scores mean nothing
    world = tmp_path / "w"
    assert not list(world.rglob("promoted.jsonl")) and not list(world.rglob("current.json"))
    assert (world / "probes" / "example-teo.jsonl").is_file()
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
    nowhere = str(tmp_path / "no-modal.toml")
    rc = run.main(
        ["--workdir", str(tmp_path / "w")],
        environ={"MODAL_TOKEN_ID": secret, "MODAL_CONFIG_PATH": nowhere},
    )

    captured = capsys.readouterr()
    err = captured.err
    assert rc == 2 and NO_GATE in captured.out.lower()
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


def test_no_crisis_gate_prints_even_for_a_bad_argument(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        run.main(["--no-such-flag"])

    assert NO_GATE in capsys.readouterr().out.lower()


def test_modal_login_from_modal_setup_is_accepted(tmp_path: Path) -> None:
    from anjo.sleep.examples.synthetic import live

    config = tmp_path / "modal.toml"
    config.write_text('[default]\ntoken_id = "x"\ntoken_secret = "y"\n')
    others = {name: "v" for name in ("ANJO_META_API_KEY", "BASETEN_API_KEY", "ANJO_SLEEP_MODAL_URL",
                                     "ANJO_SLEEP_MODAL_TOKEN", "ANJO_SLEEP_GLIMMER_REVISION")}

    assert live.missing_keys({**others, "MODAL_CONFIG_PATH": str(config)}) == []
    assert live.missing_keys({**others, "MODAL_CONFIG_PATH": str(tmp_path / "none")}) != []
    assert live.missing_keys({**others, "MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b",
                              "MODAL_CONFIG_PATH": str(tmp_path / "none")}) == []


def test_step_one_writes_the_persona_and_the_day_to_the_store(tmp_path: Path) -> None:
    _, _, report = _run(tmp_path)

    assert report is not None
    world = tmp_path / "w"
    person = persona.load_persona(EXAMPLE / "persona.json")
    turns = persona.load_day(EXAMPLE / "day1.jsonl", person.day)
    stored = [json.loads(line) for line in (world / "day.jsonl").read_text().splitlines()]
    assert [row["text"] for row in stored] == [turn.text for turn in turns]
    record = json.loads((world / "persona.jsonl").read_text())
    assert record["person_id"] == person.person_id
    assert not any(held.text in json.dumps(record) for held in person.held_out_preferences)


def test_the_trained_adapter_records_the_rendering_and_a_foreign_one_is_refused(tmp_path: Path) -> None:
    from anjo.sleep.serving import framing
    from anjo.sleep.serving.client import ServingError

    _, _, report = _run(tmp_path)

    assert report is not None and report.checkpoint is not None
    assert report.checkpoint.rendering == fakes.FakeGlimmer().rendering
    meta = json.loads((tmp_path / "w" / "adapters" / report.checkpoint.id / "adapter_config.json").read_text())
    assert meta[framing.RENDERING_KEY] == report.checkpoint.rendering
    from dataclasses import replace

    with pytest.raises(ServingError, match="never served under another"):
        fakes.FakeGlimmer().complete(replace(report.checkpoint, rendering="g000/low"), [], temperature=0.0)


def test_each_run_trains_under_a_new_adapter_name(tmp_path: Path) -> None:
    serving = fakes.FakeServing()

    first = run.execute(["--fake", "--workdir", str(tmp_path / "a"), "--no-chat"],
                        fakes.fake_services(serving=serving), say=lambda _line: None)
    second = run.execute(["--fake", "--workdir", str(tmp_path / "b"), "--no-chat"],
                         fakes.fake_services(serving=serving), say=lambda _line: None)

    assert first.checkpoint and second.checkpoint and first.checkpoint.id != second.checkpoint.id
    assert len(serving.on_volume - {fakes.FAKE_DONOR}) == 2


def test_an_adapter_name_already_on_the_volume_stops_the_run_with_a_clear_message(tmp_path: Path) -> None:
    serving = fakes.FakeServing()
    quiet = lambda _line: None  # noqa: E731
    run.execute(["--fake", "--workdir", str(tmp_path / "a"), "--no-chat"],
                fakes.fake_services(serving=serving), say=quiet, run_id="same")

    with pytest.raises(run.ExampleError, match="already on the volume"):
        run.execute(["--fake", "--workdir", str(tmp_path / "b"), "--no-chat"],
                    fakes.fake_services(serving=serving), say=quiet, run_id="same")


def test_fake_upload_refuses_a_name_that_exists(tmp_path: Path) -> None:
    from anjo.sleep.serving import config

    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": config.MODEL, "r": 16}))
    (folder / "adapter_model.safetensors").write_bytes(b"x")
    serving = fakes.FakeServing()
    serving.upload_adapter(folder, "p-000000000001-n1")

    with pytest.raises(config.ServeConfigError):
        serving.upload_adapter(folder, "p-000000000001-n1")


def test_the_run_waits_out_the_idle_window_so_a_cold_start_mounts_the_new_adapter(tmp_path: Path) -> None:
    serving = fakes.FakeServing()  # like Modal: nothing restarts it, only ten idle minutes do
    lines: list[str] = []
    started = serving.clock()

    report = run.execute(["--fake", "--workdir", str(tmp_path / "w"), "--no-chat"],
                         fakes.fake_services(serving=serving), say=lines.append)

    assert report.checkpoint is not None and report.checkpoint.id in serving.visible
    assert serving.clock() - started >= config.SCALEDOWN_S  # the wait, on the fake's clock
    assert any(line.startswith("waiting ") and "scale down" in line for line in lines)
    assert any(line.startswith("serving p-") for line in lines)


def test_a_fake_server_does_not_restart_on_its_own(tmp_path: Path) -> None:
    serving = fakes.FakeServing()
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text(json.dumps(fakes._adapter_config("x")))
    (folder / "adapter_model.safetensors").write_bytes(b"x")
    serving.upload_adapter(folder, "p-000000000002-n1")

    serving.sleep_now()
    serving.wake()  # no idle time has passed: the same container answers

    assert "p-000000000002-n1" not in serving.visible
    serving.clock.sleep(config.SCALEDOWN_S)
    serving.wake()
    assert "p-000000000002-n1" in serving.visible


def test_a_server_that_never_loads_the_adapter_stops_the_run_with_the_fix(tmp_path: Path) -> None:
    serving = fakes.FakeServing(scales_down=False)

    with pytest.raises(run.ExampleError, match="modal app stop"):
        run.execute(["--fake", "--workdir", str(tmp_path / "w"), "--no-chat"],
                    fakes.fake_services(serving=serving), say=lambda _line: None)


class _CountingGpu(fakes.FakeGpu):
    calls = 0

    def run(self, job: str, inputs: Any, *, timeout_s: float) -> Any:
        type(self).calls += 1
        return super().run(job, inputs, timeout_s=timeout_s)


def _services_with(gpu: Any, serving: fakes.FakeServing, spark_calls: list[int] | None = None) -> Any:
    from dataclasses import replace

    services = replace(fakes.fake_services(serving=serving), gpu=gpu)
    if spark_calls is None:
        return services

    def counted(*args: Any, **kwargs: Any) -> Any:
        spark_calls.append(1)
        return fakes.fake_spark(*args, **kwargs)

    return replace(services, spark=counted)


def test_adapter_flag_reuses_an_adapter_and_never_trains(tmp_path: Path) -> None:
    serving = fakes.FakeServing()
    gpu = _CountingGpu()
    _CountingGpu.calls = 0

    report = run.execute(
        ["--fake", "--workdir", str(tmp_path / "w"), "--no-chat", "--adapter", fakes.FAKE_DONOR],
        _services_with(gpu, serving), say=lambda _line: None,
    )

    assert _CountingGpu.calls == 0
    assert report.checkpoint is not None and report.checkpoint.id == fakes.FAKE_DONOR
    assert not {"practice", "dataset", "train"} & set(report.stages) and "arms" in report.stages


def test_an_unknown_adapter_stops_the_run_before_any_paid_step(tmp_path: Path) -> None:
    spark_calls: list[int] = []

    with pytest.raises(run.ExampleError, match="no adapter p-0000000000aa-n1"):
        run.execute(
            ["--fake", "--workdir", str(tmp_path / "w"), "--adapter", "p-0000000000aa-n1"],
            _services_with(_CountingGpu(), fakes.FakeServing(), spark_calls), say=lambda _line: None,
        )

    assert spark_calls == []


def test_a_donor_that_is_not_on_the_volume_stops_the_run_before_any_paid_step(tmp_path: Path) -> None:
    spark_calls: list[int] = []

    with pytest.raises(run.ExampleError, match="--donor: no adapter p-0000000000bb-n1"):
        run.execute(
            ["--fake", "--workdir", str(tmp_path / "w"), "--donor", "p-0000000000bb-n1"],
            _services_with(_CountingGpu(), fakes.FakeServing(), spark_calls), say=lambda _line: None,
        )

    assert spark_calls == []


@pytest.mark.parametrize("flag", ["--adapter", "--donor"])
def test_an_adapter_trained_under_another_rendering_stops_the_run_before_any_paid_step(
    tmp_path: Path, flag: str,
) -> None:
    serving = fakes.FakeServing()
    serving.configs[fakes.FAKE_DONOR] = fakes._adapter_config("g000000000000/high")
    spark_calls: list[int] = []

    with pytest.raises(run.ExampleError, match="never served under another"):
        run.execute(["--fake", "--workdir", str(tmp_path / "w"), flag, fakes.FAKE_DONOR],
                    _services_with(_CountingGpu(), serving, spark_calls), say=lambda _line: None)

    assert spark_calls == []


def test_the_probe_failure_names_the_exact_reuse_command(tmp_path: Path) -> None:
    with pytest.raises(run.ExampleError) as raised:
        run.execute(["--fake", "--workdir", str(tmp_path / "w"), "--no-chat"],
                    fakes.fake_services(serving=fakes.FakeServing(scales_down=False)),
                    say=lambda _line: None, run_id="r1")

    message = str(raised.value)
    adapter = message.rsplit("--adapter ", 1)[1].strip()
    assert f"modal app stop {config.serve_app_name(config.serve_precision())}" in message and adapter.startswith("p-")


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
        world = open_world(self.workdir / "w", "example-teo", date(2026, 3, 2))
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
    hero = json.loads((SLEEP / "personas" / "film_founder.json").read_text())["visible"]
    others = {(pid, name) for pid, name in registered if pid != hero["person_id"]}
    assert person.person_id not in {pid for pid, _ in registered}  # the example's own id
    assert person.name == hero["name"] and person.name not in {name for _, name in others}  # the hero, shown smaller
    blob = ((EXAMPLE / "persona.json").read_text() + (EXAMPLE / "day1.jsonl").read_text()).lower()
    for term in ("tessa", "marlowe", "kevin"):
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
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") in {"import_module", "find_spec"}:
                found |= {  # a module loaded by name, as `modal` is
                    arg.value.split(".")[0] for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                }
    return {n for n in found if n not in sys.stdlib_module_names and n not in {"anjo", "__future__"}}


def test_public_pyproject_covers_the_example() -> None:
    meta = tomllib.loads(_published_file("pyproject.public.toml", "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = {
        spec.split("[")[0].split(">")[0].split("=")[0].split("<")[0].split(";")[0].strip().lower()
        for spec in [*meta["dependencies"], *(s for extra in meta["optional-dependencies"].values() for s in extra)]
    }
    needed = _third_party_imports(EXAMPLE)
    assert "modal" in needed  # the scan sees the by-name import of modal too
    assert needed <= declared, sorted(needed - declared)


def test_readme_runs_the_fake_example_before_the_live_one() -> None:
    readme = _published_file("README.md", "README.md").read_text(encoding="utf-8")

    assert "## Run it yourself" in readme
    section = readme.split("## Run it yourself", 1)[1].split("\n## ", 1)[0]
    fake = section.index("python -m anjo.sleep.examples.synthetic --fake")
    live = section.index("python -m anjo.sleep.examples.synthetic --persona persona.json --day day1.jsonl")
    assert fake < live
    for name in ("MODAL_TOKEN_ID", "ANJO_META_API_KEY", "BASETEN_API_KEY"):
        assert name in section
