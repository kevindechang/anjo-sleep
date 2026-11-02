# mypy: disallow-untyped-decorators=False
"""The scripts of a detached Colab night (``colab_detached``), run for real as local processes against
a temporary folder that stands in for the VM's: no VM, no network."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from anjo.sleep.train.runners import colab_detached as detached
from anjo.sleep.train.runners.colab_detached import Reading

KEYS = ("ANJO_TRAIN_CONFIG", "ANJO_TRAIN_OUT")
WAIT_S = 20.0


def _run(script: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=WAIT_S,
                          env={**os.environ, **(env or {})}, check=True)
    return done.stdout


def _launch(remote: Path, entry: str, env: dict[str, str] | None = None) -> int:
    out = _run(detached.launch_script(str(remote), entry, KEYS),
               {"ANJO_TRAIN_CONFIG": '{"r": 16}', "ANJO_TRAIN_OUT": str(remote / "out"), **(env or {})})
    pid = detached.read_launch(out)
    assert pid is not None, out
    return pid


def _poll(remote: Path) -> Reading | None:
    return detached.read_poll(_run(detached.poll_script(str(remote))))


def _until(remote: Path, state: str) -> Reading:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        reading = _poll(remote)
        if reading is not None and reading.state == state:
            return reading
        time.sleep(0.1)
    raise AssertionError(f"no {state} reading within {WAIT_S} s")


def test_the_launcher_returns_at_once_and_the_poll_reads_the_trainings_exit_code(tmp_path: Path) -> None:
    entry = ("import json, os, sys, time\n"
             "time.sleep(1.5)\n"
             "os.makedirs(os.environ['ANJO_TRAIN_OUT'], exist_ok=True)\n"
             "seen = {key: os.environ.get(key) for key in ('ANJO_TRAIN_CONFIG', 'ANJO_TRAIN_PREV')}\n"
             "open(os.path.join(os.environ['ANJO_TRAIN_OUT'], 'seen.json'), 'w').write(json.dumps(seen))\n"
             "print('training', __file__)\n"
             "sys.exit(3)\n")
    started = time.monotonic()
    _launch(tmp_path, entry, {"ANJO_TRAIN_PREV": "a kept VM's last night"})  # not one of this night's keys
    assert time.monotonic() - started < 1.5  # it never waits for the training
    assert _poll(tmp_path) == Reading("running")
    assert _until(tmp_path, "done") == Reading("done", 3)
    seen = (tmp_path / "out" / "seen.json").read_text()
    assert seen == '{"ANJO_TRAIN_CONFIG": "{\\"r\\": 16}", "ANJO_TRAIN_PREV": null}'
    assert f"training {tmp_path / detached.ENTRY}" in (tmp_path / detached.LOG).read_text()


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_killed_training_reads_missing_and_a_new_launch_forgets_the_last_exit_code(tmp_path: Path) -> None:
    entry = ("import os, time\nos.makedirs(os.environ['ANJO_TRAIN_OUT'], exist_ok=True)\n"
             "open(os.path.join(os.environ['ANJO_TRAIN_OUT'], 'pid'), 'w').write(str(os.getpid()))\ntime.sleep(60)\n")
    _launch(tmp_path, entry)
    assert _poll(tmp_path) == Reading("running")
    deadline = time.monotonic() + WAIT_S
    while not (tmp_path / "out" / "pid").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    training = int((tmp_path / "out" / "pid").read_text())
    assert detached.MARK in _run(detached.kill_script(str(tmp_path)))
    assert _until(tmp_path, "missing") == Reading("missing")  # gone, with no exit code
    assert _gone(training)  # the training itself, the runner's child, went with the runner's group
    (tmp_path / detached.EXIT).write_text("0")  # as a kept VM's last night left it
    _launch(tmp_path, "import time\ntime.sleep(60)\n")
    assert _poll(tmp_path) == Reading("running")
    _run(detached.kill_script(str(tmp_path)))
    _until(tmp_path, "missing")


def test_under_a_kernel_that_never_reaps_its_runner_the_poll_still_reads_done(tmp_path: Path) -> None:
    """On the VM the launcher runs in the session's kernel, which lives on and never reaps the runner."""
    launcher = detached.launch_script(str(tmp_path), "import time\ntime.sleep(0.5)\n", KEYS)
    kernel = subprocess.Popen(
        [sys.executable, "-c", "import os, sys, time\nos.environ.update(ANJO_TRAIN_CONFIG='{}', ANJO_TRAIN_OUT='o')\n"
         "exec(compile(sys.stdin.read(), 'launch', 'exec'), {})\nsys.stdout.flush()\ntime.sleep(30)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert kernel.stdin is not None and kernel.stdout is not None
        kernel.stdin.write(launcher)
        kernel.stdin.close()
        assert detached.read_launch(kernel.stdout.readline()) is not None
        assert _until(tmp_path, "done") == Reading("done", 0) and kernel.poll() is None  # the kernel lives on
    finally:
        kernel.kill()
        kernel.wait()


def test_no_launch_reads_missing_and_a_kill_with_nothing_to_kill_says_so(tmp_path: Path) -> None:
    assert _poll(tmp_path) == Reading("missing")
    assert "not killed" in _run(detached.kill_script(str(tmp_path)))


def test_the_log_tail_prints_the_logs_last_lines(tmp_path: Path) -> None:
    _launch(tmp_path, "for i in range(80):\n    print('line', i)\n")
    assert _until(tmp_path, "done") == Reading("done", 0)
    lines = _run(detached.log_tail_script(str(tmp_path))).splitlines()
    assert lines[0].startswith(detached.MARK) and lines[1:] == [f"line {i}" for i in range(30, 80)]
    assert _run(detached.log_tail_script(str(tmp_path / "none"))).splitlines()[1:] == []


@pytest.mark.parametrize(("printed", "reading"), [
    (f"{detached.MARK} running\n", Reading("running")),
    (f"[colab] An update is available\n{detached.MARK} done -9\n", Reading("done", -9)),
    (f"{detached.MARK} missing\n", Reading("missing")),
    ("", None), (f"{detached.MARK} done\n", None), (f"{detached.MARK} sleeping\n", None),
    ("Traceback (most recent call last):\n", None),
])
def test_a_poll_is_read_only_off_its_marked_line(printed: str, reading: Reading | None) -> None:
    assert detached.read_poll(printed) == reading


def test_a_launch_is_read_only_off_its_marked_line() -> None:
    assert detached.read_launch(f"[colab] note\n{detached.MARK} launched 4242\n") == 4242
    assert detached.read_launch("") is None and detached.read_launch(f"{detached.MARK} launched x\n") is None


def test_the_scripts_use_only_the_standard_library() -> None:
    for script in (detached.launch_script("/content/anjo", "print(1)\n", KEYS), detached.poll_script("/x"),
                   detached.kill_script("/x"), detached.log_tail_script("/x"), detached.runner_script("/x")):
        tree = ast.parse(script)
        imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert imported and imported <= set(sys.stdlib_module_names)
