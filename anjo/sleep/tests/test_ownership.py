# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Undo, delete and what changed (Task 9): RF2 (a delete while a night runs), remote copies, undo then
train, the changes lines, and that receipts don't depend on the talker. Fakes only, no network."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.ownership.cancel import EpochFlag
from anjo.sleep.ownership.inventory import FileInventory
from anjo.sleep.ownership.lock import drain
from anjo.sleep.ownership.changes import Change, what_changed
from anjo.sleep.ownership.delete import (
    DeleteIncomplete,
    DeletePorts,
    PathEraser,
    delete_all,
    erase_stored,
)
from anjo.sleep.ownership.remote import ModalAdapters, S3Adapters
from anjo.sleep.ownership.undo import NothingToUndo, undo
from anjo.sleep.tests.ownership_fakes import FakeS3, FakeVolume
from anjo.sleep.tests.night_fakes import NIGHT, Homework, World, steps, thread
from anjo.sleep.types import CheckpointRef, NightResult

SECOND = NIGHT + timedelta(days=1)
THIRD = NIGHT + timedelta(days=2)


def ref(uid: str, night: date, name: str, parent: str | None = None) -> CheckpointRef:
    return CheckpointRef(name, uid, night, f"modal-volume://anjo-sleep-adapters/{name}", parent)


def adapter(uid: str, number: int) -> str:
    return f"p-{hashlib.sha256(uid.encode()).hexdigest()[:12]}-n{number}"


def _one(world: World, uid: str, night: date, **deps: Any) -> NightResult:
    return run_night_batch_with(world.deps(**deps), {uid: night})[uid]


# ── Fakes ─────────────────────────────────────────────────────────────────────


class Ports:
    """Ports over a ``World``'s files: the registry, the person's folder, the night logs."""

    def __init__(self, world: World, tmp: Path) -> None:
        self.world = world
        self.flag = EpochFlag(tmp / "cancel")
        self.volume = FakeVolume()
        self.bucket = FakeS3()
        self.stopped: list[str] = []
        self.inventory = FileInventory(tmp / "inventory")
        self.lock_path = tmp / "night.lock"

    def names(self, uid: str) -> frozenset[str]:
        registry = self.world.registry
        return registry.trained_names(uid) | {c.id for c in registry.history(uid)}

    def ports(self) -> DeletePorts:
        world = self.world

        class Stopper:
            name = "fake runner"

            def stop(inner, user_id: str, adapters: frozenset[str]) -> int:  # noqa: N805
                self.stopped.append(user_id)
                return 1

        return DeletePorts(
            cancel=self.flag, adapter_names=self.names, inventory=self.inventory, stoppers=(Stopper(),),
            night_lock=lambda t: drain(self.lock_path, t), lock_timeout_s=5.0,
            remotes=(ModalAdapters(self.volume), S3Adapters("bucket", self.bucket)),
            erasers=(
                PathEraser("night logs", lambda uid, _: [world.logs(uid)]),
                PathEraser("registry", lambda uid, _: [world.root / "checkpoints" / uid]),
                PathEraser("sleep files", lambda uid, _: [world.root / uid]),
            ),
        )


def _upload(p: Ports, name: str) -> None:
    p.volume.folders.add(name)
    p.bucket.keys |= {f"adapters/{name}/adapter_config.json", f"adapters/{name}/adapter_model.safetensors",
                      f"adapters/{name}/README.md"}


# ── RF2: a delete while a night runs ─────────────────────────────────────────


def test_delete_during_running_night_writes_nothing_after(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    for uid in ("p1", "p2"):
        world.plan(uid, NIGHT)
    deleted: list[Any] = []

    class DeletingTrainer:
        """Training is running when the person deletes; it then keeps writing, as a step in flight would."""

        def train(self, user_id: str, night: date, prev: Any, data: Any, name: str) -> Any:
            out = world.trainer.train(user_id, night, prev, data, name)
            if user_id == "p1":
                deleted.append(delete_all("p1", owned.ports()))
                world.evidence("p1").add(world.planned[("p1", NIGHT)])  # a write after the delete
            return out

    deps = world.deps(
        trainer=DeletingTrainer(), cancel=owned.flag.watcher(),
        purge=lambda uid: erase_stored(uid, owned.ports()),
    )
    results = run_night_batch_with(deps, {"p1": NIGHT, "p2": NIGHT})

    assert deleted and deleted[0].cancelled and deleted[0].jobs_stopped == 1
    assert results["p1"].status == "failed" and results["p1"].failed_at == "train"
    for left in (world.root / "p1", world.logs("p1"), world.root / "checkpoints" / "p1"):
        assert not left.exists(), left  # nothing of the person survives the night's last step
    assert world.notifier.queued == [] and world.registry.last_promoted("p1") is None
    assert results["p2"].status == "promoted"  # the rest of the batch goes on
    assert steps(results["p2"].log_path)[-1]["step"] == "finish"


def test_a_night_started_after_a_delete_runs_normally(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    world.plan("p1", NIGHT)
    delete_all("p1", owned.ports())  # before any night: a person with nothing yet
    deps = world.deps(cancel=owned.flag.watcher())
    assert run_night_batch_with(deps, {"p1": NIGHT})["p1"].status == "promoted"


def test_the_cancel_flag_is_seen_by_a_night_in_another_process(tmp_path: Path) -> None:
    flag = EpochFlag(tmp_path / "cancel")
    watching = flag.watcher()
    assert watching("p1") is False  # the baseline
    EpochFlag(tmp_path / "cancel").set("p1")  # another process deletes
    assert watching("p1") is True and watching("p2") is False
    assert EpochFlag(tmp_path / "cancel").watcher()("p1") is False  # a new night starts from the new counter
    with pytest.raises(ValueError):
        flag.set("../escape")


# ── The order, and a delete that can't finish ────────────────────────────────


def test_delete_runs_in_the_specs_order(tmp_path: Path) -> None:
    order: list[str] = []

    class Step:
        def __init__(self, name: str) -> None:
            self.name = name

        def set(self, user_id: str) -> None:
            order.append("cancel")

        def stop(self, user_id: str, adapters: frozenset[str]) -> int:
            order.append("stop")
            return 0

        def remove(self, adapters: frozenset[str]) -> int:
            order.append("remote")
            return 0

        def erase(self, user_id: str, adapters: frozenset[str]) -> int:
            order.append(self.name)
            return 0

    def names(uid: str) -> frozenset[str]:
        order.append("names")
        return frozenset()

    delete_all("p1", DeletePorts(cancel=Step("c"), adapter_names=names, inventory=FileInventory(tmp_path / "inv"), stoppers=(Step("s"),), remotes=(Step("r"),),
                                 erasers=(Step("files"), Step("mind workspace"))))
    assert order == ["cancel", "names", "stop", "remote", "files", "mind workspace"]  # fence first, then list


def test_a_failed_step_does_not_stop_the_rest_and_the_delete_says_so(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    world.plan("p1", NIGHT)
    _one(world, "p1", NIGHT)

    class Broken:
        name = "broken volume"

        def remove(self, adapters: frozenset[str]) -> int:
            raise OSError("the volume is unreachable")

    ports = replace(owned.ports(), remotes=(Broken(), *owned.ports().remotes))
    with pytest.raises(DeleteIncomplete) as raised:
        delete_all("p1", ports)
    assert raised.value.report.failures == ("remove broken volume: OSError",)
    assert not (world.root / "p1").exists() and not (world.root / "checkpoints" / "p1").exists()  # the rest ran


# ── Remote copies ────────────────────────────────────────────────────────────


def test_delete_removes_remote_checkpoints(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    first, second, other = adapter("p1", 1), adapter("p1", 2), adapter("p2", 1)
    world.registry.record_trained("p1", first, NIGHT)
    world.registry.promote("p1", ref("p1", NIGHT, first))
    world.registry.record_trained("p1", second, SECOND)  # trained and rejected: never promoted, still stored
    for name in (first, second, other):
        _upload(owned, name)
    world.plan("p1", NIGHT)
    world.evidence("p1").add(world.planned[("p1", NIGHT)])

    report = delete_all("p1", owned.ports())

    assert owned.volume.folders == {other} and {k.split("/")[1] for k in owned.bucket.keys} == {other}
    assert report.adapters == 2 and dict(report.remote_removed) == {"modal volume": 2, "sagemaker bucket": 2}
    assert not (world.root / "p1").exists() and not (world.root / "checkpoints" / "p1").exists()
    assert delete_all("p1", owned.ports()).adapters == 0  # again: nothing left, and no error


def test_remote_stores_refuse_a_name_that_is_not_an_adapter() -> None:
    for store in (ModalAdapters(FakeVolume({"keep"})), S3Adapters("b", FakeS3({"adapters/keep/x"}))):
        with pytest.raises(ValueError):
            store.remove(frozenset({"../../weights"}))


# ── Undo ─────────────────────────────────────────────────────────────────────


def test_undo_then_night_trains_from_restored(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    world.plan("p1", SECOND)
    world.plan("p1", THIRD, count=2)
    one = _one(world, "p1", NIGHT)
    two = _one(world, "p1", SECOND)
    assert one.checkpoint and two.checkpoint and world.registry.current("p1") == two.checkpoint
    kept = len(world.evidence("p1").stored(THIRD))

    restored = undo("p1", world.registry)

    assert restored == one.checkpoint and world.registry.current("p1") == one.checkpoint
    assert len(world.evidence("p1").stored(THIRD)) == kept  # the evidence stays
    three = _one(world, "p1", THIRD)
    assert world.trainer.calls[-1]["prev"] == one.checkpoint  # trained from the restored one
    assert three.status == "promoted" and three.checkpoint is not None
    assert three.checkpoint.parent_id == one.checkpoint.id
    assert world.trainer.calls[-1]["examples"] >= world.trainer.calls[0]["examples"]  # the undone night's lessons return


def test_undo_with_no_earlier_version_says_so(tmp_path: Path) -> None:
    world = World(tmp_path)
    with pytest.raises(NothingToUndo):
        undo("p1", world.registry)
    world.plan("p1", NIGHT)
    _one(world, "p1", NIGHT)
    with pytest.raises(NothingToUndo):
        undo("p1", world.registry)


# ── What changed ─────────────────────────────────────────────────────────────


def _score(path: Path, number: int, night: date, personal: float, subset: str | None = None,
           n: int | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    stat = {"score": personal, **({} if n is None else {"n": n})}
    (path / f"night-{number:02d}.json").write_text(json.dumps({
        "night": night.isoformat(), "arms": {"3": {"knowledge": stat, "overall": stat},
                                             "1": {"knowledge": {"score": 0.5}}},
        **({} if subset is None else {"subset": subset})}), encoding="utf-8")


def test_a_score_moves_only_against_a_night_scored_on_the_same_probes(tmp_path: Path) -> None:
    person = tmp_path / "results" / "p1"
    _score(person / "scores", 1, NIGHT, 0.50, n=65)  # the full run
    _score(person / "scores", 2, SECOND, 0.70, subset="daily-v1", n=11)  # the daily subset: 14 of its probes
    _score(person / "scores", 3, THIRD, 0.64, subset="daily-v1", n=11)
    changes = what_changed("p1", NIGHT, results=lambda uid: tmp_path / "results" / uid)
    assert [(c.night, c.kind) for c in changes] == [(THIRD, "score"), (THIRD, "score")]
    assert "0.70 to 0.64" in changes[0].line and "0.50" not in "\n".join(c.line for c in changes)
    assert all("on 11 of your own test questions" in c.line for c in changes)  # a small set says how small

    unsized = tmp_path / "results" / "p2"  # a subset move whose n isn't known is never reported
    _score(unsized / "scores", 1, SECOND, 0.70, subset="daily-v1")
    _score(unsized / "scores", 2, THIRD, 0.40, subset="daily-v1")
    assert what_changed("p2", NIGHT, results=lambda uid: tmp_path / "results" / uid) == []


def test_what_changed_reads_the_night_logs_and_the_score_moves(tmp_path: Path) -> None:
    world = World(tmp_path)
    person = tmp_path / "results" / "p1"
    where = {"logs": lambda uid: person / "nights"}
    world.plan("p1", NIGHT)
    world.plan("p1", SECOND)
    _one(world, "p1", NIGHT, **where)
    world.failing_gates["p1"] = {"collapse"}
    _one(world, "p1", SECOND, **where)
    _score(person / "scores", 1, NIGHT, 0.50)
    _score(person / "scores", 2, SECOND, 0.62)
    results = lambda uid: tmp_path / "results" / uid  # noqa: E731

    changes = what_changed("p1", NIGHT, results=results)
    kinds = [(c.night, c.kind) for c in changes]
    assert (NIGHT, "promoted") in kinds and (SECOND, "rejected") in kinds and (SECOND, "score") in kinds
    text = "\n".join(c.line for c in changes)
    assert "collapse" in text and "0.50 to 0.62" in text and "went up" in text
    assert "new reply trips" not in text  # a gate's failure reason can name a safety case: never shown
    assert all(isinstance(c, Change) for c in changes)
    assert {c.night for c in what_changed("p1", SECOND, results=results)} == {SECOND}
    assert what_changed("p1", THIRD, results=results) == []
    assert what_changed("nobody", NIGHT, results=results) == []


# ── A talker swap ────────────────────────────────────────────────────────────


def test_receipts_identical_across_a_talker_swap(tmp_path: Path) -> None:
    """Her receipts come from her state and the chooser, never from the talker: swap the model that
    words the note and the receipts are byte for byte the same (by construction, so a unit test)."""
    receipts = []
    for talker in ("glimmer", "spark open"):
        world = World(tmp_path / talker.replace(" ", "-"))
        world.plan("p1", NIGHT)
        world.targets("p1").upsert(thread("p1"))

        def words(user_id: str, topic: str, facts: Any, talker: str = talker) -> str:
            return f"{talker} words this note."

        world.writer.write = words  # type: ignore[method-assign]
        result = _one(world, "p1", NIGHT, homework=Homework(world))
        log = steps(result.log_path)
        receipts.append([(s["step"], s.get("receipt"), s.get("action")) for s in log if s["step"] in ("choose", "note")])
    assert receipts[0] == receipts[1] and receipts[0]


# ── Fence first, drain the writers, then list (RF2) ──────────────────────────


def test_a_step_that_writes_after_the_cancel_and_crashes_still_leaves_nothing(tmp_path: Path) -> None:
    """The night holds the lock; it sees the cancel, writes once more and dies before its purge. The
    delete waited for the lock, so what it wrote is listed and removed."""
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    world.plan("p1", NIGHT)
    held, late = threading.Event(), world.root / "p1" / "late.txt"

    def night() -> None:
        with drain(owned.lock_path, 5.0):
            held.set()
            while owned.flag.epoch("p1") == 0:
                threading.Event().wait(0.01)
            late.parent.mkdir(parents=True, exist_ok=True)
            late.write_text("written after the cancel")
            # ...and the process dies here, before any purge

    thread = threading.Thread(target=night)
    thread.start()
    assert held.wait(5)
    delete_all("p1", owned.ports())
    thread.join()
    assert not late.exists() and not (world.root / "p1").exists()


def test_an_adapter_registered_during_the_delete_is_still_removed(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    early, late = adapter("p1", 1), adapter("p1", 2)
    world.registry.record_trained("p1", early, NIGHT)
    _upload(owned, early)
    _upload(owned, late)
    held = threading.Event()

    def night() -> None:
        with drain(owned.lock_path, 5.0):
            held.set()
            while owned.flag.epoch("p1") == 0:
                threading.Event().wait(0.01)
            world.registry.record_trained("p1", late, SECOND)  # trained just as the delete began

    thread = threading.Thread(target=night)
    thread.start()
    assert held.wait(5)
    report = delete_all("p1", owned.ports())
    thread.join()
    assert report.adapters == 2 and owned.volume.folders == set() and owned.bucket.keys == set()


def test_a_night_that_never_leaves_fails_the_delete_after_one_retry(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    world.plan("p1", NIGHT)
    world.evidence("p1").add(world.planned[("p1", NIGHT)])
    tries: list[float] = []

    def lock(timeout: float) -> Any:
        tries.append(timeout)
        return drain(owned.lock_path, timeout)

    ports = replace(owned.ports(), lock_timeout_s=0.2, night_lock=lock)
    with drain(owned.lock_path, 1.0):  # a live night
        # flock is per open file, so a second open in this process is refused as another process would be
        with pytest.raises(DeleteIncomplete) as raised:
            delete_all("p1", ports)
    assert tries == [0.2, 0.2] and owned.stopped == ["p1"]  # the VM was stopped between the two tries
    assert "night lock: still held" in raised.value.report.failures
    assert (world.root / "p1").exists()  # nothing was deleted while a night could still write


def test_a_failed_remote_delete_is_kept_and_retried_even_after_the_registry_is_gone(tmp_path: Path) -> None:
    world = World(tmp_path)
    owned = Ports(world, tmp_path)
    name = adapter("p1", 1)
    world.registry.record_trained("p1", name, NIGHT)
    _upload(owned, name)

    class Down:
        name = "down"

        def remove(self, adapters: frozenset[str]) -> int:
            raise OSError("unreachable")

    good = owned.ports()
    with pytest.raises(DeleteIncomplete):
        delete_all("p1", replace(good, remotes=(Down(), *good.remotes)))
    assert owned.inventory.read("p1") == {name}  # kept: weights may remain remote
    assert not (world.root / "checkpoints" / "p1").exists()  # the registry is gone, the inventory is not

    assert delete_all("p1", good).adapters == 1  # a retry finds the name in the inventory
    assert owned.volume.folders == set() and owned.bucket.keys == set()
    assert owned.inventory.read("p1") == frozenset() and owned.inventory.pending() == []


def test_a_nights_cancel_baseline_is_read_before_any_state_loads(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    seen: list[str] = []

    def logs(uid: str) -> Path:
        seen.append("logs")
        return world.logs(uid)

    def cancel(uid: str) -> bool:
        seen.append("cancel")
        return False

    run_night_batch_with(world.deps(logs=logs, cancel=cancel), {"p1": NIGHT})
    assert seen[0] == "cancel"
    flag = EpochFlag(tmp_path / "flag")
    flag.set("p1")
    watching = flag.watcher()
    assert watching("p1") is False  # one read is both the baseline and the comparison
