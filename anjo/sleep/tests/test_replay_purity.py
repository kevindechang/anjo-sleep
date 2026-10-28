# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Replay's pure modules (Task 17b): ``types``, ``score`` and ``dataset`` read no clock, database,
model, file or environment, in the style of the inquiry core's purity test. Each imports only pure
stdlib and the pure modules it names; ``dataset``'s only randomness is ``random.Random`` drawn from
a seed it is given, and λ arrives as an argument (``replay.settings`` reads it)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPLAY = Path(__file__).parents[1] / "mind" / "replay"
STDLIB = frozenset({"__future__", "collections.abc", "dataclasses", "datetime", "re", "types", "typing"})
ALLOWED = {
    "types.py": STDLIB | {
        "anjo.sleep.provenance",  # SPARK_OPEN_IDS, the two providers' model ids, only
        "anjo.sleep.mind.apd.types", "anjo.sleep.mind.inquiry.types", "anjo.sleep.types",
    },
    "score.py": STDLIB | {"anjo.sleep.mind.replay.types"},
    "dataset.py": STDLIB | {
        "random",  # random.Random(seed) only
        "anjo.sleep.context.recall_common", "anjo.sleep.mind.apd.types", "anjo.sleep.mind.inquiry.types",
        "anjo.sleep.mind.replay.score", "anjo.sleep.mind.replay.types", "anjo.sleep.provenance",
        "anjo.sleep.train.dataset",
    },
}
IMPURE_CALLS = frozenset({
    "now", "utcnow", "today", "getenv", "environ", "open", "read_text", "connect", "urlopen", "time", "sleep",
    "uuid4", "post", "get_db", "token_penalty",
})


def _tree(module: str) -> ast.Module:
    return ast.parse((REPLAY / module).read_text(encoding="utf-8"))


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_replays_types_score_and_dataset_are_pure(module: str) -> None:
    tree = _tree(module)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names = [node.module or ""]
            if module == "types.py" and node.module == "anjo.sleep.provenance":
                assert [alias.name for alias in node.names] == ["SPARK_OPEN_IDS"], f"{module} uses provenance"
        else:
            names = []
        for name in names:
            assert name in ALLOWED[module], f"{module} imports {name}"
        called = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        assert called not in IMPURE_CALLS, f"{module} uses {called}"


def test_datasets_only_randomness_is_a_seeded_generator() -> None:
    for node in ast.walk(_tree("dataset.py")):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "random":
            assert node.attr == "Random", f"dataset.py uses random.{node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Random":
            assert node.args, "every generator is drawn from a seed"
