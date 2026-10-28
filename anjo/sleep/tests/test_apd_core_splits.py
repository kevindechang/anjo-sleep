# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""MIND core's splits (MIND's data): core-train, core-val and core-test, held out from each other
by field and template, with problem-free lives in each; the pinned b12 subset of core-test that B
runs on; and the CLI that generates, runs and judges them. Every model is a fake."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from anjo.sleep.mind.apd import __main__ as cli
from anjo.sleep.mind.apd import backend as backends
from anjo.sleep.mind.apd import core_families, core_splits, families, gen_batch, plan
from anjo.sleep.mind.apd.store import DataDir
from anjo.sleep.mind.apd.types import CORE_SPLITS, FAMILIES, MAX_PREVALENCE, CoreSplit
from anjo.sleep.mind.spark_provider import FIREWORKS

# Numbers a core resolution may state that are sums or products of its facts' numbers.
DERIVED = {
    "truck-generator-load": {"3,300"}, "furniture-clamp-time": {"9"}, "pottery-kiln-load": {"25"},
    "bikes-race-service": {"24"}, "photo-wall-length": {"36"}, "theatre-wall-height": {"3"},
}
CONCLUSION_WORDS = ("problem", "conflict", "mismatch", "mistake", "fixed by", "fix ", "solution", "diagnos")


def _sentences(templates: tuple[families.Template, ...]) -> set[str]:
    return {s for t in templates for s in (*t.origins, t.disambiguating, t.surface, t.resolution)}


def _templates(split: CoreSplit) -> tuple[families.Template, ...]:
    return tuple(t for name in core_families.fields(split) for t in core_families.templates_for(split, name))


def test_splits_hold_out_templates() -> None:
    # Arrange
    by_split = {split: _templates(split) for split in CORE_SPLITS}
    apd_held_out = families.TEMPLATES["held_out"]

    # Assert: no field, template or sentence is shared between the core's splits
    for a, b in ((0, 1), (0, 2), (1, 2)):
        one, two = by_split[CORE_SPLITS[a]], by_split[CORE_SPLITS[b]]
        assert {t.id for t in one}.isdisjoint({t.id for t in two})
        assert set(core_families.fields(CORE_SPLITS[a])).isdisjoint(core_families.fields(CORE_SPLITS[b]))
        assert _sentences(one).isdisjoint(_sentences(two))
    # ... nor with APD's frozen held-out test
    for split, templates in by_split.items():
        assert {t.id for t in templates}.isdisjoint({t.id for t in apd_held_out}), split
        assert set(core_families.fields(split)).isdisjoint(families.fields("held_out")), split
        for name in core_families.fields(split):  # every field can fill a life, with every family
            found = core_families.templates_for(split, name)
            assert len(found) == MAX_PREVALENCE and {t.family for t in found} == set(FAMILIES)
    # core-train keeps APD's development templates unchanged, and they are its only borrowed ones
    assert set(families.TEMPLATES["development"]) <= set(by_split["core-train"])


def test_core_split_sizes_and_problem_free_lives() -> None:
    sizes: dict[CoreSplit, tuple[int, int]] = {"core-train": (120, 15), "core-val": (20, 3), "core-test": (40, 6)}
    for split, (lives, empty) in sizes.items():
        specs = core_splits.core_specs(split)
        assert len(specs) == lives and sum(1 for s in specs if s.prevalence == 0) == empty
        assert {s.split for s in specs} == {split} and len({s.seed for s in specs}) == lives
        assert {s.field for s in specs} == set(core_families.fields(split))
        assert all(0 <= s.prevalence <= MAX_PREVALENCE for s in specs)


def test_core_specs_are_fixed() -> None:
    # The splits were drawn before any of their lives was written: a change to the draw fails here
    digest = hashlib.sha256(json.dumps([
        [s.split, s.field, s.prevalence, s.seed] for split in CORE_SPLITS for s in core_splits.core_specs(split)
    ]).encode()).hexdigest()
    assert digest[:16] == "8fe4e3a203e408b0"


def test_core_templates_follow_the_family_rules() -> None:
    for t in core_families.CORE_TEMPLATES:
        for fact in (*t.origins, t.disambiguating):
            assert not any(w in fact.lower() for w in CONCLUSION_WORDS), (t.id, fact)
            assert t.resolution not in fact
        facts = " ".join((*t.origins, t.disambiguating)).replace(",", "")
        for number in re.findall(r"\d+(?:[.,]\d+)*", t.resolution):
            assert number.replace(",", "") in facts or number in DERIVED.get(t.id, set()), (t.id, number)
        assert t.topic not in (t.id, t.resolution)


def test_b12_is_pinned_stratified_and_holds_problem_free_lives() -> None:
    # Arrange
    test = core_splits.core_specs("core-test")

    # Act
    b12 = core_splits.subset_specs("b12")
    drawn = core_splits.draw_subset(test, core_splits.B12_SIZE, core_splits.B12_EMPTY, "core-test:b12")

    # Assert: the pinned seeds are the draw's, chosen by seed alone
    assert tuple(sorted(s.seed for s in b12)) == core_splits.B12_SEEDS == tuple(sorted(s.seed for s in drawn))
    assert len(b12) == 12 and set(b12) <= set(test)
    found = Counter(s.prevalence for s in b12)
    assert found[0] >= 2  # problem-free lives
    whole = Counter(s.prevalence for s in test if s.prevalence)
    for prevalence, count in whole.items():  # the rest in proportion to the split, within one
        assert abs(found[prevalence] - 10 * count / sum(whole.values())) < 1, prevalence


def test_a_subset_names_lives_of_its_own_split_only() -> None:
    with pytest.raises(ValueError):
        core_splits.subset_specs("b13")
    with pytest.raises(SystemExit):
        cli.main(["--fake", "run", "--split", "core-val", "--subset", "b12"])
    with pytest.raises(SystemExit):
        cli.main(["--fake", "run", "--held-out", "--split", "core-test"])


def test_a_core_life_draws_only_its_splits_templates() -> None:
    backend = backends.fake_backend(FIREWORKS)
    for split in CORE_SPLITS:
        spec = next(s for s in core_splits.core_specs(split) if s.prevalence == MAX_PREVALENCE)
        made = gen_batch.generate_life(spec, backend)
        assert made.life.split == split and made.life.id in gen_batch.spec_ids(spec)
        assert {item.template for item in made.problems.items} <= {t.id for t in _templates(split)}
        assert len(made.problems.items) == MAX_PREVALENCE


def test_generate_run_and_judge_b12_on_fakes(tmp_path: Path) -> None:
    # Arrange: core-test generated
    data = tmp_path / "pilot"
    lines: list[str] = []
    assert cli.main(["--fake", "--data", str(data), "generate", "--split", "core-test"], out=lines.append) == 0
    assert len(DataDir(data).lives("core-test")) == 40

    # Act: B alone on b12, then judged
    assert cli.main(["--fake", "--data", str(data), "run", "--split", "core-test", "--subset", "b12"],
                    out=lines.append) == 0
    assert cli.main(["--fake", "--data", str(data), "judge", "--split", "core-test", "--subset", "b12"],
                    out=lines.append) == 0

    # Assert: twelve B runs, on the subset's lives, and nothing else
    runs = sorted(path.name for path in (data / "runs").iterdir())
    b12 = {life.id for life in core_splits.in_subset(DataDir(data).lives("core-test"), "b12")}
    assert len(runs) == 12 and {name.removesuffix(".B.json") for name in runs} == b12
    assert any(line.startswith("judged 12 run(s)") for line in lines)


def test_the_core_generate_plan_holds_each_life_at_its_own_prevalence(tmp_path: Path) -> None:
    lines: list[str] = []
    live = ["--live", "--prices", "1.25,4.25,1.5,2.61", "--judge-prices", "1.25,4.25,0.75,3.75"]
    assert cli.main([*live, "--data", str(tmp_path), "generate", "--split", "core-train"], out=lines.append) == 0
    text = "\n".join(lines)
    expected = sum(plan.expected_usd(plan.expected_calls(plan.surface_days(s))) for s in core_splits.core_specs("core-train"))
    assert "generate 120 lives of core-train" in text and f"Expected: about ${expected:,.2f} for 120 lives" in text
