# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The bar a contradiction method must clear (``contra_bar``) and the toy it is measured on: the
toy's conflict is a stance flip on the same attribute, which its oracle reads from the latents,
while timing and the untrained "matches no earlier event" score stay near chance. Then the pair
scorer and the ``contra`` command, on toy lives and on an export of them (CPU, offline)."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from anjo.sleep.mind.core import contra_bar, toy
from anjo.sleep.mind.core.types import LabelledStream

CHANCE = 0.15  # how far from 0.5 an uninformative score may land on ~130 to 256 candidates (about 3 sigma)


def _bar(lives: tuple[LabelledStream, ...], parts: dict[str, toy.Parts]) -> dict[str, float]:
    cands = contra_bar.candidates(lives, contra_bar.in_toy)
    oracle = {c.label_id: toy.oracle(c.knowable, c.own, parts[c.label_id]) for c in cands}
    return contra_bar.bar(cands, {"oracle": oracle})["auc"]  # type: ignore[return-value]


def test_on_the_toy_the_oracle_reads_the_conflict_and_timing_and_the_bar_sit_near_chance() -> None:
    standard = _bar(toy.lives(256, seed=7, stance=True), toy.oracle_parts(256, seed=7))
    revised = _bar(toy.revision_lives(256, seed=7, stance=True), toy.oracle_parts(256, seed=7, revisions=True))
    for aucs in (standard, revised):
        assert aucs["oracle"] >= 0.95  # the latents carry it
        assert abs(aucs["gap_first"] - 0.5) < CHANCE and abs(aucs["gap_latest"] - 0.5) < CHANCE  # timing doesn't
        assert abs(aucs["unmatched"] - 0.5) < CHANCE  # nor does resemblance to what came before
    assert standard["match_age"] < 0.5 + CHANCE  # a stale assumption resembles an older version only in revisions


def test_the_two_families_differ_only_in_the_contradiction_and_plain_is_the_default() -> None:
    contradiction = ("-plan", "-revision", "-constraint", "-raised")
    for index in range(8):
        plain, stance = toy.life(7, index), toy.life(7, index, stance=True)
        rest = [[(e.event_id, e.vec.tobytes()) for e in life.events if not e.event_id.endswith(contradiction)
                 or "-bg" in e.event_id] for life in (plain, stance)]
        assert rest[0] == rest[1]  # every draw before the contradiction is the same (its times may move)
    plain_lives = toy.lives(64, seed=7)
    assert [lab.id for life in plain_lives for lab in life.labels] == [
        lab.id for life in toy.lives(64, seed=7, stance=True) for lab in life.labels]
    # Plain: the constraint negates the plan's attribute and its decoy brings one of its own, so
    # resemblance tells it; a stale assumption resembles the first plan.
    standard = contra_bar.bar(contra_bar.candidates(toy.lives(256, seed=7), contra_bar.in_toy))["auc"]
    revised = contra_bar.bar(contra_bar.candidates(toy.revision_lives(256, seed=7), contra_bar.in_toy))["auc"]
    assert standard["unmatched"] > 0.6 and revised["match_age"] > 0.9  # type: ignore[index]


def test_a_pairs_conflict_and_its_twin_differ_in_the_oracles_reading() -> None:
    parts = {k: v for i in range(16) for k, v in toy._draft(7, i, toy.DIM, clash=True, stance=True).parts.items()}
    for pair in toy.contradiction_pairs(16, seed=7, stance=True):
        readings = []
        for life in (pair.problem, pair.benign):
            cand = next(c for c in contra_bar.candidates((life,), contra_bar.in_toy) if c.label_id == pair.label)
            readings.append(toy.oracle(cand.knowable, cand.own, parts[pair.label]))
        assert readings[0] > 0.75 and readings[1] < 0.25  # a sign flip, or none


def test_a_candidate_is_read_at_its_constraint_against_everything_before_it() -> None:
    (life,) = toy.revision_lives(1, seed=7, stance=True)
    (cand,) = contra_bar.candidates((life,), contra_bar.in_toy)
    label = next(lab for lab in life.labels if lab.id == cand.label_id)
    assert cand.knowable.event_id.endswith("-constraint") and cand.knowable.t == label.knowable_at
    assert [e.event_id.rsplit("-", 1)[-1] for e in cand.own] == ["plan", "revision"]
    assert cand.earlier == life.events[:life.events.index(cand.knowable)]
    assert cand.failed == label.failed
    assert contra_bar.candidates(toy.lives(4, seed=7, stance=True), lambda lab: False) == ()


def test_a_margin_over_the_bar_is_a_paired_interval_of_the_auc_difference() -> None:
    cands = contra_bar.candidates(toy.revision_lives(64, seed=7, stance=True), contra_bar.in_toy)
    truth = {c.label_id: float(c.failed) for c in cands}
    line = contra_bar.scored(contra_bar.unmatched, cands)
    low, high = contra_bar.margin(truth, line, cands, draws=200) or (0.0, 0.0)
    assert 0 < low <= high  # a perfect score beats a near-chance bar on every resample
    assert contra_bar.margin(line, line, cands, draws=50) == (0.0, 0.0)  # the bar never beats itself
    assert contra_bar.margin(truth, line, [c for c in cands if c.failed]) is None


def test_auc_counts_ties_half_a_missing_score_as_zero_and_needs_both_sides() -> None:
    (life,) = toy.revision_lives(1, seed=7, stance=True)
    (cand,) = contra_bar.candidates((life,), contra_bar.in_toy)
    twin = type(cand)(cand.life_id, "other", not cand.failed, cand.knowable, cand.own, cand.earlier)
    pos, neg = (cand, twin) if cand.failed else (twin, cand)
    assert contra_bar.auc({pos.label_id: 1.0, neg.label_id: 0.0}, (pos, neg)) == 1.0
    assert contra_bar.auc({pos.label_id: 1.0, neg.label_id: 1.0}, (pos, neg)) == 0.5
    assert contra_bar.auc({neg.label_id: 0.5}, (pos, neg)) == 0.0  # the conflict has no score: 0.0
    assert contra_bar.auc({}, (pos,)) is None


def test_the_untrained_scores_read_resemblance_and_hours() -> None:
    (life,) = toy.revision_lives(1, seed=7, stance=True)
    (cand,) = contra_bar.candidates((life,), contra_bar.in_toy)
    rows = np.stack([e.vec for e in cand.earlier]).astype(np.float64)
    y = cand.knowable.vec.astype(np.float64)
    best = int(np.argmax(rows @ y / (np.linalg.norm(rows, axis=1) * np.linalg.norm(y))))
    assert contra_bar.unmatched(cand) == pytest.approx(1 - float(rows[best] @ y / np.linalg.norm(rows[best]) / np.linalg.norm(y)))
    assert contra_bar.gap_first(cand) == (cand.knowable.t - cand.own[0].t) / timedelta(hours=1)
    assert contra_bar.gap_latest(cand) == (cand.knowable.t - cand.own[-1].t) / timedelta(hours=1) > 0
    assert contra_bar.match_age(cand) == (cand.knowable.t - cand.earlier[best].t) / timedelta(hours=1)


# ── The pair scorer and the command (torch) ───────────────────────────────────


def test_the_newest_matching_source_carries_the_comparison_and_padding_changes_nothing() -> None:
    torch = pytest.importorskip("torch")
    from anjo.sleep.mind.core import pair_scorer

    torch.manual_seed(0)
    scorer = pair_scorer.PairScorer(toy.DIM, 4)
    cands = contra_bar.candidates(toy.revision_lives(6, seed=7, stance=True), contra_bar.in_toy)
    one_by_one = {c.label_id: pair_scorer.Fitted(scorer, 0, 0.0, 0).scores([c])[c.label_id] for c in cands}
    together = pair_scorer.Fitted(scorer, 0, 0.0, 0).scores(cands)  # padded to the longest history
    assert together == pytest.approx(one_by_one, abs=1e-5)

    sources = torch.randn(1, 3, toy.DIM)
    with torch.no_grad():
        scorer.match_head.bias.fill_(50.0)  # every source matches: only the newest counts
        newest = scorer(sources, torch.ones(1, 3), sources[:, -1])
        alone = scorer(sources[:, -1:], torch.ones(1, 1), sources[:, -1])
    assert float(newest) == pytest.approx(float(alone), abs=1e-4)


def test_the_pair_scorer_keeps_its_best_validation_epoch() -> None:
    pytest.importorskip("torch")
    from anjo.sleep.mind.core import pair_scorer

    val = contra_bar.candidates(toy.revision_lives(16, seed=8, stance=True), contra_bar.in_toy)

    def source(epoch: int) -> tuple[contra_bar.Candidate, ...]:
        return contra_bar.candidates(toy.revision_lives(32, seed=900 + epoch, stance=True), contra_bar.in_toy)

    fitted = pair_scorer.fit(source, val, d=toy.DIM, rank=4, seed=0, epochs=4, patience=10)
    again = pair_scorer.fit(source, val, d=toy.DIM, rank=4, seed=0, epochs=4, patience=10)
    assert fitted.epochs == 4 and 0 <= fitted.best_epoch < 4
    assert fitted.val_loss == pytest.approx(pair_scorer._loss(fitted.scorer, val, 32))
    assert fitted.scores(val) == pytest.approx(again.scores(val))  # the seed fixes the run
    assert set(fitted.scores(val)) == {c.label_id for c in val}
    with pytest.raises(ValueError, match="validation"):
        pair_scorer.fit(source, (), d=toy.DIM, rank=4, seed=0, epochs=1, patience=1)


def test_contra_prints_its_plan_first_and_plan_stops_there(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    from anjo.sleep.mind.core import cli

    out = tmp_path / "contra.json"
    assert cli.main(["contra", "--out", str(out), "--plan"]) == 0
    printed = capsys.readouterr()
    plan = json.loads(printed.err)["plan"]
    assert plan["splits"]["val"]["rev"]["lives"] == 256 and plan["splits"]["test"]["std"]["candidates"] > 0
    assert "unmatched" in plan["counts_only_if"] and not out.exists()
    assert cli.main(["contra", "--out", str(out), "--core", "bad name=x"]) == 2


def test_contra_runs_unchanged_on_an_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    from anjo.sleep.mind.core import cli
    from anjo.sleep.tests.test_mind_core_cli import _export

    _export(tmp_path / "export", toy.lives(48, seed=21))
    export, out, model = tmp_path / "export", tmp_path / "contra.json", tmp_path / "core"
    assert cli.main(["train", "--export", str(export), "--latents", str(export / "latents"), "--out", str(model),
                     "--epochs", "1", "--width", "16", "--slots", "4", "--val-share", "0.4"]) == 0
    assert cli.main(["contra", "--out", str(out), "--export", str(export), "--latents", str(export / "latents"),
                     "--val-share", "0.4", "--seeds", "0", "1", "--epochs", "2", "--rank", "4",
                     "--core", f"plain={model / cli.MODEL_FILE}"]) == 0
    capsys.readouterr()
    result = json.loads(out.read_text())
    assert result["plan"]["latent_dim"] == toy.DIM and set(result["bar"]["test"]) == {"lives"}
    assert "oracle" not in result["bar"]["val"]["lives"]["auc"]  # real latents have no generating parts
    assert result["methods"]["pair_scorer"]["chosen"] in {"0", "1"} and set(result["methods"]) == {"pair_scorer", "plain"}
    assert set(result["methods"]["plain"]["beats_bar"]) == {"lives"}
