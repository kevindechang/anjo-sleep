# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Self-distillation (``ANJO_SLEEP_SDFT``; Shenfeld, Damani, Hübotter and Agrawal, arXiv 2601.19897): the
switch and its refusals, today's night byte for byte with it off, the rows built from the seeding rows'
messages (the teacher reads her day, the student the query alone), the overlap guard against probes,
held-out items, the gate, anchor and knowledge sets and her own requests, the rows' round trip into the
job, and the term itself: the seeding KL on the student's own sample, its SDFT-held (seeded, not
distilled) before-and-after reading and its budget. CPU only, on the objective tests' tiny torch model; torch tests skip without it."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from anjo.sleep.night.dreams import Terms
from anjo.sleep.night.meter import SpendMeter
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.sets.loader import SETS_DIR, load_anchors, load_collapse, load_general
from anjo.sleep.tests.night_fakes import World, own, step, window
from anjo.sleep.tests.test_sleep_objective import (
    AT,
    NIGHT,
    TODAYS_KEYS,
    _data,
    _NoLeak,
    _plan,
    _tiny,
    _Tokens,
    _torch,
    _transcript,
)
from anjo.sleep.train import colab_train, objective, sleep_job, sleep_loss, terms
from anjo.sleep.train.dataset import SeedRow
from anjo.sleep.train.lora import LoraCfg, LoraCfgError, training_config
from anjo.sleep.types import Evidence, Probe

SEED = {objective.OBJECTIVE_ENV: "sft_dpo_seed"}
ON = {**SEED, objective.SDFT_ENV: "1"}
TONE = "Short. Answer first."
PREFERENCE = "Wants the flaw named before the strength."
OWN = "Prefers sixteen millimetre film to digital for anything personal."  # in no set: only her request
# Trunk e5f21c86, before self-distillation: what today's code wrote and trained on the fixtures below.
TODAYS_JSONL_SHA = "32fd5ceb1bbc2e2e9a225845a2985e02658d55642c822046ca7ea6706143030c"
TODAYS_RECEIPT = {"anchors": 0, "seeds": 2, "seeds_dropped": {"missing": 0, "no_context": 0, "probe": 0,
                                                              "teacher_target": 0, "truncated": 0, "uncleared": 0}}
TODAYS_LOSS_KEYS = {"accumulation", "dpo", "dreams", "epochs", "grads_adapter_only", "imitation_reward",
                    "long_term_drift", "long_term_rank", "loss", "minutes", "on_policy_samples", "out_of_time", "rows",
                    "seed_kl_after", "seed_kl_before", "steps", "teacher_drift", "weights"}
TODAYS_LOSS = {"accumulation": 5, "epochs": 1, "imitation_reward": 0.0625,
               "loss": {"dpo": 0.673672080039978, "dream": None, "imitate": -0.022164523601531982,
                        "seed": 0.08597790747880936, "sft": 2.4939593076705933},
               "minutes": {"dpo": 0.0, "dream": 0.0, "imitate": 0.0, "seed": 0.0, "sft": 0.0},
               "on_policy_samples": 2, "rows": {"dpo": 1, "dream": 0, "imitate": 4, "seed": 5, "sft": 2},
               "seed_kl_after": 0.10197173804044724, "seed_kl_before": 0.105043925344944, "steps": 5,
               "teacher_drift": 0.0}
TODAYS_PARAMS = (-0.8799343407154083, -0.8013342637568712, 6.178973389789462)  # A's sum, B's sum, B's |sum|


def _said() -> list[Evidence]:
    """Today's lesson, tone request and preference: what the teacher reads and the student must hold."""
    tone = Evidence(id="e-tone", person_id="p1", day=NIGHT, kind="tone_request", text=TONE, source="chat:2",
                    held_out=False, created_at=AT, ancestry=frozenset())
    pref = Evidence(id="e-pref", person_id="p1", day=NIGHT, kind="preference", text=PREFERENCE, source="chat:3",
                    held_out=False, created_at=AT, ancestry=frozenset())
    return [own("p1", NIGHT, 0), tone, pref]


def _night(tmp_path: Path, on: bool, *, extra: list[Evidence] | None = None, sealed: list[Probe] | None = None,
           made: list[Any] | None = None) -> tuple[Any, dict[str, Any]]:
    made = made or [_transcript(0), _transcript(1)]
    world = World(tmp_path)
    world.evidence("p1").add([*_said(), *(extra or [])])
    world.transcripts("p1").add(made)
    deps = world.deps(terms=Terms(seed=True, sdft=on))
    data, receipt = terms.attach_night_terms(deps, "p1", NIGHT, world.evidence("p1"), sealed or [], _data(made),
                                             SpendMeter(spark_in_per_m=0.2, spark_out_per_m=0.6))
    return data, receipt


def _loss_batch(sdft: tuple[Any, ...] = ()) -> Any:
    return sleep_loss.Batch(
        sft=(sleep_loss.Seq((1, 2), (3, 11)), sleep_loss.Seq((1, 4), (5, 6, 11))),
        pairs=(sleep_loss.PairSeq((1, 5), (6, 11), (7, 11)),),
        seeds=tuple(sleep_loss.SeedSeq((1, i), (9, 9, 1, i), (4, 5, 11)) for i in range(2, 7)),
        sdft=sdft,
    )


# ── the switch ───────────────────────────────────────────────────────────────


def test_the_switch_off_is_todays_night_byte_for_byte(tmp_path: Path) -> None:
    for env, today in (({}, "sft_dpo"), ({objective.SDFT_ENV: "0"}, "sft_dpo"), (SEED, "sft_dpo_seed"),
                       ({**SEED, objective.SDFT_ENV: "0"}, "sft_dpo_seed"), ({**SEED, objective.SDFT_ENV: " "}, "sft_dpo_seed")):
        cfg = objective.night_cfg(env)
        assert cfg == LoraCfg(objective=today) and not cfg.sdft
        assert objective.night_terms(env) == Terms(seed=today == "sft_dpo_seed") and not objective.night_terms(env).sdft
    sent = training_config(objective.night_cfg(SEED), 12, warm=True, revision="a" * 40, night=NIGHT)
    assert set(sent) == TODAYS_KEYS | {"terms"} and sent["terms"] == {"fold_rank": None, "dreams": False}
    # The dataset the trainer is sent, and the dataset step's receipt: today's, byte for byte.
    data, receipt = _night(tmp_path, on=False)
    assert data.sdft == () and receipt == TODAYS_RECEIPT
    text = data.to_jsonl(pairs=True, terms=True)
    assert hashlib.sha256(text.encode()).hexdigest() == TODAYS_JSONL_SHA and '"sdft"' not in text


def test_the_switch_off_trains_todays_loss() -> None:
    torch = _torch()
    model = _tiny(torch)
    got = sleep_loss.train_terms(model, _loss_batch(), _plan(), timer=lambda: 0.0)
    assert set(got) == TODAYS_LOSS_KEYS and got["weights"] == sleep_loss.WEIGHTS
    assert sleep_loss.WEIGHTS == {"sft": 1.0, "dpo": 1.0, "seed": 1.0, "imitate": 0.25, "dream": 0.5}
    for key, value in TODAYS_LOSS.items():
        assert got[key] == pytest.approx(value, rel=1e-6, abs=1e-9), key
    params = (float(model.lora_A.detach().double().sum()), float(model.lora_B.detach().double().sum()),
              float(model.lora_B.detach().double().abs().sum()))
    assert params == pytest.approx(TODAYS_PARAMS, rel=1e-6)


def test_the_switch_rides_only_on_the_seed_objective_and_never_in_a_matched_control() -> None:
    cfg = objective.night_cfg(ON)
    assert cfg.sdft and cfg.objective == "sft_dpo_seed" and objective.night_terms(ON) == Terms(seed=True, sdft=True)
    with_all = objective.night_cfg({**ON, objective.DAY_DREAMS_ENV: "1", objective.CAPACITY_ENV: "1"})
    assert (with_all.sdft, with_all.day_dreams, with_all.passes) == (True, True, 3)
    assert objective.night_cfg({**ON, objective.FOLD_ENV: "1"}).fold_rank is not None
    sent = training_config(cfg, 12, warm=True, revision="a" * 40, night=NIGHT)
    assert sent["terms"] == {"fold_rank": None, "dreams": False, "sdft": True}
    assert colab_train.load_config(json.dumps(sent))["terms"]["sdft"] is True
    for bad in ({**sent, "terms": {**sent["terms"], "sdft": "yes"}}, {**sent, "match_minutes": 5.0}):
        with pytest.raises(ValueError):
            colab_train.load_config(json.dumps(bad))
    for env in ({objective.SDFT_ENV: "1"}, {objective.OBJECTIVE_ENV: "sft", objective.SDFT_ENV: "1"},
                {**SEED, objective.SDFT_ENV: "yes"}):
        with pytest.raises(objective.ObjectiveError):
            objective.night_cfg(env)
    for bad_cfg in ({"objective": "sft_dpo"}, {"objective": "sft_dpo_seed", "match_by_night": ((NIGHT, 5.0),)}):
        with pytest.raises(LoraCfgError):
            LoraCfg(sdft=True, **bad_cfg)
    assert terms.SDFT_KINDS == sleep_loss.SDFT_KINDS and terms.SDFT_REPLY_TOKENS == sleep_loss.SDFT_TOKENS
    assert sleep_job.SDFT_KIND == sleep_loss.SDFT_TERM == "sdft" and sleep_loss.SDFT_TERM not in sleep_loss.WEIGHTS


# ── the rows ─────────────────────────────────────────────────────────────────


def test_on_the_teacher_reads_her_day_and_the_student_the_query_alone(tmp_path: Path) -> None:
    data, receipt = _night(tmp_path, on=True)
    assert receipt["sdft"] == {"rows": {"knowledge": 2, "tone": 2, "taste": 2}, "sources": 2,
                               "dropped": dict.fromkeys(terms.SDFT_DROPS, 0)}
    assert {k: v for k, v in receipt.items() if k != "sdft"} == TODAYS_RECEIPT  # the seeding rows are today's
    by_source = {row.id: row for row in data.seeds}
    for row in data.sdft:
        source = by_source[row.source]
        assert row.context == source.context and TONE in row.context and PREFERENCE in row.context
        assert "they edit at night" in row.context  # the snapshot of C, then today's items: 2C's material
        assert row.query.startswith(source.message) and TONE not in row.query and PREFERENCE not in row.query
        frames = terms.SDFT_FRAMES[row.kind]
        assert row.query == terms.sdft_query(source.message, row.kind, source.id, 20261028)
        assert (row.query == source.message) == (row.kind == "knowledge") and any(row.query.endswith(f) for f in frames)
    written = [json.loads(line) for line in data.to_jsonl(pairs=True, terms=True).splitlines()]
    rows = [row for row in written if row.get("kind") == "sdft"]
    assert len(rows) == 6 and all("completion" not in row for row in rows)  # the job samples every reply
    for row in rows:
        assert [m["role"] for m in row["teacher"]] == ["system", "user"] and TONE in row["teacher"][0]["content"]
        assert [m["role"] for m in row["prompt"]] == ["user"] and row["teacher"][1] == row["prompt"][0]
        assert TONE not in json.dumps(row["prompt"]) and row["sdft_kind"] in terms.SDFT_KINDS
    # Today's split never reads them; the job reads them only when the recipe asks.
    assert colab_train.split_rows(written) == colab_train.split_rows([r for r in written if r.get("kind") != "sdft"])
    by_kind = sleep_job.term_rows(colab_train, written)
    assert len(by_kind["sdft"]) == 6 and len(by_kind["seed"]) == 2
    template = {"reasoning_strength": "low", "current_date": NIGHT.isoformat()}
    cfg = {"max_seq_length": 8192, "terms": {"fold_rank": None, "dreams": False, "sdft": True}}
    batch, too_long = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [], [], by_kind, cfg, template)
    assert len(batch.sdft) == 6 and too_long == {"seed": 0, "dream": 0, "sdft": 0}
    first = batch.sdft[0]
    assert first.prompt[:2] == (1, 8) and first.teacher[:2] == (1, 7) and first.teacher[-len(first.prompt) + 1:] == first.prompt[1:]
    assert {seq.kind for seq in batch.sdft} == set(sleep_loss.SDFT_KINDS) and first.state is None
    off, too_long = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [], [], by_kind,
                                         {**cfg, "terms": {"fold_rank": None, "dreams": False}}, template)
    assert off.sdft == () and too_long == {"seed": 0, "dream": 0}
    with pytest.raises(ValueError, match="kind and its source"):
        sleep_job.term_rows(colab_train, [{**rows[0], "source": None}])
    with pytest.raises(ValueError, match="cleaned"):
        sleep_job.term_rows(colab_train, [{**rows[0], "teacher": [{"role": "system", "content": "<|eot|>"}]}])


def test_sources_are_tonights_then_dreams_then_replay_each_message_once_and_capped() -> None:
    def row(i: str, message: str, group: str | None = None) -> SeedRow:
        return SeedRow(id=i, context="Notes: they edit at night.", message=message, reply="r", tokens=10, group=group)

    seeds = [row("old", "An old scene I shot?"), row("d0", "A dreamt question about the cut?", "dream"),
             row("t0", "Tonight's scene: the opening?"), row("t1", "Tonight's scene: the opening?")]
    made, dropped = terms.sdft_rows(seeds, frozenset({"old"}), terms.OverlapGuard([]), [], _NoLeak(), seed=3)
    assert [r.source for r in made[::3]] == ["t0", "d0", "old"] and dropped == dict.fromkeys(terms.SDFT_DROPS, 0)
    many = [row(f"t{i}", f"Scene {i} of the short: what lens?") for i in range(terms.SDFT_SOURCES_MAX + 3)]
    made, _ = terms.sdft_rows(many, frozenset(), terms.OverlapGuard([]), [], _NoLeak(), seed=3)
    assert len({r.source for r in made}) == terms.SDFT_SOURCES_MAX and len(made) == 3 * terms.SDFT_SOURCES_MAX
    short, dropped = terms.sdft_rows(seeds[:1], frozenset(), terms.OverlapGuard([]), [], _NoLeak(), seed=3,
                                     max_seq_tokens=terms.SDFT_REPLY_TOKENS)
    assert short == () and dropped["truncated"] == 3  # the sample's room counts against the length


# ── the guard ────────────────────────────────────────────────────────────────


PROBE = Probe("p1-new_work-01", "p1", "new_work", "I have a rough cut of the thesis short at nineteen minutes and the "
              "festival wants fifteen; which scene do I lose first?", None, None, ("h1",))
TASTE = Probe("p1-taste_ab-01", "p1", "taste_ab", "How is the demo cut?",
              ("A narrated four-minute walkthrough", "A ninety-second silent screen recording"), "x", ("h2",))
HELD = Evidence(id="h3", person_id="p1", day=NIGHT, kind="preference", held_out=True, created_at=AT,
                ancestry=frozenset(), source="chat:9",
                text="Submits the thesis short to three festivals he can name and never to thirty of them at once.")


def _guard() -> terms.OverlapGuard:
    asked = Evidence(id="e-own", person_id="p1", day=NIGHT, kind="preference", text=OWN, source="chat:4",
                     held_out=False, created_at=AT, ancestry=frozenset())
    return terms.sdft_guard([PROBE, TASTE], [HELD], [*_said(), asked])


def test_the_guard_refuses_probe_held_out_set_and_cue_text_exact_and_near() -> None:
    guard = _guard()
    clean = "Scene 4 of the short runs long and the editor wants a call by Friday. What lens for the reshoot?"
    assert guard.refusal(clean) is None
    assert all(guard.refusal(f"{clean}\n\n{frame}") is None for frames in terms.SDFT_FRAMES.values() for frame in frames)
    assert guard.refusal(f"Quick one. {PROBE.prompt} Thanks.") == "probe"  # exact
    near = PROBE.prompt.replace("nineteen", "twenty").replace("lose first", "cut first")
    assert near != PROBE.prompt and guard.refusal(near) == "probe"  # a near duplicate
    assert guard.refusal("For the demo: A narrated four-minute walkthrough, or not?") == "probe"  # a taste option
    assert guard.refusal("I want to submit the thesis short to three festivals he can name, sure.") == "held_out"  # cut from one
    assert guard.refusal(f"Context: {HELD.text}") == "held_out"
    assert guard.refusal(f"Odd one: {load_anchors()[0].question}") == "gate_set"
    assert guard.refusal(load_general()[0].question) == "gate_set"
    assert guard.refusal(load_collapse()[0].prompt) == "gate_set"
    knowledge = next(SETS_DIR.glob("knowledge_*.jsonl"))
    line = json.loads(knowledge.read_text(encoding="utf-8").splitlines()[0])["choices"][0]
    assert guard.refusal(f"My notes say: {line}") == "gate_set"  # the frozen knowledge set too
    assert guard.refusal(f"Remember: {TONE} So, the lens?") == "cue"  # her own request would teach the cue
    assert guard.refusal("She prefers sixteen millimetre film to digital for anything personal, so which?") == "cue"
    assert guard.refusal(f"{PREFERENCE} What's weak here?") is not None  # a persona's own line is a set item too


def test_a_row_carrying_a_probe_or_held_out_text_is_refused_and_counted(tmp_path: Path) -> None:
    leaky = [replace(_transcript(0), message=f"Before the edit: {HELD.text}"), _transcript(1),
             replace(_transcript(2), message=f"Help: {PROBE.prompt}")]
    data, receipt = _night(tmp_path, on=True, extra=[HELD], sealed=[PROBE], made=leaky)
    # Both seed (these rows skip the dataset's own decontamination); neither message ever becomes a query.
    assert {row.source for row in data.sdft} == {"t1"} and {"t0", "t2"} <= {row.id for row in data.seeds}
    assert receipt["sdft"]["dropped"]["held_out"] == 3 and receipt["sdft"]["dropped"]["probe"] == 3
    assert receipt["sdft"]["rows"] == dict.fromkeys(terms.SDFT_KINDS, 1)
    assert not any(HELD.text in row.query or PROBE.prompt in row.query for row in data.sdft)

    class Flags:
        def check(self, texts: Any, sealed: Any) -> tuple[bool, ...]:
            return tuple("lens" in text for text in texts)

    seeds = [SeedRow(id="t9", context="Notes.", message="Which lens for scene 9?", reply="r", tokens=5)]
    made, dropped = terms.sdft_rows(seeds, frozenset(), terms.OverlapGuard([]), [PROBE], Flags(), seed=1)
    assert made == () and dropped["probe"] == 3  # the decontaminator's three rules run on every query too


def test_the_night_logs_the_rows_in_its_dataset_receipt(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    deps = replace(world.deps(), terms=Terms(seed=True, sdft=True))
    result = run_night_batch_with(deps, {"p1": NIGHT}, windows={"p1": window(NIGHT)})
    receipt = step(result["p1"].log_path, "dataset")["terms"]
    assert set(receipt["sdft"]) == {"rows", "sources", "dropped"} and set(receipt["sdft"]["rows"]) == set(terms.SDFT_KINDS)
    assert sum(receipt["sdft"]["rows"].values()) + sum(receipt["sdft"]["dropped"].values()) == 3 * receipt["seeds"]


# ── the term ─────────────────────────────────────────────────────────────────


def test_the_term_is_reverse_kl_to_the_teacher_on_the_students_own_sample() -> None:
    torch = _torch()
    model = _tiny(torch)
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    start = model.lora_B.detach().clone()
    with torch.no_grad():
        model.lora_B.add_(0.5)  # tonight's training so far
    tonight = model.lora_B.detach().clone()
    model.train()
    row = sleep_loss.SdftSeq(prompt=(1, 2), teacher=(7, 8, 9, 1, 2), kind="tone", source="t0")
    torch.manual_seed(5)
    kl, reply = sleep_loss.sdft_kl(model, row, teacher, _plan())
    assert reply and len(reply) <= sleep_loss.SDFT_TOKENS and (reply[-1] == 11 or len(reply) == sleep_loss.SDFT_TOKENS)
    sampled = model.calls[:-2]
    assert sampled and all(not grad and not train and on for _, on, grad, train, _ in sampled)  # her adapter, no context
    assert sampled[0][0] == row.prompt and torch.equal(sampled[0][4], tonight)  # the student samples, tonight's adapter
    (t_ids, _, t_grad, t_train, t_b), (s_ids, _, s_grad, s_train, s_b) = model.calls[-2:]
    assert (t_ids, t_grad, t_train) == ((*row.teacher, *reply), False, False) and torch.equal(t_b, start)
    assert (s_ids, s_grad, s_train) == ((*row.prompt, *reply), True, True) and torch.equal(s_b, tonight)
    kl.backward()
    assert model.lora_B.grad is not None and model.head.weight.grad is None
    n, t_cut = len(reply), len(row.teacher)
    with torch.no_grad():
        model.lora_B.copy_(start)
        t = torch.log_softmax(model(torch.tensor([[*row.teacher, *reply]])).logits[0, t_cut - 1:t_cut - 1 + n], -1)
        model.lora_B.copy_(tonight)
        s = torch.log_softmax(model(torch.tensor([[*row.prompt, *reply]])).logits[0, 1:1 + n], -1)
    assert torch.allclose(kl.detach(), (s.exp() * (s - t)).sum(-1).mean(), atol=1e-6) and float(kl.detach()) > 0


def _sdft_rows(sources: int) -> tuple[Any, ...]:
    return tuple(sleep_loss.SdftSeq((1, 2 + i, k), (9, 8, 1, 2 + i, k), kind, f"s{i}")
                 for i in range(sources) for k, kind in enumerate(sleep_loss.SDFT_KINDS))


def test_the_job_trains_sdft_rows_on_fresh_samples_and_reads_the_sdft_held_kl(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    model = _tiny(torch)
    rows = _sdft_rows(5)
    prompts = {row.prompt for row in rows}
    drawn: list[tuple[int, ...]] = []
    real = sleep_loss.sample

    def recording(model: Any, prompt: Any, max_new: int, plan: Any, soft: Any = None) -> Any:
        if tuple(prompt) in prompts:
            assert max_new == sleep_loss.SDFT_TOKENS
            drawn.append(tuple(prompt))
        return real(model, prompt, max_new, plan, soft)

    monkeypatch.setattr(sleep_loss, "sample", recording)
    got = sleep_loss.train_terms(model, _loss_batch(rows), _plan(), timer=lambda: 0.0)
    sdft = got["sdft"]
    held, trained = sleep_loss.sdft_split(rows, _plan().seed)
    assert len(held) == 3 * sleep_loss.SDFT_HELD and len({rows[i].source for i in held}) == sleep_loss.SDFT_HELD
    assert not {rows[i].source for i in held} & {rows[i].source for i in trained}  # a held message is never distilled
    reading = sdft["seeded_not_distilled"]
    assert reading["label"] == sleep_loss.SDFT_HELD_LABEL == "SDFT-held (seeded, not distilled)"
    assert sdft["rows"] == dict.fromkeys(sleep_loss.SDFT_KINDS, 3) and reading["rows"] == dict.fromkeys(sleep_loss.SDFT_KINDS, 2)
    assert "held_out" not in json.dumps(sdft) and "held-out" not in json.dumps(sdft)  # no held-out claim anywhere
    visits = 9 * got["epochs"]
    assert sdft["on_policy_samples"] == visits and reading["samples"] == 6 and sdft["skipped_late"] == 0
    assert len(drawn) == visits + 6  # every visit drew its own sample, and each held row one
    assert set(drawn[:6]) == {rows[i].prompt for i in held}  # the held slice first, at the night's start
    assert math.isfinite(reading["kl_before"]) and reading["kl_after"] != reading["kl_before"]
    assert set(reading["kl_by_kind"]) == set(sleep_loss.SDFT_KINDS) and sdft["weight"] == sleep_loss.SDFT_WEIGHT
    assert got["weights"] == sleep_loss.WEIGHTS and got["loss"]["sdft"] is not None and "sdft" in got["minutes"]
    assert got["teacher_drift"] == 0.0 and got["grads_adapter_only"]
    assert set(got) == TODAYS_LOSS_KEYS | {"sdft"}
    few = sleep_loss.train_terms(_tiny(torch), _loss_batch(_sdft_rows(2)), _plan(), timer=lambda: 0.0)["sdft"]
    assert few["seeded_not_distilled"]["samples"] == 0 and few["seeded_not_distilled"]["kl_before"] is None
    assert few["rows"] == dict.fromkeys(sleep_loss.SDFT_KINDS, 2)


def test_the_held_sources_seed_rows_still_train_under_the_seed_term() -> None:
    torch = _torch()
    seeds = tuple(sleep_loss.SeedSeq((1, 2 + i), (9, 8, 1, 2 + i), (4, 5, 11)) for i in range(5))
    rows = tuple(sleep_loss.SdftSeq(seq.prompt, seq.teacher, "knowledge", f"s{i}") for i, seq in enumerate(seeds))
    batch = replace(_loss_batch(rows), seeds=seeds)
    held, _ = sleep_loss.sdft_split(rows, _plan().seed)
    got = sleep_loss.train_terms(_tiny(torch), batch, _plan(), timer=lambda: 0.0)
    assert len(held) == sleep_loss.SDFT_HELD and got["rows"]["seed"] == len(seeds)  # held from distillation only
    assert got["sdft"]["seeded_not_distilled"]["rows"]["knowledge"] == sleep_loss.SDFT_HELD


def test_past_its_budget_the_term_draws_nothing_and_a_matched_control_refuses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _torch()
    monkeypatch.setattr(sleep_loss, "SDFT_BUDGET_S", -1.0)
    got = sleep_loss.train_terms(_tiny(torch), _loss_batch(_sdft_rows(5)), _plan(), timer=lambda: 0.0)
    assert got["sdft"]["on_policy_samples"] == 0 and got["sdft"]["skipped_late"] == 9 * got["epochs"]
    assert got["loss"]["sdft"] is None and got["sdft"]["seeded_not_distilled"]["samples"] == 6  # still read
    with pytest.raises(ValueError, match="matched control"):
        sleep_loss.train_terms(_tiny(torch), _loss_batch(_sdft_rows(1)), _plan(stretch_s=60.0))
    with pytest.raises(ValueError, match="ask for"):
        sleep_loss.train_terms(_tiny(torch), _loss_batch((replace(_sdft_rows(1)[0], kind="style"),)), _plan())
