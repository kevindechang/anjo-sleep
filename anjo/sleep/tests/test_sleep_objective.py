# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""The sleep objective (Behrouz, Hashemi and Mirrokni, "Language Models Need Sleep", 2026): knowledge
seeding against the base with the consolidated day in context, the imitation term, the nightly fold of
the adapter, and dreams filtered by gradient norm and an anchor check. Each is behind a switch, and with
every switch off the night is today's. CPU only, on a tiny torch model; tests that need torch skip
without it."""
from __future__ import annotations

import json
import math
import random
import tarfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anjo.sleep.model_api import ChatResult
from anjo.sleep.night.dreams import (
    HYPOTHETICAL,
    SPEAK,
    TAKEN,
    WAIT,
    Dream,
    DreamSeed,
    GlimmerDreams,
    NoDreams,
    Terms,
    seeds_from_dream,
    state_text,
)
from anjo.sleep.night.run import run_night_batch_with
from anjo.sleep.practice.transcripts import PracticeTranscript
from anjo.sleep.provenance import GLIMMER
from anjo.sleep.serving import config
from anjo.sleep.tests.night_fakes import World, step, window
from anjo.sleep.train import colab_train, objective, sleep_fold, sleep_job, sleep_loss, terms
from anjo.sleep.train.dataset import Dataset, Example
from anjo.sleep.train.lora import LoraCfg, LoraCfgError, StepPlan, plan_steps, training_config
from anjo.sleep.types import Evidence, Probe

NIGHT = date(2026, 10, 29)
AT = datetime(2026, 10, 29, 20, 0, tzinfo=timezone.utc)
GEMINI = "gemini-3.8-flash"  # an uncleared model: its text never trains, in context or target
# The recipe's keys on 2026-10-29, before this task: an off night sends exactly these.
TODAYS_KEYS = {
    "r", "alpha", "dropout", "target_modules", "learning_rate", "seed", "max_seq_length", "packing",
    "accumulation", "epochs", "batch_size", "warm_start", "revision", "objective", "dpo_beta",
    "dpo_learning_rate", "min_optimizer_steps", "reasoning_strength", "current_date", "template_sha256",
    "rendering",
}


def _torch() -> Any:
    return pytest.importorskip("torch")


# ── the host's rows ──────────────────────────────────────────────────────────


class Transcripts:
    def __init__(self, items: Sequence[PracticeTranscript]) -> None:
        self.items = {item.id: item for item in items}

    def add(self, items: Sequence[PracticeTranscript]) -> None:
        self.items.update({item.id: item for item in items})

    def get(self, item_id: str) -> PracticeTranscript | None:
        return self.items.get(item_id)

    def for_night(self, night: date) -> list[PracticeTranscript]:
        return [item for item in self.items.values() if item.night == night]


def _transcript(i: int, *, target: str = "glimmer") -> PracticeTranscript:
    return PracticeTranscript(
        id=f"t{i}", person_id="p1", night=NIGHT, scene=i, message=f"What should I shoot scene {i} on?",
        reply=f"Your 16mm Bolex for scene {i}, as you planned.", context="Notes: they edit at night.",
        f_ids=(), ancestry=frozenset({GLIMMER}), created_at=AT, target=target,  # type: ignore[arg-type]
    )


def _data(transcripts: Sequence[PracticeTranscript]) -> Dataset:
    examples = tuple(
        Example(id=t.id, prompt=(("user", t.message),), completion=t.reply, context_dropped=True, tokens=40,
                new=True, teacher=t.target == "teacher")
        for t in transcripts
    )
    return Dataset(examples=examples, tokens=40 * len(examples), dropped={})


def _today(ancestry: frozenset[str] | None) -> Evidence:
    return Evidence(id="e1", person_id="p1", day=NIGHT, kind="preference", text="They switched the short to 16mm today.",
                    source="chat:1", held_out=False, created_at=AT, model=None, ancestry=ancestry)


def test_uncleared_context_rows_are_excluded_from_seeding() -> None:
    made = [_transcript(0), _transcript(1)]
    store = Transcripts(made)
    rows, dropped = terms.seed_rows(_data(made), store, [_today(frozenset({GEMINI}))], [], hosted_in_training=False)
    assert rows == () and dropped["uncleared"] == 2  # the day's item came from Gemini: no row trains on it
    rows, dropped = terms.seed_rows(_data(made), store, [_today(None)], [], hosted_in_training=False)
    assert rows == () and dropped["uncleared"] == 2  # unknown ancestry is uncleared too
    rows, dropped = terms.seed_rows(_data(made), store, [_today(frozenset())], [], hosted_in_training=False)
    assert [row.id for row in rows] == ["t0", "t1"] and dropped["uncleared"] == 0  # the person's own words
    assert "16mm today" in rows[0].context and "they edit at night" in rows[0].context  # notes + today
    # The teacher reads the day; the student reads the message alone, and both score the same reply.
    data = replace(_data(made), seeds=rows)
    written = [json.loads(line) for line in data.to_jsonl(pairs=True, terms=True).splitlines()]
    seed = next(row for row in written if row.get("kind") == "seed")
    assert [m["role"] for m in seed["teacher"]] == ["system", "user"] and "16mm today" in seed["teacher"][0]["content"]
    assert [m["role"] for m in seed["prompt"]] == ["user"] and seed["teacher"][1] == seed["prompt"][0]
    assert "16mm today" not in json.dumps(seed["prompt"]) + json.dumps(seed["completion"])
    off = [json.loads(line) for line in data.to_jsonl(pairs=True).splitlines()]
    assert not any("kind" in row for row in off)


def test_only_tonights_glimmer_replies_seed() -> None:
    made = [_transcript(0), _transcript(1, target="teacher")]
    rows, dropped = terms.seed_rows(_data(made), Transcripts(made), [_today(frozenset())], [],
                                    hosted_in_training=False)
    assert [row.id for row in rows] == ["t0"] and dropped["teacher_target"] == 1  # Spark Open's reply isn't on-policy
    bare = replace(made[0], context=None)
    rows, dropped = terms.seed_rows(_data([bare]), Transcripts([bare]), [], [], hosted_in_training=False)
    assert rows == () and dropped["no_context"] == 1  # nothing to consolidate: the teacher would be the student


def test_seeding_never_reads_minds_live_threads() -> None:
    """Weights hold the consolidated day; the channel (or its text stand-in) holds the live threads. The
    teacher's context is notes plus today's items, and a dream's thread is the student's input only."""
    made = [_transcript(0)]
    rows, _ = terms.seed_rows(_data(made), Transcripts(made), [_today(frozenset())], [], hosted_in_training=False)
    assert "thread" not in rows[0].context.lower()
    dream = Dream(id="d0", state_ref="s0", thread="Open: whether the grant lands (risk 0.7)", message="Any news?",
                  reply="Not yet. I'd plan both ways.", ancestry=frozenset({GLIMMER}), kind="boundary",
                  event_ids=("e1",))
    unanchored = replace(dream, id="d1", event_ids=())
    kept, dropped = terms.dream_rows([dream, unanchored], [], _NoLeak(), hosted_in_training=False)
    assert [row.id for row in kept] == ["d0"] and dropped["unanchored"] == 1  # a dream rests on a real event
    written = json.loads(replace(_data(made), dreams=kept).to_jsonl(terms=True).splitlines()[-1])
    assert written["kind"] == "dream" and written["prompt"][0] == {"role": "system", "content": dream.thread}
    assert written["seed_kind"] == "boundary" and dream.thread not in json.dumps(written["completion"])


class _NoLeak:
    def check(self, texts: Sequence[str], sealed: Sequence[Probe]) -> tuple[bool, ...]:
        return (False,) * len(texts)


# ── switches ─────────────────────────────────────────────────────────────────


def test_switches_off_are_todays_night(tmp_path: Path) -> None:
    for env in ({}, {objective.FOLD_ENV: "0", objective.DREAMS_ENV: "0"}, {"ANJO_SLEEP_MATCH_MINUTES": "6.5"}):
        cfg = objective.night_cfg(env)
        assert cfg == LoraCfg(objective="sft_dpo") and cfg.fold_rank is None and not cfg.dreams
        assert objective.night_terms(env) == Terms() and not (Terms().seed or Terms().dreams)
    sent = training_config(LoraCfg(objective="sft_dpo"), 12, warm=True, revision="a" * 40, night=NIGHT)
    assert set(sent) == TODAYS_KEYS
    made = [_transcript(0)]
    rows, _ = terms.seed_rows(_data(made), Transcripts(made), [_today(frozenset())], [], hosted_in_training=False)
    plain = _data(made)
    assert replace(plain, seeds=rows).to_jsonl(pairs=True) == plain.to_jsonl(pairs=True)
    # The training script reads kind rows only under sft_dpo_seed; today's split never sees them.
    written = [json.loads(line) for line in replace(plain, seeds=rows).to_jsonl(pairs=True, terms=True).splitlines()]
    assert colab_train.split_rows(written) == colab_train.split_rows(
        [json.loads(line) for line in plain.to_jsonl(pairs=True).splitlines()])
    # A night with the switches off logs no terms and sends the trainer today's dataset.
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    deps = world.deps()
    assert deps.terms == Terms() and isinstance(deps.dreams, NoDreams)
    result = run_night_batch_with(deps, {"p1": NIGHT}, windows={"p1": window(NIGHT)})
    assert result["p1"].status == "promoted" and "terms" not in step(result["p1"].log_path, "dataset")


def test_unwired_dreams_say_so_in_the_dataset_receipt(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.plan("p1", NIGHT)
    deps = replace(world.deps(), terms=Terms(seed=True, dreams=True))
    result = run_night_batch_with(deps, {"p1": NIGHT}, windows={"p1": window(NIGHT)})
    receipt = step(result["p1"].log_path, "dataset")["terms"]
    assert receipt["dreams"] == terms.DREAMS_NOT_WIRED == "not wired" and receipt["anchors"] == 0


def test_the_night_command_sets_the_terms_only_when_a_switch_is_on(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from anjo.sleep.night import __main__ as command

    seen: list[Any] = []
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(command, "_wiring", lambda path: lambda trainer, options: World(tmp_path).deps())
    def record(deps: Any, nights: Any, **kw: Any) -> dict[str, Any]:
        seen.append(deps)
        return {}

    monkeypatch.setattr(command, "run_night_batch_with", record)
    for name in (objective.OBJECTIVE_ENV, objective.FOLD_ENV, objective.DREAMS_ENV):
        monkeypatch.delenv(name, raising=False)
    command.execute(wiring="w:x", night=NIGHT, users=["u1"], personas={"u1": "founder"})
    monkeypatch.setenv(objective.OBJECTIVE_ENV, "sft_dpo_seed")
    monkeypatch.setenv(objective.DREAMS_ENV, "1")
    command.execute(wiring="w:x", night=NIGHT, users=["u1"], personas={"u1": "founder"})
    assert [deps.terms for deps in seen] == [Terms(), Terms(seed=True, dreams=True)]


def test_fold_and_dreams_need_the_seed_objective_and_bad_values_stop_the_night() -> None:
    on = {objective.OBJECTIVE_ENV: "sft_dpo_seed", objective.FOLD_ENV: "1", objective.DREAMS_ENV: "1"}
    cfg = objective.night_cfg(on)
    assert (cfg.objective, cfg.fold_rank, cfg.dreams) == ("sft_dpo_seed", config.MAX_LORA_RANK, True)
    assert objective.night_terms(on) == Terms(seed=True, dreams=True)
    for bad in ({objective.FOLD_ENV: "1"}, {objective.DREAMS_ENV: "1"}, {objective.FOLD_ENV: "yes"}):
        with pytest.raises(objective.ObjectiveError):
            objective.night_cfg(bad)
    with pytest.raises(LoraCfgError):
        LoraCfg(objective="sft", fold_rank=32)
    with pytest.raises(LoraCfgError):
        LoraCfg(objective="sft_dpo_seed", fold_rank=8)  # the long-term slot holds at least a night


def test_capacity_is_the_servers_most_rank_and_three_passes_one_row_a_step() -> None:
    seed = {objective.OBJECTIVE_ENV: "sft_dpo_seed"}
    cfg = objective.night_cfg({**seed, objective.CAPACITY_ENV: "1"})
    assert (cfg.objective, cfg.r, cfg.passes, cfg.fold_rank) == ("sft_dpo_seed", config.MAX_LORA_RANK, 3, None)
    assert objective.night_terms({**seed, objective.CAPACITY_ENV: "1"}) == Terms(seed=True)  # the same rows
    off = objective.night_cfg({**seed, objective.CAPACITY_ENV: "0"})
    assert off == objective.night_cfg(seed) == LoraCfg(objective="sft_dpo_seed") and off.passes is None
    assert plan_steps(40, cfg) == StepPlan(accumulation=1, epochs=3, steps_per_epoch=40)
    sent = training_config(cfg, 12, warm=True, revision="a" * 40, night=NIGHT)
    assert sent["terms"] == {"fold_rank": None, "dreams": False, "passes": 3} and (sent["r"], sent["alpha"]) == (32, 64)
    assert (sent["accumulation"], sent["epochs"]) == (1, 3) and sent["learning_rate"] == cfg.warm_lr  # the recipe's rates
    assert colab_train.load_config(json.dumps(sent))["terms"]["passes"] == 3
    for bad in (0, 5, True, "3"):
        with pytest.raises(ValueError):
            colab_train.load_config(json.dumps({**sent, "terms": {**sent["terms"], "passes": bad}}))
    assert training_config(off, 12, warm=True, revision="a" * 40, night=NIGHT)["terms"] == {"fold_rank": None, "dreams": False}
    plain = objective.night_cfg({objective.CAPACITY_ENV: "1"})  # today's objective: the host's step plan carries it
    sent_plain = training_config(plain, 12, warm=False, revision="a" * 40, night=NIGHT)
    assert (plain.objective, sent_plain["r"], sent_plain["accumulation"], sent_plain["epochs"]) == ("sft_dpo", 32, 1, 3)
    assert "terms" not in sent_plain and colab_train.load_config(json.dumps(sent_plain))["epochs"] == 3
    for bad_env in ({**seed, objective.CAPACITY_ENV: "1", objective.FOLD_ENV: "1"}, {**seed, objective.CAPACITY_ENV: "yes"}):
        with pytest.raises(objective.ObjectiveError):
            objective.night_cfg(bad_env)
    for bad_cfg in ({"passes": 0}, {"passes": 5}, {"passes": 3, "match_by_night": ((NIGHT, 5.0),)}):
        with pytest.raises(LoraCfgError):
            LoraCfg(objective="sft_dpo", **bad_cfg)


def test_the_job_plans_the_seed_loss_from_the_recipe_passes_included() -> None:
    sent = training_config(LoraCfg(objective="sft_dpo_seed", r=32, passes=3), 12, warm=False, revision="a" * 40,
                           night=NIGHT)
    plan = sleep_job.seed_plan(sleep_loss, sent, eot=11, pad=0, long_term_rank=0, begun=0.0)
    assert (plan.passes, plan.min_steps, plan.lr, plan.long_term_rank, plan.stretch_s) == (3, 8, 2e-4, 0, 0.0)
    plain = training_config(LoraCfg(objective="sft_dpo_seed"), 12, warm=False, revision="a" * 40, night=NIGHT)
    assert sleep_job.seed_plan(sleep_loss, plain, eot=11, pad=0, long_term_rank=0, begun=0.0).passes == 0


def test_the_seed_recipe_reaches_the_script_and_a_folded_night_starts_a_fresh_slot() -> None:
    cfg = LoraCfg(objective="sft_dpo_seed", fold_rank=32, dreams=True)
    sent = training_config(cfg, 12, warm=True, revision="a" * 40, night=NIGHT)
    assert sent["terms"] == {"fold_rank": 32, "dreams": True} and sent["learning_rate"] == cfg.lr
    assert training_config(replace(cfg, fold_rank=None), 12, warm=True, revision="a" * 40,
                           night=NIGHT)["learning_rate"] == cfg.warm_lr
    loaded = colab_train.load_config(json.dumps(sent))
    assert loaded["objective"] == "sft_dpo_seed" and "dpo" not in colab_train.build_settings(loaded)
    with pytest.raises(ValueError):
        colab_train.load_config(json.dumps({**sent, "terms": {"fold_rank": 64, "dreams": True}}))


def test_colab_ships_the_seed_objectives_loss_files_at_modals_paths(tmp_path: Path) -> None:
    from anjo.sleep.train import modal_job
    from anjo.sleep.train.runners import colab

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("")
    inputs = {"workdir": str(tmp_path), "dataset_path": str(dataset), "config": {"objective": "sft_dpo_seed"}}
    archive = tmp_path / colab.SEED_ARCHIVE
    assert colab._uploads(inputs) == [
        (dataset, f"{colab.REMOTE_DIR}/{dataset.name}"), (archive, colab.SEED_ARCHIVE_ENTRY)]
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == sorted(remote.lstrip("/") for _, remote in modal_job.TRAIN_FILES)
    assert colab._uploads({**inputs, "config": {"objective": "sft_dpo"}}) == [
        (dataset, f"{colab.REMOTE_DIR}/{dataset.name}")]


class _Tokens:
    """A stand-in tokenizer: a message's ids are its role's mark then its words' lengths; ``<|eot|>`` is 11."""

    def apply_chat_template(self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool,
                            return_dict: bool, **template: str) -> list[int]:
        ids = [1]
        for message in messages:
            ids += [{"system": 7, "user": 8}[message["role"]], *(len(w) for w in message["content"].split())]
        return [*ids, 9] if add_generation_prompt else ids

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [5] if text == colab_train.USER_CHANNEL else [len(w) for w in text.split()]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return 11


def test_the_script_feeds_each_term_the_rows_the_server_would_render(tmp_path: Path) -> None:
    made = [_transcript(0)]
    rows, _ = terms.seed_rows(_data(made), Transcripts(made), [_today(frozenset())], [], hosted_in_training=False)
    dream = Dream(id="d0", state_ref="s0", thread="Open: the grant", message="Any news?", reply="Not yet.",
                  ancestry=frozenset({GLIMMER}), kind="random", event_ids=("e1",))
    dreams, _ = terms.dream_rows([dream], [], _NoLeak(), hosted_in_training=False)
    data = replace(_data(made), seeds=rows, dreams=dreams, anchors=terms.anchor_rows(3, n=2))
    written = [json.loads(line) for line in data.to_jsonl(pairs=True, terms=True).splitlines()]
    plain, pair_rows = colab_train.split_rows(written)
    by_kind = sleep_job.term_rows(colab_train, written)
    assert [len(by_kind[k]) for k in sleep_job.KINDS] == [1, 1, 2] and len(plain) == 1 and not pair_rows
    cfg = {"max_seq_length": 8192, "terms": {"fold_rank": None, "dreams": True}}
    template = {"reasoning_strength": "low", "current_date": NIGHT.isoformat()}
    batch, too_long = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [(row, {}) for row in plain], [], by_kind,
                                           cfg, template)
    seed = batch.seeds[0]
    assert seed.prompt[:2] == (1, 8) and seed.teacher[:2] == (1, 7)  # the student reads the message, the teacher the day first
    assert seed.prompt[-2:] == (9, 5) and seed.reply[-1] == 11 and seed.reply == batch.sft[0].reply
    assert batch.dreams[0].prompt[:2] == (1, 7) and batch.dream_groups == ("random",)
    assert all(11 not in choice for item in batch.anchors for choice in item.choices)  # scored open, as the gate does
    assert too_long == {"seed": 0, "dream": 0}
    off, _ = sleep_job.seed_batch(colab_train, sleep_loss, _Tokens(), [], [], by_kind,
                                  {**cfg, "terms": {"fold_rank": None, "dreams": False}}, template)
    assert off.dreams == () and off.dream_groups == ()
    with pytest.raises(ValueError, match="kind"):
        sleep_job.term_rows(colab_train, [{**written[-1], "kind": "nightmare"}])
    with pytest.raises(ValueError, match="cleaned"):
        sleep_job.term_rows(colab_train, [{**by_kind["seed"][0], "teacher": [{"role": "system", "content": "<|eot|>"}]}])
    loaded = colab_train.load_loss(Path(sleep_loss.__file__).parent)  # as the job loads it under python -I
    assert loaded.WEIGHTS == sleep_loss.WEIGHTS
    with pytest.raises(ValueError, match="beside"):
        colab_train.load_loss(tmp_path)


# ── the loss, on a tiny model ────────────────────────────────────────────────


def _tiny(torch: Any, vocab: int = 12, dim: int = 6, rank: int = 2) -> Any:
    """A causal toy with a LoRA on its head: the logits read a running mean of the tokens so far, so
    a context changes them, and ``disable_adapter`` switches the LoRA off as PEFT's does. Each forward
    pass records its ids, whether the adapter was on, whether a gradient was, whether it was in train
    mode, and the LoRA's B as it stood."""
    nn = torch.nn

    Module: Any = nn.Module

    class Tiny(Module):  # type: ignore[misc]  # torch is Any here
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.emb = nn.Embedding(vocab, dim)
            self.head = nn.Linear(dim, vocab)
            self.lora_A = nn.Parameter(torch.randn(rank, dim) * 0.3)
            self.lora_B = nn.Parameter(torch.randn(vocab, rank) * 0.3)
            for p in (*self.emb.parameters(), *self.head.parameters()):
                p.requires_grad_(False)
            self.enabled = True
            self.calls: list[tuple[tuple[int, ...], bool, bool, bool, Any]] = []

        @contextmanager
        def disable_adapter(self) -> Iterator[None]:
            self.enabled = False
            try:
                yield
            finally:
                self.enabled = True

        def forward(self, input_ids: Any, logits_to_keep: int = 0, **kw: Any) -> Any:
            self.calls.append((tuple(input_ids[0].tolist()), self.enabled, torch.is_grad_enabled(), self.training,
                               self.lora_B.detach().clone()))
            h = self.emb(input_ids)
            h = h.cumsum(dim=1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
            logits = self.head(h)
            if self.enabled:
                logits = logits + 2.0 * (h @ self.lora_A.T) @ self.lora_B.T
            return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

        def generate(self, input_ids: Any, max_new_tokens: int, eos_token_id: int, **kw: Any) -> Any:
            ids = input_ids
            for _ in range(max_new_tokens):
                probs = torch.softmax(self(ids).logits[:, -1], dim=-1)
                nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
                if int(nxt) == eos_token_id:
                    break
            return ids

    return Tiny()


def _plan(**changes: Any) -> Any:
    return replace(sleep_loss.Plan(lr=0.05, min_steps=2, beta=0.1, seed=3, eot=11, pad=0, dpo_lr=1e-4), **changes)


def test_seeding_teacher_is_the_long_term_adapter_plus_context() -> None:
    """The teacher is the adapter the night started from (the long-term one), with the day in
    context, no gradient and no dropout; the student is tonight's adapter on the message alone."""
    torch = _torch()
    model = _tiny(torch)
    long_term = model.lora_B.detach().clone()
    teacher = sleep_loss.Teacher([model.lora_A, model.lora_B])
    with torch.no_grad():
        model.lora_B.add_(0.5)  # what tonight's training has done so far
    tonight = model.lora_B.detach().clone()
    model.train()
    row = sleep_loss.SeedSeq(prompt=(1, 2), teacher=(7, 8, 9, 1, 2), reply=(3, 4, 11))
    kl = sleep_loss.seed_kl(model, row, teacher)
    (t_ids, t_on, t_grad, t_train, t_b), (s_ids, s_on, s_grad, s_train, s_b) = model.calls
    assert (t_ids, t_on, t_grad, t_train) == ((7, 8, 9, 1, 2, 3, 4, 11), True, False, False)  # the day first
    assert torch.equal(t_b, long_term)  # the long-term adapter, not the base and not tonight's
    assert (s_ids, s_on, s_grad, s_train) == ((1, 2, 3, 4, 11), True, True, True) and torch.equal(s_b, tonight)
    assert torch.equal(model.lora_B, tonight) and model.training  # tonight's adapter is back, training resumes
    kl.backward()
    assert model.lora_B.grad is not None and model.head.weight.grad is None
    # Reverse KL(student || teacher) over the reply's positions, by hand:
    with torch.no_grad():
        model.lora_B.copy_(long_term)
        t = torch.log_softmax(model(torch.tensor([[7, 8, 9, 1, 2, 3, 4, 11]])).logits[0, 4:7], -1)
        model.lora_B.copy_(tonight)
        s = torch.log_softmax(model(torch.tensor([[1, 2, 3, 4, 11]])).logits[0, 1:4], -1)
    assert torch.allclose(kl.detach(), (s.exp() * (s - t)).sum(-1).mean(), atol=1e-6) and float(kl.detach()) > 0
    # Tonight's adapter still the night's start and no context: teacher and student agree, nothing to learn.
    same = sleep_loss.SeedSeq(prompt=(1, 2), teacher=(1, 2), reply=(3, 4))
    start = sleep_loss.Teacher([model.lora_A, model.lora_B])
    assert float(sleep_loss.seed_kl(model, same, start).detach()) == pytest.approx(0.0, abs=1e-7)


def test_with_the_fold_the_teacher_has_the_long_term_ranks_on_and_tonights_slot_off() -> None:
    torch = _torch()
    model = _tiny(torch, rank=3)
    sleep_loss.load_leading(model, {"lora_A.weight": model.lora_A[:2].detach().clone(),
                                    "lora_B.weight": model.lora_B[:, :2].detach().clone()}, rank=2, adapter="")
    long_term = model.lora_B.detach().clone()
    assert not long_term[:, 2:].any()  # the fresh slot starts empty
    seeds = tuple(sleep_loss.SeedSeq((1, i), (9, 9, 1, i), (4, 5, 11)) for i in range(2, 6))
    # Four updates: the first is at rate 0 (today's warmup), so a later student pass sees the slot filled.
    sleep_loss.train_terms(model, sleep_loss.Batch(seeds=seeds), _plan(long_term_rank=2, min_steps=3))
    taught = [b for ids, on, grad, _, b in model.calls if ids[:2] == (9, 9) and on and not grad]
    trained = [b for ids, on, grad, _, b in model.calls if ids[:2] != (9, 9) and on and grad]
    assert len(taught) >= len(seeds) and all(torch.equal(b, long_term) for b in taught)
    assert any(b[:, 2:].any() for b in trained)  # the student had tonight's slot on
    assert model.lora_B[:, 2:].any() and torch.equal(model.lora_B[:, :2], long_term[:, :2])


def test_seeding_trains_only_the_adapter() -> None:
    torch = _torch()
    model = _tiny(torch)
    base = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" not in name}
    lora = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}
    batch = sleep_loss.Batch(
        sft=(sleep_loss.Seq((1, 2), (3, 11)),),
        pairs=(sleep_loss.PairSeq((1, 5), (6, 11), (7, 11)),),
        seeds=tuple(sleep_loss.SeedSeq((1, i), (9, 9, 1, i), (4, 5, 11)) for i in range(2, 6)),
    )
    receipt = sleep_loss.train_terms(model, batch, _plan())
    assert all(torch.equal(p, base[name]) for name, p in model.named_parameters() if name in base)
    assert any(not torch.equal(p, lora[name]) for name, p in model.named_parameters() if name in lora)
    assert receipt["steps"] >= 2 and receipt["rows"]["seed"] == 4 and receipt["rows"]["imitate"] >= 1
    assert set(receipt["minutes"]) >= {"sft", "dpo", "seed", "imitate"} and receipt["weights"] == sleep_loss.WEIGHTS
    assert receipt["grads_adapter_only"] and receipt["teacher_drift"] == 0.0  # adapter off, context on: unmoved
    assert receipt["seed_kl_after"] != receipt["seed_kl_before"]
    model.head.weight.requires_grad_(True)  # anything but the adapter trainable: refused before a step
    with pytest.raises(ValueError, match="adapter"):
        sleep_loss.train_terms(model, batch, _plan())


def test_with_seeding_on_the_dpo_step_is_todays() -> None:
    """The DPO term keeps today's pass: its own AdamW at today's DPO rate (never the main block's,
    500x higher here), today's accumulation and epochs rule, and so today's number of updates."""
    torch = _torch()
    pairs = tuple(sleep_loss.PairSeq((1, i), (6, 11), (7, 11)) for i in range(2, 7))
    batch = sleep_loss.Batch(
        sft=(sleep_loss.Seq((1, 2), (3, 11)),), pairs=pairs,
        seeds=tuple(sleep_loss.SeedSeq((1, i), (9, 9, 1, i), (4, 5, 11)) for i in range(2, 6)),
    )
    plan = _plan(min_steps=4)
    model = _tiny(torch)
    dpo = sleep_loss.train_terms(model, batch, plan)["dpo"]
    cfg = {"min_optimizer_steps": plan.min_steps, "batch_size": 1}
    accumulation, epochs = colab_train.dpo_plan(len(pairs), cfg)  # today's pass, as the script plans it
    assert (dpo["accumulation"], dpo["epochs"]) == (accumulation, epochs) and dpo["lr"] == plan.dpo_lr
    assert dpo["steps"] == epochs * -(-len(pairs) // accumulation)
    # Adam moves each weight by about its rate a step: today's DPO rate, not the main block's.
    assert 0.1 * plan.dpo_lr <= dpo["max_update"] <= 3.2 * plan.dpo_lr < plan.lr / 100
    with_dpo_lr = sleep_loss.train_terms(_tiny(torch), batch, replace(plan, dpo_lr=plan.lr))["dpo"]
    assert with_dpo_lr["max_update"] > 10 * dpo["max_update"]  # what sharing the SFT rate would have done


def test_with_passes_the_main_block_takes_one_row_a_step_each_pass_and_the_dpo_block_is_todays() -> None:
    torch = _torch()
    pairs = tuple(sleep_loss.PairSeq((1, i), (6, 11), (7, 11)) for i in range(2, 7))
    batch = sleep_loss.Batch(
        sft=tuple(sleep_loss.Seq((1, i), (3, 11)) for i in range(2, 9)), pairs=pairs,
        seeds=tuple(sleep_loss.SeedSeq((1, i), (9, 9, 1, i), (4, 5, 11)) for i in range(2, 6)),
    )
    plan = _plan(min_steps=4)
    step_plan = sleep_loss.train_terms(_tiny(torch), batch, plan)
    passes = sleep_loss.train_terms(_tiny(torch), batch, replace(plan, passes=3))
    rows = sum(passes["rows"][term] for term in ("sft", "seed", "imitate"))
    assert step_plan["accumulation"] > 1 and step_plan["epochs"] == 1  # the step plan: a few accumulated steps
    assert (passes["accumulation"], passes["epochs"]) == (1, 3)
    assert passes["steps"] - passes["dpo"]["steps"] == 3 * rows  # one row a step, three times over
    assert passes["dpo"] == {**step_plan["dpo"], "max_update": passes["dpo"]["max_update"]}  # DPO keeps today's plan
    with pytest.raises(ValueError, match="time budget"):
        sleep_loss.train_terms(_tiny(torch), batch, replace(plan, passes=3, stretch_s=60.0))


def test_the_schedule_is_todays_cosine_with_warmup_step_for_step() -> None:
    """The main and DPO blocks' rate is the one today's trainer gives (transformers' cosine with a
    ``ceil(3%)``-step warmup, 0 on the first step), so a seeded night gets no extra rate over today's."""
    torch = _torch()
    transformers = pytest.importorskip("transformers")
    for total in (8, 16, 40, 67, 100):
        warm = math.ceil(sleep_loss.WARMUP * total)
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        today = transformers.get_cosine_schedule_with_warmup(optimizer, warm, total).lr_lambdas[0]
        ours = sleep_loss.schedule(total)
        assert [ours(step) for step in range(total)] == pytest.approx([today(step) for step in range(total)], abs=1e-12)
    assert sleep_loss.schedule(8)(0) == 0.0 and sleep_loss.schedule(8)(8) == sleep_loss.DREAM_RATE


def test_the_imitation_reward_is_edit_similarity_cut_at_z0() -> None:
    assert sleep_loss.edit_distance((1, 2, 3), (1, 3)) == 1
    assert sleep_loss.edit_reward((1, 2, 3, 4), (1, 2, 3, 4)) == 1.0
    assert sleep_loss.edit_reward((1, 2, 3, 9), (1, 2, 3, 4)) == pytest.approx(0.75)
    assert sleep_loss.edit_reward((9, 9, 9, 9), (1, 2, 3, 4)) == 0.0  # past z0: no partial credit
    assert sleep_loss.edit_reward((), ()) == 1.0


def test_the_imitation_gradient_follows_the_better_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """REINFORCE with a leave-one-out baseline: the better-scored completion gains log-probability.
    (On a random model every reward is 0 and no gradient flows, as the CPU smoke saw.)"""
    torch = _torch()
    model = _tiny(torch)
    rewards = iter([1.0, 0.0])
    monkeypatch.setattr(sleep_loss, "edit_reward", lambda sample, target, z0=0.5: next(rewards))
    row = sleep_loss.SeedSeq(prompt=(1, 2), teacher=(1, 2), reply=(3, 4, 5, 6, 11))
    loss, mean = sleep_loss.imitation_loss(model, row, random.Random(0), _plan())
    assert mean == 0.5 and loss is not None and loss.requires_grad
    loss.backward()
    assert model.lora_B.grad is not None and float(model.lora_B.grad.abs().sum()) > 0
    assert sleep_loss.imitation_loss(model, replace(row, reply=(3,)), random.Random(0), _plan()) == (None, None)


def test_dream_filter_keeps_topk_plus_random() -> None:
    norms = [0.1, 5.0, 0.3, 4.0, 0.2, 0.9, 0.05]
    kept = sleep_loss.select_dreams(norms, k=2, b=2, seed=7)
    assert kept[:2] == [1, 3] and len(kept) == 4 and len(set(kept)) == 4  # the top two by gradient norm first
    assert set(kept[2:]) <= {0, 2, 4, 5, 6}  # then two of the rest at random, seeded
    assert sleep_loss.select_dreams(norms, k=2, b=2, seed=7) == kept
    assert len({tuple(sleep_loss.select_dreams(norms, k=2, b=2, seed=s)[2:]) for s in range(20)}) > 1
    assert sleep_loss.select_dreams([1.0, 2.0], k=3, b=2, seed=0) == [1, 0]  # fewer than k + b: all, ranked
    assert sleep_loss.select_dreams([], k=3, b=2, seed=0) == []


def test_a_dream_batch_is_kept_only_when_the_anchor_check_holds() -> None:
    torch = _torch()
    anchors = tuple(sleep_loss.AnchorSeq((1, i), ((3,), (4,)), 0) for i in range(2, 5))
    dreams = tuple(sleep_loss.Seq((2, i), (4, 4, 11)) for i in range(5))  # all teach the wrong answer
    batch = sleep_loss.Batch(sft=(sleep_loss.Seq((1, 2), (3, 11)),), dreams=dreams, anchors=anchors)
    model = _tiny(torch)
    receipt = sleep_loss.train_terms(model, batch, _plan(dreams=True))
    dream = receipt["dreams"]
    assert dream["candidates"] == 5 and dream["kept"] == min(5, sleep_loss.DREAM_TOP_K + sleep_loss.DREAM_RANDOM)
    after = sleep_loss.anchor_score(model, anchors)
    assert dream["block_kept"] == (dream["anchor_after"] >= dream["anchor_before"] - sleep_loss.ANCHOR_TOLERANCE)
    assert after == pytest.approx(dream["anchor_after"] if dream["block_kept"] else dream["anchor_before"], abs=1e-6)
    with pytest.raises(ValueError, match="anchor"):
        sleep_loss.train_terms(_tiny(torch), replace(batch, anchors=()), _plan(dreams=True))


def test_the_receipt_gives_the_dream_filters_choices_by_seed_kind() -> None:
    """Boundary seeds against random-thread seeds: the cheap evidence that targeting matters."""
    groups = ("boundary", "random", "boundary", "random", "boundary")
    norms, kept = [5.0, 1.0, 4.0, 2.0, 0.5], [0, 2, 3]  # top 2 (both boundary), then one at random
    by = sleep_loss.by_group(groups, norms, kept, k=2)
    assert by["boundary"] == {"candidates": 3, "kept": 2, "keep_rate": round(2 / 3, 4), "top_k": 2,
                              "norm_mean": pytest.approx(9.5 / 3)}
    assert by["random"]["kept"] == 1 and by["random"]["top_k"] == 0 and by["random"]["keep_rate"] == 0.5
    torch = _torch()
    anchors = (sleep_loss.AnchorSeq((1, 2), ((3,), (4,)), 0),)
    dreams = tuple(sleep_loss.Seq((2, i), (4, 4, 11)) for i in range(4))
    batch = sleep_loss.Batch(sft=(sleep_loss.Seq((1, 2), (3, 11)),), dreams=dreams, anchors=anchors,
                             dream_groups=("boundary", "boundary", "random", "random"))
    receipt = sleep_loss.train_terms(_tiny(torch), batch, _plan(dreams=True))
    assert set(receipt["dreams"]["by_seed"]) == {"boundary", "random"}
    assert sum(g["candidates"] for g in receipt["dreams"]["by_seed"].values()) == 4
    with pytest.raises(ValueError, match="seed kind"):
        sleep_loss.train_terms(_tiny(torch), replace(batch, dream_groups=("boundary",)), _plan(dreams=True))


# ── the fold ─────────────────────────────────────────────────────────────────

PREFIX = "base_model.model.model.language_model.layers.{}.self_attn.q_proj"


def _adapter(torch: Any, folder: Path, k: int, d_in: int = 40, d_out: int = 36, layers: int = 2) -> dict[str, Any]:
    torch.manual_seed(1)
    tensors = {}
    for layer in range(layers):
        tensors[f"{PREFIX.format(layer)}.lora_A.weight"] = torch.randn(k, d_in)
        tensors[f"{PREFIX.format(layer)}.lora_B.weight"] = torch.randn(d_out, k)
    folder.mkdir(parents=True)
    sleep_fold.write_safetensors(tensors, folder / "adapter_model.safetensors")
    meta = {"peft_type": "LORA", "r": k, "lora_alpha": 2 * k, "base_model_name_or_path": config.MODEL,
            "target_modules": colab_train.lora_targets(("q_proj",)), "rank_pattern": {}, "alpha_pattern": {},
            "use_dora": False, "task_type": "CAUSAL_LM", "anjo_rendering": "g1/medium"}
    (folder / "adapter_config.json").write_text(json.dumps(meta))
    return tensors


def test_fold_keeps_a_servable_plain_lora(tmp_path: Path) -> None:
    torch = _torch()
    folder = tmp_path / "p-0123456789ab-n3"
    before = _adapter(torch, folder, k=48)  # tonight's 16 on top of a long-term 32
    sleep_fold.fold_folder(folder, rank=32)
    config.check_adapter(folder, folder.name)  # what the server and the upload accept
    meta = json.loads((folder / "adapter_config.json").read_text())
    assert (meta["r"], meta["lora_alpha"], meta["rank_pattern"], meta["alpha_pattern"]) == (32, 64, {}, {})
    assert meta["anjo_rendering"] == "g1/medium" and meta["use_dora"] is False
    after = sleep_fold.read_safetensors(folder / "adapter_model.safetensors")
    assert set(after) == set(before)  # the same LoRA pairs, nothing added
    for layer in range(2):
        a, b = after[f"{PREFIX.format(layer)}.lora_A.weight"], after[f"{PREFIX.format(layer)}.lora_B.weight"]
        assert a.shape == (32, 40) and b.shape == (36, 32)
        full = before[f"{PREFIX.format(layer)}.lora_B.weight"] @ before[f"{PREFIX.format(layer)}.lora_A.weight"]
        tail = torch.linalg.svdvals(full)[32:]
        assert torch.linalg.norm(full - b @ a) == pytest.approx(float(torch.linalg.norm(tail)), rel=1e-3)  # best rank 32
    small = tmp_path / "p-0123456789ab-n1"
    first = _adapter(torch, small, k=16)
    sleep_fold.fold_folder(small, rank=32)  # a first night fits: kept whole, at its own rank
    kept = sleep_fold.read_safetensors(small / "adapter_model.safetensors")
    name = PREFIX.format(0)
    assert kept[f"{name}.lora_A.weight"].shape == (16, 40)
    assert torch.allclose(kept[f"{name}.lora_B.weight"] @ kept[f"{name}.lora_A.weight"],
                          first[f"{name}.lora_B.weight"] @ first[f"{name}.lora_A.weight"], atol=1e-4)


def test_fold_keeps_the_long_term_part_frozen_while_the_night_trains() -> None:
    torch = _torch()
    model = _tiny(torch, rank=3)
    a, b = model.lora_A.detach().clone(), model.lora_B.detach().clone()
    batch = sleep_loss.Batch(sft=tuple(sleep_loss.Seq((1, i), (3, 11)) for i in range(2, 6)))
    receipt = sleep_loss.train_terms(model, batch, _plan(long_term_rank=2))
    assert receipt["long_term_drift"] == 0.0
    assert torch.equal(model.lora_A[:2], a[:2]) and torch.equal(model.lora_B[:, :2], b[:, :2])  # long-term held
    assert not torch.equal(model.lora_B[:, 2:], b[:, 2:])  # tonight's slot learned


def test_the_long_term_adapter_loads_into_the_leading_slots() -> None:
    torch = _torch()
    model = _tiny(torch, rank=3)
    saved = {"lora_A.weight": torch.ones(2, 6), "lora_B.weight": torch.full((12, 2), 2.0)}
    sleep_loss.load_leading(model, saved, rank=2, adapter="")
    assert torch.equal(model.lora_A[:2], torch.ones(2, 6)) and torch.equal(model.lora_B[:, :2], torch.full((12, 2), 2.0))
    assert torch.equal(model.lora_B[:, 2:], torch.zeros(12, 1))  # tonight's slot starts at zero: the fold's reset


def test_the_loss_file_is_standalone() -> None:
    import ast

    tree = ast.parse(Path(sleep_loss.__file__).read_text(encoding="utf-8"))
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])}
    assert not top & {"torch", "transformers", "peft", "trl", "anjo", "safetensors"}


# ── dreams from MIND's states ────────────────────────────────────────────────


class _Glimmer:
    def __init__(self, cut: set[int] | None = None) -> None:
        self.calls: list[tuple[Any, list[dict[str, str]], int | None]] = []
        self.cut = cut or set()

    def complete(self, adapter: Any, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None,
                 max_tokens: int | None = None) -> ChatResult:
        self.calls.append((adapter, messages, seed))
        n = len(self.calls)
        return ChatResult(text=f"text {n}", tool_calls=(), model=GLIMMER, input_tokens=10, output_tokens=5,
                          finish_reason="length" if n in self.cut else "stop")


def _seed(branch: str, hypothetical: bool, events: tuple[str, ...] = ("e1", "e2"), kind: str = "boundary") -> DreamSeed:
    return DreamSeed(state_ref="s1", thread="Open: the grant decision", event_ids=events, branch=branch,
                     hypothetical=hypothetical, kind=kind, ancestry=frozenset())  # type: ignore[arg-type]


def test_glimmer_dreams_speak_from_minds_states() -> None:
    seeds = [_seed("speak", False), _seed("wait", True), _seed("speak", False, events=())]
    glimmer = _Glimmer(cut={4})
    source = GlimmerDreams(glimmer, lambda uid, night: seeds, seed=5, per_state=2)
    made = source.dream("p1", NIGHT, None)
    assert source.available and len(glimmer.calls) == 8  # a message, then her reply, per candidate; none unanchored
    assert len(made) == 3  # the reply cut at its cap is no dream
    taken, imagined = made[0], made[-1]
    assert taken.thread == f"Open: the grant decision\n{TAKEN}: speak"
    assert imagined.thread == f"Open: the grant decision\n{HYPOTHETICAL}: wait"  # the branch she didn't take
    assert taken.ancestry == frozenset({GLIMMER})  # her own text, on a thread from the person's own words
    assert (taken.kind, taken.event_ids, taken.state_ref) == ("boundary", ("e1", "e2"), "s1")
    assert len({dream.id for dream in made}) == 3
    reply_messages = glimmer.calls[1][1]
    assert reply_messages[0] == {"role": "system", "content": taken.thread} and reply_messages[1]["content"] == "text 1"
    assert NoDreams().dream("p1", NIGHT, None) == () and not NoDreams().available
    with pytest.raises(ValueError, match="kind"):
        _seed("speak", False, kind="nightmare")


def test_minds_night_receipt_maps_to_dream_seeds() -> None:
    """MIND's kept states (slot4's night receipt shape, faked here): a boundary is an imagined branch
    of her decision, a picked state an imagined continuation; the text comes from the real events."""
    receipt = {
        "state": "slept", "decision": {"speak": False},
        "boundaries": [{"state": "b1", "threads": [2, 4], "rests_on": ["e1", "e2"], "steps": 3, "speak": True, "slot": 4},
                       {"state": "b2", "threads": [1], "rests_on": ["e3"], "steps": 1, "speak": False, "slot": None},
                       {"state": "b3", "threads": [5], "rests_on": [], "steps": 2, "speak": True, "slot": 5}],
        "dreams": [{"state": "d1", "threads": [2], "rests_on": ["e1"]}, {"state": "d2", "threads": [3], "rests_on": []}],
    }
    read: list[tuple[str, ...]] = []

    def describe(event_ids: Sequence[str]) -> tuple[str, frozenset[str] | None]:
        read.append(tuple(event_ids))
        return f"Open: {' and '.join(event_ids)}", frozenset({"person"})

    seeds = seeds_from_dream(receipt, describe)
    assert [(s.state_ref, s.kind) for s in seeds] == [("b1", "boundary"), ("b2", "boundary"), ("d1", "picked")]
    assert read == [("e1", "e2"), ("e3",), ("e1",)]  # a state resting on no real event is dropped unread
    speak, wait, picked = seeds
    assert all(s.hypothetical for s in seeds) and speak.event_ids == ("e1", "e2") and speak.ancestry == frozenset({"person"})
    assert (speak.branch, wait.branch, picked.branch) == (SPEAK, WAIT, "")
    assert speak.slots == (4, 2) and wait.slots == (1,) and picked.slots == (2,)  # the turned-to thread first
    assert state_text(speak) == f"Open: e1 and e2\n{HYPOTHETICAL}: {SPEAK}" and "4" not in state_text(speak)
    assert state_text(picked) == f"Open: e1\n{HYPOTHETICAL}"  # a dream is never told as the day
    assert seeds_from_dream({"state": "slept"}, describe) == ()  # a night that kept no dream
