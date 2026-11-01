# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
"""The final lives' preparation (Task 12b steps 2-4): the freeze registration, the reveal and draw,
the expansion into persona files and the sealed probes. Tmp files, a tmp ledger and fakes only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anjo.sleep.budget import Ledger
from anjo.sleep.lives import draw_cli, prep
from anjo.sleep.lives.draw import LIVES_PATH, draw_lives
from anjo.sleep.model_api import ChatResult
from anjo.sleep.personas.model import load_person_file
from anjo.sleep.prereg import ledger, register_freeze
from anjo.sleep.probes import seal_cli
from anjo.sleep.probes.generate import COUNTS, load_sealed
from anjo.sleep.sim import expand_cli
from anjo.sleep.tests.test_expand import Client
from anjo.sleep.tests.test_probe_generate import FakeChat

SEED_RAW = b"x7Qp-ZkR9mW2vL0aB3nC8dE5fG1hJ4kL6mN0pQ2rS4tU6vW8yZ0aB2cD4eF6gHi\n"  # 64 bytes, not hex
LIVES_SHA = ledger.sha256_file(LIVES_PATH)
PRICES = {"ANJO_SLEEP_PRICES": "1.0,4.0,2.0,2.0"}


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ANJO_SLEEP_SPARK", raising=False)
    seed = tmp_path / "draw_seed.txt"
    seed.write_bytes(SEED_RAW)
    ledger.register("seed_hash", hashlib.sha256(SEED_RAW).hexdigest(), "sealed until Nov 1")
    ledger.register("lives", LIVES_SHA, "lives.json (before night 1)")
    return tmp_path


def _draw(data):
    return draw_cli.run(data / "draw_seed.txt", hashlib.sha256(SEED_RAW).hexdigest(), LIVES_SHA)


# ── the reveal and the draw ──────────────────────────────────────────────────


def test_seed_bytes_become_one_big_endian_integer_without_the_newline():
    assert len(SEED_RAW) == 64
    assert draw_cli.seed_from_bytes(SEED_RAW) == int.from_bytes(SEED_RAW[:-1], "big")
    assert draw_cli.seed_from_bytes(SEED_RAW) == draw_cli.seed_from_bytes(SEED_RAW[:-1])
    with pytest.raises(draw_cli.DrawRefused):
        draw_cli.seed_from_bytes(b"\n")


def test_draw_registers_reveal_and_draw_with_the_donor_rule(data):
    out = _draw(data)
    expected = draw_lives(draw_cli.seed_from_bytes(SEED_RAW), expected_sha256=LIVES_SHA)
    assert out["pair"] == [expected.pair[0].id, expected.pair[1].id] and out["single"] == expected.single.id
    assert out["donors"] == expected.donors
    kinds = [e.kind for e in ledger.read_entries()]
    assert kinds == ["seed_hash", "lives", "seed_reveal", "final_draw"]
    reveal, entry = ledger.read_entries()[2:]
    assert reveal.sha256 == hashlib.sha256(SEED_RAW).hexdigest()
    assert json.loads(reveal.note)["seed_text"] == SEED_RAW[:-1].decode()
    assert entry.sha256 == hashlib.sha256(entry.note.encode()).hexdigest()
    got = draw_cli.registered_draw()
    assert got.lives == (out["pair"][0], out["pair"][1], out["single"]) and got.donors == out["donors"]


def test_dry_run_checks_and_writes_nothing(data):
    before = ledger.read_entries()
    out = draw_cli.run(data / "draw_seed.txt", hashlib.sha256(SEED_RAW).hexdigest(), LIVES_SHA, dry_run=True)
    assert out["dry_run"] is True and ledger.read_entries() == before


@pytest.mark.parametrize("bad", ["hash", "lives_sha", "no_seed_entry", "no_lives_entry"])
def test_draw_refuses_any_mismatch_before_writing(data, bad):
    if bad in ("no_seed_entry", "no_lives_entry"):
        path = ledger.ledger_path()
        path.unlink()
        ledger.register("lives" if bad == "no_seed_entry" else "seed_hash",
                        LIVES_SHA if bad == "no_seed_entry" else hashlib.sha256(SEED_RAW).hexdigest(), "x")
    before = ledger.read_entries()
    registered = "0" * 64 if bad == "hash" else hashlib.sha256(SEED_RAW).hexdigest()
    lives = "1" * 64 if bad == "lives_sha" else LIVES_SHA
    with pytest.raises(draw_cli.DrawRefused):
        draw_cli.run(data / "draw_seed.txt", registered, lives)
    assert ledger.read_entries() == before


def test_the_draw_happens_once(data):
    _draw(data)
    with pytest.raises(draw_cli.DrawRefused, match="once"):
        _draw(data)
    assert draw_cli.main(["--seed-file", str(data / "draw_seed.txt"), "--registered-hash",
                          hashlib.sha256(SEED_RAW).hexdigest(), "--lives-sha", LIVES_SHA]) == 2


def test_registered_draw_refuses_an_edited_note(data):
    _draw(data)
    path = ledger.ledger_path()
    lines = path.read_text().splitlines()
    row = json.loads(lines[-1])
    row["note"] = row["note"].replace('"single":"', '"single":"Z')
    path.write_text("\n".join([*lines[:-1], json.dumps(row)]) + "\n")
    with pytest.raises((draw_cli.DrawRefused, ledger.LedgerError)):
        draw_cli.registered_draw()


def test_draw_cli_prints_json(data, capsys):
    assert draw_cli.main(["--seed-file", str(data / "draw_seed.txt"), "--registered-hash",
                          hashlib.sha256(SEED_RAW).hexdigest(), "--lives-sha", LIVES_SHA, "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


# ── the expansion ────────────────────────────────────────────────────────────


class FakeSpark:
    """``baseten_api.chat``'s shape, answering with the expand test's valid life file."""

    def __init__(self, broken_for: dict[str, int] | None = None):
        self.inner = Client()
        self.broken_for = dict(broken_for or {})
        self.calls: list[dict] = []

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None, max_tokens=None, seed=None):
        self.calls.append({"model": model, "max_tokens": max_tokens, "seed": seed, "temperature": temperature})
        result = self.inner.complete(None, messages, temperature=temperature or 0.0, seed=seed)
        life = json.loads(result.text)["visible"]["person_id"]
        if self.broken_for.get(life, 0) > 0:
            self.broken_for[life] -= 1
            return ChatResult("not json", (), model, 100, 200, "stop")
        return ChatResult(result.text, (), model, 100, 200, "stop")


def _expand(data, spark, budget=None):
    personas = data / "personas"
    personas.mkdir(exist_ok=True)
    return expand_cli.run(client=expand_cli.SparkOpenClient(spark), personas_dir=personas, budget=budget,
                          environ=PRICES), personas


def test_expansion_writes_loadable_personas_and_registers_them(data):
    _draw(data)
    spark = FakeSpark()
    budget = Ledger(data / "sleep" / "budget.jsonl")
    out, personas = _expand(data, spark, budget)
    lives = draw_cli.registered_draw().lives
    assert [o.life for o in out] == list(lives) and {o.status for o in out} == {"written"}
    for life in lives:
        pf = load_person_file(personas / f"{life}.json")
        assert pf.visible.person_id == life
    assert {c["seed"] for c in spark.calls} == {expand_cli.EXPAND_SEED}
    assert {c["max_tokens"] for c in spark.calls} == {expand_cli.EXPAND_MAX_OUT_TOKENS}
    assert [e.kind for e in ledger.read_entries()].count("final_persona") == 3
    assert not budget.open_holds() and sum(budget.settlements().values()) > 0
    again, _ = _expand(data, spark, budget)
    assert {o.status for o in again} == {"done"} and len(spark.calls) == 3


def test_a_refused_expansion_retries_on_the_next_seed_and_is_recorded(data):
    _draw(data)
    first = draw_cli.registered_draw().lives[0]
    spark = FakeSpark({first: 1})
    out, _ = _expand(data, spark)
    assert out[0].status == "written" and out[0].attempts == 2
    refused = [e for e in ledger.read_entries() if e.kind == "final_expand_refused"]
    assert len(refused) == 1 and json.loads(refused[0].note)["seed"] == expand_cli.EXPAND_SEED
    assert refused[0].sha256 == hashlib.sha256(b"not json").hexdigest()


def test_a_life_refused_every_attempt_is_not_written_and_exits_3(data):
    _draw(data)
    first = draw_cli.registered_draw().lives[0]
    out, personas = _expand(data, FakeSpark({first: 99}))
    assert out[0].status == "refused" and not (personas / f"{first}.json").exists()
    assert [o.status for o in out[1:]] == ["written", "written"]
    spark = FakeSpark()
    again, _ = _expand(data, spark)  # out of attempts across runs: no call is sent for it
    assert again[0].status == "refused" and spark.calls == []


def test_a_stray_persona_file_stops_the_run_before_any_call(data):
    _draw(data)
    personas = data / "personas"
    personas.mkdir()
    (personas / f"{draw_cli.registered_draw().lives[1]}.json").write_text("{}")
    spark = FakeSpark()
    with pytest.raises(expand_cli.ExpandRunRefused):
        _expand(data, spark)
    assert spark.calls == []


def test_expansion_without_a_draw_is_refused(data):
    with pytest.raises(expand_cli.ExpandRunRefused):
        _expand(data, FakeSpark())


def test_spark_client_refuses_an_adapter():
    with pytest.raises(ValueError):
        expand_cli.SparkOpenClient(FakeSpark()).complete("adapter", [], temperature=0.0)


def test_a_stale_final_prep_hold_is_settled_at_its_bound(data):
    budget = Ledger(data / "sleep" / "budget.jsonl")
    from anjo.sleep.night.holds import hold, usd
    hold(budget, f"{prep.HOLD_PREFIX}expand:x:s1", ("baseten_sleep_usd",), usd(0.5))
    with prep.swept(budget) as n:
        assert n == 1
    assert not budget.open_holds()


# ── the probes ───────────────────────────────────────────────────────────────


class MetaChat:
    """``meta_api.chat``'s shape over the probe test's fake writer."""

    def __init__(self, mutate=None):
        self.inner = FakeChat(mutate)
        self.max_tokens: set = set()

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None, max_tokens=None):
        self.max_tokens.add(max_tokens)
        return self.inner(model, messages, tools, timeout_s, temperature=temperature)


def _expanded(data):
    _draw(data)
    _, personas = _expand(data, FakeSpark())
    return personas


def test_probes_are_sealed_for_every_drawn_life_and_registered(data):
    personas = _expanded(data)
    sealed = data / "sealed"
    chat = MetaChat()
    budget = Ledger(data / "sleep" / "budget.jsonl")
    out = seal_cli.run(chat=seal_cli.MeteredChat(chat), budget=budget, personas_dir=personas,
                       sealed_dir=sealed, environ=PRICES)
    assert {o.status for o in out} == {"sealed"}
    for life in draw_cli.registered_draw().lives:
        probes = load_sealed(sealed / f"{life}.jsonl")
        assert {k: sum(p.kind == k for p in probes) for k in COUNTS} == COUNTS
        assert all(p.person_id == life for p in probes)
    assert chat.max_tokens == {seal_cli.PROBE_MAX_OUT_TOKENS}
    assert {c["model"] for c in chat.inner.calls} == {seal_cli.PROBE_MODEL}
    assert [e.kind for e in ledger.read_entries()].count("final_sealed") == 3
    assert not budget.open_holds()
    again = seal_cli.run(chat=seal_cli.MeteredChat(MetaChat()), budget=budget, personas_dir=personas,
                         sealed_dir=sealed, environ=PRICES)
    assert {o.status for o in again} == {"done"}


def test_an_unregistered_sealed_file_is_never_overwritten(data):
    personas = _expanded(data)
    sealed = data / "sealed"
    sealed.mkdir()
    stray = sealed / f"{draw_cli.registered_draw().lives[2]}.jsonl"
    stray.write_text("old\n")
    chat = MetaChat()
    with pytest.raises(seal_cli.SealRunRefused):
        seal_cli.run(chat=seal_cli.MeteredChat(chat), budget=None, personas_dir=personas, sealed_dir=sealed,
                     environ=PRICES)
    assert stray.read_text() == "old\n" and chat.inner.calls == []


def test_probes_refuse_an_edited_persona_file(data):
    personas = _expanded(data)
    path = personas / f"{draw_cli.registered_draw().lives[0]}.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(seal_cli.SealRunRefused):
        seal_cli.run(chat=seal_cli.MeteredChat(MetaChat()), budget=None, personas_dir=personas,
                     sealed_dir=data / "sealed", environ=PRICES)


def test_a_probe_refusal_is_recorded_and_not_sealed(data):
    personas = _expanded(data)
    out = seal_cli.run(chat=seal_cli.MeteredChat(MetaChat(lambda kind, rows: rows[:-1])), budget=None,
                       personas_dir=personas, sealed_dir=data / "sealed", environ=PRICES)
    assert {o.status for o in out} == {"refused"}
    assert [e.kind for e in ledger.read_entries()].count("final_probes_refused") == 3
    assert not (data / "sealed").exists() or not list((data / "sealed").iterdir())


def test_probes_need_a_sealed_dir(data, monkeypatch):
    monkeypatch.delenv("ANJO_SLEEP_SEALED_DIR", raising=False)
    with pytest.raises(seal_cli.SealRunRefused):
        seal_cli.run(chat=seal_cli.MeteredChat(MetaChat()), budget=None, environ=PRICES)


# ── the freeze registration ──────────────────────────────────────────────────


def test_freeze_registers_final_report_expansion_and_the_recipe(data):
    env = {"ANJO_SLEEP_OBJECTIVE": "sft_dpo_seed", "ANJO_SLEEP_CAPACITY": "1", "ANJO_SLEEP_JUDGE": "deepseek"}
    items = register_freeze.compute_items(env)
    kinds = {i.kind: i for i in items}
    assert {"final_report", "prompt_expand", "final_draw_cli", "final_expand_cli", "final_seal_cli",
            "donor_rule", "final_recipe"} <= set(kinds)
    assert not [i for i in items if i.missing]
    recipe = json.loads(kinds["final_recipe"].label)
    assert set(recipe) == set(register_freeze.RECIPE_ENV)
    assert recipe["ANJO_SLEEP_OBJECTIVE"] == "sft_dpo_seed" and recipe["ANJO_SLEEP_HARNESS"] is None
    assert kinds["final_recipe"].sha256 == hashlib.sha256(kinds["final_recipe"].label.encode()).hexdigest()
    assert kinds["final_report"].sha256 == ledger.sha256_file(Path(register_freeze.SLEEP) / "reports/final.py")


def test_freeze_registration_appends_once(data, monkeypatch):
    monkeypatch.setenv("ANJO_SLEEP_OBJECTIVE", "sft_dpo_seed")
    assert register_freeze.main(["--register"]) == 0
    n = len(ledger.read_entries())
    assert register_freeze.main(["--register"]) == 0
    assert len(ledger.read_entries()) == n
    notes = [e.note for e in ledger.read_entries() if e.kind == "final_recipe"]
    assert len(notes) == 1 and notes[0].endswith("(at the freeze)")
    assert register_freeze.main([]) == 0



# ── retries sample: attempt 1 greedy, attempts 2+ at the retry temperature ──


class GreedyBreaks(FakeSpark):
    """Spark Open whose greedy reply breaks a rule and whose sampled one doesn't; records every call."""

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None, max_tokens=None, seed=None):
        self.calls.append({"model": model, "messages": messages, "tools": tools, "timeout_s": timeout_s,
                           "temperature": temperature, "max_tokens": max_tokens, "seed": seed})
        if not temperature:
            return ChatResult("not json", (), model, 100, 200, "stop")
        return ChatResult(self.inner.complete(None, messages, temperature=temperature, seed=seed).text, (), model,
                          100, 200, "stop")


def test_expansion_attempt_1_is_the_greedy_request_and_retries_sample(data):
    from anjo.sleep.lives.draw import load_pairs
    from anjo.sleep.personas.model import PLANNED_NIGHTS
    from anjo.sleep.sim import expand
    _draw(data)
    spark = GreedyBreaks()
    out, personas = _expand(data, spark)
    assert {o.status for o in out} == {"written"} and {o.attempts for o in out} == {2}
    first = draw_cli.registered_draw().lives[0]
    pair = next(p for p in load_pairs() if any(lf.id == first for lf in p.lives))
    life = next(lf for lf in pair.lives if lf.id == first)
    assert spark.calls[0] == {
        "model": "meta-models/muse-spark-open-1026", "tools": None, "timeout_s": expand_cli.EXPAND_TIMEOUT_S,
        "messages": [{"role": "user", "content": expand._prompt(life, pair, PLANNED_NIGHTS)}],
        "temperature": 0.0, "max_tokens": expand_cli.EXPAND_MAX_OUT_TOKENS, "seed": expand_cli.EXPAND_SEED}
    assert spark.calls[1] == {**spark.calls[0], "temperature": expand_cli.RETRY_TEMPERATURE,
                              "seed": expand_cli.EXPAND_SEED + 1}
    entries = ledger.read_entries()
    refused = [json.loads(e.note) for e in entries if e.kind == "final_expand_refused"]
    assert [(n["attempt"], n["temperature"]) for n in refused] == [(1, 0.0)] * 3
    written = [json.loads(e.note) for e in entries if e.kind == "final_persona"]
    assert [(n["attempt"], n["temperature"], n["seed"]) for n in written] == \
        [(2, expand_cli.RETRY_TEMPERATURE, expand_cli.EXPAND_SEED + 1)] * 3
    assert load_person_file(personas / f"{first}.json").visible.person_id == first


def test_expansion_attempt_3_also_samples_at_the_retry_temperature(data):
    _draw(data)
    first = draw_cli.registered_draw().lives[0]
    spark = FakeSpark({first: 99})
    _expand(data, spark)
    temps = [(c["seed"], c["temperature"]) for c in spark.calls[:3]]
    assert temps == [(expand_cli.EXPAND_SEED, 0.0), (expand_cli.EXPAND_SEED + 1, expand_cli.RETRY_TEMPERATURE),
                     (expand_cli.EXPAND_SEED + 2, expand_cli.RETRY_TEMPERATURE)]
    notes = [json.loads(e.note) for e in ledger.read_entries() if e.kind == "final_expand_refused"]
    assert [n["temperature"] for n in notes] == [0.0, expand_cli.RETRY_TEMPERATURE, expand_cli.RETRY_TEMPERATURE]


class GreedyMeta(MetaChat):
    """The Meta API probe writer, its greedy replies one probe short, its sampled ones whole."""

    def __call__(self, model, messages, tools, timeout_s, *, temperature=None, max_tokens=None):
        self.inner.mutate = None if temperature else (lambda kind, rows: rows[:-1])
        return super().__call__(model, messages, tools, timeout_s, temperature=temperature, max_tokens=max_tokens)


def _seal(data, personas, chat):
    return seal_cli.run(chat=seal_cli.MeteredChat(chat), budget=None, personas_dir=personas,
                        sealed_dir=data / "sealed", environ=PRICES)


def test_probe_attempt_1_is_greedy_and_a_retry_samples_and_seals(data):
    personas = _expanded(data)
    chat = GreedyMeta()
    assert {o.status for o in _seal(data, personas, chat)} == {"refused"}
    assert {c["temperature"] for c in chat.inner.calls} == {0.0}
    assert len(chat.inner.calls) == 3  # each greedy attempt stops at its first, short reply
    assert {o.status for o in _seal(data, personas, chat)} == {"sealed"}
    later = chat.inner.calls[3:]
    assert len(later) == 3 * seal_cli.PROBE_CALLS and {c["temperature"] for c in later} == {seal_cli.RETRY_TEMPERATURE}
    entries = ledger.read_entries()
    refused = [json.loads(e.note) for e in entries if e.kind == "final_probes_refused"]
    assert [(n["attempt"], n["temperature"]) for n in refused] == [(1, 0.0)] * 3
    sealed = [json.loads(e.note) for e in entries if e.kind == "final_sealed"]
    assert [(n["attempt"], n["temperature"], n["seed"]) for n in sealed] == \
        [(2, seal_cli.RETRY_TEMPERATURE, seal_cli.PROBE_SEED)] * 3
    for life in draw_cli.registered_draw().lives:
        assert load_sealed(data / "sealed" / f"{life}.jsonl")


def test_probe_attempt_1_request_is_unchanged_and_attempt_3_samples(data):
    from anjo.sleep.probes.generate import TEMPERATURE, generate_probes
    personas = _expanded(data)
    first = draw_cli.registered_draw().lives[0]
    direct = FakeChat()
    generate_probes(load_person_file(personas / f"{first}.json"), seal_cli.PROBE_SEED, chat=direct,
                    model=seal_cli.PROBE_MODEL)
    chat = MetaChat(lambda kind, rows: rows[:-1])
    for _ in range(seal_cli.PROBE_ATTEMPTS):
        _seal(data, personas, chat)
    calls = chat.inner.calls
    assert calls[:1] == direct.calls[:1] and {c["temperature"] for c in calls[:3]} == {TEMPERATURE}
    by_attempt = [{c["temperature"] for c in calls[i * 3:(i + 1) * 3]} for i in range(3)]
    assert by_attempt == [{0.0}, {seal_cli.RETRY_TEMPERATURE}, {seal_cli.RETRY_TEMPERATURE}]
    notes = [json.loads(e.note) for e in ledger.read_entries() if e.kind == "final_probes_refused"]
    assert sorted((n["attempt"], n["temperature"]) for n in notes if n["life"] == first) == \
        [(1, 0.0), (2, seal_cli.RETRY_TEMPERATURE), (3, seal_cli.RETRY_TEMPERATURE)]
