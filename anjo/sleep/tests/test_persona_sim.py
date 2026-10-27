from pathlib import Path

from anjo.sleep.personas import model as m
from anjo.sleep.sim import persona_sim as sim

PERSONAS = Path(m.__file__).parent


class FakeModel:
    def __init__(self, end_after=None):
        self.systems, self.histories, self.end_after = [], [], end_after

    def complete(self, system, messages, *, temperature, seed):
        self.systems.append(system)
        self.histories.append([dict(x) for x in messages])
        n = len(self.histories)
        if self.end_after is not None and n > self.end_after:
            return "[END]"
        return f"message {n} about my film"


class FakeTransport:
    def __init__(self):
        self.sent = []

    def send(self, session_id, text, *, cleared):
        self.sent.append((session_id, text, cleared))
        return f"reply to {text}"


def run(name="film_student", day=3, seed=5, **kw):
    pf = m.load_person_file(PERSONAS / f"{name}.json")
    model, transport = FakeModel(**kw), FakeTransport()
    turns = sim.simulate_day(pf.visible, day, seed, model=model, transport=transport)
    return pf, model, transport, turns


def test_simulator_never_receives_held_out():
    pf, model, transport, turns = run()
    seen = " ".join(model.systems + [x["content"] for h in model.histories for x in h])
    seen += " ".join(t.text for t in turns) + " ".join(s[1] for s in transport.sent)
    for s in pf.held_out.preferences:
        assert s.text not in seen
    for w in pf.held_out.new_work:
        assert w.text not in seen
    for t in pf.held_out.taste_ab:
        assert t.prompt not in seen
    assert sim.simulate_day.__annotations__["person"] == "Persona"


def test_held_out_absent_from_history_memory_and_context_after_a_simulated_day():
    for name in ("film_student", "film_student_donor"):
        for day in range(8):
            pf, model, transport, turns = run(name, day)
            blob = " ".join(t.text for t in turns) + " ".join(model.systems)
            held = [s.text for s in pf.held_out.preferences] + [
                w.text for w in pf.held_out.new_work
            ]
            assert not [h for h in held if h in blob]


def test_every_request_is_in_a_cleared_session():
    _, _, transport, _ = run()
    assert transport.sent and all(c is True for _, _, c in transport.sent)
    assert len({s for s, _, _ in transport.sent}) == 1


def test_turns_alternate_and_are_numbered():
    _, _, _, turns = run()
    assert [t.role for t in turns[:4]] == ["user", "assistant", "user", "assistant"]
    assert [t.index for t in turns] == list(range(len(turns)))
    assert 4 <= len(turns) // 2 <= 9


def test_day_ends_when_model_says_end():
    _, _, transport, turns = run(end_after=2)
    assert len(turns) == 4 and len(transport.sent) == 2


def test_same_seed_same_day_is_reproducible():
    a = run(seed=9)[1].systems
    assert a == run(seed=9)[1].systems
    assert a != run(seed=10)[1].systems


def test_dated_plan_appears_on_its_day():
    pf, model, _, _ = run(day=3)
    plan = next(p for p in pf.visible.plans if p.day == 3)
    assert plan.text in model.systems[0]
    _, model2, _, _ = run(day=4)
    assert plan.text not in model2.systems[0]


def test_model_history_starts_with_user_turn_and_alternates():
    _, model, _, _ = run()
    last = model.histories[-1]
    assert [x["role"] for x in last][:3] == ["user", "assistant", "user"]
