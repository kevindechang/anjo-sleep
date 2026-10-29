# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False, disallow-any-generics=False, warn-return-any=False
"""taste_ab by log-probability forced choice (Task 2): both orders, no judge, no sampling."""
from __future__ import annotations

from dataclasses import replace

import pytest

from anjo.sleep.probes.logprob import LABELS, TasteChoice, expected_margin, forced_choice, present
from anjo.sleep.tests.sleep_fakes import FakeLogprobs, ckpt, options, taste


def _shown_first(messages) -> str:
    text = messages[-1]["content"]
    return text.split("A. ", 1)[1].split("\n", 1)[0]


def _picks(option: str):
    """A model that always wants the option's text, whichever letter it sits under."""
    def prefers(adapter, messages):
        text = messages[-1]["content"]
        a = text.split("A. ", 1)[1].split("\n", 1)[0]
        return "A" if a == option else "B"
    return prefers


def test_taste_scored_by_logprob_in_both_orders():
    p = taste()
    client = FakeLogprobs(_picks(options(p)[0]))
    choice = forced_choice(client, ckpt("p1"), None, p)
    shown = [_shown_first(c["messages"]) for c in client.calls]
    assert shown == [options(p)[0]] * 2 + [options(p)[1]] * 2
    assert {c["continuation"] for c in client.calls} == set(LABELS)
    assert choice.picked == options(p)[0] and not choice.flipped
    assert choice.correct(p.expected) is True


def test_picking_the_other_option_in_both_orders_is_wrong():
    p = taste()
    choice = forced_choice(FakeLogprobs(_picks(options(p)[1])), None, None, p)
    assert choice.picked == options(p)[1] and choice.correct(p.expected) is False


def test_an_order_flip_is_a_tie():
    p = taste()
    choice = forced_choice(FakeLogprobs(lambda adapter, messages: "A"), None, None, p)  # always "A"
    assert choice.picked is None and choice.flipped
    assert choice.correct(p.expected) is None


def test_equal_logprobs_in_one_order_are_a_tie():
    class Flat:
        def continuation_logprobs(self, adapter, messages, continuation):
            return (-1.0,)

    assert forced_choice(Flat(), None, None, taste()).picked is None


def test_label_logprob_sums_the_tokens_of_the_label():
    class TwoTokens:
        def continuation_logprobs(self, adapter, messages, continuation):
            return (-0.5, -0.5) if continuation == "A" else (-0.9,)

    choice = forced_choice(TwoTokens(), None, None, taste())
    assert choice.order_1 == (-1.0, -0.9)


def test_system_context_goes_first_and_options_are_in_the_user_turn():
    p = taste()
    msgs = present(p, (0, 1), "CTX")
    assert msgs[0] == {"role": "system", "content": "CTX"} and msgs[1]["role"] == "user"
    assert options(p)[0] in msgs[1]["content"] and p.prompt in msgs[1]["content"]
    assert present(p, (0, 1), None)[0]["role"] == "user"


def test_expected_margin_leans_toward_the_first_option_in_both_orders():
    p = taste()  # expected is options[0]: label A in (0, 1), label B in (1, 0)
    choice = TasteChoice((-0.1, -2.0), (-2.0, -0.1), options(p)[0])
    assert expected_margin(choice, p) == pytest.approx(1.9)


def test_expected_margin_leans_toward_the_second_option_in_both_orders():
    base = taste()
    p = replace(base, expected=options(base)[1])  # label B in (0, 1), label A in (1, 0)
    choice = TasteChoice((-2.0, -0.1), (-0.1, -2.0), options(p)[1])
    assert expected_margin(choice, p) == pytest.approx(1.9)


def test_expected_margin_averages_the_two_orders():
    p = taste()  # +1.0 in (0, 1), -2.0 in (1, 0)
    choice = TasteChoice((0.0, -1.0), (-1.0, -3.0), options(p)[0])
    assert expected_margin(choice, p) == pytest.approx(-0.5)


def test_expected_margin_is_none_without_an_expected_option():
    p = taste()
    choice = TasteChoice((-0.1, -2.0), (-2.0, -0.1), options(p)[0])
    assert expected_margin(choice, replace(p, expected=None)) is None
    assert expected_margin(choice, replace(p, expected="not one of the options")) is None
