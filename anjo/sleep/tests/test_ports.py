# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Every port in spec §12.1 is a Protocol with its full method signatures (Tasks 2, 3, 5, 7a, 17)."""
from __future__ import annotations

import inspect
from typing import Protocol

import pytest

from anjo.sleep import ports

# port -> method -> parameter names after self
SIGNATURES: dict[str, dict[str, tuple[str, ...]]] = {
    "EvidenceSource": {"extract_day": ("user_id", "day", "window")},
    "EvidenceStore": {
        "add": ("items",),
        "for_day": ("day",),
        "trainable": ("until",),
        "stored": ("until",),
        "held_out": ("day",),
        "search": ("query", "as_of", "k"),
        "export": (),
    },
    "TargetStore": {
        "upsert": ("target",),
        "open_targets": ("as_of",),
        "mark_acted": ("target_id", "revision"),
        "add_result": ("result",),
        "results": (),
        "settle": ("target_id",),
        "record_expectation": ("expectation",),
        "resolve_expectation": ("expectation_id", "outcome"),
        "expectations": (),
        "digest": (),
        "replay_digest": (),
        "export": (),
    },
    "ContextBuilder": {
        "build": ("user_id", "probe"),
        "recall": ("user_id", "query", "k", "budget_tokens"),
    },
    "Notifier": {"queue": ("user_id", "message", "source"), "push": ("user_id", "message")},
    "SafetyGate": {"check": ("ckpt",)},
    "Decontaminator": {"check": ("texts", "sealed")},
    "GpuRunner": {"run": ("job", "inputs", "timeout_s")},
    "IntentionSource": {"intentions": ("user_id", "as_of")},
    "Chooser": {"choose": ("opportunity", "wants", "ctx"), "derive": ("snapshot",)},
}


def _methods(port: type) -> set[str]:
    """Public methods only: typing.Protocol adds its own ``__init__``."""
    return {
        name for name, value in vars(port).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }


def test_every_port_in_the_spec_exists() -> None:
    assert set(SIGNATURES) <= set(vars(ports))


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_port_is_a_protocol_with_its_methods(name: str) -> None:
    port = getattr(ports, name)
    assert Protocol in port.__mro__, f"{name} is not a typing.Protocol"
    assert _methods(port) == set(SIGNATURES[name])
    for method, params in SIGNATURES[name].items():
        signature = inspect.signature(getattr(port, method))
        assert tuple(signature.parameters)[1:] == params, f"{name}.{method}"
        assert signature.return_annotation is not inspect.Signature.empty, f"{name}.{method}"


@pytest.mark.parametrize("name", sorted(SIGNATURES))
def test_port_is_checkable_at_runtime(name: str) -> None:
    port = getattr(ports, name)
    stub = type(f"Fake{name}", (), {m: (lambda self, *a, **k: None) for m in SIGNATURES[name]})
    assert isinstance(stub(), port)
    assert not isinstance(object(), port)
