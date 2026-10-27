"""Field checks shared by the inquiry's frozen types (``types`` and ``ops``). Pure.

Each ``check_*`` raises ``ValueError`` (``TypeError`` for a wrong container) naming the field,
never its value, so a model's text never reaches a log through an error message. Each
``freeze_*`` copies a field into an immutable value on a frozen dataclass while it is built.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from types import MappingProxyType

def check_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def check_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def check_count(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a whole number, at least 0")


def check_unit(name: str, value: object) -> None:
    if not 0.0 <= check_number(name, value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def check_one_of(name: str, value: object, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")


def freeze_ids(obj: object, name: str, *, empty_ok: bool = True) -> None:
    """Copy a sequence of ids into a tuple; a bare string is refused, not split into letters."""
    raw = getattr(obj, name)
    if isinstance(raw, str):
        raise TypeError(f"{name} must be a sequence of ids, not a string")
    value = tuple(raw)
    if not value and not empty_ok:
        raise ValueError(f"{name} must not be empty")
    for item in value:
        check_text(name, item)
    object.__setattr__(obj, name, value)


def freeze_tuple(obj: object, name: str, kind: type | tuple[type, ...]) -> None:
    raw = getattr(obj, name)
    if isinstance(raw, str):
        raise TypeError(f"{name} must be a sequence, not a string")
    value = tuple(raw)
    if not all(isinstance(item, kind) for item in value):
        raise TypeError(f"{name} holds an item of the wrong type")
    object.__setattr__(obj, name, value)


def freeze_map(obj: object, name: str, check: Callable[[str, object], object]) -> None:
    """Copy a mapping of hypothesis id to number into a read-only mapping, checking each value."""
    raw = getattr(obj, name)
    if raw is None:
        return
    value = dict(raw)
    for key, number in value.items():
        check_text(f"{name} key", key)
        check(f"a value in {name}", number)
    object.__setattr__(obj, name, MappingProxyType({key: float(v) for key, v in value.items()}))


def check_positive(name: str, value: object) -> None:
    if check_number(name, value) <= 0.0:
        raise ValueError(f"{name} must be positive")


