"""Who plays full practice's roles (``ANJO_SLEEP_PRACTICE_ROLES``). Unset, Spark Open plays all three,
as every night has. The setting is comma-separated ``role=player`` pairs:

- ``person=glimmer``: Glimmer's base model writes the scenes (the person's messages) on the serving
  GPU that is awake for the night anyway, at the person's cap, instead of a Spark Open call. ``glimmer``
  alone means this.
- ``critic=deepseek``: DeepSeek V4.1 Flash on Fireworks, non-thinking, scores the candidates of a
  synthetic person only; a real person's critic stays Spark Open. A critic that may not train is safe
  here because its words never travel: the teacher writes from F and the message alone, and the
  critic's reply is read for its scores and the fixed category (the letters that miss a constraint)
  only (``critics``). A cut verdict drops its scene. Gemini is never a critic: it is the A/B's judge.
- ``teacher=lazy``: Spark Open stays the teacher, asked only for a scene where every Glimmer candidate
  misses a constraint, the only scene where it can become the target; the critic then judges again with
  it in. A (best, worst) pair with the teacher as the worst is lost where it is never asked.

Glimmer never grades its own replies: the critic is Spark Open or DeepSeek (Kevin, 2026-10-28; the
coordinator, 2026-10-29). Each switch becomes a default only after ``tools.practice_ab`` shows it isn't worse.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, get_args

ROLES_ENV: Final = "ANJO_SLEEP_PRACTICE_ROLES"
Person = Literal["spark", "glimmer"]
Critic = Literal["spark", "deepseek"]
Teacher = Literal["always", "lazy"]
_CHOICES: Final[Mapping[str, tuple[str, ...]]] = {
    "person": get_args(Person), "critic": get_args(Critic), "teacher": get_args(Teacher)}
_SHORTHAND: Final = {"glimmer": "person=glimmer"}


class RolesError(ValueError):
    """``ANJO_SLEEP_PRACTICE_ROLES`` names a role or a player practice doesn't have."""


@dataclass(frozen=True)
class Roles:
    person: Person = "spark"
    critic: Critic = "spark"
    teacher: Teacher = "always"


TODAY: Final = Roles()


def practice_roles(environ: Mapping[str, str] | None = None) -> Roles:
    """The roles the night's practice runs with; today's when the setting is unset or blank."""
    env = os.environ if environ is None else environ
    raw = env.get(ROLES_ENV, "").strip()
    raw = _SHORTHAND.get(raw, raw)
    found: dict[str, str] = {}
    for part in filter(None, (piece.strip() for piece in raw.split(","))):
        role, _, player = (side.strip() for side in part.partition("="))
        if role not in _CHOICES or player not in _CHOICES[role] or role in found:
            choices = "; ".join(f"{name}={'|'.join(options)}" for name, options in _CHOICES.items())
            raise RolesError(f"{ROLES_ENV} is comma-separated role=player pairs, each role once: {choices}")
        found[role] = player
    return Roles(**found)  # type: ignore[arg-type]
