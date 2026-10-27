"""The live wiring: your own keys, Baseten, the Meta API, Modal to train and Modal to serve.

Every key is read from the environment here, once, and goes only into the clients that need it.
A missing one stops the run before anything is created or paid for, naming the variable (never a
value). ``--fake`` needs none of this.
"""
from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from anjo.sleep import baseten_api, meta_api
from anjo.sleep.examples.synthetic.modal_train import ModalTrainRunner
from anjo.sleep.examples.synthetic.services import ExampleError, Services
from anjo.sleep.serving.backend import MODAL_TOKEN_ENV, MODAL_URL_ENV, ModalServing
from anjo.sleep.serving.config import REVISION_ENV

BASETEN_KEYS: Final = ("BASETEN_API_KEY", baseten_api.KEY_ENV)
MODAL_CONFIG_ENV: Final = "MODAL_CONFIG_PATH"
NEEDED: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (meta_api.KEY_ENV, (meta_api.KEY_ENV,)),
    ("BASETEN_API_KEY", BASETEN_KEYS),
    (MODAL_URL_ENV, (MODAL_URL_ENV,)),
    (MODAL_TOKEN_ENV, (MODAL_TOKEN_ENV,)),
    (REVISION_ENV, (REVISION_ENV,)),
)
HELP: Final = (
    "Set them (see 'Run it yourself' in the README), or run with --fake to see the whole flow with no keys."
)


class MissingKeys(ExampleError):
    """One or more settings are unset. The message lists their names only."""


def modal_logged_in(env: Mapping[str, str]) -> bool:
    """Modal credentials: both token variables, or a login from ``modal setup`` (``~/.modal.toml``,
    or the file ``MODAL_CONFIG_PATH`` names)."""
    if env.get("MODAL_TOKEN_ID", "").strip() and env.get("MODAL_TOKEN_SECRET", "").strip():
        return True
    path = Path(env.get(MODAL_CONFIG_ENV, "").strip() or Path.home() / ".modal.toml")
    try:
        return "token_id" in path.read_text(encoding="utf-8")
    except OSError:
        return False


def missing_keys(env: Mapping[str, str]) -> list[str]:
    """The names of the settings that are unset or blank."""
    absent = [label for label, names in NEEDED if not any(env.get(name, "").strip() for name in names)]
    return absent if modal_logged_in(env) else ["MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (or run `modal setup`)", *absent]


def build_services(environ: Mapping[str, str] | None = None) -> Services:
    """The live ``Services``, or ``MissingKeys`` before anything else happens."""
    env = os.environ if environ is None else environ
    missing = missing_keys(env)
    if missing:
        raise MissingKeys(f"missing {', '.join(missing)}. {HELP}")
    if importlib.util.find_spec("modal") is None:
        raise ExampleError("the modal package is not installed: pip install \".[modal]\", or run with --fake")
    if not env.get(baseten_api.KEY_ENV, "").strip():
        os.environ[baseten_api.KEY_ENV] = env["BASETEN_API_KEY"].strip()  # baseten_api reads its own name
    serving = ModalServing(env[MODAL_URL_ENV].strip(), env[MODAL_TOKEN_ENV].strip())
    return Services(
        spark=baseten_api.chat, judge=meta_api.chat, glimmer=serving.client(), serving=serving,
        gpu=ModalTrainRunner(), revision=env[REVISION_ENV].strip(),
    )
