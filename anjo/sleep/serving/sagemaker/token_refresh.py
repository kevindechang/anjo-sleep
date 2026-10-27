"""The dev host's AWS identity, with no AWS key anywhere: a Google ID token for the VM's service
account, written where the AWS SDK reads it (``AWS_WEB_IDENTITY_TOKEN_FILE``).

    python -m anjo.sleep.serving.sagemaker.token_refresh --out <token file>

The GCP metadata server mints the token (audience ``AUDIENCE``). With ``AWS_ROLE_ARN`` naming the
``anjo-sleep-devhost`` role, boto3 trades it for short-lived credentials through
``AssumeRoleWithWebIdentity``, against Google's built-in provider ``accounts.google.com``; the
role's trust policy accepts only this service account and this audience (deploy/aws). A token
lasts an hour, and botocore reads the file again each time it refreshes its credentials, so a
systemd timer runs this every 10 minutes (deploy/dev_host.md).

The file is replaced atomically, mode 600. The token is never printed or logged.
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Final

import httpx

AUDIENCE: Final = "anjo-sleep-devhost-aws"
METADATA_URL: Final = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
TIMEOUT_S: Final = 5.0
_JWT = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


class TokenError(RuntimeError):
    """No token could be fetched or written. The message never holds the token."""


def fetch(*, transport: httpx.BaseTransport | None = None) -> str:
    """A fresh ID token from the metadata server (reachable only from the VM)."""
    try:
        with httpx.Client(transport=transport, timeout=TIMEOUT_S) as http:
            reply = http.get(METADATA_URL, params={"audience": AUDIENCE, "format": "full"},
                             headers={"Metadata-Flavor": "Google"})
    except httpx.HTTPError as error:
        raise TokenError(f"the metadata server did not answer ({type(error).__name__})") from None
    if reply.status_code != 200:
        raise TokenError(f"the metadata server returned HTTP {reply.status_code}")
    token: str = reply.text.strip()
    if not _JWT.fullmatch(token):
        raise TokenError("the metadata server sent something that is not a token")
    return token


def write(token: str, out: Path) -> None:
    """Replace ``out`` atomically with the token, readable by its owner only."""
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, pending = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.")
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="ascii") as file:
            file.write(token)
        os.replace(pending, out)
    except OSError as error:
        Path(pending).unlink(missing_ok=True)
        raise TokenError(f"the token file could not be written ({type(error).__name__})") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the VM's Google ID token for AWS web identity.")
    parser.add_argument("--out", type=Path, required=True, help="the file AWS_WEB_IDENTITY_TOKEN_FILE names")
    args = parser.parse_args(argv)
    try:
        write(fetch(), args.out)
    except TokenError as error:
        raise SystemExit(str(error)) from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
