"""One keep-alive ping for a Colab session, run on the ``colab`` CLI's own python, isolated:

    <the CLI's python> -I colab_ping.py <session>

2026-11-02: a Colab A100 running a training ``exec`` was reclaimed about 35 minutes in, while the
local ``colab exec`` waited on with no connection, and the CLI (google-colab-cli 0.7.4) sends no
keep-alive. This is the one Google's CLI design notes describe: ``GET <colab>/tun/m/<endpoint>/
keep-alive/`` with ``X-Colab-Tunnel: Google`` on the user's OAuth session, a read timeout counting
as a ping. ``ColabRunner`` runs it every ``KEEPALIVE_PERIOD_S`` while a training ``exec`` runs.

It imports only ``colab_cli`` and the standard library, so it runs in the CLI's own environment.
Exit 0: pinged (or the read timed out). Exit 2: a bad command line. Exit 3: no session by that
name on this machine. Any other failure is left uncaught, so the process exits 1. It prints
nothing on success.
"""
import sys
from urllib.parse import urljoin

from colab_cli import client  # type: ignore[import-not-found]  # the CLI's environment, not this repo's
from colab_cli.common import State  # type: ignore[import-not-found]

HEADERS = {"X-Colab-Tunnel": "Google"}
REQUEST_TIMEOUT_S = 20.0  # under the runner's 30-second hard timeout for the whole process
# The CLI's OAuth session is requests' (``colab_cli.client`` imports it): a read that timed out
# reached Colab, which is all a keep-alive needs. A connect timeout is a failure.
READ_TIMEOUT = client.requests.exceptions.ReadTimeout
USAGE = 2
UNKNOWN_SESSION = 3


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: colab_ping.py <session>", file=sys.stderr)
        return USAGE
    state = State()
    session = state.store.get(argv[0])
    if session is None:
        print("no such session on this machine", file=sys.stderr)
        return UNKNOWN_SESSION
    url = urljoin(state.client.colab_domain, f"{client.TUN_ENDPOINT}/{session.endpoint}/keep-alive/")
    try:
        state.client._issue_request(url, headers=dict(HEADERS), timeout=REQUEST_TIMEOUT_S)
    except READ_TIMEOUT:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
