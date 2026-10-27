# mypy: disallow-untyped-decorators=False
# (pyproject's follow_imports = "skip" hides pytest's own types, so its decorators look untyped)
"""Tests are offline in both repos: the private guard, or the public repo's own (offline.py)."""
from __future__ import annotations

import importlib.util
import socket
from typing import Any

import pytest

from anjo.sleep.tests import offline

PRIVATE_ANJO = importlib.util.find_spec("anjo.core") is not None


def test_the_public_guard_refuses_non_loopback_traffic(monkeypatch: pytest.MonkeyPatch) -> None:
    reached: list[Any] = []

    def connect(self: socket.socket, address: Any) -> None:
        reached.append(address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        reached.append(address)
        return 0

    def sendto(self: socket.socket, data: bytes, *args: Any) -> int:
        reached.append(args[-1])
        return len(data)

    # Stand-ins for the real calls, so the test checks this guard alone and opens nothing.
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    offline.guard_sockets(monkeypatch)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        with pytest.raises(offline.NetworkBlocked, match="192.0.2.1"):
            tcp.connect(("192.0.2.1", 443))
        with pytest.raises(offline.NetworkBlocked):
            tcp.connect_ex(("192.0.2.1", 443))
        tcp.connect(("127.0.0.1", 9))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        with pytest.raises(offline.NetworkBlocked):
            udp.sendto(b"x", ("192.0.2.1", 53))
    assert reached == [("127.0.0.1", 9)]


def test_a_non_loopback_connect_raises() -> None:
    with pytest.raises(OSError) as caught:
        socket.create_connection(("192.0.2.1", 9), timeout=0.5)
    assert type(caught.value).__name__ == "NetworkBlocked"


def test_the_public_guard_steps_aside_for_the_private_one() -> None:
    if not PRIVATE_ANJO:
        pytest.skip("the private guard exists only in the private repo")
    assert offline.private_guard_active()
    assert socket.socket.connect.__module__ == "tests.network_guard"
