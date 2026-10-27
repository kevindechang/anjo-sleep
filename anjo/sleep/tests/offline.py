"""The public repo's network guard: tests never reach a non-loopback address.

The private repo loads ``tests.network_guard`` through pytest's ``addopts``; the public repo has
no such plugin, so ``anjo/sleep/tests/conftest.py`` applies this one to every test unless the
private guard is already active. A TCP or UDP connect or ``sendto`` to a non-loopback address
raises ``NetworkBlocked``; loopback and Unix sockets stay open. DNS lookups are not intercepted.
"""
from __future__ import annotations

import ipaddress
import socket
import sys
from typing import Any

import pytest


class NetworkBlocked(OSError):
    """A test talked to a non-loopback address."""


def _is_loopback(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check(sock: socket.socket, address: object) -> None:
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if not _is_loopback(host):
        raise NetworkBlocked(f"tests are offline: blocked traffic to {address!r}; fake the call")


def private_guard_active() -> bool:
    guard = sys.modules.get("tests.network_guard")
    return guard is not None and bool(guard.is_active())


def guard_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse non-loopback traffic until ``monkeypatch`` undoes it at the end of the test."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto

    def connect(self: socket.socket, address: Any) -> None:
        _check(self, address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        _check(self, address)
        return real_connect_ex(self, address)

    def sendto(self: socket.socket, data: Any, *args: Any) -> int:
        _check(self, args[-1])  # sendto(data, address) or sendto(data, flags, address)
        return int(real_sendto(self, data, *args))

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
