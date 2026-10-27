"""The visit token and the client bucket (Task 13a).

A visit starts once the visitor confirms they are 18 or older. The server then mints a visit token,
``v1.<nonce>.<issued>.<mac>``: an HMAC over a random nonce and the time, and nothing else. Every
chat route needs a valid token under ``VISIT_TTL_S`` old, so the confirmation is enforced on the
server, not only in the page.

``client_key`` is what the limits count a client by: an HMAC of the address, so the saved counters
hold no address. An IPv6 address counts by its /64, the block one subscriber usually holds. Every
address that isn't globally routable (private, loopback, link-local, shared, documentation, or not
an address at all) shares one bucket. On the public host such an address is a proxy's own or a
forged one, and it must never be a way around the per-client limits: if the proxy stops passing
real addresses, every visitor lands in that one bucket and the limits close, rather than open.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
from datetime import datetime
from typing import Final

VISIT_TTL_S: Final = 2 * 60 * 60
CLOCK_SKEW_S: Final = 60
NON_GLOBAL: Final = "non-global"
_VERSION: Final = "v1"
_NONCE = re.compile(r"[A-Za-z0-9_-]{16,64}")


def _mac(secret: bytes, *parts: str) -> str:
    if not secret:
        raise ValueError("the demo key is empty")
    return hmac.new(secret, "\x1f".join(parts).encode(), hashlib.sha256).hexdigest()[:32]


def _bucket(address: str) -> str:
    try:
        ip = ipaddress.ip_address(address.strip())
    except ValueError:
        return NON_GLOBAL
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        mapped = ip.ipv4_mapped
        return str(mapped) if mapped.is_global else NON_GLOBAL
    if not ip.is_global:
        return NON_GLOBAL
    if isinstance(ip, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def client_key(address: str, secret: bytes) -> str:
    """The keyed bucket the limits count ``address`` in."""
    return _mac(secret, "client", _bucket(address))


def mint_visit(secret: bytes, nonce: str, now: datetime) -> str:
    if not _NONCE.fullmatch(nonce):
        raise ValueError("a visit nonce is 16 to 64 URL-safe characters")
    issued = str(int(now.timestamp()))
    return f"{_VERSION}.{nonce}.{issued}.{_mac(secret, 'visit', nonce, issued)}"


def visit_nonce(token: str, secret: bytes, now: datetime) -> str | None:
    """The visit's nonce when ``token`` is one this server minted and it hasn't expired, else None."""
    parts = token.split(".")
    if len(parts) != 4 or parts[0] != _VERSION:
        return None
    _version, nonce, issued, mac = parts
    if not _NONCE.fullmatch(nonce) or not issued.isdigit():
        return None
    if not hmac.compare_digest(mac, _mac(secret, "visit", nonce, issued)):
        return None
    age = now.timestamp() - int(issued)
    if age < -CLOCK_SKEW_S or age > VISIT_TTL_S:
        return None
    return nonce
