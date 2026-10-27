# mypy: disallow-untyped-decorators=False, disallow-untyped-defs=False, disallow-incomplete-defs=False, disallow-untyped-calls=False
import base64
import json

import pytest

from anjo.sleep.prereg import ledger

H = "a" * 64


@pytest.fixture
def path(tmp_path, monkeypatch):
    monkeypatch.setenv("ANJO_DATA_DIR", str(tmp_path))
    return ledger.ledger_path()


def test_ledger_lives_under_data_dir(path, tmp_path):
    assert path == tmp_path / "sleep" / "ledger.jsonl"


def test_ledger_hash_matches_file(path, tmp_path):
    f = tmp_path / "x.md"
    f.write_text("prompt body")
    e = ledger.register_file("prompt", f, "x")
    assert e.sha256 == ledger.sha256_file(f)
    assert ledger.verify() == 1


def test_register_chains_entries(path):
    a = ledger.register("a", H, "one")
    b = ledger.register("b", "b" * 64, "two")
    assert b.prev_hash == a.entry_hash and (a.seq, b.seq) == (0, 1)
    assert ledger.verify() == 2


def test_ledger_verification_fails_if_an_old_entry_is_edited(path):
    ledger.register("a", H, "one")
    ledger.register("b", "b" * 64, "two")
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["sha256"] = "c" * 64
    path.write_text(json.dumps(first) + "\n" + lines[1] + "\n")
    with pytest.raises(ledger.LedgerError):
        ledger.verify()
    with pytest.raises(ledger.LedgerError):
        ledger.register("c", H, "refuses to extend a broken chain")


def test_verification_fails_if_an_entry_is_removed(path):
    for i in range(3):
        ledger.register("k", H, str(i))
    lines = path.read_text().splitlines()
    path.write_text(lines[0] + "\n" + lines[2] + "\n")
    with pytest.raises(ledger.LedgerError):
        ledger.verify()


def test_register_rejects_bad_hash(path):
    with pytest.raises(ledger.LedgerError):
        ledger.register("a", "xyz", "n")


class FakeGithub:
    def __init__(self, existing=None):
        self.calls = []
        self.remote = existing  # text or None

    def request(self, method, url, *, headers, json_body):
        self.calls.append((method, url, dict(headers), json_body))
        if method == "GET":
            if self.remote is None:
                return ledger.HttpResponse(404, {})
            enc = base64.b64encode(self.remote.encode()).decode()
            return ledger.HttpResponse(200, {"content": enc, "sha": "blob1"})
        self.remote = base64.b64decode(json_body["content"]).decode()
        return ledger.HttpResponse(201, {"commit": {"sha": "commit123"}})


def test_push_prereg_appends_to_branch_with_token_from_env(path, monkeypatch):
    monkeypatch.setenv(ledger.TOKEN_ENV, "secret-token")
    e1 = ledger.register("a", H, "one")
    client = FakeGithub()
    assert ledger.push_prereg([e1], client) == "commit123"
    put = client.calls[-1]
    assert put[0] == "PUT" and put[3]["branch"] == "prereg" and "sha" not in put[3]
    assert "kevindechang/anjo-sleep" in put[1]
    assert put[2]["Authorization"] == "Bearer secret-token"
    assert "secret-token" not in json.dumps(put[3])

    e2 = ledger.register("b", "b" * 64, "two")
    ledger.push_prereg([e1, e2], client)
    assert client.calls[-1][3]["sha"] == "blob1"
    assert len(client.remote.splitlines()) == 2


def test_push_prereg_refuses_without_token(path, monkeypatch):
    monkeypatch.delenv(ledger.TOKEN_ENV, raising=False)
    e = ledger.register("a", H, "one")
    with pytest.raises(ledger.PushError):
        ledger.push_prereg([e], FakeGithub())


def test_push_prereg_error_does_not_leak_token(path, monkeypatch):
    monkeypatch.setenv(ledger.TOKEN_ENV, "secret-token")
    e = ledger.register("a", H, "one")

    class Failing(FakeGithub):
        def request(self, method, url, *, headers, json_body):
            if method == "PUT":
                return ledger.HttpResponse(422, {"message": "bad"})
            return super().request(method, url, headers=headers, json_body=json_body)

    with pytest.raises(ledger.PushError) as err:
        ledger.push_prereg([e], Failing())
    assert "secret-token" not in str(err.value)


def test_push_prereg_rejects_remote_that_diverges(path, monkeypatch):
    monkeypatch.setenv(ledger.TOKEN_ENV, "t")
    e = ledger.register("a", H, "one")
    other = json.dumps({**json.loads(path.read_text()), "note": "tampered"})
    with pytest.raises(ledger.LedgerError):
        ledger.push_prereg([e], FakeGithub(existing=other + "\n"))
