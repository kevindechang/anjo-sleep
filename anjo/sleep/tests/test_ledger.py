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
        self.branch_status = 200
        self.auth_status = 0

    def request(self, method, url, *, headers, json_body):
        self.calls.append((method, url, dict(headers), json_body))
        if method == "GET" and "/branches/" in url:
            return ledger.HttpResponse(self.branch_status, {})
        if method == "GET" and self.auth_status:
            return ledger.HttpResponse(self.auth_status, {})
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


def test_verify_fails_when_file_missing_and_expected(path):
    assert ledger.verify() == 0
    with pytest.raises(ledger.LedgerError):
        ledger.verify(expect_exists=True)


def test_verify_fails_on_seq_gap_and_reorder(path):
    for i in range(3):
        ledger.register("k", H, str(i))
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    with pytest.raises(ledger.LedgerError):
        ledger.verify()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(ledger.LedgerError):
        ledger.verify()


def test_verify_against_remote_head_catches_truncation(path):
    ledger.register("a", H, "one")
    last = ledger.register("b", "b" * 64, "two")
    assert ledger.verify(remote_head=last.entry_hash) == 2
    path.write_text(path.read_text().splitlines()[0] + "\n")  # a valid but truncated chain
    assert ledger.verify() == 1
    with pytest.raises(ledger.LedgerError):
        ledger.verify(remote_head=last.entry_hash)


def test_push_records_head_and_next_verify_compares(path, monkeypatch):
    monkeypatch.setenv(ledger.TOKEN_ENV, "t")
    ledger.register("a", H, "one")
    e2 = ledger.register("b", "b" * 64, "two")
    client = FakeGithub()
    ledger.push_prereg(ledger.read_entries(), client)
    assert ledger.read_pushed_head() == e2.entry_hash
    assert ledger.verify(remote_head=ledger.read_pushed_head()) == 2


def test_push_refuses_remote_that_lost_the_recorded_head(path, monkeypatch):
    monkeypatch.setenv(ledger.TOKEN_ENV, "t")
    e1 = ledger.register("a", H, "one")
    ledger.register("b", "b" * 64, "two")
    ledger.push_prereg(ledger.read_entries(), FakeGithub())
    truncated = FakeGithub(existing=json.dumps(json.loads(path.read_text().splitlines()[0])) + "\n")
    with pytest.raises(ledger.PushError):
        ledger.push_prereg([e1], truncated)


@pytest.mark.parametrize("bad", [H + "\n", H.upper(), H[:-1], " " + H])
def test_sha256_with_trailing_newline_or_wrong_shape_is_refused(path, bad):
    with pytest.raises(ledger.LedgerError):
        ledger.register("a", bad, "n")


def test_two_registers_in_sequence_keep_one_valid_chain(path):
    for i in range(4):
        ledger.register("k", H, str(i))
    assert ledger.verify(expect_exists=True) == 4
    assert [e.seq for e in ledger.read_entries()] == [0, 1, 2, 3]


def test_push_404_without_branch_fails_closed(path, monkeypatch):
    monkeypatch.setenv(ledger.TOKEN_ENV, "t")
    e = ledger.register("a", H, "one")
    client = FakeGithub()
    client.branch_status = 404
    with pytest.raises(ledger.PushError):
        ledger.push_prereg([e], client)


@pytest.mark.parametrize("status", [401, 403])
def test_push_401_or_403_always_raises(path, monkeypatch, status):
    monkeypatch.setenv(ledger.TOKEN_ENV, "t")
    e = ledger.register("a", H, "one")
    client = FakeGithub(existing="")
    client.auth_status = status
    with pytest.raises(ledger.PushError):
        ledger.push_prereg([e], client)


def test_push_put_body_carries_the_noreply_identity_and_no_other_email(path, monkeypatch):
    import re

    monkeypatch.setenv(ledger.TOKEN_ENV, "t")
    e = ledger.register("a", H, "one")
    client = FakeGithub()
    ledger.push_prereg([e], client)
    body = client.calls[-1][3]
    ident = {"name": "Chang Chia Wei", "email": "196453186+kevindechang@users.noreply.github.com"}
    assert body["author"] == ident and body["committer"] == ident
    assert set(re.findall(r"[\w.+-]+@[\w.-]+", json.dumps(body))) == {ident["email"]}
