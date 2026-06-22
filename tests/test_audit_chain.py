"""Tests for the tamper-evident SHA-256 audit chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fhir_mcp.audit as audit_mod
from fhir_mcp.audit import audit, verify_chain


@pytest.fixture(autouse=True)
def isolated_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect audit output to a temp file and reset the chain tip."""
    monkeypatch.setattr(audit_mod, "_AUDIT_PATH", tmp_path / "audit.jsonl")
    monkeypatch.setattr(audit_mod, "_prev_hash", "GENESIS")


def _records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit(actor="agent:test", action="test_action", reason="unit test",
          target_ids=["r1"])
    recs = _records(path)
    assert len(recs) == 1
    r = recs[0]
    assert r["actor"] == "agent:test"
    assert r["action"] == "test_action"
    assert r["prev_hash"] == "GENESIS"
    assert r["outcome"] == "ok"


def test_chain_links(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    for i in range(4):
        audit(actor="agent:test", action=f"action_{i}", reason="chain test",
              target_ids=[str(i)])
    recs = _records(path)
    assert len(recs) == 4
    prev = "GENESIS"
    for r in recs:
        assert r["prev_hash"] == prev, (
            f"Chain broken at record action={r['action']}: "
            f"expected prev_hash={prev[:16]}..."
        )
        prev = hashlib.sha256(
            json.dumps(r, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()


def test_verify_chain_passes(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    for i in range(3):
        audit(actor="agent:test", action=f"a{i}", reason="verify test")
    assert verify_chain(path) is True


def test_verify_detects_field_mutation(tmp_path: Path) -> None:
    """Mutating any field in a record breaks the chain."""
    path = tmp_path / "audit.jsonl"
    audit(actor="agent:test", action="original", reason="write")
    audit(actor="agent:test", action="second", reason="write")
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["actor"] = "evil-actor"  # tamper
    lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    assert verify_chain(path) is False


def test_verify_detects_record_insertion(tmp_path: Path) -> None:
    """Inserting a fake record between two real ones breaks the chain."""
    path = tmp_path / "audit.jsonl"
    audit(actor="agent:test", action="a1", reason="write")
    audit(actor="agent:test", action="a2", reason="write")
    lines = path.read_text().splitlines()
    fake = json.dumps(
        {"action": "steal", "actor": "attacker", "outcome": "ok",
         "prev_hash": "GENESIS", "reason": "bad", "target_ids": [], "ts": "X"},
        separators=(",", ":"), sort_keys=True,
    )
    lines.insert(1, fake)
    path.write_text("\n".join(lines) + "\n")
    assert verify_chain(path) is False


def test_extra_fields_preserved(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    audit(actor="agent:test", action="propose", reason="test",
          extra={"write_id": "pw-abc123"})
    r = _records(path)[0]
    assert r["write_id"] == "pw-abc123"
