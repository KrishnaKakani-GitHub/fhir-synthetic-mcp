"""Tests for the store: read path + the gated propose → approve/reject flow.

We test the store directly (not through MCP) because the store holds the
logic that matters for safety. The MCP layer is a thin wrapper over it.

Fixture: SQLite database seeded from the synthetic JSON fixture.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from fhir_mcp.models import PendingWriteStatus, ProposedObservation
from fhir_mcp.store import FhirStore, StoreError

DATA_SRC = Path(__file__).resolve().parents[1] / "data" / "synthetic_patients.json"


@pytest.fixture()
def store(tmp_path: Path) -> FhirStore:
    """SQLite store seeded from the synthetic JSON fixture."""
    s = FhirStore(tmp_path / "test.db")
    s.import_from_json(DATA_SRC)
    return s


# --- Reads --------------------------------------------------------------------


def test_list_patient_ids(store: FhirStore) -> None:
    assert "pat-001" in store.get_patient_ids()


def test_get_patient_ok(store: FhirStore) -> None:
    assert store.get_patient("pat-001").mrn == "SYN-0001"


def test_get_patient_unknown_raises(store: FhirStore) -> None:
    with pytest.raises(StoreError):
        store.get_patient("pat-999")


def test_list_observations(store: FhirStore) -> None:
    obs = store.list_observations("pat-001")
    assert len(obs) == 2
    assert {o.display for o in obs} == {"Heart rate", "Systolic blood pressure"}


# --- Gated write path ---------------------------------------------------------


def _proposal(patient_id: str = "pat-001", value: float = 98.6) -> ProposedObservation:
    return ProposedObservation(
        patient_id=patient_id,
        code="8310-5",
        display="Body temperature",
        value=value,
        unit="degF",
        effective_date=date(2026, 6, 1),
    )


def test_propose_does_not_commit(store: FhirStore) -> None:
    before = len(store.list_observations("pat-001"))
    store.stage_write(_proposal())
    assert len(store.list_observations("pat-001")) == before
    assert len(store.list_pending()) == 1


def test_propose_rejects_negative_value(store: FhirStore) -> None:
    with pytest.raises(StoreError):
        store.stage_write(_proposal(value=-5))


def test_propose_unknown_patient_raises(store: FhirStore) -> None:
    with pytest.raises(StoreError):
        store.stage_write(_proposal(patient_id="pat-999"))


def test_approve_commits_and_persists(store: FhirStore) -> None:
    pending = store.stage_write(_proposal())
    obs = store.approve_write(pending.write_id, approver="dr.smith")
    # Present in live observations
    assert any(o.id == obs.id for o in store.list_observations("pat-001"))
    # Status + approver recorded
    pw = store.get_pending(pending.write_id)
    assert pw.status == PendingWriteStatus.approved
    assert pw.decided_by == "dr.smith"
    # Durable: a fresh store on the same DB file sees the committed row
    store2 = FhirStore(store._db_path)
    assert any(o.id == obs.id for o in store2.list_observations("pat-001"))


def test_reject_does_not_commit(store: FhirStore) -> None:
    before = len(store.list_observations("pat-001"))
    pending = store.stage_write(_proposal())
    store.reject_write(pending.write_id, approver="dr.smith")
    assert len(store.list_observations("pat-001")) == before
    assert store.get_pending(pending.write_id).status == PendingWriteStatus.rejected


def test_cannot_approve_twice(store: FhirStore) -> None:
    pending = store.stage_write(_proposal())
    store.approve_write(pending.write_id, approver="dr.smith")
    with pytest.raises(StoreError):
        store.approve_write(pending.write_id, approver="dr.smith")
