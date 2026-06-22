"""Data store: SQLite backend with deterministic validation gate.

Same public interface as v1 (FhirStore). The write gate
(propose → stage → approve/reject) is unchanged — only the
persistence layer is replaced and validator.py is now called.

Design decisions:
- sqlite3 stdlib only (no ORM, no external deps)
- WAL mode: concurrent reads don't block writes
- Foreign keys: prevents orphaned observations
- Pending writes are in-memory only (intentional) — proposals are
  never persisted until a human approves them
- validator.validate_observation() is called in stage_write() before
  the proposal enters the queue — this is the deterministic gate

PHI NOTE: All PHI touchpoints are in this module only. The store
deliberately never logs record contents — that responsibility lives
in audit.py (IDs only).
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from .models import (
    DataStore,
    Observation,
    Patient,
    PendingWrite,
    PendingWriteStatus,
    ProposedObservation,
)
from .validator import ValidationError, validate_observation


class StoreError(Exception):
    """Raised for store-level problems (missing records, invalid state)."""


class FhirStore:
    """SQLite-backed FHIR store with in-memory gated write queue.

    A threading lock guards mutations. The pending-write queue is
    intentionally in-memory only — unapproved proposals are never
    persisted, so they cannot leak into the database on restart.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._pending: dict[str, PendingWrite] = {}
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    birth_date  TEXT NOT NULL,
                    gender      TEXT NOT NULL,
                    mrn         TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS observations (
                    id             TEXT PRIMARY KEY,
                    patient_id     TEXT NOT NULL REFERENCES patients(id),
                    code           TEXT NOT NULL,
                    display        TEXT NOT NULL,
                    value          REAL NOT NULL,
                    unit           TEXT NOT NULL,
                    effective_date TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_obs_patient
                    ON observations(patient_id);
                """
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_patient_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM patients ORDER BY id"
            ).fetchall()
        return [r["id"] for r in rows]

    def get_patient(self, patient_id: str) -> Patient:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
        if row is None:
            raise StoreError(f"Unknown patient_id: {patient_id}")
        return Patient(**dict(row))

    def list_observations(self, patient_id: str) -> list[Observation]:
        self.get_patient(patient_id)  # raises StoreError if unknown
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM observations WHERE patient_id = ? ORDER BY effective_date",
                (patient_id,),
            ).fetchall()
        return [
            Observation(
                id=r["id"],
                patient_id=r["patient_id"],
                code=r["code"],
                display=r["display"],
                value=r["value"],
                unit=r["unit"],
                effective_date=date.fromisoformat(r["effective_date"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Gated write: stage → approve / reject
    # ------------------------------------------------------------------

    def stage_write(self, proposed: ProposedObservation) -> PendingWrite:
        """Stage a proposed observation. Does NOT commit. Returns the ticket.

        Deterministic validation order:
          1. Patient exists check (structural)
          2. Negative value guard (basic sanity)
          3. Non-empty code check (basic sanity)
          4. LOINC registry + value range + unit (clinical)

        Any failure raises StoreError. The proposal never enters the queue.
        """
        self.get_patient(proposed.patient_id)  # patient must exist
        if proposed.value < 0:
            raise StoreError("Observation value cannot be negative.")
        if not proposed.code.strip():
            raise StoreError("Observation code is required.")

        # Deterministic clinical validation gate
        result = validate_observation(proposed)
        if not result.ok:
            raise StoreError(
                "Proposal rejected by deterministic validation: "
                + "; ".join(result.violations)
            )

        with self._lock:
            write_id = f"pw-{uuid.uuid4().hex[:8]}"
            pending = PendingWrite(
                write_id=write_id,
                proposed=proposed,
                validation_warnings=result.warnings or [],
            )
            self._pending[write_id] = pending
            return pending

    def get_pending(self, write_id: str) -> PendingWrite:
        pending = self._pending.get(write_id)
        if pending is None:
            raise StoreError(f"Unknown write_id: {write_id}")
        return pending

    def list_pending(self) -> list[PendingWrite]:
        return [
            w for w in self._pending.values()
            if w.status == PendingWriteStatus.pending
        ]

    def approve_write(self, write_id: str, approver: str) -> Observation:
        """Commit a staged write. Only reachable after explicit human approval."""
        with self._lock:
            pending = self.get_pending(write_id)
            if pending.status != PendingWriteStatus.pending:
                raise StoreError(
                    f"Write {write_id} already {pending.status.value}."
                )
            obs = Observation(
                id=f"obs-{uuid.uuid4().hex[:8]}",
                patient_id=pending.proposed.patient_id,
                code=pending.proposed.code,
                display=pending.proposed.display,
                value=pending.proposed.value,
                unit=pending.proposed.unit,
                effective_date=pending.proposed.effective_date,
            )
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO observations
                        (id, patient_id, code, display, value, unit, effective_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        obs.id, obs.patient_id, obs.code, obs.display,
                        obs.value, obs.unit, str(obs.effective_date),
                    ),
                )
            pending.status = PendingWriteStatus.approved
            pending.decided_at = datetime.now(timezone.utc)
            pending.decided_by = approver
            return obs

    def reject_write(self, write_id: str, approver: str) -> PendingWrite:
        with self._lock:
            pending = self.get_pending(write_id)
            if pending.status != PendingWriteStatus.pending:
                raise StoreError(
                    f"Write {write_id} already {pending.status.value}."
                )
            pending.status = PendingWriteStatus.rejected
            pending.decided_at = datetime.now(timezone.utc)
            pending.decided_by = approver
            return pending

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def import_from_json(self, json_path: Path) -> None:
        """Seed the database from the legacy JSON fixture.

        Uses INSERT OR IGNORE so it is safe to re-run against an
        already-populated database.
        """
        import json
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        data = DataStore.model_validate(raw)
        with self._connect() as conn:
            for p in data.patients:
                conn.execute(
                    """INSERT OR IGNORE INTO patients
                       (id, name, birth_date, gender, mrn)
                       VALUES (?, ?, ?, ?, ?)""",
                    (p.id, p.name, str(p.birth_date), p.gender.value, p.mrn),
                )
            for o in data.observations:
                conn.execute(
                    """INSERT OR IGNORE INTO observations
                       (id, patient_id, code, display, value, unit, effective_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        o.id, o.patient_id, o.code, o.display,
                        o.value, o.unit, str(o.effective_date),
                    ),
                )
