"""Data store: SQLite backend with deterministic validation gate.

Same public interface as v1 (FhirStore). The write gate
(propose → stage → approve/reject) is unchanged.

Design decisions:
- sqlite3 stdlib only (no ORM, no external deps)
- WAL mode: concurrent reads don\'t block writes
- Foreign keys: prevents orphaned observations
- Pending writes are in-memory only (intentional) — proposals are
  never persisted until a human approves them
- validator.validate_observation() called in stage_write() — deterministic gate

Field-level encryption (PHI at rest):
- Set FHIR_MCP_ENCRYPTION_KEY to a Fernet key to encrypt PHI fields.
- Encrypted fields: name, mrn, birth_date (all directly identifying).
- Observation values/codes are not encrypted (clinical, not identifying).
- Generate a key: python -c "from fhir_mcp.store import generate_encryption_key; print(generate_encryption_key())"
- Dev mode: encryption off when key not set. Never run dev mode with real PHI.

PHI NOTE: All PHI touchpoints are in this module only. The store
deliberately never logs record contents — that responsibility lives
in audit.py (IDs only).
"""
from __future__ import annotations

import logging
import os
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

_logger = logging.getLogger("fhir_mcp.store")
_ENCRYPTION_KEY = os.environ.get("FHIR_MCP_ENCRYPTION_KEY", "").strip()


class StoreError(Exception):
    """Raised for store-level problems (missing records, invalid state)."""


# ---------------------------------------------------------------------------
# Field-level encryption helpers
# ---------------------------------------------------------------------------


def generate_encryption_key() -> str:
    """Generate a new Fernet encryption key. Run once; store securely.

    Usage::

        python -c "from fhir_mcp.store import generate_encryption_key; print(generate_encryption_key())"

    Save the output as FHIR_MCP_ENCRYPTION_KEY in your secrets manager.
    Losing this key means losing access to all encrypted PHI records.
    """
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def _get_fernet():
    """Return a Fernet instance if encryption key is set, else None."""
    if not _ENCRYPTION_KEY:
        return None
    try:
        from cryptography.fernet import Fernet
        key = _ENCRYPTION_KEY.encode() if isinstance(_ENCRYPTION_KEY, str) else _ENCRYPTION_KEY
        return Fernet(key)
    except Exception as e:
        _logger.error("Failed to initialise Fernet encryption: %s", e)
        raise


def _encrypt(value: str) -> str:
    """Encrypt a PHI string field. Returns plaintext if no key set (dev mode)."""
    f = _get_fernet()
    if f is None:
        return value
    return f.encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    """Decrypt a PHI string field. Returns value unchanged if no key set.

    Handles migration: if value is already plaintext (pre-encryption records)
    decryption fails gracefully and the raw value is returned.
    """
    f = _get_fernet()
    if f is None:
        return value
    try:
        return f.decrypt(value.encode()).decode()
    except Exception:
        _logger.warning("Decryption failed — returning raw value (plaintext migration?)")
        return value


# ---------------------------------------------------------------------------
# FhirStore
# ---------------------------------------------------------------------------


class FhirStore:
    """SQLite-backed FHIR store with in-memory gated write queue.

    A threading lock guards mutations. The pending-write queue is
    intentionally in-memory only — unapproved proposals are never
    persisted, so they cannot leak into the database on restart.

    PHI fields (name, mrn, birth_date) are encrypted at rest when
    FHIR_MCP_ENCRYPTION_KEY is set.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._pending: dict[str, PendingWrite] = {}
        if _ENCRYPTION_KEY:
            _logger.info("Field-level encryption enabled for PHI fields")
        else:
            _logger.warning(
                "FHIR_MCP_ENCRYPTION_KEY not set — PHI stored in plaintext (dev mode)"
            )
        self._init_db()

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
        return Patient(
            id=row["id"],
            name=_decrypt(row["name"]),
            birth_date=_decrypt(row["birth_date"]),
            gender=row["gender"],
            mrn=_decrypt(row["mrn"]),
        )

    def list_observations(self, patient_id: str) -> list[Observation]:
        self.get_patient(patient_id)
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
        """Stage a proposed observation. Does NOT commit. Returns the ticket."""
        self.get_patient(proposed.patient_id)
        if proposed.value < 0:
            raise StoreError("Observation value cannot be negative.")
        if not proposed.code.strip():
            raise StoreError("Observation code is required.")

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
                raise StoreError(f"Write {write_id} already {pending.status.value}.")
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
                raise StoreError(f"Write {write_id} already {pending.status.value}.")
            pending.status = PendingWriteStatus.rejected
            pending.decided_at = datetime.now(timezone.utc)
            pending.decided_by = approver
            return pending

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def import_from_json(self, json_path: Path) -> None:
        """Seed the database from the JSON fixture.

        Uses INSERT OR IGNORE so it is safe to re-run.
        PHI fields are encrypted if FHIR_MCP_ENCRYPTION_KEY is set.
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
                    (
                        p.id,
                        _encrypt(p.name),
                        _encrypt(str(p.birth_date)),
                        p.gender.value,
                        _encrypt(p.mrn),
                    ),
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
