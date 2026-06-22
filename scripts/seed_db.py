#!/usr/bin/env python3
"""Initialise the SQLite database from the synthetic JSON fixture.

Usage:
    python scripts/seed_db.py
    FHIR_MCP_DB=path/to/custom.db python scripts/seed_db.py

Safe to re-run — uses INSERT OR IGNORE so existing rows are not overwritten.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhir_mcp.store import FhirStore

_REPO = Path(__file__).resolve().parents[1]
_JSON_SRC = _REPO / "data" / "synthetic_patients.json"
_DB_PATH = Path(os.environ.get("FHIR_MCP_DB", str(_REPO / "data" / "fhir.db")))


def main() -> None:
    if not _JSON_SRC.exists():
        print(f"ERROR: JSON fixture not found: {_JSON_SRC}", file=sys.stderr)
        sys.exit(1)

    print(f"Seeding {_DB_PATH} from {_JSON_SRC}")
    store = FhirStore(_DB_PATH)
    store.import_from_json(_JSON_SRC)

    patients = store.get_patient_ids()
    print(f"✓ {len(patients)} patient(s): {', '.join(patients)}")
    for pid in patients:
        obs = store.list_observations(pid)
        print(f"  {pid}: {len(obs)} observation(s)")
    print(f"Database: {_DB_PATH}")


if __name__ == "__main__":
    main()
