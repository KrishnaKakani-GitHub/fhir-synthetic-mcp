"""Loader for real MIMIC-IV demo discharge notes (PHI-safe).

Reads de-identified discharge summaries from the PhysioNet MIMIC-IV demo
subset (https://physionet.org/content/mimic-iv-demo/ -- free account, no CITI
training). These notes are UNLABELED free text: they have no gold diagnosis /
treatment / lab / procedure codes, so they power a *demonstration* of the
Meditron CDM agent, not a scored benchmark.

Distinction
-----------
  Scored benchmark  -> needs gold labels -> cdm dataset CDMCase (synthetic now,
                       DUA-gated MIMIC-IV-Ext-CDM later). Graded by score_case.
  Unscored demo     -> this module. Real de-identified notes, no answer key.
                       Meditron proposes; nothing is graded.

PHI / safety
------------
These are DE-IDENTIFIED notes, but treat all `text` as sensitive:
  * NEVER log, print to logs, or persist `text` -- use `note_id` for audit refs.
  * The demo runner prints model OUTPUT (proposed codes), never echoes note text
    beyond a short truncated preview the operator explicitly opts into.
  * Read-only: no writes, no patient store, governance invariant preserved.

The full DUA-gated MIMIC-IV-Ext-CDM (CITI training + signed DUA) is the
legitimate *labeled* source; it drops into the scored path via --notes-dir.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DemoNote:
    """One de-identified MIMIC-IV demo discharge note (UNLABELED).

    PHI NOTE: `text` is de-identified clinical text. Never log or persist it;
    use `note_id` for all audit references.
    """
    note_id: str
    subject_id: str       # de-identified patient ID
    hadm_id: str          # de-identified hospital admission ID
    text: str             # discharge summary free text (NO gold labels)

    def preview(self, n: int = 160) -> str:
        """Short, length-capped preview for operator display only."""
        t = " ".join(self.text.split())
        return (t[:n] + "…") if len(t) > n else t


class DemoNotesLoader:
    """Loads MIMIC-IV demo discharge notes from a local directory.

    Expects PhysioNet `discharge.csv` (note_id, subject_id, hadm_id, ... , text)
    or a directory of `.txt` files (filename stem -> note_id). The operator
    downloads the demo subset themselves; this never fetches data.
    """

    def __init__(self, notes_dir: str | Path):
        self.notes_dir = Path(notes_dir)
        if not self.notes_dir.exists():
            raise FileNotFoundError(
                f"MIMIC demo notes dir not found: {self.notes_dir}. "
                "Download the demo subset from "
                "https://physionet.org/content/mimic-iv-demo/ (free account)."
            )

    def load(self, limit: int | None = None) -> list[DemoNote]:
        """Load notes from discharge.csv if present, else *.txt files."""
        csv_path = self.notes_dir / "discharge.csv"
        if csv_path.exists():
            notes = self._load_csv(csv_path)
        else:
            notes = self._load_txt()
        if limit is not None:
            notes = notes[:limit]
        # Audit-safe: log count + ids only, never text.
        log.info("Loaded %d MIMIC demo notes: %s",
                 len(notes), [n.note_id for n in notes])
        return notes

    def _load_csv(self, path: Path) -> list[DemoNote]:
        notes: list[DemoNote] = []
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                notes.append(DemoNote(
                    note_id=row.get("note_id") or f"note-{i:04d}",
                    subject_id=row.get("subject_id", ""),
                    hadm_id=row.get("hadm_id", ""),
                    text=row.get("text", ""),
                ))
        return notes

    def _load_txt(self) -> list[DemoNote]:
        notes: list[DemoNote] = []
        for i, p in enumerate(sorted(self.notes_dir.glob("*.txt"))):
            notes.append(DemoNote(
                note_id=p.stem or f"note-{i:04d}",
                subject_id="",
                hadm_id="",
                text=p.read_text(encoding="utf-8"),
            ))
        return notes
