"""MIMIC-IV discharge summary loader.

PHI NOTE
--------
MIMIC-IV is de-identified clinical data from Beth Israel Deaconess Medical
Center, distributed via PhysioNet under a data use agreement.

  Full dataset  : requires PhysioNet account + CITI training + signed DUA
                  https://physionet.org/content/mimiciv/
  Demo subset   : 100 patients, free with PhysioNet account (no CITI required)
                  https://physionet.org/content/mimic-iv-demo/

NEVER commit MIMIC CSV files to this repository.
NEVER log raw note text — log note IDs only.
This module is safe when used with generate_synthetic_notes() (zero PHI).
When loading real MIMIC-IV files, treat all text as sensitive.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


@dataclass
class MIMICNote:
    """One MIMIC-IV discharge summary note.

    PHI NOTE: When loaded from real MIMIC-IV files, `text` is de-identified
    clinical text. Never log, print, or persist `text` in production.
    Use `note_id` for all audit references.
    """
    note_id: str
    subject_id: str      # de-identified patient ID
    hadm_id: str         # de-identified hospital admission ID
    text: str            # discharge summary text
    chartdate: str = ""
    is_synthetic: bool = False


class MIMICLoader:
    """Load MIMIC-IV discharge summaries from a PhysioNet data directory.

    Directory layout expected (MIMIC-IV-Note or MIMIC-IV Demo):
      <notes_dir>/
        discharge.csv   or   note/discharge.csv

    Args:
        notes_dir: Path to the MIMIC-IV notes directory.
        max_notes:  Cap the number of notes loaded (default 100 for demo).
    """

    _CANDIDATE_PATHS = ["discharge.csv", "note/discharge.csv"]

    def __init__(self, notes_dir: str | Path, max_notes: int = 100) -> None:
        self.notes_dir = Path(notes_dir)
        self.max_notes = max_notes

    def _find_discharge_csv(self) -> Path | None:
        for rel in self._CANDIDATE_PATHS:
            p = self.notes_dir / rel
            if p.exists():
                return p
        return None

    def load(self) -> list[MIMICNote]:
        """Load notes from disk. Falls back to synthetic notes if no CSV found."""
        csv_path = self._find_discharge_csv()
        if csv_path is None:
            log.warning(
                "No MIMIC-IV discharge.csv found under %s. "
                "Returning synthetic notes for CI/demo.",
                self.notes_dir,
            )
            return generate_synthetic_notes()
        return list(self._iter_notes(csv_path))

    def _iter_notes(self, csv_path: Path) -> Iterator[MIMICNote]:
        log.info("Loading MIMIC-IV notes from %s (max=%d)", csv_path, self.max_notes)
        count = 0
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if count >= self.max_notes:
                    break
                note_id = row.get("note_id", f"note_{count}")
                yield MIMICNote(
                    note_id=note_id,
                    subject_id=row.get("subject_id", ""),
                    hadm_id=row.get("hadm_id", ""),
                    text=row.get("text", ""),
                    chartdate=row.get("chartdate", ""),
                    is_synthetic=False,
                )
                count += 1
        log.info("Loaded %d MIMIC-IV notes", count)


# ---------------------------------------------------------------------------
# Synthetic note generator (CI / demo — zero PHI)
# ---------------------------------------------------------------------------

# Each entry: (note_id, condition_label, discharge_text_with_labs)
# Lab values are fictional but clinically plausible. No real patient data.
_SYNTHETIC_TEMPLATES: list[tuple[str, str, str]] = [
    (
        "SYN001", "PNH",
        """
Discharge Summary — Synthetic Patient SYN001
Diagnosis: Paroxysmal Nocturnal Hemoglobinuria (PNH)

Laboratory Results:
  Hemoglobin: 7.8 g/dL
  LDH: 1420 U/L
  Haptoglobin: <10 mg/dL (undetectable)
  Reticulocytes: 4.2 %
  Platelet count: 98 x10^9/L
  Creatinine: 1.1 mg/dL

Assessment: PNH confirmed by flow cytometry (CD55/CD59 deficient clone >50%).
Started eculizumab therapy. Human approval obtained prior to medication order.
"""
    ),
    (
        "SYN002", "T2DM",
        """
Discharge Summary — Synthetic Patient SYN002
Diagnosis: Type 2 Diabetes Mellitus, poorly controlled

Laboratory Results:
  HbA1c: 9.4 %
  Glucose (fasting): 248 mg/dL
  Creatinine: 1.3 mg/dL
  eGFR: 54 mL/min/1.73m2
  Potassium: 4.1 mEq/L
  Sodium: 138 mEq/L

Started insulin glargine 20 units QHS. Metformin continued.
"""
    ),
    (
        "SYN003", "HTN",
        """
Discharge Summary — Synthetic Patient SYN003
Diagnosis: Hypertensive urgency

Vital Signs on admission:
  Blood pressure: 192/114 mmHg
  Heart rate: 88 bpm
  SpO2: 97%

Laboratory Results:
  Creatinine: 1.5 mg/dL
  Potassium: 3.8 mEq/L
  Sodium: 141 mEq/L

Amlodipine 10 mg daily started. Blood pressure at discharge: 148/88 mmHg.
"""
    ),
    (
        "SYN004", "HF",
        """
Discharge Summary — Synthetic Patient SYN004
Diagnosis: Acute decompensated heart failure (EF 30%)

Laboratory Results:
  BNP: 1840 pg/mL
  NT-proBNP: 4200 pg/mL
  Sodium: 132 mEq/L
  Creatinine: 1.8 mg/dL
  Hemoglobin: 10.2 g/dL
  Potassium: 4.6 mEq/L

Furosemide IV administered. Weight decreased 4.2 kg. BNP on discharge: 620 pg/mL.
"""
    ),
    (
        "SYN005", "CKD",
        """
Discharge Summary — Synthetic Patient SYN005
Diagnosis: Chronic Kidney Disease Stage 4

Laboratory Results:
  Creatinine: 3.6 mg/dL
  eGFR: 18 mL/min/1.73m2
  Potassium: 5.3 mEq/L
  Sodium: 136 mEq/L
  Hemoglobin: 9.1 g/dL (anemia of CKD)
  Glucose: 102 mg/dL

Nephrology follow-up arranged. AV fistula planning discussed.
"""
    ),
    (
        "SYN006", "AFib",
        """
Discharge Summary — Synthetic Patient SYN006
Diagnosis: Atrial fibrillation with rapid ventricular response

Vital Signs:
  Heart rate: 136 bpm on admission, 74 bpm at discharge
  Blood pressure: 122/78 mmHg
  SpO2: 95%

Laboratory Results:
  Troponin I: 0.04 ng/mL
  Sodium: 139 mEq/L
  Potassium: 3.9 mEq/L
  Creatinine: 1.0 mg/dL

Rate-controlled with metoprolol. Apixaban started for stroke prophylaxis.
"""
    ),
    (
        "SYN007", "SickleCell",
        """
Discharge Summary — Synthetic Patient SYN007
Diagnosis: Sickle Cell Disease, acute vaso-occlusive crisis

Laboratory Results:
  Hemoglobin: 6.4 g/dL
  Reticulocytes: 12.8 %
  LDH: 680 U/L
  WBC: 14.2 x10^9/L
  Platelet count: 420 x10^9/L
  Creatinine: 0.9 mg/dL

PRBC transfusion administered (2 units, type-matched). Hydroxyurea continued.
"""
    ),
    (
        "SYN008", "COPD",
        """
Discharge Summary — Synthetic Patient SYN008
Diagnosis: COPD exacerbation

Vital Signs:
  SpO2: 84% on room air (admission), 93% on 2L NC (discharge)
  Blood pressure: 138/86 mmHg

Laboratory Results:
  WBC: 12.1 x10^9/L
  Hemoglobin: 15.8 g/dL
  Creatinine: 0.8 mg/dL
  Sodium: 143 mEq/L
  Potassium: 3.7 mEq/L

Azithromycin + prednisone course. Albuterol + ipratropium nebulizers.
"""
    ),
    (
        "SYN009", "Gaucher",
        """
Discharge Summary — Synthetic Patient SYN009
Diagnosis: Gaucher Disease Type 1

Laboratory Results:
  Hemoglobin: 9.6 g/dL
  Platelet count: 62 x10^9/L
  Ferritin: 840 ng/mL
  LDH: 390 U/L
  WBC: 3.8 x10^9/L

Beta-glucocerebrosidase activity confirmed low. Imiglucerase ERT initiated.
Human approval obtained for enzyme replacement order.
"""
    ),
    (
        "SYN010", "RA",
        """
Discharge Summary — Synthetic Patient SYN010
Diagnosis: Rheumatoid Arthritis, active disease

Laboratory Results:
  WBC: 8.4 x10^9/L
  Hemoglobin: 10.8 g/dL
  Creatinine: 0.9 mg/dL
  Glucose: 118 mg/dL
  Sodium: 137 mEq/L
  Potassium: 4.0 mEq/L

Methotrexate 15 mg weekly + folic acid. Prednisone taper prescribed.
"""
    ),
]


def generate_synthetic_notes() -> list[MIMICNote]:
    """Return 10 synthetic discharge summaries for CI and demo use.

    Notes cover all major conditions in the ontology crosswalk with
    known LOINC-coded lab observations embedded in the text.
    Zero real patient data. Safe to use without PhysioNet access.
    """
    return [
        MIMICNote(
            note_id=note_id,
            subject_id=f"SUBJ_{note_id}",
            hadm_id=f"ADM_{note_id}",
            text=text.strip(),
            chartdate="2024-01-01",
            is_synthetic=True,
        )
        for note_id, _, text in _SYNTHETIC_TEMPLATES
    ]
