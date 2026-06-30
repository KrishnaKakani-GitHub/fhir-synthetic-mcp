"""PHI-safe loader for MIMIC-IV demo *structured* tables (coded, no free text).

The openly-available MIMIC-IV demo (https://physionet.org/content/mimic-iv-demo/)
is a 100-patient subset that EXCLUDES free-text clinical notes. It ships as CSV
tables under hosp/ and icu/. This loader reads the *coded* hosp/ tables and
builds per-admission structured CDM cases:

    diagnoses_icd.csv  -> gold ICD diagnoses (per hadm_id)
    labevents.csv      -> labs ordered, mapped to LOINC (or label) via d_labitems
    prescriptions.csv  -> drugs administered (drug names; RxNorm not native)
    procedures_icd.csv -> procedures (ICD-PCS)

There are NO discharge notes here, so there is no free text to feed a model as a
"presentation". Instead, each admission becomes a structured case the agent can
reason over (e.g. given labs + meds, propose the diagnosis), compared against the
coded gold labels already present in the tables.

PHI / safety -- this is the protection architecture, enforced in code:
  * IDs only: subject_id / hadm_id are de-identified integer ciphers, but we
    treat them as sensitive. Logs emit hadm_id only; never row contents.
  * No raw clinical values are logged or printed by this module.
  * Date columns (charttime, admittime, ...) are NEVER read or surfaced --
    even shifted dates are excluded from anything that leaves the process.
  * Read-only: opens CSVs for reading; never writes patient data anywhere.
  * The caller is responsible for keeping the data dir OUTSIDE the repo; this
    module refuses to load from inside the repository tree (see _reject_in_repo).

License: Open Data Commons ODbL v1.0. Cite Johnson et al. (2023), MIMIC-IV
Clinical Database Demo v2.2, https://doi.org/10.13026/dp1f-ex47.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Columns we will NEVER read into memory or surface (date/PHI-adjacent).
_FORBIDDEN_COLUMNS = frozenset({
    "admittime", "dischtime", "deathtime", "charttime", "storetime",
    "starttime", "stoptime", "dob", "dod", "edregtime", "edouttime",
})


@dataclass
class StructuredCase:
    """One admission as a structured CDM case (coded, no free text)."""
    hadm_id: str
    diagnosis_icd: list[str] = field(default_factory=list)   # gold ICD codes
    labs: list[str] = field(default_factory=list)            # LOINC or lab label
    drugs: list[str] = field(default_factory=list)           # drug names
    procedures_icd: list[str] = field(default_factory=list)  # ICD-PCS procedures
    icd_version: str = ""                                     # "9", "10", or "9/10"

    def summary_line(self) -> str:
        """Audit-safe one-liner: counts only, never raw values."""
        ver = f" icd_ver={self.icd_version}" if self.icd_version else ""
        return (f"hadm_id={self.hadm_id} dx={len(self.diagnosis_icd)} "
                f"labs={len(self.labs)} drugs={len(self.drugs)} "
                f"proc={len(self.procedures_icd)}{ver}")


def _reject_in_repo(data_dir: Path) -> None:
    """Refuse to load PHI from inside the repo tree (prevents accidental commit)."""
    repo_root = Path(__file__).resolve().parents[1]
    try:
        data_dir.resolve().relative_to(repo_root)
    except ValueError:
        return  # outside repo -> OK
    raise ValueError(
        f"Refusing to load MIMIC data from inside the repo ({data_dir}). "
        "Keep PhysioNet data OUTSIDE the repository so it can never be committed. "
        "Move it to e.g. ~/mimic-demo and point --demo-dir there."
    )


def _read_csv_safe(path: Path) -> list[dict[str, str]]:
    """Read a CSV, dropping any forbidden (date/PHI-adjacent) columns."""
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({k: v for k, v in row.items()
                         if k not in _FORBIDDEN_COLUMNS})
    return rows


class DemoStructuredLoader:
    """Builds StructuredCase objects from the MIMIC-IV demo hosp/ CSV tables."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        _reject_in_repo(self.data_dir)
        self.hosp = self.data_dir / "hosp"
        if not self.hosp.exists():
            raise FileNotFoundError(
                f"Expected hosp/ tables under {self.data_dir}. Download the demo "
                "from https://physionet.org/content/mimic-iv-demo/ and unzip it "
                "OUTSIDE the repo."
            )

    def _lab_map(self) -> dict[str, str]:
        """itemid -> LOINC code if present, else the lab label.

        In the MIMIC demo, d_labitems.loinc_code is frequently blank, so a
        strict LOINC-only map drops most labs. We fall back to the human
        label (e.g. "Hemoglobin") so labs are represented either way; LOINC
        is preferred when available. (d_labitems has no patient data.)
        """
        out: dict[str, str] = {}
        for row in _read_csv_safe(self.hosp / "d_labitems.csv"):
            itemid = row.get("itemid", "")
            loinc = (row.get("loinc_code", "") or "").strip()
            label = (row.get("label", "") or "").strip()
            if itemid and (loinc or label):
                out[itemid] = loinc or label
        return out

    def load(self, limit: int | None = None) -> list[StructuredCase]:
        lab_by_item = self._lab_map()

        dx: dict[str, list[str]] = defaultdict(list)
        icd_versions: dict[str, set[str]] = defaultdict(set)
        for row in _read_csv_safe(self.hosp / "diagnoses_icd.csv"):
            hadm = row.get("hadm_id", "")
            code = row.get("icd_code", "")
            ver = (row.get("icd_version", "") or "").strip()
            if hadm and code:
                dx[hadm].append(code.strip())
                if ver:
                    icd_versions[hadm].add(ver)

        labs: dict[str, set[str]] = defaultdict(set)
        for row in _read_csv_safe(self.hosp / "labevents.csv"):
            hadm = row.get("hadm_id", "")
            itemid = row.get("itemid", "")
            lab = lab_by_item.get(itemid)
            if hadm and lab:
                labs[hadm].add(lab)

        drugs: dict[str, set[str]] = defaultdict(set)
        for row in _read_csv_safe(self.hosp / "prescriptions.csv"):
            hadm = row.get("hadm_id", "")
            drug = (row.get("drug", "") or "").strip()
            if hadm and drug:
                drugs[hadm].add(drug)

        proc: dict[str, list[str]] = defaultdict(list)
        for row in _read_csv_safe(self.hosp / "procedures_icd.csv"):
            hadm = row.get("hadm_id", "")
            code = row.get("icd_code", "")
            if hadm and code:
                proc[hadm].append(code.strip())

        hadm_ids = sorted(set(dx) | set(labs) | set(drugs) | set(proc))
        if limit is not None:
            hadm_ids = hadm_ids[:limit]

        cases = [
            StructuredCase(
                hadm_id=h,
                diagnosis_icd=dx.get(h, []),
                labs=sorted(labs.get(h, set())),
                drugs=sorted(drugs.get(h, set())),
                procedures_icd=proc.get(h, []),
                icd_version="/".join(sorted(icd_versions.get(h, set()))),
            )
            for h in hadm_ids
        ]
        # Audit-safe: log count + ids only, never row contents.
        log.info("Loaded %d MIMIC demo structured cases (hadm_ids: %s)",
                 len(cases), [c.hadm_id for c in cases])
        return cases
