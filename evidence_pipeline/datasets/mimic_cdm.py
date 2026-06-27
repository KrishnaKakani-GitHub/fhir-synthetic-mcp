"""MIMIC-IV Clinical Decision Making (MIMIC-CDM) dataset loader.

Source
------
Hager, P., Jungmann, F., Holland, R. et al.
  "Evaluation and mitigation of the limitations of large language models
   in clinical decision-making."
  Nature Medicine (2024). https://doi.org/10.1038/s41591-024-03097-1

Dataset (PhysioNet, free account required):
  https://physionet.org/content/mimic-iv-ext-cdm/
  No CITI training required for CDM dataset.

What MIMIC-CDM measures
------------------------
Given a patient's clinical presentation (history, labs, imaging, notes),
can an LLM make correct clinical decisions across 4 axes:
  1. Diagnosis     -- primary ICD-10 diagnosis
  2. Treatment     -- medications / interventions (RxNorm / procedure codes)
  3. Lab ordering  -- LOINC-coded lab tests to order
  4. Procedures    -- CPT-coded procedures to order

This is the eval that turns the evidence pipeline from "retrieves evidence"
into "grades whether the LLM makes correct decisions using that evidence."

PHI NOTE: MIMIC-CDM is derived from de-identified MIMIC-IV data.
Treat all loaded case data as sensitive.
Log case_id only, never raw clinical text.
COMPLIANCE: Must have PhysioNet DUA for full CDM dataset access.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CDMCase:
    """One MIMIC-CDM clinical decision making case.

    Axes match the four scored dimensions in Hager et al. 2024.
    """
    case_id: str
    patient_id: str           # de-identified
    hadm_id: str              # de-identified hospital admission
    presentation: str         # clinical summary (de-identified or synthetic)
    # Gold labels
    diagnosis_icd: list[str]  # expected ICD-10 codes
    treatment_rxnorm: list[str]   # expected RxNorm drug IDs
    labs_loinc: list[str]     # expected LOINC codes to order
    procedures_cpt: list[str] # expected CPT codes
    # Metadata
    primary_condition: str = ""
    is_synthetic: bool = False


@dataclass
class CDMScore:
    """Per-axis CDM score for one case. Range [0.0, 1.0]."""
    case_id: str
    diagnosis_score: float      # F1 over ICD-10 codes
    treatment_score: float      # F1 over RxNorm IDs
    lab_ordering_score: float   # F1 over LOINC codes
    procedure_score: float      # F1 over CPT codes

    @property
    def composite(self) -> float:
        return (
            self.diagnosis_score +
            self.treatment_score +
            self.lab_ordering_score +
            self.procedure_score
        ) / 4.0

    def to_dict(self) -> dict[str, float]:
        return {
            "case_id": self.case_id,
            "diagnosis_score": round(self.diagnosis_score, 4),
            "treatment_score": round(self.treatment_score, 4),
            "lab_ordering_score": round(self.lab_ordering_score, 4),
            "procedure_score": round(self.procedure_score, 4),
            "composite": round(self.composite, 4),
        }


def _f1(predicted: list[str], gold: list[str]) -> float:
    """Token-level F1 between predicted and gold code sets."""
    if not gold:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    pred_set = set(p.strip().upper() for p in predicted)
    gold_set = set(g.strip().upper() for g in gold)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_case(
    case: CDMCase,
    predicted_icd: list[str],
    predicted_rxnorm: list[str],
    predicted_loinc: list[str],
    predicted_cpt: list[str],
) -> CDMScore:
    """Score LLM predictions against CDM gold labels across all 4 axes."""
    return CDMScore(
        case_id=case.case_id,
        diagnosis_score=_f1(predicted_icd, case.diagnosis_icd),
        treatment_score=_f1(predicted_rxnorm, case.treatment_rxnorm),
        lab_ordering_score=_f1(predicted_loinc, case.labs_loinc),
        procedure_score=_f1(predicted_cpt, case.procedures_cpt),
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class CDMLoader:
    """Load MIMIC-CDM cases from a PhysioNet CDM data directory.

    Expected layout (MIMIC-IV-Ext-CDM v1.0):
      <data_dir>/cases/*.json

    Falls back to generate_synthetic_cdm_cases() if directory not found.

    PhysioNet CDM dataset:
      https://physionet.org/content/mimic-iv-ext-cdm/
    """

    def __init__(self, data_dir: str | Path | None = None,
                 max_cases: int = 100) -> None:
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_cases = max_cases

    def load(self) -> list[CDMCase]:
        if self.data_dir is None or not self.data_dir.exists():
            log.warning(
                "MIMIC-CDM data_dir not found (%s). "
                "Returning synthetic CDM cases for CI/demo.",
                self.data_dir,
            )
            return generate_synthetic_cdm_cases()
        return self._load_from_dir()

    def _load_from_dir(self) -> list[CDMCase]:
        cases_dir = self.data_dir / "cases"
        if not cases_dir.exists():
            cases_dir = self.data_dir
        json_files = sorted(cases_dir.glob("*.json"))[: self.max_cases]
        cases = []
        for p in json_files:
            try:
                with p.open() as f:
                    data = json.load(f)
                cases.append(CDMCase(
                    case_id=str(data.get("case_id", p.stem)),
                    patient_id=str(data.get("subject_id", "")),
                    hadm_id=str(data.get("hadm_id", "")),
                    presentation=data.get("presentation", ""),
                    diagnosis_icd=data.get("diagnosis_icd", []),
                    treatment_rxnorm=data.get("treatment_rxnorm", []),
                    labs_loinc=data.get("labs_loinc", []),
                    procedures_cpt=data.get("procedures_cpt", []),
                    primary_condition=data.get("primary_condition", ""),
                    is_synthetic=False,
                ))
            except Exception as exc:
                log.warning("Skipping %s: %s", p, exc)
        log.info("Loaded %d CDM cases from %s", len(cases), self.data_dir)
        return cases


# ---------------------------------------------------------------------------
# Synthetic CDM cases (CI + demo -- zero PHI)
# ---------------------------------------------------------------------------

_CDM_SYNTHETIC: list[dict] = [
    {
        "case_id": "CDM001",
        "patient_id": "SYNCDM001",
        "hadm_id": "ADMCDM001",
        "primary_condition": "PNH",
        "presentation": (
            "Synthetic: 34-year-old with fatigue, dark urine, abdominal pain. "
            "Hemoglobin 7.8 g/dL, LDH 1420 U/L, haptoglobin undetectable. "
            "Flow cytometry: CD55/CD59 deficient clone 62%. "
            "No thrombosis. QUESTION: What is the diagnosis, treatment, key labs, procedures?"
        ),
        "diagnosis_icd": ["D59.5"],
        "treatment_rxnorm": ["727910"],         # eculizumab
        "labs_loinc": ["718-7", "2532-0", "13945-1", "777-3"],  # Hgb, LDH, haptoglobin, platelets
        "procedures_cpt": ["86850"],            # flow cytometry
        "is_synthetic": True,
    },
    {
        "case_id": "CDM002",
        "patient_id": "SYNCDM002",
        "hadm_id": "ADMCDM002",
        "primary_condition": "T2DM",
        "presentation": (
            "Synthetic: 58-year-old with poorly controlled diabetes. "
            "HbA1c 9.4%, fasting glucose 248 mg/dL, eGFR 54 mL/min. "
            "QUESTION: Diagnosis, medication adjustment, monitoring labs?"
        ),
        "diagnosis_icd": ["E11", "E11.65"],
        "treatment_rxnorm": ["6809", "274783"],  # metformin, insulin glargine
        "labs_loinc": ["4548-4", "2345-7", "48642-3"],  # HbA1c, glucose, eGFR
        "procedures_cpt": [],
        "is_synthetic": True,
    },
    {
        "case_id": "CDM003",
        "patient_id": "SYNCDM003",
        "hadm_id": "ADMCDM003",
        "primary_condition": "HF",
        "presentation": (
            "Synthetic: 71-year-old with dyspnea, lower extremity edema. "
            "BNP 1840 pg/mL, EF 30% on echo. Creatinine 1.8 mg/dL. "
            "QUESTION: Diagnosis, diuresis plan, monitoring?"
        ),
        "diagnosis_icd": ["I50", "I50.9"],
        "treatment_rxnorm": ["4603", "41493"],  # furosemide, carvedilol
        "labs_loinc": ["42637-9", "2160-0", "2823-3"],  # BNP, creatinine, potassium
        "procedures_cpt": ["93306"],            # echocardiogram
        "is_synthetic": True,
    },
    {
        "case_id": "CDM004",
        "patient_id": "SYNCDM004",
        "hadm_id": "ADMCDM004",
        "primary_condition": "CKD",
        "presentation": (
            "Synthetic: 66-year-old with CKD stage 4. Creatinine 3.6, eGFR 18. "
            "Potassium 5.3, hemoglobin 9.1 (anemia of CKD). "
            "QUESTION: Diagnosis, management, monitoring labs?"
        ),
        "diagnosis_icd": ["N18.9", "N18.4"],
        "treatment_rxnorm": ["203150"],         # epoetin alfa
        "labs_loinc": ["2160-0", "48642-3", "718-7", "2823-3"],
        "procedures_cpt": [],
        "is_synthetic": True,
    },
    {
        "case_id": "CDM005",
        "patient_id": "SYNCDM005",
        "hadm_id": "ADMCDM005",
        "primary_condition": "SickleCell",
        "presentation": (
            "Synthetic: 24-year-old with SCD in vaso-occlusive crisis. "
            "Hemoglobin 6.4, reticulocytes 12.8%, LDH 680. Severe pain. "
            "QUESTION: Diagnosis, pain management, transfusion decision?"
        ),
        "diagnosis_icd": ["D57.1"],
        "treatment_rxnorm": ["202462", "8782"],  # hydroxyurea, morphine
        "labs_loinc": ["718-7", "17849-1", "2532-0"],
        "procedures_cpt": ["36430"],            # blood transfusion
        "is_synthetic": True,
    },
    {
        "case_id": "CDM006",
        "patient_id": "SYNCDM006",
        "hadm_id": "ADMCDM006",
        "primary_condition": "HTN",
        "presentation": (
            "Synthetic: 55-year-old with BP 192/114 mmHg, headache, no end-organ damage. "
            "Creatinine 1.5. QUESTION: Diagnosis, acute management, outpatient plan?"
        ),
        "diagnosis_icd": ["I10", "R03.0"],
        "treatment_rxnorm": ["17767", "214354"],  # amlodipine, lisinopril
        "labs_loinc": ["2160-0", "2951-2", "2823-3"],
        "procedures_cpt": [],
        "is_synthetic": True,
    },
    {
        "case_id": "CDM007",
        "patient_id": "SYNCDM007",
        "hadm_id": "ADMCDM007",
        "primary_condition": "AFib",
        "presentation": (
            "Synthetic: 68-year-old with new AFib, HR 136 bpm. Troponin 0.04. "
            "No prior anticoagulation. CHA2DS2-VASc 4. "
            "QUESTION: Rate control, anticoagulation decision, monitoring?"
        ),
        "diagnosis_icd": ["I48.91"],
        "treatment_rxnorm": ["41493", "1364435"],  # metoprolol, apixaban
        "labs_loinc": ["49563-0", "2160-0"],
        "procedures_cpt": ["93000"],            # ECG
        "is_synthetic": True,
    },
    {
        "case_id": "CDM008",
        "patient_id": "SYNCDM008",
        "hadm_id": "ADMCDM008",
        "primary_condition": "COPD",
        "presentation": (
            "Synthetic: 72-year-old with COPD exacerbation. SpO2 84% on room air, "
            "increased dyspnea and purulent sputum. WBC 12.1. "
            "QUESTION: Diagnosis, acute management, discharge plan?"
        ),
        "diagnosis_icd": ["J44.9", "J44.1"],
        "treatment_rxnorm": ["7512", "10154"],  # albuterol, prednisone
        "labs_loinc": ["59408-5", "6690-2"],
        "procedures_cpt": ["94010"],            # spirometry
        "is_synthetic": True,
    },
    {
        "case_id": "CDM009",
        "patient_id": "SYNCDM009",
        "hadm_id": "ADMCDM009",
        "primary_condition": "Gaucher",
        "presentation": (
            "Synthetic: 31-year-old with Gaucher Disease Type 1. "
            "Hemoglobin 9.6, platelets 62, ferritin 840. "
            "Beta-glucocerebrosidase activity low. QUESTION: Treatment initiation?"
        ),
        "diagnosis_icd": ["E75.22"],
        "treatment_rxnorm": ["1294103"],        # imiglucerase
        "labs_loinc": ["718-7", "777-3", "2276-4"],
        "procedures_cpt": ["86255"],            # enzyme activity assay
        "is_synthetic": True,
    },
    {
        "case_id": "CDM010",
        "patient_id": "SYNCDM010",
        "hadm_id": "ADMCDM010",
        "primary_condition": "RA",
        "presentation": (
            "Synthetic: 47-year-old with active RA, morning stiffness >1 hr, "
            "multiple swollen joints. DAS28 score 5.1. "
            "QUESTION: Disease activity assessment, DMARD escalation?"
        ),
        "diagnosis_icd": ["M05.9", "M06.9"],
        "treatment_rxnorm": ["41493"],          # methotrexate (RxNorm placeholder)
        "labs_loinc": ["718-7", "6690-2"],
        "procedures_cpt": [],
        "is_synthetic": True,
    },
]


def generate_synthetic_cdm_cases() -> list[CDMCase]:
    """Return 10 synthetic CDM cases for CI use. Zero real patient data."""
    return [
        CDMCase(
            case_id=d["case_id"],
            patient_id=d["patient_id"],
            hadm_id=d["hadm_id"],
            presentation=d["presentation"],
            diagnosis_icd=d["diagnosis_icd"],
            treatment_rxnorm=d["treatment_rxnorm"],
            labs_loinc=d["labs_loinc"],
            procedures_cpt=d["procedures_cpt"],
            primary_condition=d["primary_condition"],
            is_synthetic=True,
        )
        for d in _CDM_SYNTHETIC
    ]
