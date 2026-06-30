"""Run the MIMIC-CDM evaluation, or a structured demo on real MIMIC demo data.

Two modes
---------
1. SCORED BENCHMARK (default) -- Meditron/mock on synthetic labeled CDM cases,
   graded by the Hager 4-axis rubric. Produces F1 scores.

    python evals/run_cdm.py --model mock        # CI default; deterministic 1.000
    python evals/run_cdm.py --model meditron    # local Meditron-7B via Ollama

2. STRUCTURED DEMO -- Meditron reads REAL de-identified MIMIC-IV *demo*
   structured data (coded labs + drugs per admission; the demo has NO free-text
   notes) and proposes a diagnosis. The demo's coded gold ICD diagnoses are
   present (mixed ICD-9 / ICD-10), shown for reference, unscored.

    python evals/run_cdm.py --model meditron --demo-dir /path/OUTSIDE/repo/mimic-demo

The MIMIC-IV demo excludes free-text notes, so there is no "discharge note" to
read. We use the coded hosp/ tables (diagnoses_icd, labevents->LOINC/label,
prescriptions, procedures_icd) instead.

The 'meditron' backend is LOCAL-ONLY. Prerequisites:
    brew install ollama && brew services start ollama
    ollama pull meditron:7b      # ~3.8 GB, one-time
    pip install ollama

Demo data: download the free subset from
    https://physionet.org/content/mimic-iv-demo/   (free account, no CITI)
Keep it OUTSIDE the repo. The loader refuses to read data from inside the repo
tree so PHI can never be committed.

RESEARCH USE ONLY -- Meditron's authors recommend against clinical deployment.
PHI: read-only, no patient writes, governance invariant committed == 0 holds.
hadm_id-only logging; raw clinical values and dates are never logged or surfaced.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make evals/ and evidence_pipeline/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.mimic_cdm_eval import run_governance_cdm_eval


def _run_scored(model: str, as_json: bool) -> int:
    report = run_governance_cdm_eval(
        use_mock=(model == "mock"), backend=model,
    )
    if as_json:
        print(json.dumps(report.summary(), indent=2))
    else:
        report.print_summary()
        if model == "mock":
            print(
                "\n  NOTE: mock backend echoes gold labels -> 1.000 by "
                "construction. This verifies the harness, not model quality.\n"
                "  For a real measurement: python evals/run_cdm.py --model meditron"
            )
    return 0 if report.passed else 1


def _present(case) -> str:
    """Build a structured presentation string from coded fields (no free text)."""
    labs = ", ".join(case.labs[:12]) or "none recorded"
    drugs = ", ".join(case.drugs[:12]) or "none recorded"
    return (f"Admission with the following ordered labs: {labs}. "
            f"Medications administered: {drugs}. "
            f"Propose the most likely diagnosis as ICD-10 codes.")


def _run_demo(model: str, demo_dir: str, limit: int) -> int:
    """Structured demonstration on real MIMIC demo coded tables."""
    from evals.cdm_agent import make_backend
    from evals.demo_structured import DemoStructuredLoader

    if model != "meditron":
        print("  Demo mode is intended for --model meditron (real model on real "
              "data). The mock backend has nothing to propose here.")
    loader = DemoStructuredLoader(demo_dir)
    cases = loader.load(limit=limit)
    if not cases:
        print(f"  No structured cases found under {demo_dir}/hosp.")
        return 1

    agent = make_backend(model, gold_lookup={})

    print("\nMIMIC-IV Demo \u2014 Structured CDM Demonstration (NOT a benchmark)")
    print("=" * 68)
    print(f"  backend : {model}   admissions: {len(cases)}   source: {demo_dir}")
    print("  Real de-identified coded data (no free-text notes in the demo).")
    print("  Model proposes a diagnosis from labs + meds; gold ICD shown for")
    print("  reference only (not a scored benchmark).")
    print("=" * 68)

    for case in cases:
        resp = agent.propose(case.hadm_id, _present(case))
        print(f"\n  {case.summary_line()}")
        print(f"    model diagnosis (ICD-10) : {resp.proposed_icd or '\u2014'}")
        gold_ver = f" [ICD-{case.icd_version}]" if case.icd_version else ""
        print(f"    gold diagnosis{gold_ver} : {case.diagnosis_icd[:6] or '\u2014'}")
        if case.icd_version and "9" in case.icd_version.split("/"):
            print("      note: gold is ICD-9; model emits ICD-10 \u2014 not directly comparable.")

    print("\n  (Demonstration only \u2014 reference comparison, not a scored metric.)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MIMIC-CDM model benchmark (scored) or structured demo (unscored)")
    parser.add_argument(
        "--model", default="mock", choices=["mock", "meditron"],
        help="Backend: 'mock' (deterministic upper bound) or 'meditron' (local).",
    )
    parser.add_argument(
        "--demo-dir", default=None, metavar="DIR",
        help="Path to real MIMIC-IV demo dir (OUTSIDE repo) -> structured demo.",
    )
    parser.add_argument(
        "--limit", type=int, default=5,
        help="Max admissions to run (demo mode only; default 5).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON (scored mode).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
    )

    if args.demo_dir:
        return _run_demo(args.model, args.demo_dir, args.limit)
    return _run_scored(args.model, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
