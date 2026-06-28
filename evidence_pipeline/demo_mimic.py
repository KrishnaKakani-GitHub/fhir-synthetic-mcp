"""CLI demo: run the end-to-end MIMIC-IV pipeline and print the outcome metric.

Usage
-----
  # With real MIMIC-IV Demo files (free PhysioNet account required):
  python evidence_pipeline/demo_mimic.py --notes-dir /path/to/mimic-iv-demo/note

  # With real MIMIC-CDM dataset:
  python evidence_pipeline/demo_mimic.py --cdm-dir /path/to/mimic-iv-ext-cdm

  # Without any MIMIC-IV files (uses synthetic data, zero PHI):
  python evidence_pipeline/demo_mimic.py

Expected output (synthetic 10 notes):
  Extracted 62 LOINC-coded observations from 10 synthetic discharge notes,
  100% validated, 0% rejected by deterministic gate,
  0 committed without human approval.

PhysioNet access:
  MIMIC-IV Demo  : physionet.org/content/mimic-iv-demo/  (free account, no CITI)
  MIMIC-IV-Ext-CDM: physionet.org/content/mimic-iv-ext-cdm/ (free account, no CITI)
  Full MIMIC-IV  : physionet.org/content/mimiciv/ (CITI + DUA required)

PHI NOTE: This script never logs raw text. All output uses note IDs + codes.
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the MIMIC-IV evidence pipeline and print outcome metric."
    )
    parser.add_argument(
        "--notes-dir",
        default=None,
        help="Path to MIMIC-IV notes directory containing discharge.csv. "
             "Omit to use 10 synthetic notes.",
    )
    parser.add_argument(
        "--cdm-dir",
        default=None,
        help="Path to MIMIC-IV-Ext-CDM directory for LLM decision-making eval. "
             "Omit to use synthetic CDM cases.",
    )
    parser.add_argument(
        "--max-notes", type=int, default=100,
        help="Maximum discharge notes to process (default 100).",
    )
    args = parser.parse_args(argv)

    # --- Stage 1-4: LOINC extraction + deterministic gate ---
    if args.notes_dir:
        from evidence_pipeline.datasets.mimic import MIMICLoader
        loader = MIMICLoader(args.notes_dir, max_notes=args.max_notes)
        notes = loader.load()
        print(f"Loaded {len(notes)} MIMIC-IV notes from {args.notes_dir}")
    else:
        from evidence_pipeline.datasets.mimic import generate_synthetic_notes
        notes = generate_synthetic_notes()
        print(f"Using {len(notes)} synthetic discharge notes (zero PHI).")

    from evidence_pipeline.pipeline.end_to_end import run_pipeline
    metrics, gate = run_pipeline(notes)
    metrics.print_metric()
    print(f"  Pending human review queue : {gate.pending_count} observations")
    print(f"  Audit log entries          : {len(gate.audit_log())}")

    # --- Stage 5: CDM LLM reasoning eval ---
    print()
    print("=" * 72)
    print("  MIMIC-CDM: LLM Clinical Decision Making Eval")
    print("  Rubric: Hager et al., Nature Medicine 2024")
    print("=" * 72)
    from evidence_pipeline.evals.clinical_decision import run_cdm_eval
    from evidence_pipeline.datasets.mimic_cdm import CDMLoader
    cdm_loader = CDMLoader(data_dir=args.cdm_dir)
    cdm_cases = cdm_loader.load()
    cdm_report = run_cdm_eval(cdm_cases)
    cdm_report.print_summary()


if __name__ == "__main__":
    main(sys.argv[1:])
