"""Run the MIMIC-CDM evaluation with a selectable model backend.

Usage
-----
    python evals/run_cdm.py --model mock        # CI default; deterministic 1.000
    python evals/run_cdm.py --model meditron    # local Meditron-7B via Ollama

The 'meditron' backend is LOCAL-ONLY. Prerequisites:
    brew install ollama          # or see https://ollama.com/download
    ollama serve                 # start the local server (separate terminal)
    ollama pull meditron:7b      # ~3.8 GB, one-time
    pip install ollama           # the Python client

RESEARCH USE ONLY -- Meditron's authors recommend against clinical deployment.
This benchmarks the model on synthetic CDM cases (zero PHI). Read-and-grade
only: no patient writes, governance invariant committed == 0 holds.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="MIMIC-CDM model benchmark")
    parser.add_argument(
        "--model", default="mock", choices=["mock", "meditron"],
        help="Backend: 'mock' (deterministic upper bound) or 'meditron' (local).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON summary.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
    )

    report = run_governance_cdm_eval(
        use_mock=(args.model == "mock"), backend=args.model,
    )

    if args.json:
        print(json.dumps(report.summary(), indent=2))
    else:
        report.print_summary()
        if args.model == "mock":
            print(
                "\n  NOTE: mock backend echoes gold labels -> 1.000 by "
                "construction. This verifies the harness, not model quality.\n"
                "  For a real measurement: python evals/run_cdm.py --model meditron"
            )

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
