"""Three-layer evaluation stack for the evidence pipeline.

Presents the existing eval layers as one labeled capability / safety / product
stack, following the DeepMind-style taxonomy for evaluating AI systems:

    Capability  -- what can the pipeline do?
                   entity-linking: top-k accuracy, MRR, coverage
                   (evidence_pipeline/evals/entity_linking.py + runner.py)

    Safety      -- does it avoid harmful / ungrounded output?
                   grounding score (zero unattributed claims) + safety sweep
                   (evidence_pipeline/evals/grounding.py, safety.py)

    Product     -- does it work in the actual application?
                   end-to-end golden-case axis scoring
                   (evidence_pipeline/evals/product.py)

This module adds NO new eval logic -- it aggregates the reports the three
layers already produce and renders them as a single stack. Each layer remains
independently runnable; this is the unified view.

Reasoning frameworks cited per layer:
  Capability -> BioEL entity linking. Sung, Jeon, Lee, Kang (2020),
                Biomedical Entity Representations with Synonym Marginalization,
                arXiv:2005.00239.
  Safety     -> FACTS Grounding. Jacovi et al. (2025), arXiv:2501.03200.
  Product    -> custom product evals over MedQuAD golden cases.

PHI: all layers run on synthetic / public NIH (MedQuAD) data. Zero patient
data. Read-only; no writes, governance invariant preserved upstream.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any

from evidence_pipeline.evals.grounding import ground_crosswalk
from evidence_pipeline.evals.product import CI_THRESHOLD, run_product_eval
from evidence_pipeline.evals.runner import run_smoke
from evidence_pipeline.evals.safety import run_safety_sweep

log = logging.getLogger(__name__)

# Gates per layer. Capability uses top-1 on the covered crosswalk; safety
# requires a clean sweep + full grounding; product uses the existing composite.
CAPABILITY_TOP1_GATE = 1.0
GROUNDING_GATE = 1.0


@dataclass
class StackReport:
    """Aggregated three-layer stack result."""
    capability: dict[str, Any]
    safety: dict[str, Any]
    product: dict[str, Any]

    @property
    def all_passed(self) -> bool:
        return (
            self.capability["passed"]
            and self.safety["passed"]
            and self.product["passed"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stack": "capability/safety/product",
            "all_passed": self.all_passed,
            "layers": {
                "capability": self.capability,
                "safety": self.safety,
                "product": self.product,
            },
        }

    def print_summary(self) -> None:
        def mark(ok: bool) -> str:
            return "PASS" if ok else "FAIL"

        c, s, p = self.capability, self.safety, self.product
        print("\nEvidence Pipeline — 3-Layer Eval Stack (capability/safety/product)")
        print("=" * 66)
        print(f"  [1] CAPABILITY  {mark(c['passed'])}   (BioEL entity linking, Sung et al. 2020)")
        print(f"        top-1 accuracy : {c['top_1_accuracy']:.3f}")
        print(f"        top-5 accuracy : {c['top_5_accuracy']:.3f}")
        print(f"        MRR            : {c['mrr']:.3f}")
        print(f"        coverage       : {c['coverage']:.3f}")
        print(f"  [2] SAFETY      {mark(s['passed'])}   (FACTS grounding + safety sweep)")
        print(f"        grounding score: {s['grounding_score']:.3f}  (fully_grounded={s['fully_grounded']})")
        print(f"        safety sweep   : {s['safety_checks_passed']} passed, {s['safety_violations']} violations")
        print(f"  [3] PRODUCT     {mark(p['passed'])}   (end-to-end golden-case eval)")
        print(f"        composite      : {p['mean_composite']:.3f}  (gate >= {p['ci_threshold']})")
        print("=" * 66)
        print(f"  STACK: {'ALL LAYERS PASS' if self.all_passed else 'ONE OR MORE LAYERS FAIL'}")


def run_stack() -> StackReport:
    """Run all three layers and aggregate into one stack report."""
    # ---- Layer 1: Capability (entity linking) ----
    el = run_smoke()
    top1 = el.top_k_accuracy(1)
    capability = {
        "layer": "capability",
        "framework": "BioEL entity linking (Sung et al. 2020)",
        "top_1_accuracy": round(top1, 4),
        "top_3_accuracy": round(el.top_k_accuracy(3), 4),
        "top_5_accuracy": round(el.top_k_accuracy(5), 4),
        "mrr": round(el.mrr, 4),
        "coverage": round(el.coverage, 4),
        "passed": top1 >= CAPABILITY_TOP1_GATE,
    }
    log.info("[STACK] capability top-1=%.3f mrr=%.3f", top1, el.mrr)

    # ---- Layer 2: Safety (grounding + safety sweep) ----
    grounding = ground_crosswalk()
    sweep = run_safety_sweep()
    safety_passed = (
        grounding.fully_grounded
        and grounding.mean_grounding_score >= GROUNDING_GATE
        and sweep.passed
    )
    safety = {
        "layer": "safety",
        "framework": "FACTS grounding (Jacovi et al. 2025) + safety sweep",
        "grounding_score": round(grounding.mean_grounding_score, 4),
        "fully_grounded": grounding.fully_grounded,
        "grounding_violations": grounding.total_violations,
        "safety_checks_passed": sweep.summary()["passed"],
        "safety_violations": sweep.total_violations,
        "passed": safety_passed,
    }
    log.info("[STACK] safety grounding=%.3f violations=%d",
             grounding.mean_grounding_score, sweep.total_violations)

    # ---- Layer 3: Product (end-to-end golden cases) ----
    prod = run_product_eval()
    product = {
        "layer": "product",
        "framework": "end-to-end MedQuAD golden-case eval",
        "mean_composite": round(prod.mean_composite, 4),
        "ci_threshold": prod.ci_threshold,
        "passed": prod.mean_composite >= prod.ci_threshold,
    }
    log.info("[STACK] product composite=%.3f", prod.mean_composite)

    return StackReport(capability=capability, safety=safety, product=product)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evidence pipeline 3-layer eval stack (capability/safety/product)")
    parser.add_argument("--json", action="store_true",
                        help="Emit machine-readable JSON instead of the summary.")
    parser.add_argument("--output", default=None, help="Save JSON report to a file.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    report = run_stack()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"\nReport saved: {args.output}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
