"""Clinical Evidence Intelligence Pipeline — 3-layer eval stack.

Layer 1 — Capability (entity linking)
  entity_linking   EvalResult, EvalReport, top_k_accuracy, mrr
  runner           run_smoke(), run_full(corpus_dir)

Layer 2 — Safety (hallucination + PHI guard)
  safety           SafetyResult, SafetyReport, run_safety_sweep()

Layer 3 — Product (AMIE-style multi-axis golden dataset)
  product          ProductResult, ProductReport, run_product_eval()

Grounding (FACTS Grounding baseline)
  grounding        GroundingResult, GroundingReport,
                   ground_output(), ground_crosswalk()

Reference eval taxonomies
  Google DeepMind FACTS Grounding: deepmind.google/research/evals/
  Google DeepMind AMIE: research.google/blog/amie-gains-vision-...
"""
from evidence_pipeline.evals.entity_linking import (
    EvalReport,
    EvalResult,
    mrr,
    top_k_accuracy,
)
from evidence_pipeline.evals.grounding import (
    GroundingReport,
    GroundingResult,
    GroundingViolation,
    ground_crosswalk,
    ground_output,
)
from evidence_pipeline.evals.product import (
    CI_THRESHOLD,
    GOLDEN_DATASET,
    AxisScores,
    GoldenCase,
    ProductReport,
    ProductResult,
    run_product_eval,
)
from evidence_pipeline.evals.safety import (
    SafetyReport,
    SafetyResult,
    SafetyViolation,
    run_safety_sweep,
)

__all__ = [
    # Layer 1
    "EvalResult", "EvalReport", "top_k_accuracy", "mrr",
    # Layer 2
    "SafetyViolation", "SafetyResult", "SafetyReport", "run_safety_sweep",
    # Layer 3
    "GoldenCase", "AxisScores", "ProductResult", "ProductReport",
    "run_product_eval", "GOLDEN_DATASET", "CI_THRESHOLD",
    # Grounding
    "GroundingViolation", "GroundingResult", "GroundingReport",
    "ground_output", "ground_crosswalk",
]
