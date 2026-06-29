"""Tests for the three-layer eval stack aggregator (evals/stack.py)."""
from __future__ import annotations

from evidence_pipeline.evals.stack import StackReport, run_stack


def test_stack_runs_all_three_layers():
    report = run_stack()
    assert report.capability["layer"] == "capability"
    assert report.safety["layer"] == "safety"
    assert report.product["layer"] == "product"


def test_capability_layer_reports_entity_linking_metrics():
    report = run_stack()
    c = report.capability
    for key in ("top_1_accuracy", "top_5_accuracy", "mrr", "coverage"):
        assert key in c
        assert 0.0 <= c[key] <= 1.0


def test_safety_layer_reports_grounding_and_sweep():
    report = run_stack()
    s = report.safety
    assert "grounding_score" in s and 0.0 <= s["grounding_score"] <= 1.0
    assert "safety_violations" in s
    assert isinstance(s["fully_grounded"], bool)


def test_product_layer_reports_composite():
    report = run_stack()
    p = report.product
    assert "mean_composite" in p
    assert p["mean_composite"] >= p["ci_threshold"]


def test_stack_all_passed_is_conjunction():
    report = run_stack()
    expected = (
        report.capability["passed"]
        and report.safety["passed"]
        and report.product["passed"]
    )
    assert report.all_passed is expected


def test_stack_to_dict_shape():
    report = run_stack()
    d = report.to_dict()
    assert d["stack"] == "capability/safety/product"
    assert set(d["layers"]) == {"capability", "safety", "product"}
    assert isinstance(d["all_passed"], bool)


def test_stack_passes_on_synthetic_data():
    """On the synthetic/demo corpus all three layers should pass."""
    report = run_stack()
    assert report.all_passed is True
