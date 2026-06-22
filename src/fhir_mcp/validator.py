"""Deterministic clinical validation gate.

Called from store.stage_write() before a proposal can enter the pending queue.
This is a hard gate — it cannot be bypassed by the agent.

Validation rules are loaded from data/loinc_rules.json at module import time.
The file path can be overridden with the FHIR_MCP_LOINC_RULES env var for
testing or custom deployments.

Validation checks (in order):
  1. Code is in the LOINC registry (unknown codes rejected by default)
  2. Value is not below the absolute reject threshold (physiologically impossible)
  3. Value is within the defined min-max range
  4. Unit matches the canonical unit for the code
  5. Flag: value is above the clinical review threshold (allowed but noted)

Design note: validating unit strings with exact equality is intentional.
Allowing unit aliasing (e.g. 'beats/min' for '/min') opens a class of silent
misinterpretation bugs. Callers must normalise units before proposing.

PHI NOTE: Validation operates on code + value + unit only. No patient data
is accessed here. This module has no PHI touchpoints.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import ProposedObservation

_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "loinc_rules.json"
)
_RULES_PATH = Path(
    os.environ.get("FHIR_MCP_LOINC_RULES", str(_DEFAULT_RULES_PATH))
)


def _load_rules(path: Path) -> dict[str, Any]:
    """Load LOINC rules from JSON, stripping comment keys."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


_RULES: dict[str, Any] = _load_rules(_RULES_PATH)


@dataclass
class ValidationResult:
    """Outcome of a single validation run."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rule: dict[str, Any] | None = None  # matched rule, if any

    def __bool__(self) -> bool:
        return self.ok


class ValidationError(Exception):
    """Raised when a proposed observation fails deterministic validation."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__("Validation failed: " + "; ".join(violations))
        self.violations = violations


def validate_observation(
    proposed: ProposedObservation,
    *,
    strict_unknown: bool = True,
) -> ValidationResult:
    """Validate a proposed observation against LOINC rules.

    Args:
        proposed: The proposed observation to validate.
        strict_unknown: If True (default), reject codes not in the registry.
                        Set False for research/exploration contexts where
                        novel LOINC codes may be used before rules are added.

    Returns:
        ValidationResult with ok=True if the proposal passes, or
        ok=False with violations listed.
    """
    code = proposed.code.strip()
    value = proposed.value
    unit = proposed.unit.strip()
    violations: list[str] = []
    warnings: list[str] = []

    rule = _RULES.get(code)
    if rule is None:
        if strict_unknown:
            violations.append(
                f"LOINC code '{code}' is not in the clinical registry. "
                "Add it to data/loinc_rules.json before proposing."
            )
            return ValidationResult(ok=False, violations=violations)
        else:
            warnings.append(
                f"LOINC code '{code}' has no validation rules — "
                "proposal accepted in non-strict mode but review is advised."
            )
            return ValidationResult(ok=True, warnings=warnings)

    display = rule.get("display", code)
    expected_unit = rule.get("unit", "")
    vmin = rule.get("min")
    vmax = rule.get("max")
    reject_below = rule.get("reject_below")
    flag_above = rule.get("flag_above")

    # 1. Hard reject: below physiological minimum (e.g. negative HR)
    if reject_below is not None and value < reject_below:
        violations.append(
            f"{display} ({code}): value {value} {unit} is below the "
            f"absolute minimum ({reject_below}). This is physiologically impossible."
        )

    # 2. Range check
    if vmin is not None and value < vmin:
        violations.append(
            f"{display} ({code}): value {value} {unit} is below the valid "
            f"minimum ({vmin} {expected_unit}). Verify the observation."
        )
    elif vmax is not None and value > vmax:
        violations.append(
            f"{display} ({code}): value {value} {unit} is above the valid "
            f"maximum ({vmax} {expected_unit}). Verify the observation."
        )

    # 3. Unit check (exact match required; see module docstring)
    if expected_unit and unit != expected_unit:
        violations.append(
            f"{display} ({code}): unit '{unit}' does not match expected "
            f"'{expected_unit}'. Normalise the unit before proposing."
        )

    if violations:
        return ValidationResult(ok=False, violations=violations, rule=rule)

    # 4. Clinical flag (warning only — proposal is accepted)
    if flag_above is not None and value > flag_above:
        note = rule.get("clinical_note", "")
        warnings.append(
            f"{display} ({code}): value {value} {unit} exceeds the clinical "
            f"flag threshold ({flag_above}). {note}"
        )

    return ValidationResult(ok=True, warnings=warnings, rule=rule)


def get_rules() -> dict[str, Any]:
    """Return the loaded LOINC rules (read-only reference)."""
    return _RULES


def reload_rules(path: Path | None = None) -> None:
    """Reload rules from disk. Useful after updating loinc_rules.json."""
    global _RULES
    _RULES = _load_rules(path or _RULES_PATH)
