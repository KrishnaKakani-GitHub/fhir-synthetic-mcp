"""Pluggable CDM agent backends for the MIMIC clinical-decision-making eval.

This module supplies the *model under evaluation* for ``mimic_cdm_eval.py``.
Each backend takes a :class:`CDMCase` presentation and returns an
:class:`AgentCDMResponse` (the same dataclass the mock produces), so the eval
machinery and Hager-rubric scoring are identical regardless of backend.

Backends
--------
``mock``      Deterministic gold-label echo. CI default. No tokens, no network.
              Verifies the harness end-to-end; always scores 1.000 by construction.
``meditron``  Real Meditron-7B (Llama-2-7B medical) via a local Ollama server.
              Same model *family* as Hager et al.'s clinical 70B, sized to run
              on a laptop. LOCAL-ONLY: never runs in CI (no weights / no GPU /
              registry unreachable on the runner).

Meditron is a *base* model (no instruction tuning), so the prompt uses
few-shot exemplars and the parser is tolerant of freeform output, per EPFL's
usage guidance (in-context learning with k demonstrations).

PHI / governance
----------------
Read-and-grade only. This module never writes to a patient store and never
commits an observation, so the governance invariant (``committed == 0``) holds
by construction. For real MIMIC-CDM cases, only ``case_id`` is logged -- never
the clinical presentation text.

RESEARCH USE ONLY. Meditron's authors (EPFL) explicitly recommend against
clinical deployment without further alignment and testing. This harness
benchmarks the model on synthetic cases; it is not a clinical tool.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)

# Imported lazily-friendly: the dataclass mirrors mimic_cdm_eval.AgentCDMResponse
# to avoid a circular import (eval imports agent, not vice versa).


@dataclass
class AgentCDMResponse:
    """Parsed model output for one CDM case. Mirrors the eval's dataclass."""
    case_id: str
    proposed_icd: list[str] = field(default_factory=list)
    proposed_rxnorm: list[str] = field(default_factory=list)
    proposed_loinc: list[str] = field(default_factory=list)
    proposed_cpt: list[str] = field(default_factory=list)
    raw_response: str = ""      # debug only; never logged in production
    is_mocked: bool = False
    backend: str = "mock"


class CDMBackend(Protocol):
    """A model backend: presentation -> structured CDM codes."""

    name: str

    def propose(self, case_id: str, presentation: str) -> AgentCDMResponse: ...


# ---------------------------------------------------------------------------
# Mock backend (CI default)
# ---------------------------------------------------------------------------

class MockBackend:
    """Echoes gold labels. Deterministic upper bound; verifies wiring only."""

    name = "mock"

    def __init__(self, gold_lookup: dict[str, dict[str, list[str]]]):
        # gold_lookup[case_id] = {"icd": [...], "rxnorm": [...], ...}
        self._gold = gold_lookup

    def propose(self, case_id: str, presentation: str) -> AgentCDMResponse:
        g = self._gold.get(case_id, {})
        return AgentCDMResponse(
            case_id=case_id,
            proposed_icd=list(g.get("icd", [])),
            proposed_rxnorm=list(g.get("rxnorm", [])),
            proposed_loinc=list(g.get("loinc", [])),
            proposed_cpt=list(g.get("cpt", [])),
            is_mocked=True,
            backend="mock",
        )


# ---------------------------------------------------------------------------
# Meditron backend (local Ollama, research-only)
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a clinical decision support assistant. Given a patient "
    "presentation, propose the most likely diagnosis, treatments, lab orders, "
    "and procedures. Respond ONLY with a JSON object using these exact keys: "
    '"diagnosis_icd" (list of ICD-10 codes), "treatment_rxnorm" (list of '
    'RxNorm ingredient IDs), "labs_loinc" (list of LOINC codes), '
    '"procedures_cpt" (list of CPT codes). Use [] for none. No prose.'
)

# One worked exemplar to anchor the base model's output format (k=1 few-shot).
_FEWSHOT_USER = (
    "Presentation: 58-year-old with crushing substernal chest pain radiating "
    "to the left arm, diaphoresis, ST elevation on ECG.\n"
    "Respond with the JSON object."
)
_FEWSHOT_ASSISTANT = (
    '{"diagnosis_icd": ["I21.3"], "treatment_rxnorm": ["1191", "32968"], '
    '"labs_loinc": ["10839-9", "2157-6"], "procedures_cpt": ["92941"]}'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class MeditronBackend:
    """Meditron-7B via a local Ollama server. LOCAL-ONLY; never in CI.

    Requires:
      * Ollama installed and running   (`brew install ollama`, then `ollama serve`)
      * Model pulled                   (`ollama pull meditron:7b`)
      * `pip install ollama`           (the Python client)
    """

    name = "meditron"

    def __init__(
        self,
        model: str = "meditron:7b",
        host: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature
        self.timeout_s = timeout_s
        try:
            import ollama  # noqa: F401
        except ImportError as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "MeditronBackend needs the ollama client: pip install ollama, "
                "and a running Ollama server with `ollama pull meditron:7b`."
            ) from e

    def propose(self, case_id: str, presentation: str) -> AgentCDMResponse:
        import ollama

        client = ollama.Client(host=self.host, timeout=self.timeout_s)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _FEWSHOT_USER},
            {"role": "assistant", "content": _FEWSHOT_ASSISTANT},
            {"role": "user",
             "content": f"Presentation: {presentation}\nRespond with the JSON object."},
        ]
        try:
            resp = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature},
            )
            raw = resp["message"]["content"]
        except Exception as e:  # network / server / model errors
            log.warning("[CDM] case_id=%s meditron call failed: %s", case_id, type(e).__name__)
            return AgentCDMResponse(case_id=case_id, raw_response="", backend="meditron")

        parsed = self._parse(raw)
        # Log case_id + axis counts only -- never the presentation or raw clinical text.
        log.info(
            "[CDM] case_id=%s backend=meditron icd=%d rx=%d loinc=%d cpt=%d",
            case_id, len(parsed.proposed_icd), len(parsed.proposed_rxnorm),
            len(parsed.proposed_loinc), len(parsed.proposed_cpt),
        )
        parsed.case_id = case_id
        return parsed

    @staticmethod
    def _parse(raw: str) -> AgentCDMResponse:
        """Tolerant parse: extract the first JSON object from freeform output."""
        out = AgentCDMResponse(case_id="", raw_response=raw, backend="meditron")
        m = _JSON_RE.search(raw or "")
        if not m:
            return out
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return out

        def _codes(key: str) -> list[str]:
            v = data.get(key, [])
            if isinstance(v, str):
                v = [v]
            return [str(x).strip() for x in v if str(x).strip()]

        out.proposed_icd = _codes("diagnosis_icd")
        out.proposed_rxnorm = _codes("treatment_rxnorm")
        out.proposed_loinc = _codes("labs_loinc")
        out.proposed_cpt = _codes("procedures_cpt")
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_backend(
    name: str,
    gold_lookup: dict[str, dict[str, list[str]]] | None = None,
    **kwargs,
) -> CDMBackend:
    """Construct a backend by name: 'mock' | 'meditron'."""
    name = name.lower()
    if name == "mock":
        return MockBackend(gold_lookup or {})
    if name == "meditron":
        return MeditronBackend(**kwargs)
    raise ValueError(f"Unknown CDM backend: {name!r}. Use 'mock' or 'meditron'.")
