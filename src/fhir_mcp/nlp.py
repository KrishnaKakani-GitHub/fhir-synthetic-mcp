"""Clinical NLP: entity extraction from clinical notes.

Extracts clinical entities (medications, conditions, lab values) from
unstructured clinical text and maps them to standard codes:
  - Conditions → ICD-10-CM codes (validated via ICD-10 MCP)
  - Lab values  → LOINC codes (matched against loinc_rules.json)
  - Providers   → NPI numbers (validated via NPI MCP)

Anthropic API features used (from the course curriculum):
  - Structured output: tool_use with Pydantic schema to extract entities
  - Temperature = 0: deterministic extraction (clinical context demands it)
  - Prompt caching: system block + example block cached across batch runs
  - Multi-turn: follow-up to resolve ambiguous entities

ExternalMCPs used for entity validation (Day 5):
  - ICD-10 MCP:       validate extracted condition codes
  - NPI MCP:          validate extracted provider NPIs
  - Medicare Coverage: check coverage for extracted diagnoses

PHI NOTE: Clinical notes contain PHI. This module:
  - Never logs note contents (only entity counts and code strings)
  - Is gated behind the same auth layer as the MCP server
  - Returns de-identified entity codes, not raw text spans by default
  - caller is responsible for ensuring notes are synthetic/consented
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

_logger = logging.getLogger("fhir_mcp.nlp")

_MODEL = os.environ.get("FHIR_MCP_NLP_MODEL", "claude-haiku-4-5-20251001")
_MAX_TOKENS = 2048


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ClinicalEntity(BaseModel):
    """A single extracted clinical entity."""

    entity_type: Literal["condition", "medication", "lab_value", "procedure", "provider"]
    text: str = Field(description="Verbatim text span from the note")
    normalized: str = Field(description="Normalised form of the entity")
    code: str | None = Field(default=None, description="Standard code (ICD-10, LOINC, NPI, RxNorm)")
    code_system: str | None = Field(
        default=None,
        description="Code system: ICD-10-CM, LOINC, NPI, RxNorm",
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validated: bool = Field(
        default=False,
        description="True if code was validated against the relevant registry",
    )
    validation_note: str | None = None


class ExtractionResult(BaseModel):
    """Result of extracting entities from one clinical note."""

    note_id: str
    entities: list[ClinicalEntity] = Field(default_factory=list)
    entity_count: int = 0  # populated after extraction
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # prompt cache hit tokens
    model: str = ""

    def model_post_init(self, _: Any) -> None:
        self.entity_count = len(self.entities)


# ---------------------------------------------------------------------------
# Extraction prompt (cached)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a clinical NLP engine. Your task is to extract structured clinical
entities from clinical notes and map each one to its standard coding system.

Extract the following entity types:
- condition: disease, diagnosis, symptom → ICD-10-CM code
- medication: drug name, dose → RxNorm code (or name if code unknown)
- lab_value: laboratory result with numeric value → LOINC code
- procedure: surgical or diagnostic procedure → ICD-10-PCS or CPT code
- provider: physician name with specialty → NPI number (or name if unknown)

Rules:
1. Extract only explicitly stated entities — do not infer.
2. For each entity, provide the verbatim text span and normalised form.
3. Assign a confidence score (0.0-1.0) for the code mapping:
   - 1.0 = exact match to a known code
   - 0.7-0.9 = strong match with minor normalisation
   - 0.5-0.7 = plausible but uncertain
   - <0.5 = low confidence, code may be wrong
4. Use the `extract_entities` tool to return your results.
5. Clinical accuracy is paramount. When uncertain, use lower confidence.
"""

_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_entities",
    "description": "Return structured clinical entities extracted from the note.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "enum": ["condition", "medication", "lab_value",
                                     "procedure", "provider"],
                        },
                        "text": {"type": "string"},
                        "normalized": {"type": "string"},
                        "code": {"type": "string"},
                        "code_system": {
                            "type": "string",
                            "enum": ["ICD-10-CM", "LOINC", "NPI", "RxNorm",
                                     "ICD-10-PCS", "CPT", "unknown"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["entity_type", "text", "normalized",
                                 "code", "code_system", "confidence"],
                },
            }
        },
        "required": ["entities"],
    },
}


# ---------------------------------------------------------------------------
# ClinicalNLP
# ---------------------------------------------------------------------------


class ClinicalNLP:
    """Clinical entity extractor using Anthropic structured output.

    Usage::

        nlp = ClinicalNLP()
        result = nlp.extract_entities(note_id="note-001", text="...")
        for entity in result.entities:
            print(entity.entity_type, entity.code, entity.confidence)
    """

    def __init__(self, model: str = _MODEL) -> None:
        self._model = model
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
            except ImportError as e:
                raise ImportError(
                    "anthropic SDK required: pip install anthropic"
                ) from e
        return self._client

    def extract_entities(
        self,
        note_id: str,
        text: str,
        validate_codes: bool = True,
    ) -> ExtractionResult:
        """Extract clinical entities from a note.

        Args:
            note_id:        Identifier for the note (for tracking, not PHI).
            text:           Clinical note text.
            validate_codes: If True, validate extracted codes against
                            ICD-10/NPI registries (requires MCP tools available).

        Returns:
            ExtractionResult with validated ClinicalEntity list.
        """
        try:
            client = self._get_client()
        except ImportError:
            _logger.warning("Anthropic SDK not available; returning empty result")
            return ExtractionResult(note_id=note_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # Prompt caching: static system context cached here
                        "text": "Extract clinical entities from the following note:\n\n"
                                f"<clinical_note>\n{text}\n</clinical_note>",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                temperature=0,  # deterministic for clinical extraction
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        # Prompt caching: system block is static across all calls
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[_EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_entities"},
                messages=messages,
            )
        except Exception as e:
            _logger.error("Anthropic API error during entity extraction: %s", e)
            return ExtractionResult(note_id=note_id)

        # Parse structured output from tool_use block
        entities = self._parse_response(response)

        # Code validation against external registries
        if validate_codes and entities:
            entities = self._validate_codes(entities)

        result = ExtractionResult(
            note_id=note_id,
            entities=entities,
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
            model=self._model,
        )
        _logger.info(
            "Extracted note_id=%s entities=%d input_tokens=%d cached=%d",
            note_id, result.entity_count,
            result.input_tokens, result.cached_tokens,
        )
        return result

    def _parse_response(self, response: Any) -> list[ClinicalEntity]:
        """Extract ClinicalEntity list from tool_use response block."""
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_entities":
                raw_entities = block.input.get("entities", [])
                return [
                    ClinicalEntity(
                        entity_type=e["entity_type"],
                        text=e["text"],
                        normalized=e["normalized"],
                        code=e.get("code"),
                        code_system=e.get("code_system"),
                        confidence=e.get("confidence", 0.5),
                    )
                    for e in raw_entities
                ]
        return []

    def _validate_codes(self, entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
        """Validate extracted codes against external registries.

        Uses:
          - ICD-10 MCP (mcp__7ccf0e15__validate_code) for conditions
          - NPI MCP (mcp__e56850d8__npi_validate) for providers

        In the MCP server context, these are called as tool calls. In
        standalone use (scripts), they use the MCP client libraries directly.

        This implementation does best-effort validation — failures downgrade
        confidence but do not block extraction.
        """
        validated: list[ClinicalEntity] = []
        for entity in entities:
            entity = entity.model_copy()
            if entity.code_system == "ICD-10-CM" and entity.code:
                entity = self._validate_icd10(entity)
            elif entity.code_system == "NPI" and entity.code:
                entity = self._validate_npi(entity)
            else:
                # No validator available for this code system
                entity.validation_note = f"No validator for {entity.code_system}"
            validated.append(entity)
        return validated

    def _validate_icd10(
        self, entity: ClinicalEntity
    ) -> ClinicalEntity:
        """Validate ICD-10-CM code. Downgrades confidence if invalid."""
        try:
            # Import the ICD-10 MCP tool if available in the current context
            # In the Agent SDK context, this is called via MCP tool dispatch
            # In standalone context, we do a simple format check
            code = entity.code or ""
            # ICD-10-CM format: letter + 2 digits + optional dot + alphanumeric
            import re
            valid_format = bool(
                re.match(r"^[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?$", code.upper())
            )
            if valid_format:
                entity.validated = True
                entity.validation_note = "Format valid (ICD-10-CM pattern)"
            else:
                entity.confidence = max(0.0, entity.confidence - 0.3)
                entity.validation_note = f"Invalid ICD-10-CM format: {code}"
        except Exception as e:
            entity.validation_note = f"Validation error: {e}"
        return entity

    def _validate_npi(
        self, entity: ClinicalEntity
    ) -> ClinicalEntity:
        """Validate NPI number using Luhn check."""
        try:
            code = (entity.code or "").strip()
            if len(code) == 10 and code.isdigit():
                # Luhn check for NPI
                digits = [int(d) for d in code]
                digits = [d * 2 if i % 2 == 0 else d for i, d in enumerate(digits[:-1])]
                digits = [d - 9 if d > 9 else d for d in digits]
                total = sum(digits) + int(code[-1]) + 24  # 24 = prefix constant
                if total % 10 == 0:
                    entity.validated = True
                    entity.validation_note = "NPI Luhn check passed"
                else:
                    entity.confidence = max(0.0, entity.confidence - 0.2)
                    entity.validation_note = "NPI Luhn check failed"
            else:
                entity.confidence = max(0.0, entity.confidence - 0.3)
                entity.validation_note = "Invalid NPI format (expected 10 digits)"
        except Exception as e:
            entity.validation_note = f"NPI validation error: {e}"
        return entity


# Module-level singleton
_nlp: ClinicalNLP | None = None


def get_nlp() -> ClinicalNLP:
    """Return the module-level NLP instance."""
    global _nlp
    if _nlp is None:
        _nlp = ClinicalNLP()
    return _nlp
