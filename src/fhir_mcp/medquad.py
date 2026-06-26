"""MedQuAD dataset loader — NIH clinical QA corpus.

Loads the Medical Question Answering Dataset (MedQuAD) into typed Pydantic v2
models.  MedQuAD contains 47,457 QA pairs drawn from 12 NIH websites including
cancer.gov, niddk.nih.gov, GARD (rare diseases), and MedlinePlus.

Dataset sources:
  XML  (original):  https://github.com/abachaa/MedQuAD          (CC BY 4.0)
  CSV  (converted): https://github.com/avery-lockwood/MedQuAD-CSVs

This loader supports both distribution formats.  The CSV port is preferred for
ingestion speed; the XML original is used when full annotation fidelity (UMLS
CUI, semantic type) is required.

Why MedQuAD for this project:
  - Real NIH clinician / consumer questions across common AND rare diseases
  - Built-in UMLS Concept Unique Identifier (CUI) gold labels for entity-
    linking evaluation (used in evals/ harness, PR #3)
  - GARD source flag surfaces the rare-disease question subset the JD names
  - CC BY 4.0 license: clean for a public portfolio project with attribution

Reference for citation (required by PhysioNet/NIH attribution norms):
  Abacha, A.B. & Demner-Fushman, D. (2019).  A Question-Entailment Approach
  to Question Answering.  BMC Bioinformatics 20, 511.
  https://doi.org/10.1186/s12859-019-3119-8

PHI NOTE: MedQuAD contains de-identified public NIH content.  No patient data
is present.  This module has zero PHI touchpoints.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

_logger = logging.getLogger("fhir_mcp.medquad")

# ---------------------------------------------------------------------------
# Environment-variable override (mirrors FHIR_MCP_LOINC_RULES pattern)
# ---------------------------------------------------------------------------
_DEFAULT_CORPUS_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "medquad"
)
_CORPUS_DIR = Path(
    os.environ.get("FHIR_MCP_MEDQUAD_DIR", str(_DEFAULT_CORPUS_DIR))
)

# NIH sources that indicate rare-disease content
_RARE_DISEASE_SOURCES: frozenset[str] = frozenset({
    "GARD",
    "genetic_and_rare_diseases",
    "rarediseases",
})

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

QuestionType = Literal[
    "information",
    "treatment",
    "diagnosis",
    "causes",
    "prevention",
    "inheritance",
    "symptoms",
    "exams and tests",
    "outlook",
    "complications",
    "support groups",
    "considerations",
    "research",
    "stages",
    "susceptibility",
    "frequency",
    "other",
]


class MedQuADItem(BaseModel):
    """A single QA pair from the MedQuAD corpus."""

    item_id: str = Field(description="Unique identifier within the corpus")
    question: str = Field(description="The clinical question text")
    answer: str | None = Field(
        default=None,
        description="NIH-authored answer text; None when removed for copyright",
    )
    question_type: str = Field(
        default="other",
        description="Question category (information, treatment, diagnosis, …)",
    )
    focus: str | None = Field(
        default=None,
        description="Medical entity the question is about (e.g. 'Type 2 Diabetes')",
    )
    cui: str | None = Field(
        default=None,
        description="UMLS Concept Unique Identifier for the focus entity",
    )
    semantic_type: str | None = Field(
        default=None,
        description="UMLS semantic type (e.g. 'Disease or Syndrome')",
    )
    source: str | None = Field(
        default=None,
        description="NIH source site (e.g. 'GARD', 'cancer.gov')",
    )
    source_url: str | None = Field(
        default=None,
        description="URL of the source NIH page",
    )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_answered(self) -> bool:
        """True when the item has a non-empty answer.

        ~16,407 of 47,457 MedQuAD pairs retain full answers after
        MedlinePlus copyright removal.  Use this to filter the eval set.
        """
        return bool(self.answer and self.answer.strip())

    @property
    def has_gold_cui(self) -> bool:
        """True when a UMLS CUI is available for entity-linking evaluation."""
        return bool(self.cui and self.cui.strip())

    @property
    def is_rare_disease(self) -> bool:
        """True when the question originates from a rare-disease source (GARD)."""
        if not self.source:
            return False
        src_lower = self.source.lower()
        return any(tag in src_lower for tag in _RARE_DISEASE_SOURCES)


# ---------------------------------------------------------------------------
# Loader result
# ---------------------------------------------------------------------------


@dataclass
class LoadResult:
    """Outcome of a corpus load operation."""

    items: list[MedQuADItem] = field(default_factory=list)
    total: int = 0
    answered: int = 0
    with_cui: int = 0
    rare_disease: int = 0
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """True when at least one item was loaded successfully."""
        return self.total > 0

    def __repr__(self) -> str:
        return (
            f"LoadResult(total={self.total}, answered={self.answered}, "
            f"with_cui={self.with_cui}, rare_disease={self.rare_disease}, "
            f"errors={len(self.errors)})"
        )


# ---------------------------------------------------------------------------
# MedQuADLoader
# ---------------------------------------------------------------------------


class MedQuADLoader:
    """Loads MedQuAD QA pairs from a local corpus directory.

    Supports two distribution formats:
      CSV  (one file per source, or a merged file)
      XML  (one .xml file per source, original abachaa/MedQuAD layout)

    Usage::

        loader = MedQuADLoader()
        result = loader.load()                       # all items
        eval_set = loader.eval_items()               # answered + has_gold_cui
        rare = loader.filter(is_rare_disease=True)   # GARD subset
    """

    def __init__(self, corpus_dir: Path | None = None) -> None:
        self._dir = corpus_dir or _CORPUS_DIR
        self._cache: list[MedQuADItem] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, *, force_reload: bool = False) -> LoadResult:
        """Load all QA pairs from the corpus directory.

        Tries CSV first (faster), then XML.  Returns an empty LoadResult
        if the corpus directory does not exist — never raises.

        Args:
            force_reload: Bypass the in-memory cache and re-read disk.

        Returns:
            LoadResult with all loaded MedQuADItem objects and summary stats.
        """
        if self._cache is not None and not force_reload:
            return self._make_result(self._cache)

        if not self._dir.exists():
            _logger.warning(
                "MedQuAD corpus directory not found: %s — "
                "set FHIR_MCP_MEDQUAD_DIR or download the corpus.",
                self._dir,
            )
            return LoadResult()

        items: list[MedQuADItem] = []
        errors: list[str] = []

        csv_files = sorted(self._dir.rglob("*.csv"))
        xml_files = sorted(self._dir.rglob("*.xml"))

        if csv_files:
            _logger.info("Loading %d CSV file(s) from %s", len(csv_files), self._dir)
            for csv_path in csv_files:
                loaded, errs = self._load_csv(csv_path)
                items.extend(loaded)
                errors.extend(errs)
        elif xml_files:
            _logger.info("Loading %d XML file(s) from %s", len(xml_files), self._dir)
            for xml_path in xml_files:
                loaded, errs = self._load_xml(xml_path)
                items.extend(loaded)
                errors.extend(errs)
        else:
            _logger.warning(
                "No .csv or .xml files found in %s", self._dir
            )

        self._cache = items
        result = self._make_result(items)
        result.errors = errors
        _logger.info(
            "MedQuAD loaded: %s",
            result,
        )
        return result

    def eval_items(self) -> list[MedQuADItem]:
        """Return items suitable for entity-linking eval: answered + has_gold_cui."""
        return [i for i in self.load().items if i.is_answered and i.has_gold_cui]

    def filter(
        self,
        *,
        question_type: str | None = None,
        is_rare_disease: bool | None = None,
        is_answered: bool | None = None,
        has_gold_cui: bool | None = None,
        source: str | None = None,
    ) -> list[MedQuADItem]:
        """Return a filtered subset of the corpus.

        All filters are AND-combined.  Pass None to skip a filter.

        Args:
            question_type:   Filter by question category string.
            is_rare_disease: True → GARD/rare-disease items only.
            is_answered:     True → items with non-empty answers.
            has_gold_cui:    True → items with UMLS CUI annotations.
            source:          Filter by source site substring (case-insensitive).
        """
        items = self.load().items

        def _keep(item: MedQuADItem) -> bool:
            if question_type is not None and item.question_type != question_type:
                return False
            if is_rare_disease is not None and item.is_rare_disease != is_rare_disease:
                return False
            if is_answered is not None and item.is_answered != is_answered:
                return False
            if has_gold_cui is not None and item.has_gold_cui != has_gold_cui:
                return False
            if source is not None:
                if item.source is None or source.lower() not in item.source.lower():
                    return False
            return True

        return [i for i in items if _keep(i)]

    def iter_items(self) -> Iterator[MedQuADItem]:
        """Iterate over all items without holding the full list in caller memory."""
        yield from self.load().items

    # ------------------------------------------------------------------
    # CSV parser
    # ------------------------------------------------------------------

    def _load_csv(
        self, path: Path
    ) -> tuple[list[MedQuADItem], list[str]]:
        """Parse a MedQuAD CSV file.

        Expected columns (subset — extras are ignored):
          qtype, Question, Answer, source, Focus, CUI, SemanticType, url

        The CSV port (avery-lockwood/MedQuAD-CSVs) uses these headers.
        """
        items: list[MedQuADItem] = []
        errors: list[str] = []
        stem = path.stem

        try:
            with path.open(encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    try:
                        item = MedQuADItem(
                            item_id=f"{stem}_{idx}",
                            question=_strip(row.get("Question") or row.get("question", "")),
                            answer=_strip(row.get("Answer") or row.get("answer")) or None,
                            question_type=_strip(
                                row.get("qtype") or row.get("question_type", "other")
                            ) or "other",
                            focus=_strip(row.get("Focus") or row.get("focus")) or None,
                            cui=_strip(row.get("CUI") or row.get("cui")) or None,
                            semantic_type=_strip(
                                row.get("SemanticType")
                                or row.get("semantic_type")
                            ) or None,
                            source=_strip(row.get("source") or row.get("Source")) or stem,
                            source_url=_strip(row.get("url") or row.get("URL")) or None,
                        )
                        if item.question:
                            items.append(item)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{path.name}:{idx}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Failed to open {path.name}: {exc}")

        _logger.debug("CSV %s: %d items, %d errors", path.name, len(items), len(errors))
        return items, errors

    # ------------------------------------------------------------------
    # XML parser
    # ------------------------------------------------------------------

    def _load_xml(
        self, path: Path
    ) -> tuple[list[MedQuADItem], list[str]]:
        """Parse an original MedQuAD XML file.

        Handles the abachaa/MedQuAD structure where each file represents
        one NIH source document with QAPairs children.
        """
        items: list[MedQuADItem] = []
        errors: list[str] = []
        stem = path.stem

        try:
            tree = ET.parse(path)  # noqa: S314 — local file, not user-supplied
            root = tree.getroot()
        except ET.ParseError as exc:
            errors.append(f"XML parse error in {path.name}: {exc}")
            return items, errors

        # Source-level metadata lives on the root or a Document element
        source = (
            root.findtext("Source")
            or root.get("source")
            or stem
        )
        source_url = root.findtext("URL") or root.get("url")

        # Focus / CUI annotations
        focus = root.findtext("Focus") or root.findtext(".//Focus")
        cui = (
            root.findtext(".//CUI")
            or root.findtext(".//cui")
        )
        semantic_type = (
            root.findtext(".//SemanticType")
            or root.findtext(".//semantictype")
        )

        # QAPair elements may be direct children or under QAPairs
        qa_pairs = root.findall(".//QAPair")
        for pair in qa_pairs:
            pid = pair.get("pid", "0")
            q_elem = pair.find("Question")
            a_elem = pair.find("Answer")
            if q_elem is None:
                continue
            question = _strip(q_elem.text)
            if not question:
                continue
            qtype = q_elem.get("qtype", "other")
            answer = _strip(a_elem.text) if a_elem is not None else None

            try:
                item = MedQuADItem(
                    item_id=f"{stem}_{pid}",
                    question=question,
                    answer=answer or None,
                    question_type=qtype or "other",
                    focus=_strip(focus) or None,
                    cui=_strip(cui) or None,
                    semantic_type=_strip(semantic_type) or None,
                    source=_strip(source) or stem,
                    source_url=_strip(source_url) or None,
                )
                items.append(item)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path.name}:pid={pid}: {exc}")

        _logger.debug("XML %s: %d items, %d errors", path.name, len(items), len(errors))
        return items, errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_result(items: list[MedQuADItem]) -> LoadResult:
        return LoadResult(
            items=items,
            total=len(items),
            answered=sum(1 for i in items if i.is_answered),
            with_cui=sum(1 for i in items if i.has_gold_cui),
            rare_disease=sum(1 for i in items if i.is_rare_disease),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip(value: str | None) -> str:
    """Strip whitespace from a nullable string; return empty string on None."""
    return (value or "").strip()


# ---------------------------------------------------------------------------
# Module-level singleton (mirrors get_nlp() / get_rules() pattern)
# ---------------------------------------------------------------------------

_loader: MedQuADLoader | None = None


def get_loader(corpus_dir: Path | None = None) -> MedQuADLoader:
    """Return the module-level MedQuADLoader instance.

    Args:
        corpus_dir: Override the corpus directory.  If None (default),
                    uses FHIR_MCP_MEDQUAD_DIR env var or data/medquad/.
    """
    global _loader
    if _loader is None or corpus_dir is not None:
        _loader = MedQuADLoader(corpus_dir=corpus_dir)
    return _loader
