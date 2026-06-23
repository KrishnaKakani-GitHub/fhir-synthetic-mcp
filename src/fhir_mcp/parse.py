"""Nemotron Parse integration — raw document → structured text.

Nemotron Parse (NVIDIA NIM) overcomes traditional OCR limitations:
  - Multi-column layout support (prior auth letters, EOB statements)
  - Table extraction with spatial grounding (formulary tables, benefit grids)
  - Reading-order reconstruction (critical for insurance documents)
  - Markdown formatting with structure preserved

This module is the first stage of the end-to-end pipeline:

  Raw PDF/Doc
    → Nemotron Parse (this module)  → structured markdown
    → ClinicalNLP.extract_entities()  → ICD-10/LOINC/NPI entities
    → LOINC deterministic validator    → accept / reject / warn
    → propose_observation              → staged write
    → human approve_write              → committed observation
    → SHA-256 audit chain              → tamper-evident record

API: NVIDIA NIM (OpenAI-compatible)
  https://integrate.api.nvidia.com/v1/chat/completions
  Model: nvidia/nemotron-parse
  Auth:  NVIDIA_API_KEY environment variable

NIM accepts JPEG, PNG, BMP, TIFF, WEBP — NOT raw PDFs.
PDFs are rendered to per-page PNG images via pypdfium2 before sending.

For PHI compliance, deploy NIM self-hosted on-premises so no document
content leaves your infrastructure. The cloud NIM endpoint is suitable
for synthetic or de-identified documents only.

PHI NOTE: If NVIDIA_API_KEY is set and documents contain PHI, use the
self-hosted NIM endpoint (NEMOTRON_PARSE_BASE_URL env var).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_logger = logging.getLogger("fhir_mcp.parse")

_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
_BASE_URL = os.environ.get(
    "NEMOTRON_PARSE_BASE_URL",
    "https://integrate.api.nvidia.com/v1",
)
_MODEL = "nvidia/nemotron-parse"
_TIMEOUT = 60  # seconds
_MAX_PAGES = 10  # cap to avoid token limits on large documents

_PARSE_SYSTEM_PROMPT = (
    "You are a clinical document parser. Extract and structure all content "
    "from the provided document. Preserve table structure as markdown tables. "
    "Maintain reading order. Extract: patient demographics, diagnosis codes, "
    "procedure codes, medication lists, authorization decisions, dates, and "
    "provider information. Output clean, structured markdown."
)


class ParseResult:
    """Result from Nemotron Parse."""

    def __init__(
        self,
        structured_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = _MODEL,
        parse_method: str = "nemotron_parse",
        page_count: int = 0,
    ) -> None:
        self.structured_text = structured_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.parse_method = parse_method
        self.page_count = page_count
        self.output_word_count = len(structured_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "structured_text": self.structured_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "parse_method": self.parse_method,
            "page_count": self.page_count,
            "output_word_count": self.output_word_count,
        }


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def _is_pdf(source: str | Path | bytes) -> bool:
    """Detect PDF by URL extension, file extension, or magic bytes."""
    if isinstance(source, bytes):
        return source[:4] == b"%PDF"
    return str(source).lower().endswith(".pdf")


def _download_url(url: str) -> bytes:
    """Fetch a remote URL to bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "fhir-mcp/1.0"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return resp.read()


def _pdf_to_page_images(pdf_bytes: bytes) -> list[str]:
    """Render PDF pages to base64 PNG strings via pypdfium2.

    Returns one base64 PNG string per page, capped at _MAX_PAGES.
    pypdfium2 is a pure-Python wheel with bundled libpdfium — no
    system dependencies required.

    Install: pip install pypdfium2
    """
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pypdfium2 is required to parse PDF files. "
            "Install it: pip install pypdfium2"
        ) from e

    pdf = pdfium.PdfDocument(pdf_bytes)
    page_images: list[str] = []
    n_pages = min(len(pdf), _MAX_PAGES)

    for i in range(n_pages):
        page = pdf[i]
        bitmap = page.render(scale=2)  # 2x scale improves OCR quality
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        page_images.append(base64.b64encode(buf.getvalue()).decode())

    _logger.info("Rendered %d/%d PDF pages to PNG", n_pages, len(pdf))
    return page_images


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_document(
    source: str | Path | bytes,
    document_type: str = "clinical",
) -> ParseResult:
    """Parse a clinical document using Nemotron Parse.

    Args:
        source: One of:
          - str/Path: local file path (PDF, PNG, JPG, TIFF)
          - bytes:    raw document bytes
          - str starting with 'http': public URL (PDF or image)
        document_type: Hint for the parser. One of:
          'clinical', 'prior_auth', 'eob', 'treatment_plan', 'lab_report'

    Returns:
        ParseResult with structured_text (markdown) ready for NLP extraction.

    PDFs are converted to per-page PNG images before sending to NIM,
    since the NIM API accepts JPEG/PNG/BMP/TIFF/WEBP but not raw PDFs.

    PHI NOTE: Document content is sent to NVIDIA NIM. Use a self-hosted
    NIM endpoint (NEMOTRON_PARSE_BASE_URL) for documents containing PHI.
    """
    if not _NVIDIA_API_KEY:
        _logger.warning(
            "NVIDIA_API_KEY not set — falling back to raw text extraction. "
            "Set NVIDIA_API_KEY to enable Nemotron Parse."
        )
        return _fallback_parse(source)

    # --- Resolve source to bytes or image_url list -------------------------
    content_items: list[dict[str, Any]] = []
    page_count = 0

    if isinstance(source, str) and source.startswith("http"):
        if _is_pdf(source):
            # Download PDF, render to PNG pages
            _logger.info("Downloading PDF from URL for page rendering")
            pdf_bytes = _download_url(source)
            page_b64s = _pdf_to_page_images(pdf_bytes)
            page_count = len(page_b64s)
            content_items = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in page_b64s
            ]
        else:
            # Direct image URL (PNG, JPEG, etc.)
            page_count = 1
            content_items = [
                {"type": "image_url", "image_url": {"url": source}}
            ]

    elif isinstance(source, (str, Path)) and not str(source).startswith("http"):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        doc_bytes = path.read_bytes()
        if _is_pdf(source) or _is_pdf(doc_bytes):
            page_b64s = _pdf_to_page_images(doc_bytes)
            page_count = len(page_b64s)
            content_items = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in page_b64s
            ]
        else:
            # Local image file
            media_type = _infer_media_type(path)
            doc_b64 = base64.b64encode(doc_bytes).decode()
            page_count = 1
            content_items = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{doc_b64}"},
                }
            ]

    elif isinstance(source, bytes):
        if _is_pdf(source):
            page_b64s = _pdf_to_page_images(source)
            page_count = len(page_b64s)
            content_items = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in page_b64s
            ]
        else:
            doc_b64 = base64.b64encode(source).decode()
            page_count = 1
            content_items = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{doc_b64}"},
                }
            ]
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    if not content_items:
        return _fallback_parse(source)

    # --- Build NIM request -------------------------------------------------
    system_prompt = f"{_PARSE_SYSTEM_PROMPT} Document type: {document_type}."
    content_items.append(
        {"type": "text", "text": "Parse this document and return structured markdown."}
    )

    payload = json.dumps({
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_items},
        ],
        "max_tokens": 4096,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_NVIDIA_API_KEY}",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Nemotron Parse API error {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        _logger.warning("Nemotron Parse unreachable: %s — using fallback", e)
        return _fallback_parse(source)

    structured_text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    usage = data.get("usage", {})

    _logger.info(
        "Nemotron Parse complete: pages=%d output_words=%d input_tokens=%d",
        page_count,
        len(structured_text.split()),
        usage.get("prompt_tokens", 0),
    )

    return ParseResult(
        structured_text=structured_text,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=_MODEL,
        parse_method="nemotron_parse_nim",
        page_count=page_count,
    )


def _infer_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def _fallback_parse(source: str | Path | bytes) -> ParseResult:
    """Fallback: return raw text when NVIDIA_API_KEY is not set."""
    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8", errors="replace")
        except Exception:
            text = "[Binary content — set NVIDIA_API_KEY for structured parsing]"
    elif isinstance(source, (str, Path)) and not str(source).startswith("http"):
        path = Path(source)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = f"[Could not read {path} — set NVIDIA_API_KEY for structured parsing]"
    else:
        text = str(source)

    return ParseResult(
        structured_text=text,
        parse_method="raw_text_fallback",
    )
