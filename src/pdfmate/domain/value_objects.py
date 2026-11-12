"""Immutable domain value objects for PDF operations and pedagogical scoring."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field, field_validator


class ImageCodec(str, Enum):
    """Supported image compression codecs for PDF optimization."""

    JPEG = "JPEG"
    JPEG2000 = "JPEG2000"
    LOSSLESS_FLATE = "FLATE"


class SplitMode(str, Enum):
    """Supported PDF splitting modes."""

    BOOKMARKS = "bookmarks"
    PAGE_RANGES = "page_ranges"
    FIXED_INTERVAL = "fixed_interval"


class CompressionProfile(BaseModel):
    """Immutable value object defining PDF compression parameters."""

    model_config = {"frozen": True}

    codec: ImageCodec = ImageCodec.JPEG2000
    dpi: int = Field(default=150, ge=72, le=600)
    target_max_mb: float = Field(default=10.0, gt=0.1)
    strip_unused_objects: bool = True
    garbage_collection_level: int = Field(default=4, ge=0, le=4)

    @field_validator("dpi")
    @classmethod
    def validate_dpi(cls, value: int) -> int:
        """Ensure DPI is within practical bounds for document legibility."""
        if value < 72 or value > 600:
            raise ValueError("DPI must be between 72 and 600.")
        return value


class RedactionMask(BaseModel):
    """Immutable value object specifying a pattern or bounding box for redaction."""

    model_config = {"frozen": True}

    pattern: str = Field(default=r"\b\d{8,10}\b")
    replacement_label: str = "[REDACTED]"
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    fill_color_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    redact_handwritten_id: bool = True


class SplitStrategy(BaseModel):
    """Immutable value object specifying how a PDF should be partitioned."""

    model_config = {"frozen": True}

    mode: SplitMode = SplitMode.BOOKMARKS
    pages_per_chunk: int = Field(default=10, ge=1)
    preserve_toc: bool = True


class LiteracyScore(BaseModel):
    """Pedagogical score measuring user understanding of PDF structure & choices."""

    model_config = {"frozen": True}

    overall_score: float = Field(default=92.0, ge=0.0, le=100.0)
    compression_efficiency: float = Field(default=95.0, ge=0.0, le=100.0)
    structural_preservation: float = Field(default=90.0, ge=0.0, le=100.0)
    privacy_hygiene: float = Field(default=100.0, ge=0.0, le=100.0)
    rationale: str = (
        "High literacy: Preserved vector fonts and bookmark hierarchy while applying "
        "true stream redaction and JPEG2000 150 DPI downsampling."
    )


class AuditEntry(BaseModel):
    """Immutable audit log record for compliance and local provenance."""

    model_config = {"frozen": True}

    timestamp: str
    workflow_id: str
    node_id: str
    action: str
    parameter_hash: str
    result_path: str
