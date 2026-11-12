"""Configuration package for PDFMate AI."""

from __future__ import annotations

from pdfmate.config.settings import (
    HardwareProfile,
    HardwareTier,
    PDFMateSettings,
    QuantizationProfile,
    get_settings,
)

__all__ = [
    "HardwareProfile",
    "HardwareTier",
    "PDFMateSettings",
    "QuantizationProfile",
    "get_settings",
]
