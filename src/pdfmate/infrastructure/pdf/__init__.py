"""PDF processing infrastructure powered by PyMuPDF."""

from __future__ import annotations

from pdfmate.infrastructure.pdf.engine import PyMuPDFEngine
from pdfmate.infrastructure.pdf.plugin_registry import (
    CompressPlugin,
    INodePlugin,
    MergePlugin,
    NodePluginRegistry,
    OCRPlugin,
    RedactPlugin,
    ScanPlugin,
    SplitPlugin,
    SummarizePlugin,
)

__all__ = [
    "CompressPlugin",
    "INodePlugin",
    "MergePlugin",
    "NodePluginRegistry",
    "OCRPlugin",
    "PyMuPDFEngine",
    "RedactPlugin",
    "ScanPlugin",
    "SplitPlugin",
    "SummarizePlugin",
]
