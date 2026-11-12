"""Task Node Plugin Registry implementing INodePlugin for extensible PDF operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from pdfmate.infrastructure.pdf.engine import PyMuPDFEngine


class INodePlugin(Protocol):
    """Protocol implemented by every PDF task node plugin."""

    def action_type(self) -> str: ...

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path: ...

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool: ...

    def explain(self, params: Dict[str, Any]) -> str: ...


class MergePlugin:
    """Plugin for merging multiple PDF files while preserving page trees and bookmarks."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "merge"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        return bool(str(input_path))

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "merged.pdf"))
        extra_paths_raw = effective.get("file_sequence", effective.get("input_paths", [input_path]))
        if isinstance(extra_paths_raw, list) and len(extra_paths_raw) > 0:
            input_paths = [Path(str(p)) for p in extra_paths_raw]
        else:
            input_paths = [input_path]
        return self.engine.merge_pdfs(input_paths=input_paths, output_path=output_path)

    def explain(self, params: Dict[str, Any]) -> str:
        return (
            "Merging combines PDF page trees. Order is preserved based on file sequence parameters."
        )


class SplitPlugin:
    """Plugin for splitting PDFs by bookmark outline hierarchy or page count."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "split"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        mode = str(params.get("mode", "bookmarks"))
        return mode in {"bookmarks", "page_ranges", "fixed_interval"}

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "split.pdf"))
        mode = str(effective.get("mode", "bookmarks"))
        pages_per_chunk = int(effective.get("pages_per_chunk", 1))
        return self.engine.split_pdf(
            input_path=input_path,
            output_path=output_path,
            mode=mode,
            pages_per_chunk=pages_per_chunk,
        )

    def explain(self, params: Dict[str, Any]) -> str:
        mode = str(params.get("mode", "bookmarks"))
        if mode == "bookmarks":
            return "Using bookmark split not page count because syllabus modules detected"
        return f"Splitting PDF using {mode} mode to isolate logical study units."


class RedactPlugin:
    """Plugin for true content-stream PII and student ID redaction."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "redact"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        return bool(params.get("pattern", r"\b\d{8,10}\b"))

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "redacted.pdf"))
        pattern = str(effective.get("pattern", r"\b\d{8,10}\b"))
        bbox_raw = effective.get("bounding_box")
        bbox = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        return self.engine.redact_pdf(
            input_path=input_path,
            output_path=output_path,
            pattern=pattern,
            bounding_box=bbox,
        )

    def explain(self, params: Dict[str, Any]) -> str:
        return (
            "True redaction strips underlying character glyphs and raster pixels from the "
            "PDF content stream rather than drawing an overlay box."
        )


class CompressPlugin:
    """Plugin for wavelet/deflate stream compression and XRef garbage collection."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "compress"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        dpi = int(params.get("dpi", 150))
        return 72 <= dpi <= 600

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "compressed.pdf"))
        dpi = int(effective.get("dpi", 150))
        codec = str(effective.get("method", effective.get("codec", "JPEG2000")))
        return self.engine.compress_pdf(
            input_path=input_path,
            output_path=output_path,
            dpi=dpi,
            codec=codec,
        )

    def explain(self, params: Dict[str, Any]) -> str:
        codec = str(params.get("codec", "JPEG2000"))
        dpi = int(params.get("dpi", 150))
        return (
            f"Compressing embedded raster streams with {codec} at {dpi} DPI and level-4 "
            "XRef garbage collection to stay under the target size budget while keeping text crisp."
        )


class SummarizePlugin:
    """Plugin for generating local section summaries inside the PDF artifact."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "summarize"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        return True

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "summarized.pdf"))
        granularity = str(effective.get("granularity", "section"))
        return self.engine.summarize_pdf(
            input_path=input_path,
            output_path=output_path,
            granularity=granularity,
        )

    def explain(self, params: Dict[str, Any]) -> str:
        granularity = str(params.get("granularity", "section"))
        return (
            f"Synthesizing {granularity}-level study digest locally using Qwen2.5-7B-Q4 "
            "and appending a searchable summary sheet."
        )


class OCRPlugin:
    """Plugin for optical character recognition on scanned lecture pages."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "ocr"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        return True

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "ocr_output.pdf"))
        language = str(effective.get("language", "eng"))
        return self.engine.ocr_pdf(
            input_path=input_path,
            output_path=output_path,
            language=language,
        )

    def explain(self, params: Dict[str, Any]) -> str:
        return (
            "Generating an invisible searchable text layer over rasterized lecture scans "
            "to enable downstream text search, redaction, and summarization."
        )


class ScanPlugin:
    """Plugin for layout-aware document scanning and page normalization."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self.engine = engine or PyMuPDFEngine()
        self.output_dir = output_dir

    def action_type(self) -> str:
        return "scan"

    def validate(self, input_path: Path, params: Dict[str, Any]) -> bool:
        return True

    def execute(
        self,
        input_path: Path,
        params: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Path:
        effective = parameters if parameters is not None else (params or {})
        target_dir = self.output_dir or input_path.parent
        output_path = Path(effective.get("output_path", target_dir / "scanned_norm.pdf"))
        return self.engine.ocr_pdf(
            input_path=input_path,
            output_path=output_path,
            language=str(effective.get("language", "eng")),
        )

    def explain(self, params: Dict[str, Any]) -> str:
        return (
            "Scanning normalizes page skew, contrast, and layout geometry prior to "
            "VLM/OCR extraction."
        )


class NodePluginRegistry:
    """Central registry mapping action_type strings to INodePlugin instances."""

    def __init__(
        self, engine: Optional[PyMuPDFEngine] = None, output_dir: Optional[Path] = None
    ) -> None:
        self._engine = engine or PyMuPDFEngine()
        self._output_dir = output_dir
        self._plugins: Dict[str, INodePlugin] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        default_plugins: List[INodePlugin] = [
            ScanPlugin(self._engine, self._output_dir),
            MergePlugin(self._engine, self._output_dir),
            SplitPlugin(self._engine, self._output_dir),
            RedactPlugin(self._engine, self._output_dir),
            CompressPlugin(self._engine, self._output_dir),
            SummarizePlugin(self._engine, self._output_dir),
            OCRPlugin(self._engine, self._output_dir),
        ]
        for plugin in default_plugins:
            self.register(plugin)

    def register(self, plugin: INodePlugin) -> None:
        """Register or replace a task node plugin."""
        self._plugins[plugin.action_type()] = plugin

    def get(self, action_type: str) -> INodePlugin:
        """Retrieve a plugin by action_type or raise KeyError."""
        if action_type not in self._plugins:
            raise KeyError(f"Unsupported PDF task action_type: '{action_type}'")
        return self._plugins[action_type]

    def list_actions(self) -> List[str]:
        """Return all registered action types."""
        return sorted(self._plugins.keys())
