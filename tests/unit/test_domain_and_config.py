"""Unit tests for Domain Entities, Value Objects, and Hardware Profile Settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfmate.config.settings import (
    HardwareProfile,
    HardwareTier,
    PDFMateSettings,
    QuantizationProfile,
    get_settings,
)
from pdfmate.domain.entities import PDFDocument, TaskNode, WorkflowGraph
from pdfmate.domain.value_objects import (
    CompressionProfile,
    ImageCodec,
    LiteracyScore,
    RedactionMask,
    SplitMode,
    SplitStrategy,
)


def test_task_node_and_workflow_graph() -> None:
    n1 = TaskNode(node_id="n1", action_type="merge", parameters={"sort": "seq"})
    n2 = TaskNode(node_id="n2", action_type="redact", dependencies=["n1"])
    assert n1.is_executable() is True
    assert n2.is_executable() is True
    assert n2.is_executable(completed_nodes=set()) is False
    assert n2.is_executable(completed_nodes={"n1"}) is True

    wf = WorkflowGraph(nodes=[n1, n2], entry_nodes=["n1"])
    assert wf.node_count == 2
    assert wf.get_node_by_action("merge") == n1
    assert wf.get_node_by_action("unknown") is None
    assert wf.get_node_by_id("n2") == n2
    assert wf.get_node_by_id("missing") is None


def test_pdf_document_entity(tmp_path: Path) -> None:
    doc = PDFDocument(
        path=tmp_path / "notes.pdf",
        page_count=12,
        has_scanned_content=True,
        metadata={"author": "Student"},
    )
    assert doc.page_count == 12
    assert doc.has_scanned_content is True


def test_value_objects_immutability_and_validation() -> None:
    comp = CompressionProfile(codec=ImageCodec.JPEG2000, dpi=150, target_max_mb=10.0)
    assert comp.dpi == 150
    with pytest.raises(Exception):
        comp.dpi = 300  # type: ignore[misc]

    with pytest.raises(Exception):
        CompressionProfile(dpi=10)

    mask = RedactionMask(pattern=r"\b\d{8}\b", redact_handwritten_id=True)
    assert mask.replacement_label == "[REDACTED]"

    split = SplitStrategy(mode=SplitMode.BOOKMARKS, pages_per_chunk=5)
    assert split.preserve_toc is True

    score = LiteracyScore(overall_score=95.0)
    assert score.overall_score == 95.0


def test_hardware_profiles_and_settings(tmp_path: Path) -> None:
    t1 = HardwareProfile.for_tier(HardwareTier.TIER1_STUDENT_LAPTOP)
    assert t1.total_vram_gb == 8.0
    assert t1.quantization == QuantizationProfile.Q4_K_M

    t2 = HardwareProfile.for_tier(HardwareTier.TIER2_CREATOR_WORKSTATION)
    assert t2.total_vram_gb == 24.0
    assert t2.quantization == QuantizationProfile.Q6_K

    t3 = HardwareProfile.for_tier(HardwareTier.TIER3_CLOUD_SERVER)
    assert t3.total_vram_gb == 80.0
    assert t3.quantization == QuantizationProfile.FP16

    (tmp_path / "qwen.gguf").write_text("GGUF_STUB", encoding="utf-8")
    settings = PDFMateSettings(models_dir=tmp_path, gpu_max_vram=7.2)
    settings.verify_models_directory()
    prof = settings.get_hardware_profile()
    assert prof.max_reserved_vram_gb == 7.2

    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    empty_settings = PDFMateSettings(
        models_dir=empty_dir, llm_model_path=empty_dir / "missing.gguf"
    )
    with pytest.raises(FileNotFoundError):
        empty_settings.verify_models_directory()

    missing_settings = PDFMateSettings(models_dir=tmp_path / "nonexistent_models")
    with pytest.raises(FileNotFoundError):
        missing_settings.verify_models_directory()

    default_s = get_settings()
    assert default_s.app_name == "PDFMate AI"
