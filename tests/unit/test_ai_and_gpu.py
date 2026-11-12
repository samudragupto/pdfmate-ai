"""Unit tests for GPU Memory Manager, Local LLM Client, and Local VLM Client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pdfmate.domain.entities import WorkflowGraph
from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.ai.vlm_client import LocalVLMClient


def test_gpu_memory_manager_rtx4060_profile() -> None:
    mgr = GPUMemoryManager(max_reserved_gb=7.5, headroom_gb=0.5, vlm_offload_threshold_gb=5.5)
    with patch("subprocess.run", side_effect=OSError("no nvidia-smi")):
        avail = mgr.get_available_vram_gb()
        assert avail == 8.0
        assert mgr.is_load_safe(4.5) is True
        assert mgr.is_load_safe(7.2) is False

        mgr.reserve_for_model(4.5, "qwen2.5_7b")
        assert mgr.current_allocated_gb == 4.5
        with pytest.raises(MemoryError):
            mgr.reserve_for_model(3.5, "second_heavy_model")
        mgr.release_model("qwen2.5_7b")
        assert mgr.current_allocated_gb == 0.0

    assert mgr.compute_cpu_offload_layers(4.5, total_layers=28) == 0
    offload = mgr.compute_cpu_offload_layers(6.5, total_layers=28)
    assert offload > 0


@pytest.mark.asyncio
async def test_local_llm_client_and_fail_closed(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_dir" / "model.gguf"
    with pytest.raises(FileNotFoundError):
        LocalLLMClient(model_path=missing_model, require_model_dir=True)

    valid_model = tmp_path / "qwen.gguf"
    valid_model.write_text("GGUF_STUB", encoding="utf-8")
    client = LocalLLMClient(model_path=valid_model, require_model_dir=True)

    prompt = (
        "Merge 4 scanned lecture notes, redact handwritten student ID, "
        "split by chapter, summarize sections, compress to <10MB"
    )
    graph = await client.generate_structured(
        system="System instruction",
        prompt=prompt,
        schema_type=WorkflowGraph,
    )
    actions = [n.action_type for n in graph.nodes]
    assert "merge" in actions
    assert "redact" in actions
    assert "summarize" in actions
    assert "compress" in actions

    text_split = await client.generate_text("sys", "Explain bookmark split")
    assert "bookmark split" in text_split.lower()
    text_comp = await client.generate_text("sys", "Explain jpeg2000 compress")
    assert "jpeg2000" in text_comp.lower()
    text_red = await client.generate_text("sys", "Explain redact")
    assert "redaction" in text_red.lower()
    text_mrg = await client.generate_text("sys", "Explain merge")
    assert "merging" in text_mrg.lower()
    text_gen = await client.generate_text("sys", "General operation")
    assert "sandbox" in text_gen.lower()


@pytest.mark.asyncio
async def test_local_vlm_client_offloading(tmp_path: Path) -> None:
    mgr = GPUMemoryManager(max_reserved_gb=7.5)
    vlm = LocalVLMClient(
        gpu_manager=mgr,
        model_path=tmp_path / "vlm.gguf",
        active_footprint_gb=5.8,
        total_layers=28,
    )
    result = await vlm.detect_handwritten_regions(tmp_path / "scan.pdf", page_number=0)
    assert result["handwritten_id_detected"] is True
    assert vlm.is_loaded is False
    assert vlm.offloaded_cpu_layers > 0
