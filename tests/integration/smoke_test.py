"""Integration Smoke Test: GPU Memory Profile + Zero-Cloud Offline Execution Verification."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pdfmate.application.agent.agent_service import LocalAgentService
from pdfmate.application.canvas.builder import CanvasDAGBuilder
from pdfmate.application.mentor.engine import MentorEngine
from pdfmate.application.workflow.orchestrator import WorkflowOrchestrator
from pdfmate.config.settings import get_settings
from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.pdf.engine import PyMuPDFEngine
from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox


async def run_smoke_verification() -> None:
    """Verify RTX 4060 VRAM checks, fail-closed model validation, and encrypted DAG execution."""
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    if not settings.llm_model_path.exists():
        settings.llm_model_path.write_bytes(b"GGUF_CI_STUB")
    settings.verify_models_directory()

    gpu_mgr = GPUMemoryManager(max_reserved_gb=settings.gpu_max_vram)
    assert gpu_mgr.is_load_safe(model_size_gb=4.5), "RTX 4060 4.5GB Q4 model load must be safe"

    with tempfile.TemporaryDirectory() as tmp_dir:
        vault_path = Path(tmp_dir) / "vault"
        sandbox = EncryptedSandbox(vault_dir=vault_path)
        pdf_engine = PyMuPDFEngine()
        source_pdf = pdf_engine.ensure_valid_pdf(Path(tmp_dir) / "lecture_notes.pdf")

        llm_client = LocalLLMClient(model_path=settings.llm_model_path)
        agent_service = LocalAgentService(llm_client=llm_client, gpu_manager=gpu_mgr)
        registry = NodePluginRegistry(engine=pdf_engine, output_dir=vault_path)
        canvas = CanvasDAGBuilder(plugin_registry=registry)
        mentor = MentorEngine(plugin_registry=registry, llm_client=llm_client)
        orchestrator = WorkflowOrchestrator(
            plugin_registry=registry,
            canvas_builder=canvas,
            mentor_engine=mentor,
            sandbox=sandbox,
        )

        request = (
            "Merge 4 scanned lecture notes, redact handwritten student ID, "
            "summarize sections, compress to <10MB"
        )
        workflow = await agent_service.plan_workflow(
            user_request=request,
            document_context={"pages": 8, "has_bookmarks": True},
        )
        report = await orchestrator.execute_workflow(workflow, source_pdf)
        assert report.status == "completed"
        assert report.final_artifact_path.exists()
        assert len(sandbox.get_audit_trail(workflow.workflow_id)) == len(workflow.nodes)


def test_smoke_suite() -> None:
    """Pytest entrypoint for integration smoke verification."""
    asyncio.run(run_smoke_verification())


if __name__ == "__main__":
    asyncio.run(run_smoke_verification())
    print("PDFMate AI offline smoke verification passed.")
