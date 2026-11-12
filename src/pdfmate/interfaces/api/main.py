"""FastAPI Agent Service API for PDFMate AI (Port 8000)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pdfmate.application.agent.agent_service import LocalAgentService
from pdfmate.application.workflow.orchestrator import WorkflowOrchestrator
from pdfmate.config.settings import get_settings
from pdfmate.domain.entities import WorkflowExecutionReport, WorkflowGraph
from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.persistence.redis_state import RedisPipelineStateStore
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox


class PlanRequest(BaseModel):
    """Request payload for planning a PDF workflow DAG from natural language."""

    user_request: str
    document_context: Dict[str, object] = Field(default_factory=dict)


class ExecuteRequest(BaseModel):
    """Request payload for executing a finalized WorkflowGraph."""

    workflow: WorkflowGraph
    initial_pdf_path: str = "vault/input_lecture.pdf"


app = FastAPI(
    title="PDFMate AI Agent Service",
    version="1.0.0",
    description="Zero-cloud local agentic document mentor for Lenovo LOQ 15APH8 (RTX 4060 8GB)",
)

_settings = get_settings()
_gpu_mgr = GPUMemoryManager(max_reserved_gb=_settings.gpu_max_vram)
_llm_client = LocalLLMClient(
    model_path=_settings.llm_model_path,
    require_model_dir=False,
)
_agent_service = LocalAgentService(llm_client=_llm_client, gpu_manager=_gpu_mgr)
_state_store = RedisPipelineStateStore(redis_url=_settings.redis_url)
_sandbox = EncryptedSandbox(vault_dir=_settings.vault_dir)
_orchestrator = WorkflowOrchestrator(sandbox=_sandbox, state_store=_state_store)


@app.get("/health")
async def health_check() -> Dict[str, object]:
    """Container health check reporting GPU VRAM headroom and zero-cloud compliance."""
    hw = _settings.get_hardware_profile()
    return {
        "status": "healthy",
        "laptop_class": hw.laptop_class,
        "gpu": hw.gpu_name,
        "available_vram_gb": round(_gpu_mgr.get_available_vram_gb(), 2),
        "max_reserved_vram_gb": _gpu_mgr.max_reserved_gb,
        "offline_strict": _settings.offline_strict,
    }


@app.post("/plan", response_model=WorkflowGraph)
async def plan_workflow_endpoint(payload: PlanRequest) -> WorkflowGraph:
    """Translate a natural language request into a validated WorkflowGraph DAG."""
    try:
        return await _agent_service.plan_workflow(
            user_request=payload.user_request,
            document_context=payload.document_context,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc


@app.post("/execute", response_model=WorkflowExecutionReport)
async def execute_workflow_endpoint(payload: ExecuteRequest) -> WorkflowExecutionReport:
    """Execute a finalized WorkflowGraph topologically inside the encrypted sandbox."""
    try:
        return await _orchestrator.execute_workflow(
            workflow=payload.workflow,
            initial_pdf_path=Path(payload.initial_pdf_path),
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/workflow/{workflow_id}")
async def get_workflow_state(workflow_id: str) -> Dict[str, object]:
    """Fetch current execution state for a workflow from Redis/memory persistence."""
    state = _state_store.load_state(workflow_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found.")
    return state
