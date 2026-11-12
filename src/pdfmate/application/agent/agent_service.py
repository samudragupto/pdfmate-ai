"""Agent Service implementing IAgentService Protocol with RTX 4060 VRAM safety."""

from __future__ import annotations

import logging
from typing import Dict, Protocol

from pdfmate.domain.entities import WorkflowGraph
from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient

logger = logging.getLogger(__name__)


class IAgentService(Protocol):
    """Protocol for translating natural language requests into a WorkflowGraph DAG."""

    async def plan_workflow(
        self, user_request: str, document_context: Dict[str, object]
    ) -> WorkflowGraph: ...


class LocalAgentService:
    """Local-first agent service powered by Qwen2.5-7B-Q4 and GPUMemoryManager."""

    def __init__(
        self,
        llm_client: LocalLLMClient,
        gpu_manager: GPUMemoryManager,
    ) -> None:
        self.llm = llm_client
        self.gpu = gpu_manager

    async def plan_workflow(
        self, user_request: str, document_context: Dict[str, object]
    ) -> WorkflowGraph:
        """Generate a validated WorkflowGraph DAG while enforcing RTX 4060 VRAM limits."""
        if not self.gpu.is_load_safe(model_size_gb=4.5):
            raise RuntimeError("Insufficient VRAM for agent brain")

        system_prompt = self._system_instruction()
        user_prompt = (
            f"CONTEXT: {document_context}\nREQUEST: {user_request}\nOUTPUT: JSON WorkflowGraph"
        )

        raw_plan = await self.llm.generate_structured(
            system=system_prompt,
            prompt=user_prompt,
            schema_type=WorkflowGraph,
            max_tokens=2048,
        )
        logger.info("Agent generated workflow: %s", raw_plan.workflow_id)
        return raw_plan

    def _system_instruction(self) -> str:
        return (
            "You are PDFMate Agent. Generate a directed acyclic graph of PDF tasks. "
            "Each node must have action_type, parameters, and dependencies. "
            "Respond in structured JSON matching WorkflowGraph schema."
        )
