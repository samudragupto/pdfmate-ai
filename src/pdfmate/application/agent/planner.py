"""Natural Language to Directed Acyclic Graph (NL-to-DAG) Planner."""

from __future__ import annotations

from typing import Dict, Optional

from pdfmate.application.agent.agent_service import IAgentService
from pdfmate.application.canvas.builder import CanvasDAGBuilder
from pdfmate.domain.entities import PDFDocument, WorkflowGraph


class NLToDAGPlanner:
    """Translates student natural language instructions and PDF metadata into a validated DAG."""

    def __init__(
        self,
        agent_service: IAgentService,
        canvas_builder: Optional[CanvasDAGBuilder] = None,
    ) -> None:
        self.agent_service = agent_service
        self.canvas_builder = canvas_builder or CanvasDAGBuilder()

    async def create_validated_plan(
        self,
        user_request: str,
        document: Optional[PDFDocument] = None,
        extra_context: Optional[Dict[str, object]] = None,
    ) -> WorkflowGraph:
        """Build and validate a topological WorkflowGraph from natural language."""
        context: Dict[str, object] = {}
        if document is not None:
            context["path"] = str(document.path)
            context["page_count"] = document.page_count
            context["has_scanned_content"] = document.has_scanned_content
            context["document_metadata"] = document.metadata
        if extra_context:
            context.update(extra_context)

        workflow = await self.agent_service.plan_workflow(
            user_request=user_request,
            document_context=context,
        )
        self.canvas_builder.validate_dag(workflow)
        return workflow
