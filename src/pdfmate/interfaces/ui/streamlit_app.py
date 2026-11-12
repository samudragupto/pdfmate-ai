"""Streamlit Task Canvas UI Adapter for PDFMate AI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from pdfmate.application.agent.agent_service import LocalAgentService
from pdfmate.application.canvas.builder import CanvasDAGBuilder
from pdfmate.application.mentor.engine import MentorEngine
from pdfmate.application.workflow.orchestrator import WorkflowOrchestrator
from pdfmate.config.settings import get_settings
from pdfmate.domain.entities import TaskNode, WorkflowExecutionReport, WorkflowGraph
from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox


class TaskCanvasController:
    """Testable UI controller managing NL planning, node parameter edits, and execution."""

    def __init__(self, vault_dir: Optional[Path] = None) -> None:
        self.settings = get_settings()
        self.gpu_manager = GPUMemoryManager(max_reserved_gb=self.settings.gpu_max_vram)
        self.llm_client = LocalLLMClient(
            model_path=self.settings.llm_model_path,
            require_model_dir=False,
        )
        self.agent_service = LocalAgentService(
            llm_client=self.llm_client,
            gpu_manager=self.gpu_manager,
        )
        self.canvas_builder = CanvasDAGBuilder()
        self.mentor_engine = MentorEngine()
        self.sandbox = EncryptedSandbox(vault_dir=vault_dir or self.settings.vault_dir)
        self.orchestrator = WorkflowOrchestrator(
            canvas_builder=self.canvas_builder,
            mentor_engine=self.mentor_engine,
            sandbox=self.sandbox,
        )

    async def generate_canvas_plan(
        self, user_request: str, document_context: Optional[Dict[str, object]] = None
    ) -> WorkflowGraph:
        """Generate a DAG plan from user NL request."""
        return await self.agent_service.plan_workflow(
            user_request=user_request,
            document_context=document_context or {},
        )

    def edit_node(
        self, workflow: WorkflowGraph, node_id: str, new_params: Dict[str, object]
    ) -> WorkflowGraph:
        """Edit parameters of a node on the interactive Task Canvas."""
        return self.canvas_builder.update_node_parameters(workflow, node_id, new_params)

    def ask_why(self, node: TaskNode) -> str:
        """Query the Mentor Engine for a pedagogical explanation before execution."""
        return self.mentor_engine.explain_node_sync(node, node.parameters)

    def hardware_privacy_status(self) -> str:
        """Return hardware privacy lock indicator string."""
        vram_free = self.gpu_manager.get_available_vram_gb()
        return f"GPU Running — 0% Cloud (RTX 4060 Free VRAM: {vram_free:.1f}GB)"

    async def run_workflow(
        self, workflow: WorkflowGraph, input_pdf: Path
    ) -> WorkflowExecutionReport:
        """Execute the finalized workflow DAG topologically."""
        return await self.orchestrator.execute_workflow(workflow, input_pdf)


def main() -> None:
    """Render the Streamlit Task Canvas interactive interface."""
    st.set_page_config(page_title="PDFMate AI - Local Document Mentor", layout="wide")
    st.title("PDFMate AI: Local-First Agentic Document Mentor")

    controller = TaskCanvasController()
    st.caption(f"Target Profile: Lenovo LOQ 15APH8 | {controller.hardware_privacy_status()}")
    beginner_mode = st.checkbox("Beginner Mode (Pause & Explain Nodes)", value=True)
    default_prompt = (
        "Merge 4 scanned lecture notes, redact handwritten student ID, "
        "summarize sections, compress to <10MB"
    )
    user_prompt = st.text_area("Natural Language Task Request", value=default_prompt)

    if st.button("Generate Task Canvas DAG"):
        plan = asyncio.run(
            controller.generate_canvas_plan(
                user_request=user_prompt,
                document_context={"source_files": 4, "has_bookmarks": True},
            )
        )
        st.session_state["workflow_plan"] = plan

    if "workflow_plan" in st.session_state:
        plan_obj: WorkflowGraph = st.session_state["workflow_plan"]
        st.subheader("Interactive Task Canvas Nodes")
        rf_elements = controller.canvas_builder.to_react_flow_elements(plan_obj)
        st.json(rf_elements)

        if beginner_mode:
            for node in plan_obj.nodes:
                st.markdown(f"**Why {node.action_type}?** {controller.ask_why(node)}")

        if st.button("Confirm and Execute DAG"):
            sample_input = controller.sandbox.vault_dir / "lecture_input.pdf"
            report = asyncio.run(controller.run_workflow(plan_obj, sample_input))
            st.success(f"Execution Complete | PDF Literacy Score: {report.literacy_score:.1f}/100")
            for node_rep in report.node_reports:
                st.info(f"[{node_rep.action_type.upper()}] Mentor: {node_rep.mentor_explanation}")


if __name__ == "__main__":
    main()
