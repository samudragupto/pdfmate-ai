"""Async State Machine Workflow Orchestrator for topological DAG execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdfmate.application.canvas.builder import CanvasDAGBuilder
from pdfmate.application.mentor.engine import MentorEngine
from pdfmate.domain.entities import (
    NodeExecutionReport,
    TaskNode,
    WorkflowExecutionReport,
    WorkflowGraph,
)
from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry
from pdfmate.infrastructure.persistence.redis_state import RedisPipelineStateStore
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox


class WorkflowOrchestrator:
    """Executes WorkflowGraph DAGs topologically with parallel independent node dispatch."""

    def __init__(
        self,
        plugin_registry: Optional[NodePluginRegistry] = None,
        canvas_builder: Optional[CanvasDAGBuilder] = None,
        mentor_engine: Optional[MentorEngine] = None,
        sandbox: Optional[EncryptedSandbox] = None,
        state_store: Optional[RedisPipelineStateStore] = None,
    ) -> None:
        self.plugin_registry = plugin_registry or NodePluginRegistry()
        self.canvas_builder = canvas_builder or CanvasDAGBuilder(self.plugin_registry)
        self.mentor_engine = mentor_engine or MentorEngine(self.plugin_registry)
        self.sandbox = sandbox or EncryptedSandbox()
        self.state_store = state_store or RedisPipelineStateStore()

    async def execute(
        self,
        graph: WorkflowGraph,
        initial_pdf_path: Optional[Path] = None,
    ) -> WorkflowExecutionReport:
        """Execute a WorkflowGraph DAG and write results to EncryptedSandbox."""
        resolved_input = initial_pdf_path or (self.sandbox.vault_dir / "input_lecture.pdf")
        return await self.execute_workflow(graph, resolved_input)

    async def execute_workflow(
        self,
        workflow: WorkflowGraph,
        initial_pdf_path: Path,
    ) -> WorkflowExecutionReport:
        """Execute nodes in topological generations and return final artifact + literacy score."""
        generations = self.canvas_builder.get_topological_generations(workflow)
        node_outputs: Dict[str, Path] = {}
        all_reports: List[NodeExecutionReport] = []

        self.state_store.save_state(
            workflow.workflow_id,
            {
                "workflow_id": workflow.workflow_id,
                "status": "running",
                "nodes": {},
            },
        )

        last_output_path = initial_pdf_path

        for layer in generations:
            tasks = [
                self._execute_single_node(
                    workflow_id=workflow.workflow_id,
                    node=node,
                    default_input=last_output_path,
                    node_outputs=node_outputs,
                )
                for node in layer
            ]
            layer_reports = await asyncio.gather(*tasks)
            for report in layer_reports:
                node_outputs[report.node_id] = report.output_path
                last_output_path = report.output_path
                all_reports.append(report)

        literacy = self.mentor_engine.evaluate_literacy_score(workflow)
        final_state = self.state_store.load_state(workflow.workflow_id) or {}
        final_state["status"] = "completed"
        final_state["final_artifact_path"] = str(last_output_path)
        final_state["literacy_score"] = literacy.overall_score
        self.state_store.save_state(workflow.workflow_id, final_state)

        return WorkflowExecutionReport(
            workflow_id=workflow.workflow_id,
            final_artifact_path=last_output_path,
            node_reports=all_reports,
            literacy_score=literacy.overall_score,
            status="completed",
        )

    async def _execute_single_node(
        self,
        workflow_id: str,
        node: TaskNode,
        default_input: Path,
        node_outputs: Dict[str, Path],
    ) -> NodeExecutionReport:
        """Execute a TaskNode with Mentor explanation, sandbox encryption, and audit logging."""
        if node.dependencies and node.dependencies[-1] in node_outputs:
            input_path = node_outputs[node.dependencies[-1]]
        else:
            input_path = default_input

        plugin = self.plugin_registry.get(node.action_type)
        params: Dict[str, Any] = dict(node.parameters)
        if "output_path" not in params:
            params["output_path"] = str(
                self.sandbox.vault_dir / f"{workflow_id}_{node.node_id}_{node.action_type}.pdf"
            )

        explanation = await self.mentor_engine.explain_node(node)
        is_valid = plugin.validate(input_path, params)
        if not is_valid:
            raise ValueError(f"Validation failed for node '{node.node_id}' ({node.action_type}).")

        output_path = await asyncio.to_thread(
            self.sandbox.execute_in_sandbox,
            plugin.execute,
            input_path,
            params,
        )

        audit_entry = self.sandbox.record_audit_entry(
            workflow_id=workflow_id,
            node_id=node.node_id,
            action=node.action_type,
            parameters=params,
            result_path=output_path,
        )

        self.state_store.update_node_progress(
            workflow_id=workflow_id,
            node_id=node.node_id,
            status="completed",
            mentor_explanation=explanation,
            output_path=str(output_path),
        )

        return NodeExecutionReport(
            node_id=node.node_id,
            action_type=node.action_type,
            output_path=output_path,
            mentor_explanation=explanation,
            status="completed",
            validation_passed=is_valid,
            metadata={"audit_hash": audit_entry.parameter_hash},
        )
