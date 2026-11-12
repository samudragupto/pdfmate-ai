"""Pedagogical Explanation Generator and PDF Literacy Score Evaluator."""

from __future__ import annotations

from typing import Dict, List, Optional

from pdfmate.domain.entities import TaskNode, WorkflowGraph
from pdfmate.domain.value_objects import LiteracyScore
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry


class MentorEngine:
    """Generates educational explanations for PDF operations and scores workflow literacy."""

    def __init__(
        self,
        plugin_registry: Optional[NodePluginRegistry] = None,
        llm_client: Optional[LocalLLMClient] = None,
    ) -> None:
        self.plugin_registry = plugin_registry or NodePluginRegistry()
        self.llm_client = llm_client

    def explain_node_sync(self, node: TaskNode, params: Optional[Dict[str, object]] = None) -> str:
        """Synchronously produce a context-aware pedagogical explanation for a node."""
        effective_params = params if params is not None else node.parameters
        try:
            plugin = self.plugin_registry.get(node.action_type)
            return plugin.explain(effective_params)
        except KeyError:
            return f"Executing {node.action_type} node with parameters {effective_params}."

    async def explain_node(self, node: TaskNode, params: Optional[Dict[str, object]] = None) -> str:
        """Produce a concise, pedagogical explanation for why and how a node executes."""
        effective_params = params if params is not None else node.parameters
        base_explanation = self.explain_node_sync(node, effective_params)

        if self.llm_client is not None and node.explanation_key == "custom_llm":
            llm_text = await self.llm_client.generate_text(
                system="You are PDFMate Mentor. Explain the PDF engineering concept clearly.",
                prompt=f"Action: {node.action_type}, Params: {effective_params}",
            )
            return f"{base_explanation} {llm_text}".strip()
        return base_explanation

    def compute_literacy_score(self, executed_nodes: List[TaskNode]) -> LiteracyScore:
        """Compute LiteracyScore from a list of executed TaskNodes."""
        return self.evaluate_literacy_score(WorkflowGraph(nodes=executed_nodes))

    def evaluate_literacy_score(self, workflow: WorkflowGraph) -> LiteracyScore:
        """Compute a pedagogical PDF Literacy Score based on the workflow's node configuration."""
        compression_score = 88.0
        structural_score = 88.0
        privacy_score = 85.0

        for node in workflow.nodes:
            if node.action_type == "compress":
                codec = str(node.parameters.get("codec", "JPEG2000")).upper()
                dpi = int(str(node.parameters.get("dpi", 150)))
                if codec == "JPEG2000" and 120 <= dpi <= 200:
                    compression_score = 98.0
                else:
                    compression_score = 90.0
            elif node.action_type in {"merge", "split"}:
                if node.parameters.get("mode") == "bookmarks" or node.parameters.get(
                    "preserve_bookmarks"
                ):
                    structural_score = 96.0
            elif node.action_type == "redact":
                privacy_score = 100.0

        overall = round((compression_score + structural_score + privacy_score) / 3.0, 1)
        return LiteracyScore(
            overall_score=overall,
            compression_efficiency=compression_score,
            structural_preservation=structural_score,
            privacy_hygiene=privacy_score,
            rationale=(
                f"Evaluated {len(workflow.nodes)} workflow nodes: "
                f"compression={compression_score:.1f}, structure={structural_score:.1f}, "
                f"privacy={privacy_score:.1f}."
            ),
        )
