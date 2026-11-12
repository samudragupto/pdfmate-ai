"""Domain entities for PDFMate AI: TaskNode, WorkflowGraph, and PDFDocument."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, computed_field


class TaskNode(BaseModel):
    """Represents a single atomic PDF operation in the workflow DAG."""

    node_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str
    parameters: Dict[str, object] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    explanation_key: Optional[str] = None

    def is_executable(self, completed_nodes: Optional[Set[str]] = None) -> bool:
        """Check if this node's dependencies are satisfied."""
        if len(self.dependencies) == 0:
            return True
        if completed_nodes is None:
            return all(d in self.dependencies for d in self.dependencies)
        return all(d in completed_nodes for d in self.dependencies)


class WorkflowGraph(BaseModel):
    """Directed acyclic graph of TaskNode operations."""

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nodes: List[TaskNode] = Field(default_factory=list)
    entry_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def node_count(self) -> int:
        """Total number of task nodes in the workflow."""
        return len(self.nodes)

    def get_node_by_action(self, action: str) -> Optional[TaskNode]:
        """Return the first node matching the given action_type."""
        for node in self.nodes:
            if node.action_type == action:
                return node
        return None

    def get_node_by_id(self, node_id: str) -> Optional[TaskNode]:
        """Return the node matching the given node_id."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None


class PDFDocument(BaseModel):
    """Domain representation of a PDF document under analysis or transformation."""

    path: Path
    page_count: int
    has_scanned_content: bool = False
    metadata: Dict[str, object] = Field(default_factory=dict)


class NodeExecutionReport(BaseModel):
    """Execution telemetry and mentor output for a single TaskNode."""

    node_id: str
    action_type: str
    output_path: Path
    mentor_explanation: str
    status: str = "completed"
    validation_passed: bool = True
    metadata: Dict[str, object] = Field(default_factory=dict)


class WorkflowExecutionReport(BaseModel):
    """Aggregated outcome of executing a WorkflowGraph."""

    workflow_id: str
    final_artifact_path: Path
    node_reports: List[NodeExecutionReport] = Field(default_factory=list)
    literacy_score: float = 100.0
    status: str = "completed"
