"""NetworkX DAG validation, topological sorting, and React-Flow serialization."""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from pdfmate.domain.entities import TaskNode, WorkflowGraph
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry


class CanvasDAGBuilder:
    """Builds, edits, and validates NetworkX workflow graphs for the Task Canvas."""

    def __init__(self, plugin_registry: Optional[NodePluginRegistry] = None) -> None:
        self.plugin_registry = plugin_registry or NodePluginRegistry()
        self._active_workflow: Optional[WorkflowGraph] = None
        self._compiler = LocalLLMClient(require_model_dir=False)

    def build(self, user_request: str) -> WorkflowGraph:
        """Build and validate a WorkflowGraph DAG directly from a natural language request."""
        workflow = self._compiler._compile_workflow_from_prompt(user_request)
        self.validate_dag(workflow)
        self._active_workflow = workflow
        return workflow

    def update_node(
        self,
        node_id: str,
        params: Dict[str, object],
        workflow: Optional[WorkflowGraph] = None,
    ) -> WorkflowGraph:
        """Interactive parameter edit on a node before execution."""
        target_wf = workflow or self._active_workflow
        if target_wf is None:
            raise ValueError("No active WorkflowGraph available to update.")
        updated = self.update_node_parameters(target_wf, node_id, params)
        self._active_workflow = updated
        return updated

    def build_networkx_graph(self, workflow: WorkflowGraph) -> nx.DiGraph:
        """Construct a NetworkX DiGraph from a domain WorkflowGraph."""
        graph: nx.DiGraph = nx.DiGraph()
        node_ids = {node.node_id for node in workflow.nodes}
        for node in workflow.nodes:
            graph.add_node(
                node.node_id,
                action_type=node.action_type,
                parameters=node.parameters,
            )
        for node in workflow.nodes:
            for dep_id in node.dependencies:
                if dep_id not in node_ids:
                    raise ValueError(
                        f"Node '{node.node_id}' references unknown dependency '{dep_id}'."
                    )
                graph.add_edge(dep_id, node.node_id)
        return graph

    def validate_dag(self, workflow: WorkflowGraph) -> bool:
        """Verify acyclicity, dependency existence, and plugin registration."""
        if not workflow.nodes:
            raise ValueError("WorkflowGraph must contain at least one TaskNode.")
        supported_actions = set(self.plugin_registry.list_actions())
        for node in workflow.nodes:
            if node.action_type not in supported_actions:
                raise ValueError(
                    f"Node '{node.node_id}' uses unregistered action '{node.action_type}'."
                )
        digraph = self.build_networkx_graph(workflow)
        if not nx.is_directed_acyclic_graph(digraph):
            raise ValueError("WorkflowGraph contains a cycle and is not a valid DAG.")
        return True

    def get_topological_generations(self, workflow: WorkflowGraph) -> List[List[TaskNode]]:
        """Return topological layers of TaskNodes so each layer can execute in parallel."""
        self.validate_dag(workflow)
        digraph = self.build_networkx_graph(workflow)
        node_map: Dict[str, TaskNode] = {n.node_id: n for n in workflow.nodes}
        generations: List[List[TaskNode]] = []
        for layer_ids in nx.topological_generations(digraph):
            layer_nodes = [node_map[nid] for nid in sorted(layer_ids)]
            generations.append(layer_nodes)
        return generations

    def update_node_parameters(
        self,
        workflow: WorkflowGraph,
        node_id: str,
        updated_parameters: Dict[str, object],
    ) -> WorkflowGraph:
        """Return a new WorkflowGraph with updated parameters on the specified node."""
        new_nodes: List[TaskNode] = []
        found = False
        for node in workflow.nodes:
            if node.node_id == node_id:
                found = True
                merged_params = dict(node.parameters)
                merged_params.update(updated_parameters)
                new_nodes.append(
                    TaskNode(
                        node_id=node.node_id,
                        action_type=node.action_type,
                        parameters=merged_params,
                        dependencies=list(node.dependencies),
                        explanation_key=node.explanation_key,
                    )
                )
            else:
                new_nodes.append(node)
        if not found:
            raise KeyError(f"TaskNode '{node_id}' not found in WorkflowGraph.")
        return WorkflowGraph(
            workflow_id=workflow.workflow_id,
            nodes=new_nodes,
            entry_nodes=list(workflow.entry_nodes),
            metadata=dict(workflow.metadata),
        )

    def to_react_flow_elements(self, workflow: WorkflowGraph) -> Dict[str, List[Dict[str, object]]]:
        """Serialize WorkflowGraph into React-Flow compatible nodes and edges for Streamlit."""
        rf_nodes: List[Dict[str, object]] = []
        rf_edges: List[Dict[str, object]] = []
        for idx, node in enumerate(workflow.nodes):
            rf_nodes.append(
                {
                    "id": node.node_id,
                    "type": "default",
                    "data": {
                        "label": f"{node.action_type.upper()} ({node.node_id})",
                        "action_type": node.action_type,
                        "parameters": node.parameters,
                    },
                    "position": {"x": idx * 240, "y": 100},
                }
            )
            for dep_id in node.dependencies:
                rf_edges.append(
                    {
                        "id": f"e-{dep_id}-{node.node_id}",
                        "source": dep_id,
                        "target": node.node_id,
                        "animated": True,
                    }
                )
        return {"nodes": rf_nodes, "edges": rf_edges}
