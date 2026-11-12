"""Unit tests for Application Services, Canvas DAG Builder, Orchestrator, Worker, API, and UI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdfmate.application.agent.agent_service import LocalAgentService
from pdfmate.application.agent.planner import NLToDAGPlanner
from pdfmate.application.canvas.builder import CanvasDAGBuilder
from pdfmate.application.mentor.engine import MentorEngine
from pdfmate.application.worker import execute_pdf_node_task
from pdfmate.application.workflow.orchestrator import WorkflowOrchestrator
from pdfmate.domain.entities import PDFDocument, TaskNode, WorkflowGraph
from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.pdf.engine import PyMuPDFEngine
from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry
from pdfmate.infrastructure.persistence.redis_state import RedisPipelineStateStore
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox
from pdfmate.interfaces.api.main import app
from pdfmate.interfaces.ui.streamlit_app import TaskCanvasController


@pytest.mark.asyncio
async def test_agent_planner_canvas_and_orchestrator(tmp_path: Path) -> None:
    gpu_mgr = GPUMemoryManager(max_reserved_gb=7.5)
    llm_client = LocalLLMClient(require_model_dir=False)
    agent = LocalAgentService(llm_client=llm_client, gpu_manager=gpu_mgr)
    builder = CanvasDAGBuilder(NodePluginRegistry(output_dir=tmp_path))
    planner = NLToDAGPlanner(agent_service=agent, canvas_builder=builder)

    pdf_engine = PyMuPDFEngine()
    sample_pdf = pdf_engine.ensure_valid_pdf(tmp_path / "lecture_notes.pdf")
    doc_entity: PDFDocument = pdf_engine.inspect_document(sample_pdf)

    user_req = (
        "Merge 4 scanned lecture notes, redact handwritten student ID, "
        "summarize sections, compress to <10MB"
    )
    workflow = await planner.create_validated_plan(
        user_request=user_req,
        document=doc_entity,
        extra_context={"semester": "Fall"},
    )
    assert workflow.node_count >= 4

    # Adjust Compress node parameters as in sequence diagram
    compress_node = workflow.get_node_by_action("compress")
    assert compress_node is not None
    updated_wf = builder.update_node_parameters(
        workflow,
        compress_node.node_id,
        {"codec": "JPEG2000", "dpi": 150},
    )
    rf = builder.to_react_flow_elements(updated_wf)
    assert len(rf["nodes"]) == updated_wf.node_count
    assert len(rf["edges"]) >= 1

    # Execute full topological workflow
    sandbox = EncryptedSandbox(vault_dir=tmp_path / "vault")
    state_store = RedisPipelineStateStore(redis_url="redis://127.0.0.1:63999/0")
    mentor = MentorEngine(llm_client=llm_client)
    orchestrator = WorkflowOrchestrator(
        plugin_registry=NodePluginRegistry(output_dir=tmp_path / "vault"),
        canvas_builder=builder,
        mentor_engine=mentor,
        sandbox=sandbox,
        state_store=state_store,
    )
    report = await orchestrator.execute_workflow(updated_wf, sample_pdf)
    assert report.status == "completed"
    assert report.final_artifact_path.exists()
    assert report.literacy_score >= 90.0
    assert len(sandbox.get_audit_trail(workflow.workflow_id)) == updated_wf.node_count


def test_canvas_dag_validation_errors() -> None:
    builder = CanvasDAGBuilder()
    with pytest.raises(ValueError):
        builder.validate_dag(WorkflowGraph(nodes=[]))

    bad_action_wf = WorkflowGraph(nodes=[TaskNode(node_id="n1", action_type="unsupported_op")])
    with pytest.raises(ValueError):
        builder.validate_dag(bad_action_wf)

    missing_dep_wf = WorkflowGraph(
        nodes=[TaskNode(node_id="n1", action_type="merge", dependencies=["ghost"])]
    )
    with pytest.raises(ValueError):
        builder.validate_dag(missing_dep_wf)

    cycle_wf = WorkflowGraph(
        nodes=[
            TaskNode(node_id="n1", action_type="merge", dependencies=["n2"]),
            TaskNode(node_id="n2", action_type="compress", dependencies=["n1"]),
        ]
    )
    with pytest.raises(ValueError):
        builder.validate_dag(cycle_wf)

    with pytest.raises(KeyError):
        builder.update_node_parameters(
            WorkflowGraph(nodes=[TaskNode(node_id="n1", action_type="merge")]),
            "nonexistent",
            {},
        )


def test_celery_worker_task_execution(tmp_path: Path) -> None:
    os.environ["CELERY_ALWAYS_EAGER"] = "true"
    engine = PyMuPDFEngine()
    src = engine.ensure_valid_pdf(tmp_path / "worker_in.pdf")
    res = execute_pdf_node_task(
        workflow_id="wf-celery",
        node_id="node-c1",
        action_type="compress",
        input_path_str=str(src),
        params={"codec": "JPEG2000", "dpi": 150, "output_path": str(tmp_path / "worker_out.pdf")},
        vault_dir_str=str(tmp_path / "vault"),
    )
    assert Path(res["output_path"]).exists()
    assert len(res["parameter_hash"]) == 64


@pytest.mark.asyncio
async def test_fastapi_endpoints_and_ui_controller(tmp_path: Path) -> None:
    client = TestClient(app)
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    plan_resp = client.post(
        "/plan",
        json={
            "user_request": "Merge lecture notes and compress to <10MB",
            "document_context": {"files": 2},
        },
    )
    assert plan_resp.status_code == 200
    wf_json = plan_resp.json()
    assert len(wf_json["nodes"]) >= 2

    src_pdf = PyMuPDFEngine().ensure_valid_pdf(tmp_path / "api_input.pdf")
    exec_resp = client.post(
        "/execute",
        json={
            "workflow": wf_json,
            "initial_pdf_path": str(src_pdf),
        },
    )
    assert exec_resp.status_code == 200
    exec_data = exec_resp.json()
    assert exec_data["status"] == "completed"

    state_resp = client.get(f"/workflow/{wf_json['workflow_id']}")
    assert state_resp.status_code == 200

    not_found_resp = client.get("/workflow/nonexistent-id")
    assert not_found_resp.status_code == 404

    # Test Streamlit TaskCanvasController & CanvasDAGBuilder.build / update_node
    controller = TaskCanvasController(vault_dir=tmp_path / "ui_vault")
    built_wf = controller.canvas_builder.build("Scan, OCR, Split, Summarize, Compress, Merge")
    updated_built = controller.canvas_builder.update_node(
        built_wf.nodes[0].node_id, {"deskew": False}
    )
    assert updated_built.nodes[0].parameters["deskew"] is False
    score_vo = controller.mentor_engine.compute_literacy_score(updated_built.nodes)
    assert score_vo.overall_score >= 90.0

    assert "GPU Running — 0% Cloud" in controller.hardware_privacy_status()
    plan = await controller.generate_canvas_plan("Redact student ID and compress")
    first_id = plan.nodes[0].node_id
    assert len(controller.ask_why(plan.nodes[0])) > 10
    edited = controller.edit_node(plan, first_id, {"custom_flag": True})
    report = await controller.orchestrator.execute(edited, src_pdf)
    assert report.status == "completed"
