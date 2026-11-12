"""Celery Worker entrypoint for horizontal CPU-bound PDF task execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox

F = TypeVar("F", bound=Callable[..., Any])


class _LocalCeleryFallback:
    """Zero-dependency fallback when running locally without Celery installed."""

    def __init__(self, name: str, broker: str, backend: str) -> None:
        self.name = name
        self.broker = broker
        self.backend = backend
        self.conf: Dict[str, Any] = {}

    def task(self, name: str) -> Callable[[F], F]:
        """Decorator preserving the underlying function signature."""

        def _decorator(func: F) -> F:
            setattr(func, "task_name", name)
            return func

        return _decorator


try:
    from celery import Celery as _CeleryClass
except ImportError:
    _CeleryClass = _LocalCeleryFallback

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app: _LocalCeleryFallback = _CeleryClass(
    "pdfmate_worker",
    broker=BROKER_URL,
    backend=BROKER_URL,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_always_eager=os.environ.get("CELERY_ALWAYS_EAGER", "false").lower() == "true",
)


@celery_app.task(name="pdfmate.execute_pdf_node")
def execute_pdf_node_task(
    workflow_id: str,
    node_id: str,
    action_type: str,
    input_path_str: str,
    params: Dict[str, Any],
    vault_dir_str: str = "vault",
) -> Dict[str, Any]:
    """Execute a CPU-bound PDF plugin operation inside the encrypted sandbox."""
    sandbox = EncryptedSandbox(vault_dir=Path(vault_dir_str))
    registry = NodePluginRegistry(output_dir=sandbox.vault_dir)
    plugin = registry.get(action_type)
    input_path = Path(input_path_str)
    output_path = sandbox.execute_in_sandbox(plugin.execute, input_path, params)
    audit = sandbox.record_audit_entry(
        workflow_id=workflow_id,
        node_id=node_id,
        action=action_type,
        parameters=params,
        result_path=output_path,
    )
    return {
        "workflow_id": workflow_id,
        "node_id": node_id,
        "action_type": action_type,
        "output_path": str(output_path),
        "explanation": plugin.explain(params),
        "parameter_hash": audit.parameter_hash,
    }
