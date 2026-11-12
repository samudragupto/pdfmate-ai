"""Redis pipeline state persistence with automatic offline memory fallback."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import redis

logger = logging.getLogger(__name__)


class RedisPipelineStateStore:
    """Serializes workflow execution state to Redis for crash recovery and UI polling."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        self._memory_fallback: Dict[str, str] = {}
        self._client: Optional[redis.Redis] = None
        try:
            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=0.3,
                socket_timeout=0.3,
            )
            client.ping()
            self._client = client
        except Exception as exc:
            logger.debug("Redis unavailable (%s); using in-memory pipeline state store.", exc)
            self._client = None

    def _key(self, workflow_id: str) -> str:
        return f"pdfmate:workflow:{workflow_id}"

    def save_state(self, workflow_id: str, state: Dict[str, Any]) -> None:
        """Persist workflow state dictionary."""
        serialized = json.dumps(state, default=str)
        key = self._key(workflow_id)
        if self._client is not None:
            try:
                self._client.set(key, serialized)
                return
            except Exception as exc:
                logger.warning("Redis write failed (%s); falling back to memory.", exc)
        self._memory_fallback[key] = serialized

    def load_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve workflow state dictionary by workflow_id."""
        key = self._key(workflow_id)
        raw: Optional[str] = None
        if self._client is not None:
            try:
                val = self._client.get(key)
                raw = str(val) if val is not None else None
            except Exception as exc:
                logger.warning("Redis read failed (%s); falling back to memory.", exc)
        if raw is None:
            raw = self._memory_fallback.get(key)
        if raw is None:
            return None
        loaded: Dict[str, Any] = json.loads(raw)
        return loaded

    def update_node_progress(
        self,
        workflow_id: str,
        node_id: str,
        status: str,
        mentor_explanation: str = "",
        output_path: str = "",
    ) -> Dict[str, Any]:
        """Update execution progress for a single node in the workflow state."""
        state = self.load_state(workflow_id) or {
            "workflow_id": workflow_id,
            "status": "running",
            "nodes": {},
        }
        nodes_map = state.setdefault("nodes", {})
        nodes_map[node_id] = {
            "status": status,
            "mentor_explanation": mentor_explanation,
            "output_path": output_path,
        }
        self.save_state(workflow_id, state)
        return state
