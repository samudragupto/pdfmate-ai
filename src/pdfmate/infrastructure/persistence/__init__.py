"""Pipeline state persistence package."""

from __future__ import annotations

from pdfmate.infrastructure.persistence.redis_state import RedisPipelineStateStore

__all__ = ["RedisPipelineStateStore"]
