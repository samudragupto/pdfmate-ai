"""Local AI inference infrastructure for RTX 4060 8GB hardware profile."""

from __future__ import annotations

from pdfmate.infrastructure.ai.gpu_memory_mgr import GPUMemoryManager
from pdfmate.infrastructure.ai.llm_client import LocalLLMClient
from pdfmate.infrastructure.ai.vlm_client import LocalVLMClient

__all__ = [
    "GPUMemoryManager",
    "LocalLLMClient",
    "LocalVLMClient",
]
