"""GPU Memory Manager for NVIDIA RTX 4060 Laptop GPU (8GB VRAM)."""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404
from typing import Dict

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manages VRAM allocation and CPU layer offload calculations for RTX 4060."""

    def __init__(
        self,
        max_reserved_gb: float = 7.5,
        headroom_gb: float = 0.5,
        vlm_offload_threshold_gb: float = 5.5,
    ) -> None:
        self.max_reserved_gb = max_reserved_gb
        self.headroom_gb = headroom_gb
        self.vlm_offload_threshold_gb = vlm_offload_threshold_gb
        self._allocated_models: Dict[str, float] = {}

    @property
    def current_allocated_gb(self) -> float:
        """Return currently reserved VRAM in GB across loaded models."""
        return sum(self._allocated_models.values())

    def get_available_vram_gb(self) -> float:
        """Query free GPU memory via nvidia-smi, falling back to 8.0 GB profile."""
        nvidia_smi_bin = shutil.which("nvidia-smi") or "nvidia-smi"
        try:
            result = subprocess.run(  # nosec B603
                [
                    nvidia_smi_bin,
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            free_mb = int(result.stdout.strip().split("\n")[0])
            raw_available = free_mb / 1024.0
            return max(0.0, raw_available - self.current_allocated_gb)
        except Exception as exc:
            logger.warning("Failed to query GPU: %s. Assuming 8.0 GB.", exc)
            return max(0.0, 8.0 - self.current_allocated_gb)

    def is_load_safe(self, model_size_gb: float) -> bool:
        """Determine whether loading a model of size model_size_gb is safe within VRAM budget."""
        available = self.get_available_vram_gb()
        budget_limit = min(available, self.max_reserved_gb - self.current_allocated_gb)
        safe = (model_size_gb + self.headroom_gb) <= budget_limit
        logger.info(
            "GPU check: model=%.2fGB, available=%.2fGB, safe=%s",
            model_size_gb,
            available,
            safe,
        )
        return safe

    def reserve_for_model(self, model_size_gb: float, model_name: str = "default_model") -> None:
        """Reserve VRAM for a model or raise MemoryError if outside RTX 4060 budget."""
        if not self.is_load_safe(model_size_gb):
            raise MemoryError("GPU VRAM reservation failed for RTX 4060 profile.")
        self._allocated_models[model_name] = model_size_gb

    def release_model(self, model_name: str = "default_model") -> None:
        """Release VRAM reservation for a previously loaded model."""
        self._allocated_models.pop(model_name, None)

    def compute_cpu_offload_layers(self, model_size_gb: float, total_layers: int = 28) -> int:
        """Calculate how many transformer layers to offload to DDR5 RAM if model > threshold."""
        if model_size_gb <= self.vlm_offload_threshold_gb:
            return 0
        excess_ratio = (model_size_gb - self.vlm_offload_threshold_gb) / model_size_gb
        offload_layers = max(1, int(round(total_layers * excess_ratio)))
        return min(total_layers, offload_layers)
