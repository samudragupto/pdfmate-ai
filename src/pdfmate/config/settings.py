"""Pydantic Settings and Hardware Profiles for Lenovo LOQ 15APH8 (RTX 4060 8GB)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareTier(str, Enum):
    """Supported hardware deployment tiers."""

    TIER1_STUDENT_LAPTOP = "student_laptop_rtx4060_8gb"
    TIER2_CREATOR_WORKSTATION = "creator_workstation_rtx4090_24gb"
    TIER3_CLOUD_SERVER = "cloud_server_a100_80gb"


class QuantizationProfile(str, Enum):
    """Quantization profiles mapped to hardware tiers."""

    Q4_K_M = "Q4_K_M"
    Q6_K = "Q6_K"
    FP16 = "FP16"


class HardwareProfile(BaseModel):
    """Immutable hardware specification profile."""

    model_config = {"frozen": True}

    laptop_class: str = "Lenovo LOQ 15APH8"
    gpu_name: str = "NVIDIA RTX 4060 Laptop GPU"
    total_vram_gb: float = 8.0
    max_reserved_vram_gb: float = 7.5
    vram_headroom_gb: float = 0.5
    system_ram_gb: float = 16.0
    cpu_series: str = "AMD Ryzen HS Series"
    tier: HardwareTier = HardwareTier.TIER1_STUDENT_LAPTOP
    quantization: QuantizationProfile = QuantizationProfile.Q4_K_M
    vlm_cpu_offload_threshold_gb: float = 5.5

    @classmethod
    def for_tier(cls, tier: HardwareTier) -> HardwareProfile:
        """Build a hardware profile for a given tier."""
        if tier == HardwareTier.TIER2_CREATOR_WORKSTATION:
            return cls(
                laptop_class="Creator Workstation",
                gpu_name="NVIDIA RTX 4090",
                total_vram_gb=24.0,
                max_reserved_vram_gb=22.0,
                vram_headroom_gb=2.0,
                system_ram_gb=64.0,
                tier=tier,
                quantization=QuantizationProfile.Q6_K,
                vlm_cpu_offload_threshold_gb=18.0,
            )
        if tier == HardwareTier.TIER3_CLOUD_SERVER:
            return cls(
                laptop_class="Cloud Server",
                gpu_name="NVIDIA A100 80GB",
                total_vram_gb=80.0,
                max_reserved_vram_gb=76.0,
                vram_headroom_gb=4.0,
                system_ram_gb=256.0,
                tier=tier,
                quantization=QuantizationProfile.FP16,
                vlm_cpu_offload_threshold_gb=64.0,
            )
        return cls()


class PDFMateSettings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PDFMate AI"
    offline_strict: bool = Field(default=True, alias="OFFLINE_STRICT")
    models_dir: Path = Field(default=Path("models"), alias="MODELS_DIR")
    llm_model_path: Path = Field(
        default=Path("models/qwen2.5-7b-instruct-q4_k_m.gguf"),
        alias="LLM_MODEL_PATH",
    )
    vlm_model_path: Path = Field(
        default=Path("models/qwen2-vl-7b-instruct-q4_k_m.gguf"),
        alias="VLM_MODEL_PATH",
    )
    vault_dir: Path = Field(
        default=Path.home() / "PDFMate" / "Vault",
        alias="VAULT_DIR",
    )
    gpu_max_vram: float = Field(default=7.5, alias="GPU_MAX_VRAM")
    llm_vram_footprint_gb: float = 4.5
    vlm_vram_footprint_gb: float = 5.8
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        alias="CELERY_BROKER_URL",
    )
    agent_url: str = Field(default="http://localhost:8000", alias="AGENT_URL")
    hardware_tier: HardwareTier = Field(
        default=HardwareTier.TIER1_STUDENT_LAPTOP,
        alias="HARDWARE_TIER",
    )

    def get_hardware_profile(self) -> HardwareProfile:
        """Return the active hardware profile."""
        profile = HardwareProfile.for_tier(self.hardware_tier)
        if self.gpu_max_vram != profile.max_reserved_vram_gb:
            return HardwareProfile(
                laptop_class=profile.laptop_class,
                gpu_name=profile.gpu_name,
                total_vram_gb=profile.total_vram_gb,
                max_reserved_vram_gb=self.gpu_max_vram,
                vram_headroom_gb=profile.vram_headroom_gb,
                system_ram_gb=profile.system_ram_gb,
                cpu_series=profile.cpu_series,
                tier=profile.tier,
                quantization=profile.quantization,
                vlm_cpu_offload_threshold_gb=profile.vlm_cpu_offload_threshold_gb,
            )
        return profile

    def verify_models_directory(self) -> None:
        """Enforce fail-closed initialization if the models directory or GGUF files are missing."""
        if not self.models_dir.exists() or not self.models_dir.is_dir():
            raise FileNotFoundError(
                f"Fail-closed security error: Models directory '{self.models_dir}' is missing."
            )
        gguf_files = list(self.models_dir.glob("*.gguf"))
        if not gguf_files and not self.llm_model_path.exists():
            raise FileNotFoundError(
                f"Fail-closed security error: No GGUF model files found in '{self.models_dir}'."
            )


def get_settings() -> PDFMateSettings:
    """Factory function for PDFMateSettings."""
    return PDFMateSettings()
