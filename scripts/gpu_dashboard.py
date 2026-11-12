#!/usr/bin/env python3
"""
Live GPU Monitor for PDFMate AI Demo.
Displays nvidia-smi data in a scrolling terminal table.

Usage:
    python scripts/gpu_dashboard.py
"""

from __future__ import annotations

import os
import subprocess
import time


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _get_gpu_data() -> dict[str, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "gpu_util": f"{parts[0]}%",
            "mem_util": f"{parts[1]}%",
            "mem_used": f"{parts[2]} / {parts[3]} MiB",
            "temp": f"{parts[4]} C",
            "power": f"{parts[5]} W",
        }
    except Exception:
        return {
            "gpu_util": "N/A",
            "mem_util": "N/A",
            "mem_used": "N/A",
            "temp": "N/A",
            "power": "N/A",
        }


def main() -> None:
    _clear()
    print("=" * 50)
    print("  PDFMate AI — GPU Monitor (RTX 4060)")
    print("=" * 50)
    print("  Press Ctrl+C to exit")
    print("=" * 50)

    try:
        while True:
            data = _get_gpu_data()
            print(f"\n  GPU Utilization:  {data['gpu_util']}")
            print(f"  Memory Used:      {data['mem_used']}")
            print(f"  Memory Util:      {data['mem_util']}")
            print(f"  Temperature:      {data['temp']}")
            print(f"  Power Draw:       {data['power']}")
            print("-" * 50)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nGPU monitor stopped.")


if __name__ == "__main__":
    main()
