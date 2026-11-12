#!/usr/bin/env python3
"""
PDFMate AI Demo Launcher
Orchestrates the full 5-minute Buildathon demo on Lenovo LOQ 15APH8.

Usage:
    python scripts/demo_launcher.py --demo
    python scripts/demo_launcher.py --stop

Requires:
    - llama-server in PATH
    - Python virtual environment activated
    - Models downloaded to ./models/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LLM_MODEL = MODELS_DIR / "qwen2.5-7b-instruct-q4_k_m.gguf"
VLM_MODEL = MODELS_DIR / "qwen2-vl-7b-instruct-q4_k_m.gguf"
VAULT_DIR = PROJECT_ROOT / "vault"
LOGS_DIR = PROJECT_ROOT / "logs"

LLM_PORT = 8080
AGENT_PORT = 8000
UI_PORT = 8501

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo_launcher")

# Process handles for cleanup
_processes: list[subprocess.Popen[Any]] = []


def _cleanup() -> None:
    """Terminate all spawned processes."""
    for proc in reversed(_processes):
        if proc.poll() is None:
            logger.info("Terminating PID %d", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    logger.info("All processes terminated.")


def _install_signal_handlers() -> None:
    """Ensure cleanup runs on Ctrl+C and SIGTERM."""
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(0)))


def _wait_for_port(port: int, timeout: int = 120) -> bool:
    """Wait until a TCP port accepts connections."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def _wait_for_http(url: str, timeout: int = 120) -> bool:
    """Wait until an HTTP endpoint returns 200."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def _prewarm_model() -> None:
    """Send a trivial prompt to llama-server to load weights into VRAM."""
    import urllib.request

    url = f"http://127.0.0.1:{LLM_PORT}/completion"
    payload = json.dumps(
        {
            "prompt": "Hello",
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120):
            logger.info("Model pre-warmed. VRAM allocated.")
    except Exception as exc:
        logger.warning("Pre-warm failed (non-fatal): %s", exc)


def _start_llama_server() -> None:
    """Start llama-server for the agent brain."""
    logger.info("Starting llama-server on port %d...", LLM_PORT)
    llama_bin = shutil.which("llama-server") or r"D:\build\llama.cpp\build\bin\llama-server.exe"
    cmd = [
        llama_bin,
        "-m",
        str(LLM_MODEL),
        "--port",
        str(LLM_PORT),
        "--host",
        "127.0.0.1",
        "--n-gpu-layers",
        "35",
        "--ctx-size",
        "8192",
        "--threads",
        "8",
        "--parallel",
        "1",
        "--log-disable",
    ]
    log_file = open(LOGS_DIR / "llama_server.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    _processes.append(proc)
    if not _wait_for_port(LLM_PORT, timeout=120):
        raise RuntimeError(f"llama-server did not start on port {LLM_PORT}")
    logger.info("llama-server ready on port %d.", LLM_PORT)


def _start_agent() -> None:
    """Start the FastAPI agent service."""
    logger.info("Starting FastAPI agent on port %d...", AGENT_PORT)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "pdfmate.interfaces.api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(AGENT_PORT),
        "--log-level",
        "info",
    ]
    log_file = open(LOGS_DIR / "agent.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "LLM_SERVER_URL": f"http://127.0.0.1:{LLM_PORT}",
        },
    )
    _processes.append(proc)
    if not _wait_for_http(f"http://127.0.0.1:{AGENT_PORT}/health", timeout=30):
        raise RuntimeError(f"Agent did not start on port {AGENT_PORT}")
    logger.info("Agent ready on port %d.", AGENT_PORT)


def _start_ui() -> None:
    """Start the Streamlit Task Canvas UI."""
    logger.info("Starting Streamlit UI on port %d...", UI_PORT)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "src/pdfmate/interfaces/ui/streamlit_app.py"),
        "--server.port",
        str(UI_PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    log_file = open(LOGS_DIR / "ui.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "AGENT_URL": f"http://127.0.0.1:{AGENT_PORT}",
        },
    )
    _processes.append(proc)
    if not _wait_for_port(UI_PORT, timeout=30):
        raise RuntimeError(f"Streamlit UI did not start on port {UI_PORT}")
    logger.info("UI ready on port %d.", UI_PORT)


def _start_gpu_monitor() -> None:
    """Start a background nvidia-smi poller that writes to a log file."""
    logger.info("Starting GPU monitor...")
    cmd = [
        sys.executable,
        "-c",
        (
            "import subprocess, time\n"
            "while True:\n"
            "    subprocess.run(['nvidia-smi', "
            "'--query-gpu=utilization.gpu,memory.used,memory.total', "
            "'--format=csv,noheader,nounits', '-i', '0'])\n"
            "    time.sleep(1)\n"
        ),
    ]
    log_file = open(LOGS_DIR / "gpu_monitor.log", "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.DEVNULL)
    _processes.append(proc)


def _print_demo_checklist() -> None:
    """Print the 7-beat demo script to the terminal."""
    checklist = [
        "DEMO SCRIPT (5 minutes, 7 beats)",
        "=" * 55,
        "",
        "BEAT 1 (0:00-0:30) — HOOK",
        "  Open browser to http://localhost:8501",
        "  Point to the Privacy Lock indicator: 'GPU Running - 0% Cloud'",
        "  Physically disable Wi-Fi adapter (Fn+F5 or Settings)",
        "  Say: 'This laptop is offline. Everything runs on this GPU.'",
        "",
        "BEAT 2 (0:30-1:00) — NATURAL LANGUAGE INPUT",
        "  Type in the text box:",
        "  'I have 4 scanned lecture notes. Merge them, redact my",
        "   student ID, summarize each section, compress to under 10 MB.'",
        "  Watch nvidia-smi: VRAM spikes to ~4.5 GB (agent brain loading)",
        "",
        "BEAT 3 (1:00-1:30) — TASK CANVAS APPEARS",
        "  7 nodes render: Scan -> OCR -> Split -> Summarize ->",
        "  Compress -> Merge, plus Redact branch off Split",
        "  Say: 'Not a black box. You see every step.'",
        "",
        "BEAT 4 (1:30-2:00) — INTERACTIVE EDIT",
        "  Click the Compress node. Change JPEG2000 to Flate.",
        "  Click the Why? button on the Split node.",
        "  Mentor explains: 'Using bookmark split, not page count,",
        "   because your syllabus has 5 modules.'",
        "",
        "BEAT 5 (2:00-3:00) — EXECUTE",
        "  Hit Run. Watch nodes light up sequentially.",
        "  Redact and Summarize run in parallel (asyncio.gather).",
        "  nvidia-smi shows 60-70% GPU utilization.",
        "",
        "BEAT 6 (3:00-3:30) — LITERACY SCORE",
        "  Final PDF appears. Literacy Score: 72/100.",
        "  Tooltip: 'You now understand image compression tradeoffs",
        "   and redaction layers.'",
        "",
        "BEAT 7 (3:30-5:00) — CLOSE",
        "  Open terminal: sqlite3 vault/vault.db",
        "  Run: SELECT * FROM audit_trail LIMIT 7;",
        "  Show 7 immutable rows.",
        "  Say: 'Every operation is logged. Nothing left this machine.'",
        "",
        "=" * 55,
    ]
    print("\n".join(checklist))


def _stop_all() -> None:
    """Stop all running services."""
    logger.info("Stopping all services...")
    _cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="PDFMate AI Demo Launcher")
    parser.add_argument("--demo", action="store_true", help="Start full demo stack")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument(
        "--prewarm-only",
        action="store_true",
        help="Start llama-server and pre-warm model only",
    )
    args = parser.parse_args()

    _install_signal_handlers()

    # Ensure directories exist
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if args.stop:
        _stop_all()
        return

    if args.prewarm_only:
        _start_llama_server()
        _prewarm_model()
        logger.info("Model pre-warmed. llama-server running on port %d.", LLM_PORT)
        input("Press Enter to stop llama-server...")
        _stop_all()
        return

    if args.demo:
        logger.info("=== PDFMate AI Demo Launch ===")
        logger.info("Hardware: Lenovo LOQ 15APH8, RTX 4060 8GB, 16GB DDR5")
        logger.info("Models: Qwen2.5-7B (agent) + Qwen2-VL-7B (vision)")
        logger.info("")

        # Step 1: Start inference engine
        _start_llama_server()

        # Step 2: Pre-warm model into VRAM
        _prewarm_model()

        # Step 3: Start agent API
        _start_agent()

        # Step 4: Start UI
        _start_ui()

        # Step 5: Start GPU monitor
        _start_gpu_monitor()

        # Step 6: Print demo script
        _print_demo_checklist()

        # Step 7: Open browser
        logger.info("Opening browser to http://localhost:%d ...", UI_PORT)
        webbrowser.open(f"http://localhost:{UI_PORT}")

        logger.info("")
        logger.info("=== ALL SERVICES RUNNING ===")
        logger.info("  UI:      http://localhost:%d", UI_PORT)
        logger.info("  Agent:   http://localhost:%d/health", AGENT_PORT)
        logger.info("  LLM:     http://localhost:%d", LLM_PORT)
        logger.info("  GPU Log: %s", LOGS_DIR / "gpu_monitor.log")
        logger.info("")
        logger.info("Press Ctrl+C to stop all services.")
        logger.info("")

        # Block until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            _stop_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
