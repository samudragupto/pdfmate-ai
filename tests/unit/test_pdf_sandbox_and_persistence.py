"""Unit tests for PyMuPDF Engine, Plugin Registry, Encrypted Sandbox, and Redis State."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import fitz
import pytest

from pdfmate.infrastructure.pdf.engine import PyMuPDFEngine
from pdfmate.infrastructure.pdf.plugin_registry import NodePluginRegistry
from pdfmate.infrastructure.persistence.redis_state import RedisPipelineStateStore
from pdfmate.infrastructure.sandbox.secure_executor import EncryptedSandbox


def test_pymupdf_engine_and_plugins(tmp_path: Path) -> None:
    engine = PyMuPDFEngine()
    missing_doc = engine.inspect_document(tmp_path / "nonexistent.pdf")
    assert missing_doc.page_count == 0

    src1 = engine.ensure_valid_pdf(tmp_path / "src1.pdf")
    src2 = engine.ensure_valid_pdf(tmp_path / "src2.pdf")
    inspected = engine.inspect_document(src1)
    assert inspected.page_count == 2

    registry = NodePluginRegistry(engine=engine, output_dir=tmp_path)
    assert set(registry.list_actions()) == {
        "compress",
        "merge",
        "ocr",
        "redact",
        "scan",
        "split",
        "summarize",
    }
    with pytest.raises(KeyError):
        registry.get("nonexistent_plugin")

    # 0. Scan
    scan_plugin = registry.get("scan")
    assert scan_plugin.validate(src1, {}) is True
    assert "Scanning" in scan_plugin.explain({})
    scanned_path = scan_plugin.execute(src1, {"output_path": str(tmp_path / "scanned.pdf")})
    assert scanned_path.exists()

    # 1. Merge
    merge_plugin = registry.get("merge")
    assert merge_plugin.validate(src1, {}) is True
    assert "Merging" in merge_plugin.explain({})
    merged_path = merge_plugin.execute(
        src1,
        {"input_paths": [str(src1), str(src2)], "output_path": str(tmp_path / "merged.pdf")},
    )
    assert merged_path.exists()
    assert engine.inspect_document(merged_path).page_count == 4

    # 2. Redact (verify student ID 20269871 is truly scrubbed from PDF stream)
    redact_plugin = registry.get("redact")
    assert redact_plugin.validate(merged_path, {"pattern": r"\b\d{8}\b"}) is True
    assert "redaction" in redact_plugin.explain({}).lower()
    redacted_path = redact_plugin.execute(
        merged_path,
        {
            "pattern": r"\b\d{8}\b",
            "bounding_box": [70.0, 65.0, 200.0, 95.0],
            "output_path": str(tmp_path / "redacted.pdf"),
        },
    )
    doc = fitz.open(str(redacted_path))
    try:
        full_text = "".join(page.get_text("text") for page in doc)
        assert "20269871" not in full_text
    finally:
        doc.close()

    # 3. Split
    split_plugin = registry.get("split")
    assert split_plugin.validate(redacted_path, {"mode": "bookmarks"}) is True
    assert "Using bookmark split not page count" in split_plugin.explain({"mode": "bookmarks"})
    split_path = split_plugin.execute(
        redacted_path,
        {"mode": "bookmarks", "output_path": str(tmp_path / "split.pdf")},
    )
    assert split_path.exists()

    # 4. Summarize
    sum_plugin = registry.get("summarize")
    assert sum_plugin.validate(split_path, {}) is True
    assert "study digest" in sum_plugin.explain({})
    sum_path = sum_plugin.execute(
        split_path,
        {"granularity": "section", "output_path": str(tmp_path / "sum.pdf")},
    )
    assert sum_path.exists()

    # 5. OCR
    ocr_plugin = registry.get("ocr")
    assert ocr_plugin.validate(sum_path, {}) is True
    assert "searchable text layer" in ocr_plugin.explain({})
    ocr_path = ocr_plugin.execute(
        sum_path,
        {"language": "eng", "output_path": str(tmp_path / "ocr.pdf")},
    )
    assert ocr_path.exists()

    # 6. Compress
    comp_plugin = registry.get("compress")
    assert comp_plugin.validate(ocr_path, {"dpi": 150}) is True
    assert "JPEG2000" in comp_plugin.explain({"codec": "JPEG2000", "dpi": 150})
    comp_path = comp_plugin.execute(
        ocr_path,
        {"codec": "JPEG2000", "dpi": 150, "output_path": str(tmp_path / "comp.pdf")},
    )
    assert comp_path.exists()


def test_encrypted_sandbox_aes256_audit_and_cache(tmp_path: Path) -> None:
    sandbox = EncryptedSandbox(vault_dir=tmp_path / "vault", session_id="test-session-01")
    sample_file = tmp_path / "plain.pdf"
    sample_file.write_bytes(b"%PDF-1.7 SECRET LECTURE CONTENT")

    enc_path = sandbox.store_encrypted_artifact(sample_file)
    assert enc_path.exists()
    raw_cipher = enc_path.read_bytes()
    assert b"SECRET LECTURE CONTENT" not in raw_cipher

    decrypted = sandbox.read_encrypted_artifact(enc_path)
    assert decrypted == b"%PDF-1.7 SECRET LECTURE CONTENT"

    sec_path = sandbox.write_secure("direct_test.pdf", b"DIRECT_VAULT_BYTES")
    assert sec_path.exists()
    assert sandbox.read_secure("direct_test.pdf") == b"DIRECT_VAULT_BYTES"
    sandbox.audit_log("write_secure", "hash123", sec_path, workflow_id="wf-sec")
    assert len(sandbox.get_audit_trail(workflow_id="wf-sec")) == 1

    with pytest.raises(ValueError):
        sandbox.decrypt_bytes(b"short")

    # Immutable SQLite Audit Trail
    entry = sandbox.record_audit_entry(
        workflow_id="wf-101",
        node_id="n-1",
        action="redact",
        parameters={"pattern": "id"},
        result_path=enc_path,
    )
    trail = sandbox.get_audit_trail(workflow_id="wf-101")
    assert len(trail) == 1
    assert trail[0].parameter_hash == entry.parameter_hash
    assert len(sandbox.get_audit_trail()) == 2

    # Verify SQLite immutability triggers block UPDATE and DELETE
    conn = sqlite3.connect(str(sandbox.db_path))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM audit_trail WHERE workflow_id = 'wf-101'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE audit_trail SET action = 'tampered' WHERE workflow_id = 'wf-101'")
    finally:
        conn.close()

    # Local SQLite Embedding Cache
    assert sandbox.get_cached_embeddings("doc-abc") is None
    sandbox.cache_embeddings("doc-abc", [0.12, -0.45, 0.88])
    cached = sandbox.get_cached_embeddings("doc-abc")
    assert cached == [0.12, -0.45, 0.88]


def test_redis_pipeline_state_store_fallback() -> None:
    store = RedisPipelineStateStore(redis_url="redis://127.0.0.1:63999/0")
    assert store.load_state("missing-wf") is None

    updated = store.update_node_progress(
        workflow_id="wf-200",
        node_id="node_1",
        status="completed",
        mentor_explanation="Merged 4 lecture files",
        output_path="/tmp/merged.pdf",
    )
    assert updated["nodes"]["node_1"]["status"] == "completed"
    loaded = store.load_state("wf-200")
    assert loaded is not None
    assert loaded["workflow_id"] == "wf-200"
