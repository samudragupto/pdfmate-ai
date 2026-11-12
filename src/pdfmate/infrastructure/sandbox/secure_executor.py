"""Encrypted Vault (AES-256-GCM), immutable SQLite audit log, and local embedding cache."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from pdfmate.domain.value_objects import AuditEntry


class EncryptedSandbox:
    """Provides AES-256-GCM encrypted document storage, immutable audit logging, and caching."""

    def __init__(
        self,
        vault_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        vault_path: Optional[Path] = None,
    ) -> None:
        resolved_vault = vault_path or vault_dir or Path("vault")
        self.vault_dir = resolved_vault
        self.vault_path = resolved_vault
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        self._salt = hashlib.sha256(self.session_id.encode("utf-8")).digest()[:16]
        self._key = self._derive_hardware_session_key()
        self._aesgcm = AESGCM(self._key)
        self.db_path = self.vault_dir / "pdfmate_audit_and_cache.sqlite3"
        self._init_sqlite_tables()

    def _derive_hardware_session_key(self) -> bytes:
        """Derive a 256-bit AES key from hardware fingerprint and session salt."""
        hw_fingerprint = (
            f"{platform.node()}:{platform.machine()}:{platform.processor()}:{uuid.getnode()}"
        )
        kdf = PBKDF2HMAC(
            algorithm=SHA256(),
            length=32,
            salt=self._salt,
            iterations=100_000,
        )
        return kdf.derive(hw_fingerprint.encode("utf-8"))

    def _init_sqlite_tables(self) -> None:
        """Initialize immutable audit log table and local embedding cache in SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_trail (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        node_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        parameter_hash TEXT NOT NULL,
                        result_path TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                    BEFORE UPDATE ON audit_trail
                    BEGIN
                        SELECT RAISE(ABORT, 'Audit trail records are immutable');
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                    BEFORE DELETE ON audit_trail
                    BEGIN
                        SELECT RAISE(ABORT, 'Audit trail records are immutable');
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        doc_hash TEXT PRIMARY KEY,
                        embedding_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
        finally:
            conn.close()

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """Encrypt raw bytes using AES-256-GCM (12-byte nonce prepended)."""
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt_bytes(self, payload: bytes) -> bytes:
        """Decrypt AES-256-GCM payload (extracting 12-byte prepended nonce)."""
        if len(payload) < 28:
            raise ValueError("Invalid encrypted payload length.")
        nonce = payload[:12]
        ciphertext = payload[12:]
        return self._aesgcm.decrypt(nonce, ciphertext, None)

    def store_encrypted_artifact(self, source_path: Path, vault_name: Optional[str] = None) -> Path:
        """Read a file, encrypt with AES-256-GCM, and write `.enc` artifact in vault."""
        raw_bytes = source_path.read_bytes()
        encrypted = self.encrypt_bytes(raw_bytes)
        target_name = vault_name or f"{source_path.name}.enc"
        target_path = self.vault_dir / target_name
        target_path.write_bytes(encrypted)
        return target_path

    def read_encrypted_artifact(self, encrypted_path: Path) -> bytes:
        """Read and decrypt an `.enc` artifact from the vault."""
        payload = encrypted_path.read_bytes()
        return self.decrypt_bytes(payload)

    def write_secure(self, filename: str, data: bytes) -> Path:
        """Encrypt raw bytes with AES-256-GCM and write directly into the vault."""
        encrypted = self.encrypt_bytes(data)
        target_name = filename if filename.endswith(".enc") else f"{filename}.enc"
        target_path = self.vault_dir / target_name
        target_path.write_bytes(encrypted)
        return target_path

    def read_secure(self, filename: str) -> bytes:
        """Read and decrypt a file by filename from the encrypted vault."""
        target_name = filename if filename.endswith(".enc") else f"{filename}.enc"
        target_path = self.vault_dir / target_name
        return self.read_encrypted_artifact(target_path)

    def audit_log(
        self,
        action: str,
        params_hash: str,
        result_path: Path,
        workflow_id: str = "manual",
        node_id: str = "node",
    ) -> None:
        """Write an immutable audit entry directly using a precomputed parameter hash."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO audit_trail (
                        timestamp, workflow_id, node_id, action, parameter_hash, result_path
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_iso,
                        workflow_id,
                        node_id,
                        action,
                        params_hash,
                        str(result_path),
                    ),
                )
        finally:
            conn.close()

    def record_audit_entry(
        self,
        workflow_id: str,
        node_id: str,
        action: str,
        parameters: Dict[str, Any],
        result_path: Path,
    ) -> AuditEntry:
        """Write an immutable audit entry to local SQLite and return the domain value object."""
        canonical_params = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.sha256(canonical_params.encode("utf-8")).hexdigest()
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = AuditEntry(
            timestamp=now_iso,
            workflow_id=workflow_id,
            node_id=node_id,
            action=action,
            parameter_hash=param_hash,
            result_path=str(result_path),
        )
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO audit_trail (
                        timestamp, workflow_id, node_id, action, parameter_hash, result_path
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.timestamp,
                        entry.workflow_id,
                        entry.node_id,
                        entry.action,
                        entry.parameter_hash,
                        entry.result_path,
                    ),
                )
        finally:
            conn.close()
        return entry

    def get_audit_trail(self, workflow_id: Optional[str] = None) -> List[AuditEntry]:
        """Retrieve immutable audit entries from SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            if workflow_id is not None:
                cursor.execute(
                    """
                    SELECT timestamp, workflow_id, node_id, action, parameter_hash, result_path
                    FROM audit_trail WHERE workflow_id = ? ORDER BY id ASC
                    """,
                    (workflow_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT timestamp, workflow_id, node_id, action, parameter_hash, result_path
                    FROM audit_trail ORDER BY id ASC
                    """
                )
            rows = cursor.fetchall()
            return [
                AuditEntry(
                    timestamp=r[0],
                    workflow_id=r[1],
                    node_id=r[2],
                    action=r[3],
                    parameter_hash=r[4],
                    result_path=r[5],
                )
                for r in rows
            ]
        finally:
            conn.close()

    def cache_embeddings(self, doc_hash: str, vector: List[float]) -> None:
        """Cache document embeddings in local SQLite to avoid redundant computation."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache (doc_hash, embedding_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (doc_hash, json.dumps(vector), now_iso),
                )
        finally:
            conn.close()

    def get_cached_embeddings(self, doc_hash: str) -> Optional[List[float]]:
        """Retrieve cached document embeddings from local SQLite if present."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT embedding_json FROM embedding_cache WHERE doc_hash = ?",
                (doc_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            loaded: List[float] = json.loads(str(row[0]))
            return loaded
        finally:
            conn.close()

    def execute_in_sandbox(
        self,
        func: Callable[[Path, Dict[str, Any]], Path],
        input_path: Path,
        params: Dict[str, Any],
    ) -> Path:
        """Execute a PDF operation inside the sandbox and store an encrypted backup copy."""
        result_path = func(input_path, params)
        if result_path.exists():
            self.store_encrypted_artifact(result_path)
        return result_path
