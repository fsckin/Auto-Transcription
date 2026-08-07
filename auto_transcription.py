#!/usr/bin/env python3
"""Local, resumable Sony recorder -> Whisper -> Ollama -> Obsidian pipeline."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Sequence
import urllib.error
import urllib.parse
import urllib.request

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None


APP_NAME = "auto-transcription"
APP_VERSION = "2.1.1"
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
FINAL_STATES = {"completed", "duplicate"}
ACTIVE_STATES = {"detected", "copying", "ready", "transcribing", "transcribed", "summarizing", "summarized", "writing"}


class PipelineError(RuntimeError):
    """An expected, user-actionable pipeline failure."""


@dataclasses.dataclass
class Config:
    mount_point: Path = Path("/Volumes/IC RECORDER")
    recorder_subdir: str = "REC_FILE"
    vault: Path = Path.home() / "Obsidian" / "Main"
    recordings_folder: str = "Recordings"
    state_root: Path = Path.home() / "Library" / "Application Support" / "AutoTranscription"
    archive_audio: bool = True
    copy_audio_to_vault: bool = False
    attachments_folder: str = "Recordings/Audio"
    note_template: Path | None = None
    generate_title: bool = False
    stable_wait_seconds: float = 1.0
    import_retry_count: int = 2
    copy_timeout_seconds: int = 1800
    backend: str = "auto"
    # The small model is the default memory-safe balance for a 16 GB Mac.
    # Larger models remain available through configuration or --maximum-accuracy.
    model: str = "small"
    retry_model: str = "small"
    hybrid_retry: bool = False
    confidence_threshold: float = -0.80
    language: str | None = "en"
    initial_prompt: str = ""
    word_timestamps: bool = False
    vad_enabled: bool = True
    vad_noise_db: float = -40.0
    vad_min_silence_seconds: float = 1.0
    vad_padding_seconds: float = 0.25
    audio_filter: str = ""
    device: str = "auto"
    transcription_timeout_seconds: int = 14400
    chunk_seconds: float = 900.0
    chunk_overlap_seconds: float = 2.0
    max_direct_seconds: float = 900.0
    max_unknown_duration_bytes: int = 100 * 1024 * 1024
    summary_enabled: bool = True
    summary_model: str = "mistral"
    ollama_url: str = "http://127.0.0.1:11434"
    summary_chunk_chars: int = 24000
    summary_timeout_seconds: int = 600
    summary_retry_count: int = 3
    summary_prompt: str = ""
    ollama_keep_alive: str = "2m"
    ollama_start_timeout_seconds: int = 30
    keep_audio: bool = True
    discard_after_days: int = 0
    notify: bool = True
    prevent_sleep: bool = True
    unmount_on_success: bool = True
    nice_level: int = 20
    log_max_bytes: int = 2_000_000
    log_backups: int = 3
    corrections: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def recorder_folder(self) -> Path:
        return self.mount_point / self.recorder_subdir

    @property
    def notes_folder(self) -> Path:
        return self.vault / self.recordings_folder

    @property
    def incoming_dir(self) -> Path:
        return self.state_root / "Incoming"

    @property
    def archive_dir(self) -> Path:
        return self.state_root / "Archive"

    @property
    def processing_dir(self) -> Path:
        return self.state_root / "Processing"

    @property
    def failed_dir(self) -> Path:
        return self.state_root / "Failed"

    @property
    def transcripts_dir(self) -> Path:
        return self.state_root / "Transcripts"

    @property
    def summaries_dir(self) -> Path:
        return self.state_root / "Summaries"

    @property
    def chunks_dir(self) -> Path:
        return self.state_root / "Chunks"

    @property
    def db_path(self) -> Path:
        return self.state_root / "state.sqlite3"

    @property
    def log_path(self) -> Path:
        return self.state_root / "auto-transcription.log"

    @property
    def ollama_log_path(self) -> Path:
        return self.state_root / "ollama-serve.log"

    @property
    def lock_dir(self) -> Path:
        return self.state_root / ".pipeline.lock"


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def load_config(path: Path | None) -> Config:
    cfg = Config()
    if path is None:
        default = Path.home() / ".config" / APP_NAME / "config.toml"
        path = default if default.exists() else None
    if path is None:
        return cfg
    if not path.exists():
        raise PipelineError(f"Configuration file not found: {path}")
    if tomllib is None:
        raise PipelineError("TOML configuration requires Python 3.11 or newer")
    with path.open("rb") as handle:
        data = tomllib.load(handle)

    sections: dict[str, dict[str, str]] = {
        "paths": {
            "mount_point": "mount_point", "recorder_subdir": "recorder_subdir",
            "vault": "vault", "recordings_folder": "recordings_folder",
            "state_root": "state_root", "attachments_folder": "attachments_folder",
        },
        "output": {
            "template": "note_template", "generate_title": "generate_title",
        },
        "import": {
            "archive_audio": "archive_audio", "copy_audio_to_vault": "copy_audio_to_vault",
            "stable_wait_seconds": "stable_wait_seconds", "keep_audio": "keep_audio",
            "discard_after_days": "discard_after_days", "retry_count": "import_retry_count",
            "copy_timeout_seconds": "copy_timeout_seconds",
        },
        "transcription": {
            "backend": "backend", "model": "model", "retry_model": "retry_model",
            "hybrid_retry": "hybrid_retry", "confidence_threshold": "confidence_threshold",
            "language": "language", "initial_prompt": "initial_prompt",
            "word_timestamps": "word_timestamps", "vad_enabled": "vad_enabled",
            "vad_noise_db": "vad_noise_db", "vad_min_silence_seconds": "vad_min_silence_seconds",
            "vad_padding_seconds": "vad_padding_seconds", "audio_filter": "audio_filter",
            "device": "device", "timeout_seconds": "transcription_timeout_seconds",
            "chunk_seconds": "chunk_seconds", "chunk_overlap_seconds": "chunk_overlap_seconds",
            "max_direct_seconds": "max_direct_seconds",
        },
        "summarization": {
            "enabled": "summary_enabled", "model": "summary_model", "ollama_url": "ollama_url",
            "chunk_chars": "summary_chunk_chars", "timeout_seconds": "summary_timeout_seconds",
            "retry_count": "summary_retry_count", "prompt": "summary_prompt",
            "keep_alive": "ollama_keep_alive", "startup_timeout_seconds": "ollama_start_timeout_seconds",
        },
        "behavior": {
            "notify": "notify", "prevent_sleep": "prevent_sleep", "log_max_bytes": "log_max_bytes",
            "log_backups": "log_backups", "nice_level": "nice_level",
            "unmount_on_success": "unmount_on_success",
        },
    }
    path_fields = {"mount_point", "vault", "state_root", "note_template"}
    for section, mapping in sections.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise PipelineError(f"Configuration section [{section}] must be a table")
        for source, target in mapping.items():
            if source in values:
                value = values[source]
                setattr(cfg, target, _expand_path(value) if target in path_fields else value)
    corrections = data.get("corrections", {})
    if not isinstance(corrections, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in corrections.items()):
        raise PipelineError("[corrections] must contain string-to-string replacements")
    cfg.corrections = dict(corrections)
    return cfg


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for arg, attr in (
        ("mount", "mount_point"), ("vault", "vault"), ("state_root", "state_root"),
        ("backend", "backend"), ("model", "model"), ("language", "language"),
    ):
        value = getattr(args, arg, None)
        if value is not None:
            setattr(cfg, attr, _expand_path(value) if attr in {"mount_point", "vault", "state_root"} else value)
    if getattr(args, "no_summary", False):
        cfg.summary_enabled = False
    if getattr(args, "no_notify", False):
        cfg.notify = False
    if getattr(args, "no_vad", False):
        cfg.vad_enabled = False
    if getattr(args, "maximum_accuracy", False):
        cfg.model = "large-v3"
        cfg.hybrid_retry = False
    if getattr(args, "keep_audio", None) is not None:
        cfg.keep_audio = args.keep_audio
    if getattr(args, "discard_after_days", None) is not None:
        cfg.discard_after_days = args.discard_after_days
    return cfg


def ensure_private_directories(cfg: Config) -> None:
    if cfg.chunk_seconds <= 0:
        raise PipelineError("transcription.chunk_seconds must be greater than zero")
    if cfg.chunk_overlap_seconds < 0 or cfg.chunk_overlap_seconds >= cfg.chunk_seconds / 2:
        raise PipelineError("transcription.chunk_overlap_seconds must be non-negative and less than half a chunk")
    if cfg.max_direct_seconds <= 0 or cfg.max_direct_seconds > cfg.chunk_seconds:
        raise PipelineError("transcription.max_direct_seconds must be greater than zero and no larger than chunk_seconds")
    if cfg.max_unknown_duration_bytes <= 0:
        raise PipelineError("transcription.max_unknown_duration_bytes must be greater than zero")
    if not isinstance(cfg.nice_level, int) or not 0 <= cfg.nice_level <= 20:
        raise PipelineError("behavior.nice_level must be an integer from 0 through 20")
    if cfg.ollama_start_timeout_seconds <= 0:
        raise PipelineError("summarization.startup_timeout_seconds must be greater than zero")
    old_umask = os.umask(0o077)
    try:
        for path in (cfg.state_root, cfg.incoming_dir, cfg.processing_dir, cfg.archive_dir, cfg.failed_dir,
                     cfg.transcripts_dir, cfg.summaries_dir, cfg.chunks_dir):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        cfg.notes_folder.mkdir(parents=True, exist_ok=True)
    finally:
        os.umask(old_umask)


def configure_logging(cfg: Config, verbose: bool = False, quiet: bool = False) -> logging.Logger:
    ensure_private_directories(cfg)
    logger = logging.getLogger(APP_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = RotatingFileHandler(cfg.log_path, maxBytes=cfg.log_max_bytes, backupCount=cfg.log_backups, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if not quiet:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)
    return logger


def apply_process_priority(nice_level: int) -> int | None:
    """Lower this process priority; all subsequently spawned children inherit it."""
    if not hasattr(os, "getpriority") or not hasattr(os, "setpriority"):
        return None
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < nice_level:
        os.setpriority(os.PRIO_PROCESS, 0, nice_level)
    return os.getpriority(os.PRIO_PROCESS, 0)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(path: Path, stat: os.stat_result) -> str:
    material = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8", "surrogateescape")
    return hashlib.sha256(material).hexdigest()


def safe_component(value: str, maximum: int = 90) -> str:
    value = re.sub(r"[\x00-\x1f/:\\]+", "-", value).strip(" .-")
    value = re.sub(r"\s+", " ", value)
    return (value or "recording")[:maximum]


def yaml_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class PipelineLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> "PipelineLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                self.path.mkdir(mode=0o700)
                atomic_write(self.path / "pid", f"{os.getpid()}\n")
                self.acquired = True
                return self
            except FileExistsError:
                pid_path = self.path / "pid"
                try:
                    pid = int(pid_path.read_text(encoding="utf-8").strip())
                    os.kill(pid, 0)
                except (FileNotFoundError, ValueError, ProcessLookupError):
                    with contextlib.suppress(FileNotFoundError):
                        pid_path.unlink()
                    with contextlib.suppress(OSError):
                        self.path.rmdir()
                    continue
                except PermissionError:
                    pass
                raise PipelineError(f"Another pipeline process is already running (PID {pid})")
        raise PipelineError(f"Unable to acquire pipeline lock: {self.path}")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.acquired:
            with contextlib.suppress(FileNotFoundError):
                (self.path / "pid").unlink()
            with contextlib.suppress(OSError):
                self.path.rmdir()


@contextlib.contextmanager
def operation_timeout(seconds: int, message: str) -> Iterator[None]:
    """Bound long local operations on Unix while preserving the previous alarm."""
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handle_timeout(signum: int, frame: Any) -> None:
        raise PipelineError(message)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


class StateDB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY,
                source_fingerprint TEXT NOT NULL UNIQUE,
                source_path TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                sha256 TEXT,
                local_path TEXT,
                status TEXT NOT NULL,
                failed_stage TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                duplicate_of INTEGER REFERENCES recordings(id),
                transcript_path TEXT,
                cleaned_transcript_path TEXT,
                segments_path TEXT,
                summary_path TEXT,
                note_path TEXT,
                detected_language TEXT,
                backend TEXT,
                model TEXT,
                model_version TEXT,
                summary_model TEXT,
                generated_title TEXT,
                transcription_generation INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL,
                transcribe_seconds REAL,
                summary_seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_recordings_status ON recordings(status);
            CREATE INDEX IF NOT EXISTS idx_recordings_sha ON recordings(sha256);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        self.conn.commit()
        columns = {str(row[1]) for row in self.conn.execute("PRAGMA table_info(recordings)")}
        if "generated_title" not in columns:
            self.conn.execute("ALTER TABLE recordings ADD COLUMN generated_title TEXT")
        if "summary_model" not in columns:
            self.conn.execute("ALTER TABLE recordings ADD COLUMN summary_model TEXT")
        if "transcription_generation" not in columns:
            self.conn.execute("ALTER TABLE recordings ADD COLUMN transcription_generation INTEGER NOT NULL DEFAULT 0")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def get_by_fingerprint(self, fingerprint: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM recordings WHERE source_fingerprint=?", (fingerprint,)).fetchone()

    def get(self, record_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM recordings WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise PipelineError(f"Unknown recording id: {record_id}")
        return row

    def add_detected(self, path: Path, stat: os.stat_result, fingerprint: str) -> int:
        timestamp = now_iso()
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO recordings
               (source_fingerprint, source_path, source_name, source_size, source_mtime_ns, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'detected', ?, ?)""",
            (fingerprint, str(path), path.name, stat.st_size, stat.st_mtime_ns, timestamp, timestamp),
        )
        self.conn.commit()
        if cursor.lastrowid:
            return int(cursor.lastrowid)
        row = self.get_by_fingerprint(fingerprint)
        assert row is not None
        return int(row["id"])

    def update(self, record_id: int, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = now_iso()
        fields = ", ".join(f"{key}=?" for key in values)
        self.conn.execute(f"UPDATE recordings SET {fields} WHERE id=?", (*values.values(), record_id))
        self.conn.commit()

    def increment_attempt(self, record_id: int) -> None:
        self.conn.execute("UPDATE recordings SET attempts=attempts+1, updated_at=? WHERE id=?", (now_iso(), record_id))
        self.conn.commit()

    def find_duplicate(self, digest: str, exclude_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            """SELECT * FROM recordings
               WHERE sha256=? AND id<>? AND status IN ('ready','transcribing','transcribed','summarizing','summarized','writing','completed','duplicate')
               ORDER BY CASE status WHEN 'completed' THEN 0 WHEN 'duplicate' THEN 1 ELSE 2 END, id LIMIT 1""",
            (digest, exclude_id),
        ).fetchone()

    def pending(self, include_failed: bool = False) -> list[sqlite3.Row]:
        statuses = ["ready", "transcribed", "summarized"] + (["failed"] if include_failed else [])
        placeholders = ",".join("?" for _ in statuses)
        return list(self.conn.execute(f"SELECT * FROM recordings WHERE status IN ({placeholders}) ORDER BY source_mtime_ns, id", statuses))

    def by_hash_prefix(self, prefix: str) -> sqlite3.Row:
        rows = list(self.conn.execute("SELECT * FROM recordings WHERE sha256 LIKE ? ORDER BY id", (f"{prefix}%",)))
        if not rows:
            raise PipelineError(f"No recording matches hash prefix: {prefix}")
        if len(rows) > 1 and len({row["sha256"] for row in rows}) > 1:
            raise PipelineError(f"Hash prefix is ambiguous: {prefix}")
        return rows[0]

    def counts(self) -> dict[str, int]:
        return {str(row[0]): int(row[1]) for row in self.conn.execute("SELECT status, COUNT(*) FROM recordings GROUP BY status")}

    def recent(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM recordings ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)))

    def historical_realtime_factor(self) -> float | None:
        row = self.conn.execute(
            "SELECT SUM(transcribe_seconds), SUM(duration_seconds) FROM recordings WHERE transcribe_seconds>0 AND duration_seconds>0"
        ).fetchone()
        if row and row[0] and row[1]:
            return float(row[0]) / float(row[1])
        return None

    def migrate_legacy_hashes(self, legacy_path: Path, logger: logging.Logger) -> int:
        marker = self.conn.execute("SELECT value FROM metadata WHERE key='legacy_hashes_migrated'").fetchone()
        if marker or not legacy_path.exists():
            return 0
        count = 0
        for line in legacy_path.read_text(encoding="utf-8", errors="replace").splitlines():
            digest = line.strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            timestamp = now_iso()
            before = self.conn.total_changes
            self.conn.execute(
                """INSERT OR IGNORE INTO recordings
                   (source_fingerprint, source_path, source_name, source_size, source_mtime_ns, sha256, status,
                    created_at, updated_at, completed_at)
                   VALUES (?, ?, ?, 0, 0, ?, 'completed', ?, ?, ?)""",
                (f"legacy:{digest}", "legacy", f"legacy-{digest[:8]}", digest, timestamp, timestamp, timestamp),
            )
            count += int(self.conn.total_changes > before)
        self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES('legacy_hashes_migrated',?)", (now_iso(),))
        self.conn.commit()
        if count:
            logger.info("Migrated %d legacy processed hashes", count)
        return count


@dataclasses.dataclass(frozen=True)
class Candidate:
    path: Path
    stat: os.stat_result
    fingerprint: str


def scan_recorder(cfg: Config, db: StateDB) -> list[Candidate]:
    folder = cfg.recorder_folder
    if not folder.is_dir():
        raise PipelineError(f"Recorder folder not found: {folder}")
    candidates: list[Candidate] = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            fingerprint = source_fingerprint(path, stat)
            known = db.get_by_fingerprint(fingerprint)
            if known is None or known["status"] == "detected" or (known["status"] == "failed" and known["failed_stage"] == "copy"):
                candidates.append(Candidate(path, stat, fingerprint))
    return candidates


def wait_until_stable(candidate: Candidate, seconds: float) -> os.stat_result:
    first = candidate.path.stat()
    if seconds > 0:
        time.sleep(seconds)
    second = candidate.path.stat()
    if (first.st_size != second.st_size or first.st_mtime_ns != second.st_mtime_ns or
            candidate.stat.st_size != second.st_size or candidate.stat.st_mtime_ns != second.st_mtime_ns):
        raise PipelineError(f"Recording is still changing: {candidate.path.name}")
    return second


def copy_candidate(candidate: Candidate, cfg: Config, db: StateDB, logger: logging.Logger, dry_run: bool = False) -> str:
    if dry_run:
        logger.info("Would import: %s (%d bytes)", candidate.path, candidate.stat.st_size)
        return "would-import"
    record_id = db.add_detected(candidate.path, candidate.stat, candidate.fingerprint)
    row = db.get(record_id)
    if row["status"] == "failed" and row["failed_stage"] == "copy":
        db.update(record_id, status="detected", failed_stage=None, last_error=None)
        row = db.get(record_id)
    if row["status"] != "detected":
        return str(row["status"])
    partial: Path | None = None
    try:
        stable_stat = wait_until_stable(candidate, cfg.stable_wait_seconds)
        free = shutil.disk_usage(cfg.incoming_dir).free
        required = stable_stat.st_size + max(100 * 1024 * 1024, int(stable_stat.st_size * 0.10))
        if free < required:
            raise PipelineError(f"Not enough local disk space to import {candidate.path.name}")
        db.update(record_id, status="copying", last_error=None, failed_stage=None)
        recorded = dt.datetime.fromtimestamp(stable_stat.st_mtime).astimezone()
        stem = safe_component(candidate.path.stem)
        final_name = f"{recorded:%Y%m%d-%H%M%S}_{stem}_{candidate.fingerprint[:8]}{candidate.path.suffix.lower()}"
        destination = cfg.incoming_dir / final_name
        partial = destination.with_name(destination.name + ".partial")
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()
        with operation_timeout(cfg.copy_timeout_seconds, f"Copy timed out for {candidate.path.name}"):
            with candidate.path.open("rb") as source, partial.open("xb") as target:
                os.chmod(partial, 0o600)
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
        copied_stat = partial.stat()
        if copied_stat.st_size != stable_stat.st_size:
            raise PipelineError(f"Incomplete copy for {candidate.path.name}: expected {stable_stat.st_size}, got {copied_stat.st_size}")
        digest = sha256_file(partial)
        duplicate = db.find_duplicate(digest, record_id)
        if duplicate is not None:
            partial.unlink()
            db.update(record_id, sha256=digest, status="duplicate", duplicate_of=duplicate["id"], completed_at=now_iso())
            logger.info("Duplicate: %s (same content as record %s)", candidate.path.name, duplicate["id"])
            return "duplicate"
        os.replace(partial, destination)
        os.utime(destination, ns=(stable_stat.st_atime_ns, stable_stat.st_mtime_ns))
        duration = ffprobe_duration(destination)
        db.update(record_id, sha256=digest, local_path=str(destination), source_size=stable_stat.st_size,
                  source_mtime_ns=stable_stat.st_mtime_ns, duration_seconds=duration, status="ready")
        logger.info("Imported: %s → %s", candidate.path.name, destination.name)
        return "ready"
    except Exception as exc:
        if partial is not None:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
        db.update(record_id, status="failed", failed_stage="copy", last_error=str(exc))
        logger.error("Import failed for %s: %s", candidate.path.name, exc)
        notify_error(f"Import failed for {candidate.path.name}: {exc}", cfg)
        return "failed"


def ffprobe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        text=True, capture_output=True, timeout=30, check=False,
    )
    try:
        return float(result.stdout.strip()) if result.returncode == 0 else None
    except ValueError:
        return None


def detect_speech_clips(path: Path, cfg: Config, logger: logging.Logger) -> list[float] | None:
    """Return Whisper clip timestamps using FFmpeg silence detection."""
    if not cfg.vad_enabled or shutil.which("ffmpeg") is None:
        return None
    duration = ffprobe_duration(path)
    if not duration or duration < cfg.vad_min_silence_seconds * 2:
        return None
    filter_value = f"silencedetect=noise={cfg.vad_noise_db}dB:d={cfg.vad_min_silence_seconds}"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", filter_value, "-f", "null", "-"],
        text=True, capture_output=True, timeout=max(60, int(duration * 2)), check=False,
    )
    text = result.stderr
    starts = [float(v) for v in re.findall(r"silence_start:\s*([0-9.]+)", text)]
    ends = [float(v) for v in re.findall(r"silence_end:\s*([0-9.]+)", text)]
    silences: list[tuple[float, float]] = []
    end_index = 0
    for start in starts:
        while end_index < len(ends) and ends[end_index] <= start:
            end_index += 1
        end = ends[end_index] if end_index < len(ends) else duration
        silences.append((start, min(end, duration)))
        end_index += 1
    if not silences:
        return None
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in silences:
        if start > cursor:
            speech.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        speech.append((cursor, duration))
    padded: list[tuple[float, float]] = []
    for start, end in speech:
        start = max(0.0, start - cfg.vad_padding_seconds)
        end = min(duration, end + cfg.vad_padding_seconds)
        if end - start < 0.15:
            continue
        if padded and start <= padded[-1][1] + 0.05:
            padded[-1] = (padded[-1][0], max(padded[-1][1], end))
        else:
            padded.append((start, end))
    speech_seconds = sum(end - start for start, end in padded)
    if not padded or speech_seconds >= duration * 0.98:
        return None
    logger.debug("Silence-aware VAD selected %.1f of %.1f seconds", speech_seconds, duration)
    return [value for pair in padded for value in pair]


@contextlib.contextmanager
def preprocessed_audio(path: Path, cfg: Config) -> Iterator[Path]:
    if not cfg.audio_filter:
        yield path
        return
    if shutil.which("ffmpeg") is None:
        raise PipelineError("FFmpeg is required when audio_filter is configured")
    fd, temp_name = tempfile.mkstemp(prefix="auto-transcription-", suffix=".wav")
    os.close(fd)
    temp = Path(temp_name)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(path), "-af", cfg.audio_filter,
             "-ar", "16000", "-ac", "1", str(temp)],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise PipelineError(f"Optional audio preprocessing failed: {result.stderr.strip()}")
        yield temp
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


@contextlib.contextmanager
def extracted_audio_chunk(path: Path, start: float, duration: float, cfg: Config) -> Iterator[Path]:
    """Decode one bounded mono PCM chunk, then remove it immediately after use."""
    if shutil.which("ffmpeg") is None:
        raise PipelineError("FFmpeg is required for safe transcription of long recordings")
    estimated_bytes = int(max(1.0, duration) * 16_000 * 2)
    free = shutil.disk_usage(cfg.chunks_dir).free
    if free < estimated_bytes + 100 * 1024 * 1024:
        raise PipelineError("Not enough free space to create a bounded transcription chunk")
    fd, temp_name = tempfile.mkstemp(prefix="audio-chunk-", suffix=".wav", dir=cfg.chunks_dir)
    os.close(fd)
    temp = Path(temp_name)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{duration:.3f}",
    ]
    if cfg.audio_filter:
        command += ["-af", cfg.audio_filter]
    command += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temp)]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise PipelineError(f"Audio chunk extraction failed: {completed.stderr.strip()[-2000:]}")
        yield temp
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def offset_and_trim_chunk_result(
    result: dict[str, Any], extract_start: float, core_start: float, core_end: float,
) -> dict[str, Any]:
    """Move chunk-relative timestamps to recording time and de-duplicate overlap."""
    normalized = normalize_result(result)
    retained: list[dict[str, Any]] = []
    for segment in normalized["segments"]:
        local_start = float(segment["start"])
        local_end = float(segment["end"])
        global_start = extract_start + local_start
        global_end = extract_start + local_end
        midpoint = (global_start + global_end) / 2
        if midpoint < core_start or midpoint >= core_end:
            continue
        item = dict(segment)
        item["start"] = max(core_start, global_start)
        item["end"] = min(core_end, max(item["start"], global_end))
        words = []
        for word in segment.get("words") or []:
            shifted = dict(word)
            if "start" in shifted:
                shifted["start"] = extract_start + float(shifted["start"])
            if "end" in shifted:
                shifted["end"] = extract_start + float(shifted["end"])
            words.append(shifted)
        item["words"] = words
        retained.append(item)
    text = "".join(str(item.get("text", "")) for item in retained).strip()
    if not retained and normalized["text"] and not normalized["segments"]:
        retained = [{
            "id": 0, "start": core_start, "end": core_end, "text": normalized["text"],
            "avg_logprob": None, "no_speech_prob": None, "compression_ratio": None, "words": [],
        }]
        text = normalized["text"]
    return {"text": text, "language": normalized.get("language"), "segments": retained}


def merge_chunk_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    language = None
    for result in results:
        language = language or result.get("language")
        segments.extend(result.get("segments") or [])
    segments.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    for index, segment in enumerate(segments):
        segment["id"] = index
    text = "".join(str(segment.get("text", "")) for segment in segments).strip()
    if not text:
        text = " ".join(str(result.get("text", "")).strip() for result in results).strip()
    return {"text": text, "language": language, "segments": segments}


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    segments = []
    for index, segment in enumerate(result.get("segments") or []):
        segments.append({
            "id": segment.get("id", index),
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": str(segment.get("text", "")),
            "avg_logprob": segment.get("avg_logprob"),
            "no_speech_prob": segment.get("no_speech_prob"),
            "compression_ratio": segment.get("compression_ratio"),
            "words": segment.get("words") or [],
        })
    return {"text": str(result.get("text", "")).strip(), "language": result.get("language"), "segments": segments}


def mlx_worker_main(request_path: Path, response_path: Path) -> int:
    """Run one MLX inference job in an expendable process."""
    request = json.loads(request_path.read_text(encoding="utf-8"))
    import mlx_whisper
    result = mlx_whisper.transcribe(
        request["audio_path"], path_or_hf_repo=request["model"], verbose=None,
        hallucination_silence_threshold=1.5 if request["word_timestamps"] else None,
        language=request["language"], initial_prompt=request["initial_prompt"],
        word_timestamps=request["word_timestamps"], clip_timestamps=request["clip_timestamps"],
        no_speech_threshold=0.6, condition_on_previous_text=True,
    )
    normalized = normalize_result(result)
    atomic_write(response_path, json.dumps(normalized, ensure_ascii=False) + "\n")
    return 0


class Transcriber:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.backend = self._select_backend(cfg.backend)
        self._mock_calls = 0

    @staticmethod
    def _select_backend(requested: str) -> str:
        if requested != "auto":
            return requested
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if Transcriber.mlx_available():
                return "mlx"
        try:
            import whisper  # noqa: F401
            return "openai"
        except ImportError:
            return "cli" if shutil.which("whisper") else "missing"

    @staticmethod
    def mlx_available() -> bool:
        """Probe MLX in a child process because Metal initialization can abort."""
        probe = (
            "import mlx_whisper; import mlx.core as mx; "
            "value=mx.array([1]); mx.eval(value)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=20, check=False,
        )
        return completed.returncode == 0

    @staticmethod
    def mlx_model_name(model: str) -> str:
        mapping = {
            "turbo": "mlx-community/whisper-turbo",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
        }
        return mapping.get(model, model)

    @staticmethod
    def openai_model_name(model: str) -> str:
        return "turbo" if model == "large-v3-turbo" else model

    def _transcribe_mlx_worker(
        self, path: Path, model: str, common: dict[str, Any],
    ) -> dict[str, Any]:
        """Use process exit as a hard boundary for MLX/Metal memory reclamation."""
        with tempfile.TemporaryDirectory(prefix="mlx-worker-", dir=self.cfg.chunks_dir) as directory:
            worker_dir = Path(directory)
            request_path = worker_dir / "request.json"
            response_path = worker_dir / "response.json"
            atomic_write(request_path, json.dumps({
                "audio_path": str(path),
                "model": self.mlx_model_name(model),
                "language": common["language"],
                "initial_prompt": common["initial_prompt"],
                "word_timestamps": common["word_timestamps"],
                "clip_timestamps": common["clip_timestamps"],
            }, ensure_ascii=False) + "\n")
            command = [sys.executable, str(Path(__file__).resolve()), "_mlx-worker", str(request_path), str(response_path)]
            try:
                completed = subprocess.run(
                    command, text=True, capture_output=True, check=False,
                    timeout=self.cfg.transcription_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise PipelineError(f"MLX worker timed out for {path.name}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-3000:]
                raise PipelineError(f"MLX worker failed for {path.name}: {detail or 'unknown error'}")
            if not response_path.exists():
                raise PipelineError(f"MLX worker produced no result for {path.name}")
            try:
                return normalize_result(json.loads(response_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise PipelineError(f"MLX worker returned an invalid result for {path.name}") from exc

    def transcribe(self, path: Path, model: str, clips: Sequence[float] | None = None) -> dict[str, Any]:
        target_model = self.mlx_model_name(model) if self.backend == "mlx" else self.openai_model_name(model)
        if self.backend == "openai":
            loaded = getattr(self, "_loaded_model_name", None)
            if loaded is not None and loaded != target_model:
                self.logger.info("Unloading transcription model before switching from %s to %s", loaded, target_model)
                self.release_models()
        clip_value: str | list[float] = list(clips) if clips else "0"
        common: dict[str, Any] = {
            "language": self.cfg.language or None,
            "initial_prompt": self.cfg.initial_prompt or None,
            "word_timestamps": self.cfg.word_timestamps,
            "clip_timestamps": clip_value,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
        }
        if self.backend == "mlx":
            result = self._transcribe_mlx_worker(path, model, common)
        elif self.backend == "openai":
            import whisper
            if not hasattr(self, "_openai_models"):
                self._openai_models: dict[str, Any] = {}
            name = self.openai_model_name(model)
            if name not in self._openai_models:
                device = None if self.cfg.device == "auto" else self.cfg.device
                self.logger.info("Loading Whisper model: %s", name)
                self._openai_models[name] = whisper.load_model(name, device=device)
            result = self._openai_models[name].transcribe(str(path), verbose=False, **common)
        elif self.backend == "mock":
            self._mock_calls += 1
            result = {
                "text": f"Mock transcript for {path.stem}.", "language": self.cfg.language or "en",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": f"Mock transcript for {path.stem}.",
                              "avg_logprob": -0.2, "no_speech_prob": 0.01}],
            }
        elif self.backend == "cli":
            with tempfile.TemporaryDirectory(prefix="auto-transcription-whisper-") as directory:
                command = [
                    "whisper", str(path), "--model", self.openai_model_name(model),
                    "--output_format", "json", "--output_dir", directory, "--verbose", "False",
                    "--word_timestamps", str(self.cfg.word_timestamps),
                    "--clip_timestamps", ",".join(str(v) for v in clips) if clips else "0",
                ]
                if self.cfg.language:
                    command += ["--language", self.cfg.language]
                if self.cfg.initial_prompt:
                    command += ["--initial_prompt", self.cfg.initial_prompt]
                if self.cfg.device != "auto":
                    command += ["--device", self.cfg.device]
                completed = subprocess.run(command, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise PipelineError(f"Whisper CLI failed: {completed.stderr.strip()[-2000:]}")
                output = Path(directory) / f"{path.stem}.json"
                if not output.exists():
                    matches = list(Path(directory).glob("*.json"))
                    if len(matches) != 1:
                        raise PipelineError("Whisper CLI did not produce its JSON transcript")
                    output = matches[0]
                result = json.loads(output.read_text(encoding="utf-8"))
        else:
            raise PipelineError("No transcription backend is available. Install mlx-whisper or openai-whisper.")
        if self.backend == "openai":
            self._loaded_model_name = target_model
        return normalize_result(result)

    def version(self) -> str:
        try:
            from importlib.metadata import version
            if self.backend == "cli":
                return "command-line"
            return version("mlx-whisper" if self.backend == "mlx" else "openai-whisper")
        except Exception:
            return "unknown"

    def release_models(self) -> None:
        """Drop model references and cached accelerator allocations between pipeline phases."""
        if hasattr(self, "_openai_models"):
            self._openai_models.clear()
            del self._openai_models
        if hasattr(self, "_loaded_model_name"):
            del self._loaded_model_name
        gc.collect()

    def prepare_models(self) -> None:
        models = [self.cfg.model]
        if self.cfg.hybrid_retry and self.cfg.retry_model not in models:
            models.append(self.cfg.retry_model)
        if self.backend == "mlx":
            if not self.mlx_available():
                raise PipelineError("MLX is installed, but Metal is unavailable in this macOS session")
            from mlx_whisper.load_models import load_model
            for model in models:
                self.logger.info("Downloading/loading MLX model: %s", model)
                load_model(self.mlx_model_name(model))
        elif self.backend == "openai":
            import whisper
            for model in models:
                name = self.openai_model_name(model)
                self.logger.info("Downloading/loading Whisper model: %s", name)
                whisper.load_model(name, device=None if self.cfg.device == "auto" else self.cfg.device)
        elif self.backend == "mock":
            return
        elif self.backend == "cli":
            self.logger.warning("The existing Whisper CLI will download the selected model on its first transcription")
            return
        else:
            raise PipelineError("No transcription backend is available")

    def hybrid_retry(self, path: Path, result: dict[str, Any]) -> dict[str, Any]:
        if not self.cfg.hybrid_retry or self.cfg.model == self.cfg.retry_model:
            return result
        low: list[tuple[float, float]] = []
        for segment in result["segments"]:
            logprob = segment.get("avg_logprob")
            no_speech = segment.get("no_speech_prob")
            if logprob is not None and float(logprob) < self.cfg.confidence_threshold and (no_speech is None or float(no_speech) < 0.6):
                start = max(0.0, float(segment["start"]) - 0.35)
                end = float(segment["end"]) + 0.35
                if low and start <= low[-1][1] + 0.5:
                    low[-1] = (low[-1][0], max(low[-1][1], end))
                else:
                    low.append((start, end))
        if not low:
            return result
        self.logger.info("Retrying %d low-confidence region(s) with %s", len(low), self.cfg.retry_model)
        replacement = self.transcribe(path, self.cfg.retry_model, [v for pair in low for v in pair])
        retained = [
            segment for segment in result["segments"]
            if not any(float(segment["start"]) < end and float(segment["end"]) > start for start, end in low)
        ]
        replacements = [
            segment for segment in replacement["segments"]
            if any(float(segment["start"]) < end and float(segment["end"]) > start for start, end in low)
        ]
        merged = sorted(retained + replacements, key=lambda item: (float(item["start"]), float(item["end"])))
        return {"text": "".join(str(s["text"]) for s in merged).strip(), "language": result["language"], "segments": merged}


class OllamaSummarizer:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger

    def _generate(self, prompt: str) -> str:
        if self.cfg.summary_model == "mock":
            return "Mock summary grounded in the transcript."
        payload = json.dumps({
            "model": self.cfg.summary_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.cfg.ollama_keep_alive,
            "options": {"temperature": 0.1},
        }).encode("utf-8")
        request = urllib.request.Request(
            self.cfg.ollama_url.rstrip("/") + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        result: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(1, max(1, self.cfg.summary_retry_count) + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.summary_timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= max(1, self.cfg.summary_retry_count):
                    break
                self.logger.warning("Ollama request failed; retrying (%d/%d): %s",
                                    attempt + 1, self.cfg.summary_retry_count, exc)
                time.sleep(min(2 ** (attempt - 1), 5))
        if result is None:
            raise PipelineError(f"Ollama request failed: {last_error}") from last_error
        text = str(result.get("response", "")).strip()
        if not text:
            raise PipelineError("Ollama returned an empty summary")
        return text

    def release_model(self) -> None:
        """Ask Ollama to unload the summary model after this recording."""
        if self.cfg.summary_model == "mock":
            return
        payload = json.dumps({
            "model": self.cfg.summary_model, "prompt": "", "stream": False, "keep_alive": 0,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.cfg.ollama_url.rstrip("/") + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(30, self.cfg.summary_timeout_seconds)) as response:
                response.read()
        except Exception as exc:
            self.logger.warning("Unable to unload Ollama model %s: %s", self.cfg.summary_model, exc)

    @staticmethod
    def chunks(text: str, maximum: int) -> list[str]:
        if len(text) <= maximum:
            return [text]
        chunks: list[str] = []
        current: list[str] = []
        length = 0
        for paragraph in text.splitlines(keepends=True):
            if current and length + len(paragraph) > maximum:
                chunks.append("".join(current).strip())
                current, length = [], 0
            while len(paragraph) > maximum:
                room = maximum - length
                current.append(paragraph[:room])
                chunks.append("".join(current).strip())
                current, length = [], 0
                paragraph = paragraph[room:]
            current.append(paragraph)
            length += len(paragraph)
        if current:
            chunks.append("".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def summarize(self, timestamped_transcript: str) -> str:
        custom = self.cfg.summary_prompt.strip()
        base = custom or (
            "Create concise, structured notes using only facts explicitly present in the transcript. "
            "Do not invent names, dates, decisions, or tasks. Mark uncertain material as uncertain. "
            "Use these Markdown sections when supported by the content: Overview, Key Points, Decisions, "
            "Action Items, and Open Questions. Preserve useful transcript timestamps."
        )
        chunks = self.chunks(timestamped_transcript, self.cfg.summary_chunk_chars)
        if len(chunks) == 1:
            return self._generate(f"{base}\n\nTRANSCRIPT:\n{chunks[0]}")
        partials = []
        for index, chunk in enumerate(chunks, 1):
            self.logger.info("Summarizing transcript chunk %d/%d", index, len(chunks))
            partials.append(self._generate(
                "Extract grounded notes from this transcript section. Retain timestamps and uncertainty. "
                "Do not add information.\n\n" + chunk
            ))
        joined = "\n\n".join(f"SECTION {i}:\n{text}" for i, text in enumerate(partials, 1))
        return self._generate(f"{base}\n\nCombine these section notes without adding information or duplicating points:\n\n{joined}")

    def title(self, summary: str) -> str:
        response = self._generate(
            "Create a factual title of at most eight words for these notes. Return only the title, with no quotes, "
            "Markdown, or ending punctuation. Do not introduce information.\n\n" + summary[:12000]
        )
        return safe_component(response.strip().splitlines()[0], maximum=70)


def probe_ollama(cfg: Config, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(cfg.ollama_url.rstrip("/") + "/api/tags", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_ollama_running(cfg: Config, logger: logging.Logger) -> bool:
    """Start a detached local Ollama server when summarization needs one.

    Return True when this call started a server and False when one was already
    reachable or summarization does not use Ollama.
    """
    if not cfg.summary_enabled or cfg.summary_model == "mock":
        return False
    try:
        probe_ollama(cfg)
        return False
    except Exception as initial_error:
        parsed = urllib.parse.urlparse(cfg.ollama_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PipelineError(f"Configured remote Ollama server is unavailable: {initial_error}") from initial_error
        executable = shutil.which("ollama")
        if executable is None:
            raise PipelineError("Ollama is not running and the ollama command is not installed") from initial_error

    command = [executable, "serve"]
    if platform.system() == "Darwin" and Path("/usr/bin/nice").exists():
        command = ["/usr/bin/nice", "-n", str(cfg.nice_level), executable, "serve"]
    logger.info("Ollama is not running; starting ollama serve")
    cfg.ollama_log_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.ollama_log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    deadline = time.monotonic() + cfg.ollama_start_timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            probe_ollama(cfg)
            logger.info("Ollama is ready (PID %d)", process.pid)
            return True
        except Exception as exc:
            last_error = exc
        if process.poll() is not None:
            break
        time.sleep(0.25)
    detail = ""
    with contextlib.suppress(OSError):
        detail = cfg.ollama_log_path.read_text(encoding="utf-8", errors="replace")[-1500:].strip()
    suffix = f": {detail}" if detail else f": {last_error}"
    raise PipelineError(f"ollama serve did not become ready{suffix}")


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    for source, target in corrections.items():
        text = text.replace(source, target)
    return text


def corrected_result(result: dict[str, Any], corrections: dict[str, str]) -> dict[str, Any]:
    if not corrections:
        return result
    copied = dict(result)
    copied["text"] = apply_corrections(str(result.get("text", "")), corrections)
    copied["segments"] = []
    for segment in result.get("segments", []):
        item = dict(segment)
        item["text"] = apply_corrections(str(item.get("text", "")), corrections)
        copied["segments"].append(item)
    return copied


def timestamped_text(result: dict[str, Any]) -> str:
    if not result["segments"]:
        return result["text"]
    return "\n".join(f"[{format_timestamp(float(segment['start']))}] {str(segment['text']).strip()}" for segment in result["segments"])


DEFAULT_NOTE_TEMPLATE = """---
date: {{date}}
time_recorded: {{time_recorded}}
source: "Sony ICD-UX570"
original_file: {{original_file}}
original_path: {{original_path}}
file_hash: {{file_hash}}
audio_duration_seconds: {{duration_seconds}}
language: {{language}}
transcription_backend: {{backend}}
transcription_model: {{model}}
transcription_version: {{model_version}}
transcription_seconds: {{transcription_seconds}}
summary_model: {{summary_model}}
summary_seconds: {{summary_seconds}}
processed_at: {{processed_at}}
tags:
  - recording/transcribed
---

{{title_heading}}

{{audio_link}}

## Summary

{{summary}}

## Full Transcript

{{transcript}}
"""


def render_template(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def unique_note_path(folder: Path, base_name: str, existing: str | None = None) -> Path:
    candidate = folder / f"{base_name}.md"
    if existing and candidate == Path(existing):
        return candidate
    if not candidate.exists():
        return candidate
    for number in range(2, 10000):
        alternate = folder / f"{base_name}-{number}.md"
        if not alternate.exists():
            return alternate
    raise PipelineError(f"Unable to choose a unique note name for {base_name}")


def copy_audio_attachment(source: Path, cfg: Config, digest: str) -> tuple[Path, str]:
    folder = cfg.vault / cfg.attachments_folder
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{safe_component(source.stem)}_{digest[:8]}{source.suffix.lower()}"
    if not destination.exists():
        partial = destination.with_name(destination.name + ".partial")
        try:
            shutil.copy2(source, partial)
            if sha256_file(partial) != digest:
                raise PipelineError("Vault audio attachment failed hash verification")
            os.replace(partial, destination)
        finally:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
    relative = destination.relative_to(cfg.vault).as_posix()
    return destination, f"![[{relative}]]"


def move_state_audio(source: Path, destination_dir: Path, digest: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if destination.exists():
        if sha256_file(destination) == digest:
            if source != destination:
                source.unlink()
            return destination
        destination = destination_dir / f"{source.stem}_{digest[:8]}{source.suffix}"
    os.replace(source, destination)
    return destination


class Pipeline:
    def __init__(self, cfg: Config, db: StateDB, logger: logging.Logger):
        self.cfg = cfg
        self.db = db
        self.logger = logger
        self.transcriber: Transcriber | None = None
        self.summarizer = OllamaSummarizer(cfg, logger)
        self._ollama_preflight_done = False

    def _ensure_ollama_preflight(self) -> None:
        if self._ollama_preflight_done:
            return
        ensure_ollama_running(self.cfg, self.logger)
        self._ollama_preflight_done = True

    def _ensure_transcriber(self) -> Transcriber:
        if self.transcriber is None:
            self.transcriber = Transcriber(self.cfg, self.logger)
            self.logger.info("Transcription backend: %s", self.transcriber.backend)
        return self.transcriber

    def _release_transcriber(self) -> None:
        if self.transcriber is None:
            return
        self.transcriber.release_models()
        self.transcriber = None
        gc.collect()

    def _chunk_cache_dir(self, row: sqlite3.Row) -> Path:
        settings = {
            "format": 1,
            "generation": int(row["transcription_generation"] or 0),
            "model": self.cfg.model,
            "retry_model": self.cfg.retry_model,
            "hybrid_retry": self.cfg.hybrid_retry,
            "confidence_threshold": self.cfg.confidence_threshold,
            "language": self.cfg.language,
            "initial_prompt": self.cfg.initial_prompt,
            "word_timestamps": self.cfg.word_timestamps,
            "vad_enabled": self.cfg.vad_enabled,
            "vad_noise_db": self.cfg.vad_noise_db,
            "vad_min_silence_seconds": self.cfg.vad_min_silence_seconds,
            "vad_padding_seconds": self.cfg.vad_padding_seconds,
            "audio_filter": self.cfg.audio_filter,
            "chunk_seconds": self.cfg.chunk_seconds,
            "chunk_overlap_seconds": self.cfg.chunk_overlap_seconds,
        }
        signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return self.cfg.chunks_dir / str(row["sha256"]) / f"g{settings['generation']}-{signature}"

    def _transcribe_bounded_chunk(
        self, audio_path: Path, extract_start: float, extract_end: float,
        core_start: float, core_end: float,
    ) -> dict[str, Any]:
        transcriber = self._ensure_transcriber()
        with operation_timeout(
            self.cfg.transcription_timeout_seconds,
            f"Transcription timed out for chunk beginning at {format_timestamp(core_start)}",
        ):
            with extracted_audio_chunk(audio_path, extract_start, extract_end - extract_start, self.cfg) as chunk_path:
                clips = detect_speech_clips(chunk_path, self.cfg, self.logger)
                result = transcriber.transcribe(chunk_path, self.cfg.model, clips)
                result = transcriber.hybrid_retry(chunk_path, result)
        return offset_and_trim_chunk_result(result, extract_start, core_start, core_end)

    def import_new(self, dry_run: bool = False, missing_ok: bool = False) -> dict[str, int]:
        try:
            candidates = scan_recorder(self.cfg, self.db)
        except PipelineError:
            if not missing_ok:
                raise
            self.logger.debug("Recorder is not mounted at %s", self.cfg.recorder_folder)
            candidates = []
        counts: dict[str, int] = {"found": len(candidates), "ready": 0, "duplicate": 0, "failed": 0}
        if not candidates:
            self.logger.info("No new recordings found")
            return counts
        self.logger.info("Found %d new recording(s)", len(candidates))
        for candidate in candidates:
            result = "failed"
            attempts = 1 if dry_run else max(1, self.cfg.import_retry_count)
            for attempt in range(1, attempts + 1):
                result = copy_candidate(candidate, self.cfg, self.db, self.logger, dry_run)
                if result != "failed" or attempt == attempts:
                    break
                self.logger.warning("Retrying import for %s (%d/%d)", candidate.path.name, attempt + 1, attempts)
                time.sleep(min(2 ** (attempt - 1), 5))
            if result in counts:
                counts[result] += 1
        return counts

    def _transcribe_record(self, row: sqlite3.Row, audio_path: Path) -> tuple[dict[str, Any], float]:
        started = time.monotonic()
        duration = ffprobe_duration(audio_path)
        if duration is None:
            transcriber = self._ensure_transcriber()
            if transcriber.backend != "mock" or audio_path.stat().st_size > self.cfg.max_unknown_duration_bytes:
                raise PipelineError(
                    "Audio duration could not be determined; refusing unsafe whole-file transcription"
                )
            self.logger.debug("Audio duration is unknown; allowing the isolated mock backend")
            with operation_timeout(self.cfg.transcription_timeout_seconds, f"Transcription timed out for {row['source_name']}"):
                with preprocessed_audio(audio_path, self.cfg) as input_path:
                    clips = detect_speech_clips(input_path, self.cfg, self.logger)
                    result = transcriber.transcribe(input_path, self.cfg.model, clips)
                    result = transcriber.hybrid_retry(input_path, result)
            return result, time.monotonic() - started
        if duration <= self.cfg.max_direct_seconds:
            transcriber = self._ensure_transcriber()
            with operation_timeout(self.cfg.transcription_timeout_seconds, f"Transcription timed out for {row['source_name']}"):
                with preprocessed_audio(audio_path, self.cfg) as input_path:
                    clips = detect_speech_clips(input_path, self.cfg, self.logger)
                    result = transcriber.transcribe(input_path, self.cfg.model, clips)
                    result = transcriber.hybrid_retry(input_path, result)
            return result, time.monotonic() - started

        cache_dir = self._chunk_cache_dir(row)
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        count = max(1, math.ceil(duration / self.cfg.chunk_seconds))
        self.logger.info(
            "Long recording: %.2f hours; transcribing as %d bounded %.0f-minute chunk(s)",
            duration / 3600, count, self.cfg.chunk_seconds / 60,
        )
        results: list[dict[str, Any]] = []
        for index in range(count):
            core_start = index * self.cfg.chunk_seconds
            core_end = min(duration, core_start + self.cfg.chunk_seconds)
            extract_start = max(0.0, core_start - self.cfg.chunk_overlap_seconds)
            extract_end = min(duration, core_end + self.cfg.chunk_overlap_seconds)
            checkpoint = cache_dir / f"chunk-{index:06d}.json"
            chunk_result: dict[str, Any] | None = None
            if checkpoint.exists():
                try:
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if (saved.get("core_start") == core_start and saved.get("core_end") == core_end
                            and isinstance(saved.get("result"), dict)):
                        chunk_result = normalize_result(saved["result"])
                        self.logger.info("Chunk %d/%d already complete; resuming", index + 1, count)
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    self.logger.warning("Ignoring unreadable chunk checkpoint: %s", checkpoint.name)
            if chunk_result is None:
                self.logger.info("Transcribing chunk %d/%d (%s to %s)", index + 1, count,
                                 format_timestamp(core_start), format_timestamp(core_end))
                chunk_result = self._transcribe_bounded_chunk(
                    audio_path, extract_start, extract_end, core_start, core_end,
                )
                atomic_write(checkpoint, json.dumps({
                    "format": 1,
                    "core_start": core_start,
                    "core_end": core_end,
                    "extract_start": extract_start,
                    "extract_end": extract_end,
                    "result": chunk_result,
                }, ensure_ascii=False, indent=2) + "\n")
            results.append(chunk_result)
        result = merge_chunk_results(results)
        if not result["text"]:
            raise PipelineError("All transcription chunks completed but returned no text")
        elapsed = time.monotonic() - started
        return result, elapsed

    def process_record(self, record_id: int) -> str:
        row = self.db.get(record_id)
        if row["status"] in FINAL_STATES:
            return str(row["status"])
        audio_path = Path(row["local_path"]) if row["local_path"] else None
        if audio_path is None or not audio_path.exists():
            message = f"Local audio file is missing for {row['source_name']}"
            self.db.update(record_id, status="failed", failed_stage="transcribe", last_error=message)
            notify_error(message, self.cfg)
            return "failed"
        # Do this before loading Whisper so Ollama startup cannot overlap the
        # transcription model's peak memory use.
        self._ensure_ollama_preflight()
        digest = str(row["sha256"])
        if audio_path.parent in {self.cfg.incoming_dir, self.cfg.failed_dir}:
            audio_path = move_state_audio(audio_path, self.cfg.processing_dir, digest)
            self.db.update(record_id, local_path=str(audio_path))
        stage = "transcribe"
        summary_model_touched = False
        try:
            transcript_path = Path(row["transcript_path"]) if row["transcript_path"] else self.cfg.transcripts_dir / f"{digest}.raw.txt"
            cleaned_path = Path(row["cleaned_transcript_path"]) if row["cleaned_transcript_path"] else self.cfg.transcripts_dir / f"{digest}.clean.txt"
            segments_path = Path(row["segments_path"]) if row["segments_path"] else self.cfg.transcripts_dir / f"{digest}.segments.json"
            if not transcript_path.exists() or not segments_path.exists():
                self.db.increment_attempt(record_id)
                self.db.update(record_id, status="transcribing", failed_stage=None, last_error=None,
                               backend=self.cfg.backend, model=self.cfg.model)
                self.logger.info("Transcribing: %s", row["source_name"])
                result, elapsed = self._transcribe_record(row, audio_path)
                backend_name = self.transcriber.backend if self.transcriber else (row["backend"] or self.cfg.backend)
                model_version = self.transcriber.version() if self.transcriber else (row["model_version"] or "unknown")
                raw = result["text"].strip()
                if not raw:
                    raise PipelineError("Transcription returned no text")
                cleaned = apply_corrections(raw, self.cfg.corrections)
                atomic_write(transcript_path, raw + "\n")
                atomic_write(cleaned_path, cleaned + "\n")
                atomic_write(segments_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                duration = ffprobe_duration(audio_path)
                self.db.update(record_id, status="transcribed", transcript_path=str(transcript_path),
                               cleaned_transcript_path=str(cleaned_path), segments_path=str(segments_path),
                               detected_language=result.get("language"), backend=backend_name,
                               model=self.cfg.model, model_version=model_version,
                               duration_seconds=duration, transcribe_seconds=elapsed)
                self.logger.info("Transcribed in %.1fs", elapsed)
            else:
                result = json.loads(segments_path.read_text(encoding="utf-8"))
                raw = transcript_path.read_text(encoding="utf-8").strip()
                cleaned = cleaned_path.read_text(encoding="utf-8").strip() if cleaned_path.exists() else apply_corrections(raw, self.cfg.corrections)

            # Whisper and Ollama must never retain their large models at the same time.
            self._release_transcriber()
            stage = "summary"
            summary_path = Path(row["summary_path"]) if row["summary_path"] else self.cfg.summaries_dir / f"{digest}.md"
            summary = "_Summary generation was disabled._"
            if self.cfg.summary_enabled:
                if not summary_path.exists():
                    self.db.update(record_id, status="summarizing", failed_stage=None, last_error=None)
                    self.logger.info("Summarizing: %s", row["source_name"])
                    started = time.monotonic()
                    summary_model_touched = True
                    summary = self.summarizer.summarize(timestamped_text(corrected_result(result, self.cfg.corrections)))
                    elapsed = time.monotonic() - started
                    atomic_write(summary_path, summary.strip() + "\n")
                    self.db.update(record_id, status="summarized", summary_path=str(summary_path),
                                   summary_model=self.cfg.summary_model, summary_seconds=elapsed)
                    self.logger.info("Summarized in %.1fs", elapsed)
                else:
                    summary = summary_path.read_text(encoding="utf-8").strip()
            else:
                self.db.update(record_id, status="summarized")

            row = self.db.get(record_id)
            stage = "note"
            self.db.update(record_id, status="writing", failed_stage=None, last_error=None)
            recorded = dt.datetime.fromtimestamp(int(row["source_mtime_ns"]) / 1_000_000_000).astimezone()
            generated_title = row["generated_title"]
            if self.cfg.generate_title and not generated_title and self.cfg.summary_enabled:
                summary_model_touched = True
                generated_title = self.summarizer.title(summary)
                self.db.update(record_id, generated_title=generated_title)
            audio_link = ""
            if self.cfg.copy_audio_to_vault:
                _, audio_link = copy_audio_attachment(audio_path, self.cfg, digest)
            descriptive = safe_component(generated_title) if generated_title else safe_component(Path(row["source_name"]).stem)
            base = f"{recorded:%Y-%m-%d %H-%M-%S}_{descriptive}_{digest[:8]}"
            note_path = unique_note_path(self.cfg.notes_folder, base, row["note_path"])
            values = {
                "date": yaml_string(recorded.date().isoformat()),
                "time_recorded": yaml_string(recorded.isoformat(timespec="seconds")),
                "original_file": yaml_string(row["source_name"]),
                "original_path": yaml_string(row["source_path"]),
                "file_hash": yaml_string(digest),
                "duration_seconds": str(round(float(row["duration_seconds"]), 3)) if row["duration_seconds"] else "null",
                "language": yaml_string(row["detected_language"] or self.cfg.language or "unknown"),
                "backend": yaml_string(row["backend"] or self.cfg.backend),
                "model": yaml_string(row["model"] or self.cfg.model),
                "model_version": yaml_string(row["model_version"] or "unknown"),
                "transcription_seconds": str(round(float(row["transcribe_seconds"]), 3)) if row["transcribe_seconds"] else "null",
                "summary_model": yaml_string(row["summary_model"] or (self.cfg.summary_model if self.cfg.summary_enabled else "disabled")),
                "summary_seconds": str(round(float(row["summary_seconds"]), 3)) if row["summary_seconds"] else "null",
                "processed_at": yaml_string(dt.datetime.now().astimezone().isoformat(timespec="seconds")),
                "title_heading": f"# {generated_title}" if generated_title else "",
                "audio_link": audio_link,
                "summary": summary.strip(),
                "transcript": cleaned.strip(),
            }
            template = DEFAULT_NOTE_TEMPLATE
            if self.cfg.note_template:
                if not self.cfg.note_template.is_file():
                    raise PipelineError(f"Note template not found: {self.cfg.note_template}")
                template = self.cfg.note_template.read_text(encoding="utf-8")
            note = render_template(template, values).rstrip() + "\n"
            if note_path.exists():
                existing = note_path.read_text(encoding="utf-8")
                if existing != note:
                    note_path = unique_note_path(self.cfg.notes_folder, base)
            atomic_write(note_path, note, mode=0o644)
            destination_dir = self.cfg.archive_dir if self.cfg.archive_audio else self.cfg.incoming_dir
            final_audio = move_state_audio(audio_path, destination_dir, digest) if audio_path.parent != destination_dir else audio_path
            self.db.update(record_id, status="completed", local_path=str(final_audio), note_path=str(note_path),
                           completed_at=now_iso(), failed_stage=None, last_error=None)
            self.logger.info("Complete: %s → %s", row["source_name"], note_path.name)
            return "completed"
        except Exception as exc:
            failed_audio = audio_path
            if audio_path.exists() and audio_path.parent == self.cfg.processing_dir:
                with contextlib.suppress(Exception):
                    failed_audio = move_state_audio(audio_path, self.cfg.failed_dir, digest)
            self.db.update(record_id, status="failed", failed_stage=stage, last_error=str(exc), local_path=str(failed_audio))
            self.logger.exception("%s failed for %s: %s", stage.capitalize(), row["source_name"], exc)
            notify_error(f"{stage.capitalize()} failed for {row['source_name']}: {exc}", self.cfg)
            return "failed"
        finally:
            self._release_transcriber()
            if summary_model_touched:
                release = getattr(self.summarizer, "release_model", None)
                if release is not None:
                    release()

    def process_pending(self, include_failed: bool = False) -> dict[str, int]:
        rows = self.db.pending(include_failed=include_failed)
        counts = {"completed": 0, "failed": 0}
        if not rows:
            self.logger.info("No recordings are waiting for processing")
            return counts
        factor = self.db.historical_realtime_factor()
        if factor:
            known = sum(float(row["duration_seconds"] or 0) for row in rows)
            if known:
                self.logger.info("Estimated transcription time: about %s", dt.timedelta(seconds=int(known * factor)))
        for index, row in enumerate(rows, 1):
            self.logger.info("Processing %d/%d", index, len(rows))
            result = self.process_record(int(row["id"]))
            counts[result if result in counts else "failed"] += 1
        return counts


@contextlib.contextmanager
def prevent_sleep(enabled: bool) -> Iterator[None]:
    process: subprocess.Popen[Any] | None = None
    if enabled and platform.system() == "Darwin" and shutil.which("caffeinate"):
        process = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        if process and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)


def notify(title: str, message: str, enabled: bool) -> None:
    if not enabled or platform.system() != "Darwin" or shutil.which("osascript") is None:
        return
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_message = message.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.run(["osascript", "-e", f'display notification "{safe_message}" with title "{safe_title}"'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logging.getLogger(APP_NAME).warning("Unable to deliver notification: %s", exc)


def notify_error(message: str, cfg: Config) -> None:
    notify("Auto Transcription Error", message[:350], cfg.notify)


def notify_new_recordings(count: int, cfg: Config) -> None:
    if count <= 0:
        return
    noun = "recording" if count == 1 else "recordings"
    notify("Auto Transcription", f"Transcribing {count} new {noun}", cfg.notify)


def notify_finished(completed: int, failed: int, cfg: Config) -> None:
    if completed <= 0 and failed <= 0:
        return
    message = f"Finished: {completed} completed"
    if failed:
        message += f", {failed} failed"
    notify("Auto Transcription", message, cfg.notify)


def unmount_recorder(cfg: Config, logger: logging.Logger) -> bool:
    """Unmount the configured removable volume after a fully successful run."""
    if not cfg.unmount_on_success or platform.system() != "Darwin":
        return False
    mount = cfg.mount_point.resolve()
    if not mount.exists() or not os.path.ismount(mount):
        logger.debug("Recorder volume is no longer mounted: %s", mount)
        return False
    volumes_root = Path("/Volumes").resolve()
    if mount.parent != volumes_root:
        raise PipelineError(f"Refusing to unmount a path outside /Volumes: {mount}")
    diskutil = Path("/usr/sbin/diskutil")
    executable = str(diskutil) if diskutil.is_file() else shutil.which("diskutil")
    if not executable:
        raise PipelineError("Cannot unmount recorder because diskutil was not found")
    logger.info("Unmounting recorder volume: %s", mount)
    try:
        completed = subprocess.run(
            [executable, "unmount", str(mount)], text=True, capture_output=True,
            timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PipelineError(f"Unable to unmount recorder {mount.name}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1500:]
        raise PipelineError(f"Unable to unmount recorder {mount.name}: {detail or 'diskutil failed'}")
    logger.info("Recorder unmounted successfully: %s", mount.name)
    return True


def check_dependencies(cfg: Config, logger: logging.Logger) -> bool:
    ok = True
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("Platform: %s %s", platform.system(), platform.machine())
    for command in ("ffmpeg", "ffprobe"):
        found = shutil.which(command)
        logger.info("%-12s %s", command + ":", found or "MISSING")
        ok &= found is not None
    transcriber = Transcriber(cfg, logger)
    logger.info("Backend: %s", transcriber.backend)
    if transcriber.backend == "missing":
        ok = False
    if transcriber.backend == "mlx" and not transcriber.mlx_available():
        logger.error("MLX: installed, but Metal is unavailable in this session")
        ok = False
    if cfg.summary_enabled and cfg.summary_model != "mock":
        try:
            with urllib.request.urlopen(cfg.ollama_url.rstrip("/") + "/api/tags", timeout=5) as response:
                tags = json.loads(response.read().decode("utf-8"))
            names = {item.get("name", "").split(":")[0] for item in tags.get("models", [])}
            wanted = cfg.summary_model.split(":")[0]
            installed = wanted in names
            logger.info("Ollama model: %s (%s)", cfg.summary_model, "installed" if installed else "MISSING")
            ok &= installed
        except Exception as exc:
            logger.error("Ollama: unavailable (%s)", exc)
            ok = False
    recorder_ready = cfg.recorder_folder.is_dir() and os.access(cfg.recorder_folder, os.R_OK)
    logger.info("Recorder:    %s (%s)", cfg.recorder_folder, "readable" if recorder_ready else "not mounted")
    for label, path in (("Vault", cfg.vault), ("State", cfg.state_root)):
        writable = path.is_dir() and os.access(path, os.W_OK)
        logger.info("%-12s %s (%s)", label + ":", path, "writable" if writable else "NOT WRITABLE")
        ok &= writable
    return bool(ok)


def prepare_dependencies(cfg: Config, logger: logging.Logger) -> None:
    transcriber = Transcriber(cfg, logger)
    transcriber.prepare_models()
    if cfg.summary_enabled and cfg.summary_model != "mock":
        try:
            with urllib.request.urlopen(cfg.ollama_url.rstrip("/") + "/api/tags", timeout=5) as response:
                tags = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise PipelineError(f"Ollama is unavailable: {exc}") from exc
        names = {item.get("name", "").split(":")[0] for item in tags.get("models", [])}
        if cfg.summary_model.split(":")[0] not in names:
            if shutil.which("ollama") is None:
                raise PipelineError(f"Ollama model is missing: {cfg.summary_model}")
            logger.info("Downloading Ollama model: %s", cfg.summary_model)
            result = subprocess.run(["ollama", "pull", cfg.summary_model], check=False)
            if result.returncode != 0:
                raise PipelineError(f"Unable to download Ollama model: {cfg.summary_model}")


def reset_failed(db: StateDB, logger: logging.Logger) -> int:
    rows = list(db.conn.execute("SELECT * FROM recordings WHERE status='failed'"))
    for row in rows:
        if row["transcript_path"] and Path(row["transcript_path"]).exists():
            status = "transcribed"
        elif row["local_path"] and Path(row["local_path"]).exists():
            status = "ready"
        else:
            status = "detected"
        db.update(int(row["id"]), status=status, failed_stage=None, last_error=None)
    logger.info("Reset %d failed recording(s)", len(rows))
    return len(rows)


def recover_interrupted(db: StateDB, cfg: Config, logger: logging.Logger) -> int:
    """Return records left in an in-progress state to their last durable stage."""
    rows = list(db.conn.execute(
        "SELECT * FROM recordings WHERE status IN ('copying','transcribing','summarizing','writing')"
    ))
    for row in rows:
        if row["status"] == "copying":
            status = "detected"
        elif row["summary_path"] and Path(row["summary_path"]).exists():
            status = "summarized"
        elif row["transcript_path"] and Path(row["transcript_path"]).exists():
            status = "transcribed"
        elif row["local_path"] and Path(row["local_path"]).exists():
            status = "ready"
        else:
            status = "detected"
        db.update(int(row["id"]), status=status, failed_stage=None,
                  last_error="Recovered after an interrupted pipeline run")
    for partial in cfg.incoming_dir.glob("*.partial"):
        with contextlib.suppress(OSError):
            partial.unlink()
    if rows:
        logger.warning("Recovered %d interrupted recording(s)", len(rows))
    return len(rows)


def reset_for_reprocess(db: StateDB, hash_prefix: str, logger: logging.Logger) -> int:
    row = db.by_hash_prefix(hash_prefix)
    if not row["local_path"] or not Path(row["local_path"]).exists():
        raise PipelineError("Cannot reprocess because the local audio file is missing")
    db.update(int(row["id"]), status="ready", failed_stage=None, last_error=None,
              transcript_path=None, cleaned_transcript_path=None, segments_path=None,
              summary_path=None, note_path=None, completed_at=None,
              transcription_generation=int(row["transcription_generation"] or 0) + 1)
    logger.info("Queued for reprocessing: %s", row["source_name"])
    return int(row["id"])


def reset_application_state(cfg: Config) -> bool:
    """Delete the configured application state root after strict safety checks."""
    target = cfg.state_root.resolve()
    home = Path.home().resolve()
    cwd = Path.cwd().resolve()
    protected = {
        Path("/").resolve(), home, home / "Library", home / "Library" / "Application Support",
        Path("/tmp").resolve(), Path("/private/tmp").resolve(), cwd,
    }
    sensitive = {home, cwd, cfg.vault.resolve(), cfg.mount_point.resolve()}
    if target in protected or any(target in path.parents for path in sensitive):
        raise PipelineError(f"Refusing to reset protected or overly broad path: {target}")
    if target.exists() and not target.is_dir():
        raise PipelineError(f"State root is not a directory: {target}")
    if not target.exists():
        return False

    # Avoid retaining open log descriptors when main() is called repeatedly in
    # one Python process (for example, tests or an embedding application).
    app_logger = logging.getLogger(APP_NAME)
    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)
        handler.close()

    # The lock makes an active pipeline a hard refusal. The lock itself is
    # inside the directory and is intentionally removed with the state tree.
    with PipelineLock(target / ".pipeline.lock"):
        shutil.rmtree(target)
    return True


def cleanup_archives(cfg: Config, db: StateDB, days: int, dry_run: bool, logger: logging.Logger) -> int:
    if days <= 0:
        raise PipelineError("cleanup requires --older-than-days greater than zero")
    cutoff = time.time() - days * 86400
    removed = 0
    rows = list(db.conn.execute("SELECT * FROM recordings WHERE status='completed' AND local_path IS NOT NULL"))
    for row in rows:
        path = Path(row["local_path"])
        try:
            completed_timestamp = dt.datetime.fromisoformat(row["completed_at"]).timestamp() if row["completed_at"] else path.stat().st_mtime
        except (ValueError, OSError):
            completed_timestamp = path.stat().st_mtime if path.exists() else time.time()
        if path.parent != cfg.archive_dir or not path.exists() or completed_timestamp >= cutoff:
            continue
        if dry_run:
            logger.info("Would remove archived audio: %s", path)
        else:
            path.unlink()
            db.update(int(row["id"]), local_path=None)
            logger.info("Removed archived audio: %s", path)
        removed += 1
    return removed


def launchd_plist(cfg: Config, interval: int, script: Path, config_path: Path | None) -> str:
    args = [sys.executable, str(script)]
    if config_path:
        args += ["--config", str(config_path)]
    args += ["--quiet", "run"]
    xml_args = "\n".join(f"        <string>{xml_escape(value)}</string>" for value in args)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>local.auto-transcription</string>
    <key>ProgramArguments</key>
    <array>
{xml_args}
    </array>
    <key>StartOnMount</key><true/>
    <key>StartInterval</key><integer>{interval}</integer>
    <key>RunAtLoad</key><true/>
    <key>Nice</key><integer>20</integer>
    <key>ProcessType</key><string>Background</string>
    <key>LowPriorityIO</key><true/>
    <key>StandardOutPath</key><string>{xml_escape(str(cfg.log_path))}</string>
    <key>StandardErrorPath</key><string>{xml_escape(str(cfg.log_path))}</string>
</dict>
</plist>
"""


def xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def install_launchd(
    cfg: Config, args: argparse.Namespace, logger: logging.Logger, show_load_hint: bool = True,
) -> Path:
    output = _expand_path(args.output or "~/Library/LaunchAgents/local.auto-transcription.plist")
    content = launchd_plist(cfg, args.interval, Path(__file__).resolve(), Path(args.config).resolve() if args.config else None)
    atomic_write(output, content, mode=0o644)
    logger.info("Wrote launchd agent: %s", output)
    if show_load_hint:
        logger.info("Load it with: launchctl bootstrap gui/$(id -u) %s", output)
    return output


def activate_launchd(plist: Path, logger: logging.Logger) -> None:
    """Replace and verify the current user's LaunchAgent without prompting."""
    if platform.system() != "Darwin":
        raise PipelineError("--install is available only on macOS")
    launchctl = "/bin/launchctl"
    if not Path(launchctl).is_file():
        raise PipelineError("launchctl was not found")
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/local.auto-transcription"
    # An absent old service is expected on first installation.
    try:
        subprocess.run(
            [launchctl, "bootout", service], text=True, capture_output=True,
            timeout=30, check=False,
        )
        loaded = subprocess.run(
            [launchctl, "bootstrap", domain, str(plist)], text=True, capture_output=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PipelineError(f"Unable to install mount automation: {exc}") from exc
    if loaded.returncode != 0:
        detail = (loaded.stderr or loaded.stdout).strip()[-2000:]
        raise PipelineError(f"Unable to install mount automation: {detail or 'launchctl bootstrap failed'}")
    try:
        verified = subprocess.run(
            [launchctl, "print", service], text=True, capture_output=True,
            timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PipelineError(f"Unable to verify mount automation: {exc}") from exc
    if verified.returncode != 0:
        detail = (verified.stderr or verified.stdout).strip()[-2000:]
        raise PipelineError(f"Mount automation was loaded but could not be verified: {detail}")
    logger.info("Installed and activated mount automation: %s", service)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=APP_VERSION)
    parser.add_argument("--config", type=Path, help="TOML configuration file")
    parser.add_argument("--mount", help="Recorder mount point")
    parser.add_argument("--vault", help="Obsidian vault path")
    parser.add_argument("--state-root", help="Local pipeline state directory")
    parser.add_argument(
        "--reset-state", action="store_true",
        help="delete the configured application state directory and exit",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="install and activate automatic run-on-mount behavior, then exit",
    )
    parser.add_argument("--backend", choices=("auto", "mlx", "openai", "cli", "mock"))
    parser.add_argument("--model")
    parser.add_argument("--language", help="Known language code; use 'auto' to detect")
    parser.add_argument("--maximum-accuracy", action="store_true", help="Use large-v3 for all audio")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--keep-audio", dest="keep_audio", action="store_true", default=None)
    audio_group.add_argument("--no-keep-audio", dest="keep_audio", action="store_false")
    parser.add_argument("--discard-after-days", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("scan", help="List recordings not yet known to the pipeline")
    import_parser = sub.add_parser("import", help="Copy new recorder files into the local inbox")
    import_parser.add_argument("--dry-run", action="store_true")
    process_parser = sub.add_parser("process", help="Process the local queue")
    process_parser.add_argument("--include-failed", action="store_true")
    run_parser = sub.add_parser("run", help="Import new files, then process the local queue")
    run_parser.add_argument("--dry-run", action="store_true")
    status_parser = sub.add_parser("status", help="Show state counts and recent items")
    status_parser.add_argument("--limit", type=int, default=20)
    sub.add_parser("retry", help="Reset failed stages and resume without discarding completed work")
    reprocess = sub.add_parser("reprocess", help="Deliberately transcribe and summarize one recording again")
    reprocess.add_argument("--hash", required=True, dest="hash_prefix")
    cleanup = sub.add_parser("cleanup", help="Remove old local archived audio; notes and transcripts are retained")
    cleanup.add_argument("--older-than-days", type=int, required=True)
    cleanup.add_argument("--dry-run", action="store_true")
    sub.add_parser("check", help="Validate dependencies and configuration")
    sub.add_parser("prepare", help="Download and warm the configured local models")
    launchd = sub.add_parser("install-launchd", help="Write a run-on-mount LaunchAgent without loading it")
    launchd.add_argument("--output")
    launchd.add_argument("--interval", type=int, default=60)
    return parser


def print_status(db: StateDB, limit: int) -> None:
    counts = db.counts()
    print("State counts:")
    for status in sorted(counts):
        print(f"  {status:12} {counts[status]}")
    print("\nRecent recordings:")
    for row in db.recent(limit):
        error = f" — {row['last_error']}" if row["last_error"] else ""
        digest = (row["sha256"] or "--------")[:8]
        print(f"  {row['id']:4} {row['status']:12} {digest} {row['source_name']}{error}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    standalone_count = int(args.reset_state) + int(args.install)
    if standalone_count > 1:
        parser.error("--install and --reset-state cannot be combined")
    if standalone_count and args.command:
        parser.error("standalone options cannot be combined with a command")
    if not standalone_count and not args.command:
        parser.error("a command, --install, or --reset-state is required")
    cfg: Config | None = None
    try:
        cfg = apply_overrides(load_config(args.config), args)
        if args.reset_state:
            target = cfg.state_root.resolve()
            removed = reset_application_state(cfg)
            print(f"Deleted application state: {target}" if removed else f"Application state does not exist: {target}")
            return 0
        if cfg.language == "auto":
            cfg.language = None
        logger = configure_logging(cfg, args.verbose, args.quiet)
        try:
            actual_nice = apply_process_priority(cfg.nice_level)
            if actual_nice is not None:
                logger.debug("Process nice level: %d (child processes inherit this priority)", actual_nice)
        except OSError as exc:
            logger.warning("Unable to lower process priority to nice level %d: %s", cfg.nice_level, exc)
        ensure_private_directories(cfg)
        if args.install:
            install_args = argparse.Namespace(output=None, interval=60, config=args.config)
            plist = install_launchd(cfg, install_args, logger, show_load_hint=False)
            activate_launchd(plist, logger)
            print(f"Automatic recorder mount processing installed: {plist}")
            return 0
        with StateDB(cfg.db_path) as db:
            db.migrate_legacy_hashes(cfg.vault / ".processed_hashes", logger)
            command = args.command
            if command == "status":
                print_status(db, args.limit)
                return 0
            if command == "check":
                ok = check_dependencies(cfg, logger)
                if not ok:
                    notify_error("Dependency check failed; see the transcription log for details", cfg)
                return 0 if ok else 1
            if command == "prepare":
                prepare_dependencies(cfg, logger)
                return 0
            if command == "scan":
                candidates = scan_recorder(cfg, db)
                for candidate in candidates:
                    print(f"{candidate.stat.st_size:12}  {candidate.path}")
                print(f"{len(candidates)} new recording(s)")
                return 0
            if command == "install-launchd":
                install_launchd(cfg, args, logger)
                return 0
            with PipelineLock(cfg.lock_dir):
                recover_interrupted(db, cfg, logger)
                pipeline = Pipeline(cfg, db, logger)
                if command == "import":
                    result = pipeline.import_new(args.dry_run)
                    return 1 if result["failed"] else 0
                if command == "process":
                    with prevent_sleep(cfg.prevent_sleep):
                        result = pipeline.process_pending(args.include_failed)
                    if not cfg.keep_audio and cfg.discard_after_days > 0:
                        cleanup_archives(cfg, db, cfg.discard_after_days, False, logger)
                    notify_finished(result["completed"], result["failed"], cfg)
                    return 1 if result["failed"] else 0
                if command == "run":
                    imported = pipeline.import_new(args.dry_run, missing_ok=True)
                    if args.dry_run:
                        return 0
                    notify_new_recordings(imported["ready"], cfg)
                    with prevent_sleep(cfg.prevent_sleep):
                        processed = pipeline.process_pending()
                    if not cfg.keep_audio and cfg.discard_after_days > 0:
                        cleanup_archives(cfg, db, cfg.discard_after_days, False, logger)
                    total_failed = imported["failed"] + processed["failed"]
                    if total_failed:
                        notify_finished(processed["completed"], total_failed, cfg)
                        return 1
                    unmount_recorder(cfg, logger)
                    notify_finished(processed["completed"], 0, cfg)
                    return 0
                if command == "retry":
                    reset_failed(db, logger)
                    imported = pipeline.import_new(missing_ok=True)
                    notify_new_recordings(imported["ready"], cfg)
                    with prevent_sleep(cfg.prevent_sleep):
                        result = pipeline.process_pending()
                    if not cfg.keep_audio and cfg.discard_after_days > 0:
                        cleanup_archives(cfg, db, cfg.discard_after_days, False, logger)
                    total_failed = imported["failed"] + result["failed"]
                    notify_finished(result["completed"], total_failed, cfg)
                    return 1 if total_failed else 0
                if command == "reprocess":
                    record_id = reset_for_reprocess(db, args.hash_prefix, logger)
                    with prevent_sleep(cfg.prevent_sleep):
                        completed = pipeline.process_record(record_id) == "completed"
                    if completed and not cfg.keep_audio and cfg.discard_after_days > 0:
                        cleanup_archives(cfg, db, cfg.discard_after_days, False, logger)
                    notify_finished(1 if completed else 0, 0 if completed else 1, cfg)
                    return 0 if completed else 1
                if command == "cleanup":
                    cleanup_archives(cfg, db, args.older_than_days, args.dry_run, logger)
                    return 0
        parser.error(f"Unhandled command: {args.command}")
    except PipelineError as exc:
        logging.getLogger(APP_NAME).error("Error: %s", exc)
        notify_error(str(exc), cfg or Config())
        return 2
    except KeyboardInterrupt:
        logging.getLogger(APP_NAME).warning("Interrupted; completed stages were preserved")
        return 130
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_mlx-worker":
        raise SystemExit(mlx_worker_main(Path(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit(main())
