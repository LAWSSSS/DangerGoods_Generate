"""Atomic JSON records, SHA-256 provenance and append-only stage events."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fusion_engine.config import reject_secrets
from fusion_engine.errors import AcceptanceError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    reject_secrets(data, "record")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_event(run_dir: Path, stage: str, status: str, **details: Any) -> None:
    data = {"time_utc": utc_now(), "stage": stage, "status": status, **details}
    reject_secrets(data, "event")
    path = run_dir / "logs/events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def new_run_directory(parent: Path, task_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent / task_id / f"{stamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for folder in ("analysis", "masks", "prompts", "candidates", "outputs", "logs", "annotations"):
        (run_dir / folder).mkdir()
    return run_dir


@contextmanager
def review_lock(run_dir: Path):
    path = run_dir / ".review.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AcceptanceError("该运行正在验收；请等待当前操作结束") from exc
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
