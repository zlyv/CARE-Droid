#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(os.environ.get("ICDE_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("ICDE_OUTPUT", str(PROJECT / "outputs/care-icde-source-v1"))).resolve()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, purpose: str, hash_file: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "absolute_path": str(resolved),
        "purpose": purpose,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(resolved) if hash_file else None,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def run_command(argv: list[str], timeout: int = 20, cwd: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {"argv": argv, "timeout_seconds": timeout}
    try:
        completed = subprocess.run(
            argv, cwd=str(cwd or PROJECT), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            errors="replace", timeout=timeout, check=False,
        )
        record.update({
            "status": "COMPLETED", "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:], "stderr": completed.stderr[-20000:],
        })
    except subprocess.TimeoutExpired as exc:
        record.update({
            "status": "TIMEOUT", "returncode": None,
            "stdout": str(exc.stdout or "")[-20000:], "stderr": str(exc.stderr or "")[-20000:],
        })
    except Exception as exc:
        record.update({"status": "ERROR", "returncode": None, "error": repr(exc)})
    record["wall_seconds"] = round(time.monotonic() - started, 6)
    return record


def norm_sha(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if len(text) == 64 and all(ch in "0123456789abcdef" for ch in text) else ""


def bool_text(value: bool) -> str:
    return "PASS" if value else "FAIL"

