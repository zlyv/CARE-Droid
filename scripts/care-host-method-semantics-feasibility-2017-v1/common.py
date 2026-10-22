#!/usr/bin/env python3
"""Shared I/O and identity helpers for the CARE Host method-semantics feasibility audit.

Inputs are frozen 2017 CARE assets. Outputs are written only beneath the new audit
artifact directory. This module never mutates APKs, checkpoints, prototype banks, or
existing CARE feature caches.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

PROJECT = Path(os.environ.get("CHMS_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ.get("CHMS_OUTPUT", str(PROJECT / "artifacts/care_host_method_semantics_feasibility_2017_v1"))).resolve()
SCHEMA = "care-host-method-semantics-feasibility-2017-v1.0"
ROLE_MAP = [7, 10, 21, 19, 9, 15, 16, 0, 11, 25, 29, 6, 12, 1, 26, 28, 23, 13, 14, 2, 17, 30, 22, 4, 5, 24, 3, 18, 8, 20, 27]


def canonical(value):
    return re.sub(r"\s+", "", str(value or "")).casefold()


def stable_id(value):
    return int.from_bytes(hashlib.sha256(canonical(value).encode()).digest()[:8], "big")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe(path):
    path = Path(path).resolve()
    expected = (PROJECT / "artifacts/care_host_method_semantics_feasibility_2017_v1").resolve()
    if OUTPUT != expected:
        raise RuntimeError(f"unsafe output root: {OUTPUT}")
    if path != OUTPUT and OUTPUT not in path.parents:
        raise RuntimeError(f"write outside audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, obj):
    path = safe(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_json_gz(path, obj):
    path = safe(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields=None, compress=False):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    path = safe(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if compress else open
    with opener(tmp, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def file_record(path, purpose):
    path = Path(path).resolve()
    stat = path.stat()
    return {"absolute_path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path), "purpose": purpose}

