#!/usr/bin/env python3
"""Linux-side receiver for verified AndroZoo APK batches from Windows.

This script runs over an SSH stdin pipe, validates every TAR member against the
frozen 15,000-row manifest, SHA256-checks the received APK, and commits it to
its final year/label directory atomically without overwriting existing files.
It never executes or extracts APK contents. Designed for Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

SHA = re.compile(r"^[0-9a-f]{64}$")
MEMBER = re.compile(r"^year(2017|2018|2019|2020|2021|2022)/(benign|malware)/([0-9a-f]{64})\.apk$")
CHUNK = 1024 * 1024


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def manifest_index(path: Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"year", "label", "sha256", "apk_size"}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Manifest missing fields: {needed - set(reader.fieldnames or [])}")
        for row in reader:
            sha = row["sha256"].strip().lower()
            year = row["year"].strip()
            label = row["label"].strip()
            if not SHA.fullmatch(sha) or year not in {str(x) for x in range(2017, 2023)} or label not in {"0", "1"}:
                raise ValueError("Invalid manifest identity")
            expected_path = f"year{year}/{'benign' if label == '0' else 'malware'}/{sha}.apk"
            size = int(row.get("apk_size") or 0)
            if size < 0 or sha in result:
                raise ValueError("Invalid size or duplicate SHA in manifest")
            result[sha] = (expected_path, size)
    if len(result) != 15000:
        raise ValueError(f"Frozen manifest expected 15000 entries, found {len(result)}")
    return result


def import_archive(root: Path, manifest: dict[str, tuple[str, int]], count: int, min_free_gb: float) -> dict:
    if not 1 <= count <= 100:
        raise ValueError("Batch size must be 1..100")
    success: list[str] = []
    skipped: list[str] = []
    seen: set[str] = set()
    receiver_root = root / "apks"
    receiver_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
        for member in archive:
            if not member.isfile() or not MEMBER.fullmatch(member.name):
                raise ValueError("Rejected non-file or unsafe TAR member")
            match = MEMBER.fullmatch(member.name)
            assert match is not None
            sha = match.group(3)
            if sha in seen:
                raise ValueError(f"Duplicate SHA within incoming batch: {sha}")
            seen.add(sha)
            if sha not in manifest:
                raise ValueError(f"SHA not in frozen manifest: {sha}")
            rel, expected_size = manifest[sha]
            if rel != member.name:
                raise ValueError(f"Member path mismatches frozen manifest: {member.name}")
            if member.size < 1 or member.size > 4 * 1024**3:
                raise ValueError("Unexpected APK size")
            if expected_size > 0 and expected_size != member.size:
                raise ValueError(f"APK size mismatches frozen manifest: {sha}")
            final = receiver_root / rel
            final.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Archive member data missing")

            # Even for an existing valid file, consume the TAR stream and check
            # the incoming data hash. This ensures the ACK describes verified bytes.
            existing_good = final.exists() and hash_file(final) == sha
            if final.exists() and not existing_good:
                raise ValueError(f"Existing target APK does not match SHA; refusing overwrite: {sha}")
            if not existing_good and shutil.disk_usage(root).free < min_free_gb * 1024**3:
                raise OSError(f"Free space below {min_free_gb:g} GiB")

            part = final.with_name(f".{sha}.{os.getpid()}.incoming.part")
            h = hashlib.sha256()
            consumed = 0
            try:
                if existing_good:
                    while consumed < member.size:
                        block = source.read(min(CHUNK, member.size - consumed))
                        if not block:
                            raise IOError("Truncated TAR member")
                        h.update(block)
                        consumed += len(block)
                else:
                    fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(fd, "wb") as f:
                        while consumed < member.size:
                            block = source.read(min(CHUNK, member.size - consumed))
                            if not block:
                                raise IOError("Truncated TAR member")
                            f.write(block)
                            h.update(block)
                            consumed += len(block)
                        f.flush()
                        os.fsync(f.fileno())
                if consumed != member.size or h.hexdigest() != sha:
                    raise ValueError(f"Incoming SHA256 mismatch: {sha}")
                if existing_good:
                    skipped.append(sha)
                else:
                    # Hard-link is an atomic no-clobber commit on the same FS.
                    try:
                        os.link(part, final)
                    except FileExistsError:
                        if hash_file(final) != sha:
                            raise ValueError(f"Target created concurrently with invalid contents: {sha}")
                        skipped.append(sha)
                    else:
                        success.append(sha)
            finally:
                source.close()
                if part.exists():
                    part.unlink()

    if len(seen) != count:
        raise ValueError(f"Truncated/incomplete batch: expected {count}, observed {len(seen)}")
    summary = {"count": count, "received": success, "skipped": skipped, "all_verified": sorted(seen)}
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--expect-count", type=int, required=True)
    p.add_argument("--min-free-gb", type=float, default=25)
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir() or args.min_free_gb < 0:
        raise SystemExit("Invalid destination dataset root")
    manifest = manifest_index(root / "manifests" / "selected_all.csv")
    result = import_archive(root, manifest, args.expect_count, args.min_free_gb)
    logdir = root / "logs"
    logdir.mkdir(exist_ok=True)
    record = {"utc": datetime.now(timezone.utc).isoformat(), **result}
    with (logdir / "windows_batch_receipts.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, separators=(",", ":")) + "\n")
        log.flush()
        os.fsync(log.fileno())
    print("AZ_BATCH_ACK " + json.dumps(result, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"AZ_BATCH_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
