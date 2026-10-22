#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from common import COHORTS, DATASET, EXTRACT_ROOT, OUTPUT, dump_json


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def load_done(path: Path) -> dict[tuple[str, int], dict]:
    done = {}
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            target = Path(row.get("absolute_path", ""))
            key = (str(row.get("member", "")), int(row.get("member_size", -1)))
            if target.is_file() and target.stat().st_size == int(row.get("member_size", -2)):
                done[key] = row
    return done


def extract_one(source, target_dir: Path, expected_size: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    fd, temporary_name = tempfile.mkstemp(prefix=".cic-apk-", suffix=".tmp", dir=target_dir)
    written = 0
    try:
        with os.fdopen(fd, "wb") as output:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block); output.write(block); written += len(block)
            output.flush(); os.fsync(output.fileno())
        if written != expected_size:
            raise RuntimeError(f"archive member size changed expected={expected_size} actual={written}")
        sid = digest.hexdigest(); target = target_dir / (sid + ".apk")
        if target.exists():
            if target.stat().st_size != written:
                raise RuntimeError("SHA target exists with different size: " + str(target))
            os.unlink(temporary_name)
        else:
            os.replace(temporary_name, target)
        return sid, target
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cohort, spec in COHORTS.items():
        archive_path = DATASET / "archives" / spec["archive"]
        checkpoint = OUTPUT / "checkpoints" / f"extracted_{cohort}.jsonl"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        done = load_done(checkpoint)
        completed = 0
        with tarfile.open(archive_path, "r:gz") as archive, checkpoint.open("a", encoding="utf-8") as log:
            for member in archive:
                if not member.isfile() or not member.name.casefold().endswith(".apk"):
                    continue
                if not safe_member_name(member.name):
                    raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe member " + member.name)
                key = (member.name, int(member.size))
                row = done.get(key)
                if row is None:
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError("cannot read archive member " + member.name)
                    with source:
                        sid, target = extract_one(source, EXTRACT_ROOT / cohort, int(member.size))
                    row = {"sha256": sid, "cohort": cohort, "label": int(spec["label"]),
                           "archive": str(archive_path.resolve()), "member": member.name,
                           "member_size": int(member.size), "absolute_path": str(target.resolve())}
                    log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"); log.flush()
                    done[key] = row
                all_rows.append(row); completed += 1
                if completed % 100 == 0:
                    print(f"CIC_EXTRACT cohort={cohort} completed={completed}", flush=True)
        print(f"CIC_EXTRACT_COHORT_DONE cohort={cohort} completed={completed}", flush=True)
    dump_json(OUTPUT / "extraction_summary.json", {
        "status": "PASS", "archive_member_rows": len(all_rows),
        "unique_sha256": len({row["sha256"] for row in all_rows}),
        "extract_root": str(EXTRACT_ROOT), "archives_modified": False})
    print(f"CIC_EXTRACTION_DONE rows={len(all_rows)} unique={len({r['sha256'] for r in all_rows})}", flush=True)


if __name__ == "__main__":
    main()
