#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tarfile
from pathlib import PurePosixPath

from common import COHORTS, DATASET, EXTRACT_ROOT, OUTPUT, dump_json


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def main():
    archive_root = DATASET / "archives"
    rows = []
    unsafe = []
    total_apks = 0
    total_apk_bytes = 0
    for cohort, spec in COHORTS.items():
        path = archive_root / spec["archive"]
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: missing archive " + str(path))
        apk_count = 0
        apk_bytes = 0
        print(f"CIC_ARCHIVE_SCAN_START cohort={cohort} archive={path}", flush=True)
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.casefold().endswith(".apk"):
                    continue
                if not safe_member_name(member.name):
                    unsafe.append({"cohort": cohort, "archive": str(path), "member": member.name})
                    continue
                apk_count += 1
                apk_bytes += int(member.size)
                if apk_count % 1000 == 0:
                    print(f"CIC_ARCHIVE_SCAN_PROGRESS cohort={cohort} apks={apk_count}", flush=True)
        if apk_count == 0:
            raise RuntimeError("BLOCKED_BY_CONTRACT: no APK members in " + str(path))
        rows.append({"cohort": cohort, "archive": str(path.resolve()),
                     "archive_bytes": path.stat().st_size, "apk_members": apk_count,
                     "archive_mtime_ns": path.stat().st_mtime_ns,
                     "uncompressed_apk_bytes": apk_bytes, "binary_label": spec["label"]})
        total_apks += apk_count
        total_apk_bytes += apk_bytes
        print(f"CIC_ARCHIVE_INVENTORY cohort={cohort} apks={apk_count} bytes={apk_bytes}", flush=True)
    if unsafe:
        dump_json(OUTPUT / "unsafe_archive_members.json", unsafe)
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe APK archive members={len(unsafe)}")
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(EXTRACT_ROOT)
    reserve = 20 * (1 << 30)
    already = sum(path.stat().st_size for path in EXTRACT_ROOT.glob("*/*.apk") if path.is_file())
    required = max(total_apk_bytes - already, 0) + reserve
    audit = {"status": "PASS" if disk.free >= required else "INSUFFICIENT_DISK",
             "dataset": str(DATASET), "archive_collection": "four archives supplied by user",
             "missing_standard_cohorts_not_in_scope": ["SMS"], "cohorts": rows,
             "total_apk_members": total_apks, "total_uncompressed_apk_bytes": total_apk_bytes,
             "extract_root": str(EXTRACT_ROOT), "already_extracted_bytes": already,
             "free_bytes": disk.free, "reserve_bytes": reserve, "required_free_bytes": required,
             "labels_loaded": False}
    dump_json(OUTPUT / "archive_inventory.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(
            f"BLOCKED_BY_CONTRACT: insufficient extraction disk free={disk.free} required={required}; "
            "archive inventory was saved without extracting APKs"
        )
    print(f"CIC_ARCHIVE_INVENTORY_PASS total={total_apks} free={disk.free} required={required}", flush=True)


if __name__ == "__main__":
    main()
