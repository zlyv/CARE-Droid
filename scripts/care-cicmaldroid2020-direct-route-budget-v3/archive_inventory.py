#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tarfile
import random
from pathlib import PurePosixPath

from common import (COHORTS, DATASET, EXTRACT_ROOT, OUTPUT, SAMPLE_PER_COHORT,
                    SAMPLE_SEED, dump_json)


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def main():
    archive_root = DATASET / "archives"
    rows = []
    unsafe = []
    non_apk_examples = []
    total_apks = 0
    total_apk_bytes = 0
    total_selected_bytes = 0
    selections = {}
    for cohort_index, (cohort, spec) in enumerate(COHORTS.items()):
        path = archive_root / spec["archive"]
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: missing archive " + str(path))
        apk_count = 0
        apk_bytes = 0
        regular_files = 0
        reservoir = []
        rng = random.Random(SAMPLE_SEED + cohort_index)
        print(f"CIC_ARCHIVE_SCAN_START cohort={cohort} archive={path}", flush=True)
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                regular_files += 1
                if not safe_member_name(member.name):
                    unsafe.append({"cohort": cohort, "archive": str(path), "member": member.name})
                    continue
                stream = archive.extractfile(member)
                magic = stream.read(4) if stream is not None else b""
                if stream is not None:
                    stream.close()
                if magic != b"PK\x03\x04":
                    if len(non_apk_examples) < 20:
                        non_apk_examples.append({"cohort": cohort, "member": member.name,
                                                 "member_size": int(member.size),
                                                 "magic_hex": magic.hex()})
                    continue
                apk_count += 1
                apk_bytes += int(member.size)
                candidate = {"member": member.name, "member_size": int(member.size)}
                if len(reservoir) < SAMPLE_PER_COHORT:
                    reservoir.append(candidate)
                else:
                    replace = rng.randrange(apk_count)
                    if replace < SAMPLE_PER_COHORT:
                        reservoir[replace] = candidate
                if apk_count % 1000 == 0:
                    print(f"CIC_ARCHIVE_SCAN_PROGRESS cohort={cohort} apks={apk_count}", flush=True)
        if apk_count == 0:
            raise RuntimeError("BLOCKED_BY_CONTRACT: no APK members in " + str(path))
        reservoir.sort(key=lambda row: (row["member"], row["member_size"]))
        selected_bytes = sum(row["member_size"] for row in reservoir)
        selections[cohort] = {"archive": str(path.resolve()), "seed": SAMPLE_SEED + cohort_index,
                              "population_apks": apk_count, "selected": reservoir}
        rows.append({"cohort": cohort, "archive": str(path.resolve()),
                     "archive_bytes": path.stat().st_size, "apk_members": apk_count,
                     "regular_file_members": regular_files,
                     "archive_mtime_ns": path.stat().st_mtime_ns,
                     "uncompressed_apk_bytes": apk_bytes, "selected_apks": len(reservoir),
                     "selected_uncompressed_bytes": selected_bytes,
                     "sampling_seed": SAMPLE_SEED + cohort_index,
                     "binary_label": spec["label"]})
        total_apks += apk_count
        total_apk_bytes += apk_bytes
        total_selected_bytes += selected_bytes
        print(f"CIC_ARCHIVE_INVENTORY cohort={cohort} apks={apk_count} selected={len(reservoir)} bytes={apk_bytes}", flush=True)
    if unsafe:
        dump_json(OUTPUT / "unsafe_archive_members.json", unsafe)
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unsafe APK archive members={len(unsafe)}")
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(EXTRACT_ROOT)
    reserve = 20 * (1 << 30)
    already = sum(path.stat().st_size for path in EXTRACT_ROOT.glob("*/*.apk") if path.is_file())
    required = max(total_selected_bytes - already, 0) + reserve
    audit = {"status": "PASS" if disk.free >= required else "INSUFFICIENT_DISK",
             "dataset": str(DATASET), "archive_collection": "three usable archives; incomplete Adware skipped",
             "missing_standard_cohorts_not_in_scope": ["SMS", "Adware (incomplete archive explicitly skipped)"],
             "apk_identification": "regular tar member with ZIP local-file-header magic PK0304",
             "non_apk_regular_examples": non_apk_examples,
             "total_apk_members": total_apks, "total_uncompressed_apk_bytes": total_apk_bytes,
             "sampling": "deterministic uniform reservoir sample within each supplied cohort",
             "sample_per_cohort_cap": SAMPLE_PER_COHORT, "sample_seed": SAMPLE_SEED,
             "total_selected_apks": sum(len(value["selected"]) for value in selections.values()),
             "total_selected_uncompressed_bytes": total_selected_bytes,
             "extract_root": str(EXTRACT_ROOT), "already_extracted_bytes": already,
             "free_bytes": disk.free, "reserve_bytes": reserve, "required_free_bytes": required,
             "labels_loaded": False}
    dump_json(OUTPUT / "archive_inventory.json", audit)
    dump_json(OUTPUT / "sampled_archive_members.json", selections)
    if audit["status"] != "PASS":
        raise RuntimeError(
            f"BLOCKED_BY_CONTRACT: insufficient extraction disk free={disk.free} required={required}; "
            "archive inventory was saved without extracting APKs"
        )
    print(f"CIC_ARCHIVE_INVENTORY_PASS total={total_apks} free={disk.free} required={required}", flush=True)


if __name__ == "__main__":
    main()
