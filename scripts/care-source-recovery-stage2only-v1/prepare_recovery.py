#!/usr/bin/env python3
"""Restore small immutable source assets and build the exact 799-APK lineage."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


SHA_KEYS = ("sha256", "sample_id", "sha", "apk_sha256")
PATH_KEYS = ("absolute_path", "apk_path", "path", "file_path")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def first(row: dict, keys) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def norm(value) -> str:
    return str(value).strip().lower()


def remap_path(value: str, storage_root: Path) -> Path:
    text = str(value).strip()
    prefixes = (
        "/home/boyuanjie/datasets/",
        "/home/boyuanjie/projects/tasa-droid/datasets/",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return storage_root / "datasets" / text[len(prefix):]
    return Path(text)


def candidate_rows(dataset: Path):
    preferred = (
        "temporal_eval_manifest.csv", "extraction_manifest_unique.csv",
        "lineage_2017_2021.csv", "lineage.csv",
    )
    seen = set()
    for name in preferred:
        for path in dataset.rglob(name):
            if path not in seen:
                seen.add(path)
                yield path
    for path in dataset.rglob("*.csv"):
        if path not in seen:
            yield path


def build_path_table(dataset: Path, needed: set[str], storage_root: Path) -> dict[str, dict]:
    result = {}
    for path in candidate_rows(dataset):
        try:
            rows = read_csv(path)
        except Exception:
            continue
        if not rows or not any(key in rows[0] for key in SHA_KEYS):
            continue
        for row in rows:
            sha = norm(first(row, SHA_KEYS))
            if sha not in needed or sha in result:
                continue
            apk = first(row, PATH_KEYS)
            if apk:
                resolved = remap_path(apk, storage_root)
                if not resolved.is_absolute():
                    dataset_relative = dataset / resolved
                    storage_relative = storage_root / "datasets" / resolved
                    resolved = (
                        dataset_relative if dataset_relative.exists() else storage_relative
                    )
                result[sha] = {"row": row, "path": resolved}
        if len(result) == len(needed):
            break
    if len(result) < len(needed):
        # The stored cohort uses SHA filenames; one bounded scan is a robust
        # fallback when old manifests contain stale absolute prefixes.
        for apk in (storage_root / "datasets").rglob("*.apk"):
            sha = apk.stem.lower()
            if sha in needed and sha not in result:
                result[sha] = {"row": {}, "path": apk}
            if len(result) == len(needed):
                break
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.stat().st_size != destination.stat().st_size:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/home/boyuanjie/projects/tasa-droid"))
    parser.add_argument("--storage-root", type=Path, default=Path(
        "/mnt/HDD1/boyuanjie/tasa-droid-storage"))
    parser.add_argument("--dataset", type=Path, default=Path(
        "/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021"))
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    project = args.project.resolve()
    artifacts = project / "artifacts"
    reference_path = package / "assets" / "source_reference_799.csv"
    reference = read_csv(reference_path)
    if len(reference) != 799 or len({norm(row["sha256"]) for row in reference}) != 799:
        raise RuntimeError("bundled source reference is not the exact 799-APK cohort")
    labels = [int(row["label"]) for row in reference]
    if labels.count(0) != 400 or labels.count(1) != 399:
        raise RuntimeError("unexpected fixed source label counts")

    # Restore only small immutable files into the project tree.
    for relative in (
        "global24-a25-route-conditioned-joints7-v1-full/source_predictions.csv",
        "care-bounded-fixkeep-v3-full/source_predictions.csv",
    ):
        copy_asset(reference_path, artifacts / relative)
    copy_asset(
        package / "assets" / "multiscale_features.npz",
        artifacts / "clipped-intra-inter-structure-ablation-2017-v1-full" / "multiscale_features.npz",
    )
    p72_rows = [{
        "sha256": norm(row["sha256"]),
        "label": int(row["label"]),
        "metadata_label": "benign" if int(row["label"]) == 0 else "malware",
        "fold": int(row["fold"]),
    } for row in reference]
    write_csv(
        artifacts / "p72-verified-dependency-source-ablation" / "p72_oof_predictions.csv",
        p72_rows,
    )

    needed = {norm(row["sha256"]) for row in reference}
    table = build_path_table(args.dataset, needed, args.storage_root.resolve())
    missing = sorted(needed - set(table))
    if missing:
        raise RuntimeError("could not resolve source APK paths: " + repr(missing[:10]))
    lineage = []
    missing_files = []
    for row in reference:
        sha = norm(row["sha256"])
        found = table[sha]
        apk = found["path"].resolve()
        if not apk.is_file():
            missing_files.append(str(apk))
        source = found["row"]
        lineage.append({
            "sha256": sha,
            "absolute_path": str(apk),
            "source_id": first(source, ("source_id",)) or "v12_selective_role_v3",
            "lineage_id": first(source, ("lineage_id",)) or sha,
            "metadata_year": "2017",
            "metadata_label": "benign" if int(row["label"]) == 0 else "malware",
        })
    if missing_files:
        raise RuntimeError("resolved APK paths do not exist: " + repr(missing_files[:10]))
    lineage_path = artifacts / "care-source-recovery-stage2only-v1" / "source_lineage_2017.csv"
    write_csv(lineage_path, lineage)
    audit = {
        "source_n": 799, "benign_n": 400, "malware_n": 399,
        "years_loaded": [2017], "future_rows_loaded": False,
        "dataset_root": str(args.dataset.resolve()),
        "lineage": str(lineage_path.resolve()),
        "large_cache_root": str((args.storage_root / "care-recovery").resolve()),
    }
    (lineage_path.parent / "recovery_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared exact 799-APK 2017 source lineage:", lineage_path)


if __name__ == "__main__":
    main()
