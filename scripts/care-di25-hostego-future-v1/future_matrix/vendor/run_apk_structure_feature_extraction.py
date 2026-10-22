#!/usr/bin/env python3
"""Append compact APK archive/DEX structure features without static graphs.

The extractor is label-agnostic and never persists archive entries, DEX bytes,
or call graphs.  It reads one APK at a time and writes only bounded numeric
summaries suitable for training-fold-local filtering and standardization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


STRUCTURE_FIELDS = [
    "structure__apk_size_log",
    "structure__zip_entry_count_log",
    "structure__zip_uncompressed_size_log",
    "structure__zip_compression_ratio",
    "structure__zip_stored_entry_ratio",
    "structure__zip_encrypted_entry_ratio",
    "structure__zip_max_entry_size_ratio",
    "structure__zip_duplicate_name_count_log",
    "structure__dex_count",
    "structure__multidex_any",
    "structure__dex_uncompressed_size_log",
    "structure__dex_apk_size_ratio",
    "structure__primary_dex_fraction",
    "structure__dex_compression_ratio",
    "structure__dex_stored_ratio",
    "structure__native_library_count_log",
    "structure__native_library_size_ratio",
    "structure__native_abi_count",
    "structure__native_stored_ratio",
    "structure__asset_count_log",
    "structure__asset_size_ratio",
    "structure__asset_stored_ratio",
    "structure__embedded_payload_count_log",
    "structure__embedded_payload_size_ratio",
    "structure__resources_arsc_size_ratio",
    "structure__manifest_size_ratio",
    "structure__dex_dynamic_loader_marker_ratio",
    "structure__dex_reflection_marker_ratio",
    "structure__dex_native_load_marker_ratio",
    "structure__dex_command_exec_marker_ratio",
    "structure__dex_crypto_marker_ratio",
    "structure__dex_antidebug_marker_ratio",
]

MARKERS = {
    "dynamic_loader": (
        b"DexClassLoader", b"PathClassLoader", b"InMemoryDexClassLoader",
        b"BaseDexClassLoader",
    ),
    "reflection": (
        b"java/lang/reflect", b"Ljava/lang/Class;->forName",
        b"Ljava/lang/reflect/Method;->invoke",
    ),
    "native_load": (
        b"loadLibrary", b"Ljava/lang/System;->load",
        b"Ljava/lang/Runtime;->load",
    ),
    "command_exec": (
        b"Ljava/lang/Runtime;->exec", b"Ljava/lang/ProcessBuilder;",
        b"/system/bin/sh", b"/system/bin/su",
    ),
    "crypto": (
        b"Ljavax/crypto/Cipher;", b"Ljavax/crypto/SecretKey",
        b"Landroid/security/keystore",
    ),
    "antidebug": (
        b"isDebuggerConnected", b"TracerPid", b"/proc/self/status",
        b"ptrace",
    ),
}

PAYLOAD_SUFFIXES = {".dex", ".jar", ".apk", ".so"}
ABI_NAMES = {
    "armeabi", "armeabi-v7a", "arm64-v8a", "x86", "x86_64", "mips", "mips64"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def marker_hits(handle) -> dict[str, int]:
    hits = {name: set() for name in MARKERS}
    max_marker = max(len(marker) for values in MARKERS.values() for marker in values)
    tail = b""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        data = tail + chunk
        for family, markers in MARKERS.items():
            for index, marker in enumerate(markers):
                if index not in hits[family] and marker in data:
                    hits[family].add(index)
        tail = data[-max_marker:]
    return {family: len(values) for family, values in hits.items()}


def extract_one(apk_path: Path) -> dict[str, float]:
    apk_size = max(apk_path.stat().st_size, 1)
    with zipfile.ZipFile(apk_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        entry_count = len(infos)
        total_file = sum(info.file_size for info in infos)
        total_compressed = sum(info.compress_size for info in infos)
        stored_count = sum(info.compress_type == zipfile.ZIP_STORED for info in infos)
        encrypted_count = sum(bool(info.flag_bits & 0x1) for info in infos)
        duplicate_count = sum(value - 1 for value in Counter(
            info.filename for info in infos
        ).values() if value > 1)

        dex_infos = [
            info for info in infos
            if PurePosixPath(info.filename).name.casefold().startswith("classes")
            and PurePosixPath(info.filename).suffix.casefold() == ".dex"
        ]
        dex_file = sum(info.file_size for info in dex_infos)
        dex_compressed = sum(info.compress_size for info in dex_infos)
        dex_stored = sum(info.file_size for info in dex_infos if info.compress_type == zipfile.ZIP_STORED)
        primary = next(
            (info.file_size for info in dex_infos if info.filename.casefold() == "classes.dex"),
            0,
        )

        native_infos = [
            info for info in infos
            if info.filename.casefold().startswith("lib/")
            and PurePosixPath(info.filename).suffix.casefold() == ".so"
        ]
        native_file = sum(info.file_size for info in native_infos)
        native_stored = sum(
            info.file_size for info in native_infos
            if info.compress_type == zipfile.ZIP_STORED
        )
        abi_names = {
            PurePosixPath(info.filename).parts[1]
            for info in native_infos
            if len(PurePosixPath(info.filename).parts) > 2
            and PurePosixPath(info.filename).parts[1] in ABI_NAMES
        }

        asset_infos = [
            info for info in infos if info.filename.casefold().startswith("assets/")
        ]
        asset_file = sum(info.file_size for info in asset_infos)
        asset_stored = sum(
            info.file_size for info in asset_infos
            if info.compress_type == zipfile.ZIP_STORED
        )

        embedded_infos = []
        for info in infos:
            path = PurePosixPath(info.filename)
            suffix = path.suffix.casefold()
            conventional_dex = path.name.casefold().startswith("classes") and suffix == ".dex"
            conventional_native = info in native_infos
            if suffix in PAYLOAD_SUFFIXES and not conventional_dex and not conventional_native:
                embedded_infos.append(info)
        embedded_file = sum(info.file_size for info in embedded_infos)

        resources_size = sum(
            info.file_size for info in infos
            if info.filename.casefold() == "resources.arsc"
        )
        manifest_size = sum(
            info.file_size for info in infos
            if info.filename.casefold() == "androidmanifest.xml"
        )

        marker_count = {name: 0 for name in MARKERS}
        for info in dex_infos:
            with archive.open(info) as handle:
                current = marker_hits(handle)
            for name, value in current.items():
                marker_count[name] += value

    marker_denominators = {
        name: max(len(markers) * max(len(dex_infos), 1), 1)
        for name, markers in MARKERS.items()
    }
    return {
        "structure__apk_size_log": math.log1p(apk_size),
        "structure__zip_entry_count_log": math.log1p(entry_count),
        "structure__zip_uncompressed_size_log": math.log1p(total_file),
        "structure__zip_compression_ratio": safe_ratio(total_compressed, total_file),
        "structure__zip_stored_entry_ratio": safe_ratio(stored_count, entry_count),
        "structure__zip_encrypted_entry_ratio": safe_ratio(encrypted_count, entry_count),
        "structure__zip_max_entry_size_ratio": safe_ratio(
            max((info.file_size for info in infos), default=0), total_file
        ),
        "structure__zip_duplicate_name_count_log": math.log1p(duplicate_count),
        "structure__dex_count": float(len(dex_infos)),
        "structure__multidex_any": float(len(dex_infos) > 1),
        "structure__dex_uncompressed_size_log": math.log1p(dex_file),
        "structure__dex_apk_size_ratio": safe_ratio(dex_file, apk_size),
        "structure__primary_dex_fraction": safe_ratio(primary, dex_file),
        "structure__dex_compression_ratio": safe_ratio(dex_compressed, dex_file),
        "structure__dex_stored_ratio": safe_ratio(dex_stored, dex_file),
        "structure__native_library_count_log": math.log1p(len(native_infos)),
        "structure__native_library_size_ratio": safe_ratio(native_file, apk_size),
        "structure__native_abi_count": float(len(abi_names)),
        "structure__native_stored_ratio": safe_ratio(native_stored, native_file),
        "structure__asset_count_log": math.log1p(len(asset_infos)),
        "structure__asset_size_ratio": safe_ratio(asset_file, apk_size),
        "structure__asset_stored_ratio": safe_ratio(asset_stored, asset_file),
        "structure__embedded_payload_count_log": math.log1p(len(embedded_infos)),
        "structure__embedded_payload_size_ratio": safe_ratio(embedded_file, apk_size),
        "structure__resources_arsc_size_ratio": safe_ratio(resources_size, apk_size),
        "structure__manifest_size_ratio": safe_ratio(manifest_size, apk_size),
        **{
            f"structure__dex_{name}_marker_ratio": safe_ratio(
                marker_count[name], marker_denominators[name]
            )
            for name in MARKERS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    features_path = args.features.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with features_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("features CSV is empty")
    original_fields = list(rows[0])
    if "sha256" not in original_fields or "absolute_path" not in original_fields:
        raise SystemExit("features CSV requires sha256 and absolute_path")
    collisions = set(original_fields) & set(STRUCTURE_FIELDS)
    if collisions:
        raise SystemExit(f"structure feature columns already exist: {sorted(collisions)}")

    failures = []
    completed = []
    progress_every = max(1, args.progress_every)
    for index, row in enumerate(rows, 1):
        try:
            values = extract_one(Path(row["absolute_path"]))
        except Exception as exc:
            values = {name: 0.0 for name in STRUCTURE_FIELDS}
            failures.append({
                "sha256": row.get("sha256", ""),
                "absolute_path": row.get("absolute_path", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
        row.update({name: str(values[name]) for name in STRUCTURE_FIELDS})
        completed.append(row)
        if index % progress_every == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)} failures={len(failures)}", flush=True)

    augmented_path = output / "features_with_structure.csv"
    fields = original_fields + STRUCTURE_FIELDS
    with augmented_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(completed)
    failure_path = output / "structure_feature_failures.csv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "absolute_path", "error_type", "error_message"],
        )
        writer.writeheader()
        writer.writerows(failures)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["feature_fields"] = list(schema.get("feature_fields", original_fields)) + STRUCTURE_FIELDS
    schema["structure_feature_fields"] = STRUCTURE_FIELDS
    schema["structure_feature_extraction"] = {
        "label_agnostic": True,
        "raw_dex_persisted": False,
        "raw_archive_entries_persisted": False,
        "call_graph_built": False,
    }
    augmented_schema = output / "feature_schema_with_structure.json"
    augmented_schema.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P21_APK_ARCHIVE_DEX_STRUCTURE_FEATURE_EXTRACTION",
        "formal_training_enabled": False,
        "input_row_count": len(rows),
        "output_row_count": len(completed),
        "structure_feature_count": len(STRUCTURE_FIELDS),
        "failure_count": len(failures),
        "input_features_sha256": sha256_file(features_path),
        "outputs": {
            "features": str(augmented_path),
            "schema": str(augmented_schema),
            "failures": str(failure_path),
        },
        "notes": [
            "Features are extracted without labels.",
            "DEX bytes and archive entries are read transiently and never persisted.",
            "No call graph is built or stored.",
            "Failed APKs receive explicit zero structure features and are audited.",
        ],
    }
    audit_path = output / "structure_feature_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
