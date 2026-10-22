#!/usr/bin/env python3
"""APK identity audit using an existing Androguard installation.

The script is intentionally sequential and does not build or persist call
graphs. It writes only compact CSV/JSON audit outputs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def silence_logs() -> None:
    logging.disable(logging.INFO)
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


silence_logs()

try:
    from androguard.core.apk import APK
except Exception as exc:  # pragma: no cover - depends on server environment
    raise SystemExit(
        "无法导入 Androguard。请使用已安装它的 Python，例如 "
        "/home/boyuanjie/envs/tif/bin/python。\n"
        f"{type(exc).__name__}: {exc}"
    ) from exc


def value_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def optional_bool(apk, method_name: str) -> str:
    method = getattr(apk, method_name, None)
    if not callable(method):
        return "UNKNOWN"
    try:
        return "true" if bool(method()) else "false"
    except Exception:
        return "ERROR"


def audit_one(path: Path) -> dict[str, str]:
    result = {
        "package_name": "",
        "version_code": "",
        "version_name": "",
        "min_sdk": "",
        "target_sdk": "",
        "certificate_sha256": "",
        "signing_v1": "UNKNOWN",
        "signing_v2": "UNKNOWN",
        "signing_v3": "UNKNOWN",
        "signing_v4": "UNKNOWN",
        "manifest_status": "ERROR",
        "certificate_status": "NOT_ATTEMPTED",
        "error_type": "",
        "error_message": "",
    }
    apk = None
    try:
        apk = APK(str(path))
        result.update(
            {
                "package_name": value_text(apk.get_package()),
                "version_code": value_text(apk.get_androidversion_code()),
                "version_name": value_text(apk.get_androidversion_name()),
                "min_sdk": value_text(apk.get_min_sdk_version()),
                "target_sdk": value_text(apk.get_target_sdk_version()),
                "signing_v1": optional_bool(apk, "is_signed_v1"),
                "signing_v2": optional_bool(apk, "is_signed_v2"),
                "signing_v3": optional_bool(apk, "is_signed_v3"),
                "signing_v4": optional_bool(apk, "is_signed_v4"),
                "manifest_status": "OK",
            }
        )
        try:
            certificates = []
            for certificate in apk.get_certificates():
                digest = getattr(certificate, "sha256", None)
                if digest is not None:
                    certificates.append(value_text(digest))
            result["certificate_sha256"] = "|".join(sorted(set(certificates)))
            result["certificate_status"] = "OK" if certificates else "NO_CERTIFICATE"
        except Exception as exc:
            result["certificate_status"] = "ERROR"
            result["error_type"] = type(exc).__name__
            result["error_message"] = str(exc)[:500]
    except Exception as exc:
        result["manifest_status"] = "ERROR"
        result["certificate_status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)[:500]
    finally:
        del apk
        gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit APK identity with Androguard")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    args = parser.parse_args()

    inventory_path = args.inventory.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with inventory_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        inventory = list(csv.DictReader(handle))

    fields = [
        "source_id", "sha256", "absolute_path", "package_name", "version_code",
        "version_name", "min_sdk", "target_sdk", "certificate_sha256",
        "signing_v1", "signing_v2", "signing_v3", "signing_v4",
        "manifest_status", "certificate_status", "error_type", "error_message",
    ]
    output_csv = output / "apk_identity_androguard.csv"
    failures_csv = output / "apk_identity_failures.csv"
    rows_for_summary = []

    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle, failures_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as failure_handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        failure_writer = csv.DictWriter(failure_handle, fieldnames=fields)
        writer.writeheader()
        failure_writer.writeheader()

        for index, item in enumerate(inventory, start=1):
            free_gb = shutil.disk_usage(output).free / (1024 ** 3)
            if free_gb < args.min_free_gb:
                raise RuntimeError(
                    f"磁盘剩余空间 {free_gb:.2f} GiB，小于保护阈值 {args.min_free_gb:.2f} GiB，已停止。"
                )
            parsed = audit_one(Path(item["absolute_path"]))
            row = {
                "source_id": item.get("source_id", ""),
                "sha256": item.get("sha256", ""),
                "absolute_path": item.get("absolute_path", ""),
                **parsed,
            }
            writer.writerow(row)
            if parsed["manifest_status"] != "OK" or parsed["certificate_status"] != "OK":
                failure_writer.writerow(row)
            rows_for_summary.append(row)
            if index % 100 == 0 or index == len(inventory):
                print(f"processed {index}/{len(inventory)}", flush=True)

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P0_APK_IDENTITY_AUDIT_ANDROGUARD",
        "formal_training_enabled": False,
        "inventory_csv": str(inventory_path),
        "apk_count": len(rows_for_summary),
        "manifest_status_counts": dict(Counter(row["manifest_status"] for row in rows_for_summary)),
        "certificate_status_counts": dict(Counter(row["certificate_status"] for row in rows_for_summary)),
        "certificate_hash_joined_count": sum(bool(row["certificate_sha256"]) for row in rows_for_summary),
        "package_name_missing_count": sum(not row["package_name"] for row in rows_for_summary),
        "version_code_missing_count": sum(not row["version_code"] for row in rows_for_summary),
        "failure_row_count": sum(
            row["manifest_status"] != "OK" or row["certificate_status"] != "OK"
            for row in rows_for_summary
        ),
        "outputs": {
            "identity_csv": str(output_csv),
            "failure_csv": str(failures_csv),
        },
        "blocking_reasons": [
            "first_seen_utc_not_joined",
            "package_certificate_version_lineage_not_finalized",
            "near_duplicate_clusters_not_built",
            "static_parse_and_api_graph_coverage_not_available",
            "resolved_formal_training_config_not_generated",
        ],
        "notes": [
            "Androguard logs are suppressed; only progress and final summary are printed",
            "call graphs are not built or persisted in this phase",
            "APK objects are released after each sample",
            "disk free-space guard is checked before each APK",
        ],
    }
    audit_path = output / "apk_identity_androguard_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
