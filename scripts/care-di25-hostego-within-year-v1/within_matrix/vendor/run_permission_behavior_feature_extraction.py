#!/usr/bin/env python3
"""Append permission--sensitive-API consistency features.

Only an APK manifest is parsed.  Sensitive API evidence is read from the
existing compact feature CSV, so this stage never builds a call graph and does
not persist manifests, permissions, DEX bytes, or method signatures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
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
except Exception as exc:  # pragma: no cover - server dependent
    raise SystemExit(
        "Androguard is required; run with the existing tif environment.\n"
        f"{type(exc).__name__}: {exc}"
    ) from exc


GROUPS = {
    "sms": {
        "SEND_SMS", "RECEIVE_SMS", "READ_SMS", "WRITE_SMS",
        "RECEIVE_MMS", "RECEIVE_WAP_PUSH",
    },
    "phone": {
        "READ_PHONE_STATE", "READ_PHONE_NUMBERS", "CALL_PHONE", "ANSWER_PHONE_CALLS",
        "READ_CALL_LOG", "WRITE_CALL_LOG", "PROCESS_OUTGOING_CALLS",
    },
    "contacts_calendar": {
        "READ_CONTACTS", "WRITE_CONTACTS", "GET_ACCOUNTS", "READ_CALENDAR", "WRITE_CALENDAR",
    },
    "location": {
        "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION", "ACCESS_BACKGROUND_LOCATION",
    },
    "microphone": {"RECORD_AUDIO"},
    "camera": {"CAMERA"},
    "storage": {
        "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE", "MANAGE_EXTERNAL_STORAGE",
        "READ_MEDIA_IMAGES", "READ_MEDIA_VIDEO", "READ_MEDIA_AUDIO",
    },
    "accounts": {"AUTHENTICATE_ACCOUNTS", "USE_CREDENTIALS", "MANAGE_ACCOUNTS"},
    "overlay": {"SYSTEM_ALERT_WINDOW"},
    "boot": {"RECEIVE_BOOT_COMPLETED"},
    "install": {"REQUEST_INSTALL_PACKAGES", "INSTALL_PACKAGES", "DELETE_PACKAGES"},
    "network": {"INTERNET", "ACCESS_NETWORK_STATE", "CHANGE_NETWORK_STATE"},
    "device_admin": {"BIND_DEVICE_ADMIN"},
    "accessibility": {"BIND_ACCESSIBILITY_SERVICE"},
}

GROUP_ROLES = {
    "sms": ("sms_send_receive",),
    "phone": ("call_control_or_recording", "device_identifier_access"),
    "contacts_calendar": ("contacts_or_calendar_access",),
    "location": ("location_access",),
    "microphone": ("microphone_access", "call_control_or_recording"),
    "camera": ("camera_access",),
    "storage": ("sensitive_file_or_content_write", "file_read"),
    "accounts": ("account_or_credential_access",),
    "overlay": ("overlay_or_ui_deception",),
    "boot": ("boot_persistence",),
    "install": ("package_installation", "package_uninstallation"),
    "network": ("network_transmission_or_upload", "url_or_http_setup"),
    "device_admin": ("device_administration",),
    "accessibility": ("accessibility_control",),
}

PERMISSION_FIELDS = [
    field
    for group in GROUPS
    for field in (
        f"permission_behavior__{group}__declared",
        f"permission_behavior__{group}__declared_and_api",
        f"permission_behavior__{group}__api_without_declared",
    )
] + [
    "permission_behavior__declared_group_count",
    "permission_behavior__api_group_count",
    "permission_behavior__declared_and_api_group_count",
    "permission_behavior__api_without_declared_group_count",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_permission(value: object) -> str:
    text = str(value or "").strip().upper()
    prefix = "ANDROID.PERMISSION."
    return text[len(prefix):] if text.startswith(prefix) else text


def api_present(row: dict[str, str], roles: tuple[str, ...]) -> bool:
    for role in roles:
        for prefix in ("role_node__", "role_edge__"):
            try:
                if float(row.get(prefix + role, "0") or 0.0) > 0.0:
                    return True
            except ValueError:
                pass
    return False


def extract_one(row: dict[str, str]) -> dict[str, float]:
    apk = APK(str(Path(row["absolute_path"])))
    declared = {
        normalize_permission(permission)
        for permission in (apk.get_permissions() or [])
    }
    result: dict[str, float] = {}
    declared_count = api_count = both_count = api_only_count = 0
    for group, permissions in GROUPS.items():
        has_permission = bool(declared & permissions)
        has_api = api_present(row, GROUP_ROLES[group])
        both = has_permission and has_api
        api_only = has_api and not has_permission
        result[f"permission_behavior__{group}__declared"] = float(has_permission)
        result[f"permission_behavior__{group}__declared_and_api"] = float(both)
        result[f"permission_behavior__{group}__api_without_declared"] = float(api_only)
        declared_count += int(has_permission)
        api_count += int(has_api)
        both_count += int(both)
        api_only_count += int(api_only)
    result.update({
        "permission_behavior__declared_group_count": float(declared_count),
        "permission_behavior__api_group_count": float(api_count),
        "permission_behavior__declared_and_api_group_count": float(both_count),
        "permission_behavior__api_without_declared_group_count": float(api_only_count),
    })
    del apk
    gc.collect()
    return result


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
    fields = list(rows[0])
    required = {"sha256", "absolute_path"}
    if missing := required - set(fields):
        raise SystemExit(f"features CSV is missing columns: {sorted(missing)}")
    if set(fields) & set(PERMISSION_FIELDS):
        raise SystemExit("permission-behavior fields already exist in input")

    failures = []
    completed = []
    every = max(1, args.progress_every)
    for index, row in enumerate(rows, 1):
        try:
            values = extract_one(row)
        except Exception as exc:
            values = {field: 0.0 for field in PERMISSION_FIELDS}
            failures.append({
                "sha256": row.get("sha256", ""),
                "absolute_path": row.get("absolute_path", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
        row.update({field: str(values[field]) for field in PERMISSION_FIELDS})
        completed.append(row)
        if index % every == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)} failures={len(failures)}", flush=True)

    output_features = output / "features_with_permission_behavior.csv"
    with output_features.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + PERMISSION_FIELDS)
        writer.writeheader()
        writer.writerows(completed)
    output_failures = output / "permission_behavior_failures.csv"
    with output_failures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "absolute_path", "error_type", "error_message"],
        )
        writer.writeheader()
        writer.writerows(failures)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["feature_fields"] = list(schema.get("feature_fields", fields)) + PERMISSION_FIELDS
    schema["permission_behavior_feature_fields"] = PERMISSION_FIELDS
    schema["permission_behavior_extraction"] = {
        "label_agnostic": True,
        "manifest_only": True,
        "call_graph_built": False,
        "raw_permissions_persisted": False,
    }
    output_schema = output / "feature_schema_with_permission_behavior.json"
    output_schema.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P22_MANIFEST_PERMISSION_BEHAVIOR_CONSISTENCY",
        "formal_training_enabled": False,
        "input_row_count": len(rows),
        "output_row_count": len(completed),
        "feature_count": len(PERMISSION_FIELDS),
        "failure_count": len(failures),
        "input_features_sha256": sha256_file(features_path),
        "outputs": {
            "features": str(output_features),
            "schema": str(output_schema),
            "failures": str(output_failures),
        },
        "notes": [
            "Manifest parsing and role-consistency calculation do not use labels.",
            "Raw permissions and APK manifests are not persisted.",
            "No call graph is built or stored.",
        ],
    }
    (output / "permission_behavior_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
