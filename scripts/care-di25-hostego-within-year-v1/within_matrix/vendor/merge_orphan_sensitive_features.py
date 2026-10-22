#!/usr/bin/env python3
"""Merge compact P24 orphan-sensitive summaries into an existing feature CSV.

No APK, graph, label, manifest, method signature, or path is read here.  The
script merely joins a completed P24 summary by SHA256 and selects a predefined
label-independent subset of compact numeric fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SUMMARY_FIELDS = (
    "entry_method_count",
    "path_retained_sensitive_node_count",
    "path_excluded_sensitive_node_count",
    "entry_reachable_sensitive_node_count",
    "entry_reachable_within_h_sensitive_node_count",
    "entry_reachable_beyond_h_sensitive_node_count",
    "entry_unreachable_sensitive_node_count",
    "entry_unreachable_sensitive_node_ratio",
    "orphan_sensitive_component_count",
    "orphan_sensitive_induced_edge_count",
    "orphan_tier__0",
    "orphan_tier__1",
    "orphan_tier__2",
)

# Tiers 0--2 from sensitive_endpoint_priority_v1.yaml.  Common support roles
# (crypto, encoding, URL setup, file/database access) are deliberately absent.
HIGH_RISK_ROLES = {
    "command_execution", "dynamic_code_loading", "native_library_loading",
    "reflective_invocation", "package_installation", "package_uninstallation",
    "device_administration", "accessibility_control", "sms_send_receive",
    "call_control_or_recording", "account_or_credential_access",
    "contacts_or_calendar_access", "device_identifier_access", "location_access",
    "microphone_access", "camera_access", "clipboard_access",
    "network_transmission_or_upload", "sensitive_file_or_content_write",
    "boot_persistence", "task_or_alarm_scheduling", "background_service_control",
    "overlay_or_ui_deception", "icon_or_component_hiding",
    "process_or_wakelock_control",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_fields(summary_fields: list[str], mode: str) -> list[str]:
    present_summary = [name for name in SUMMARY_FIELDS if name in summary_fields]
    roles = sorted(name for name in summary_fields if name.startswith("orphan_role__"))
    if mode == "summary":
        return present_summary
    if mode == "high_risk_roles":
        return present_summary + [
            name for name in roles if name.removeprefix("orphan_role__") in HIGH_RISK_ROLES
        ]
    if mode == "all_roles":
        return present_summary + roles
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--orphan-summary", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("summary", "high_risk_roles", "all_roles")
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    features_path = args.features.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    orphan_path = args.orphan_summary.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with features_path.open(encoding="utf-8-sig", newline="") as handle:
        feature_reader = csv.DictReader(handle)
        feature_fields = list(feature_reader.fieldnames or [])
        feature_rows = list(feature_reader)
    with orphan_path.open(encoding="utf-8-sig", newline="") as handle:
        orphan_reader = csv.DictReader(handle)
        orphan_fields = list(orphan_reader.fieldnames or [])
        orphan_rows = list(orphan_reader)
    if not feature_rows or "sha256" not in feature_fields:
        raise SystemExit("feature CSV is empty or lacks sha256")
    if not orphan_rows or "sha256" not in orphan_fields:
        raise SystemExit("orphan summary is empty or lacks sha256")
    if any(name in feature_fields for name in orphan_fields if name.startswith("orphan_")):
        raise SystemExit("input features already contain orphan fields")

    selected = selected_fields(orphan_fields, args.mode)
    if not selected:
        raise SystemExit("no selected P24 fields are present in orphan summary")
    orphan_by_sha = {
        row["sha256"].casefold(): row
        for row in orphan_rows
        if row.get("sha256") and row.get("status") == "OK"
    }
    missing = [
        row["sha256"] for row in feature_rows
        if row.get("sha256", "").casefold() not in orphan_by_sha
    ]
    if missing:
        raise SystemExit(
            f"P24 summary does not cover {len(missing)} feature rows; first SHA256={missing[0]}"
        )

    output_rows = []
    for row in feature_rows:
        source = orphan_by_sha[row["sha256"].casefold()]
        merged = dict(row)
        for name in selected:
            merged[name] = source.get(name, "0")
        output_rows.append(merged)
    output_fields = feature_fields + selected
    features_out = output / "features_with_orphan_sensitive.csv"
    with features_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["feature_fields"] = list(schema.get("feature_fields", feature_fields)) + selected
    schema["orphan_sensitive_feature_fields"] = selected
    schema["orphan_sensitive_merge"] = {
        "mode": args.mode,
        "label_agnostic": True,
        "source_orphan_summary_sha256": sha256_file(orphan_path),
        "source_feature_sha256": sha256_file(features_path),
        "raw_graph_or_path_persisted": False,
    }
    schema_out = output / "feature_schema_with_orphan_sensitive.json"
    schema_out.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P24_ORPHAN_SENSITIVE_FEATURE_MERGE",
        "formal_training_enabled": False,
        "mode": args.mode,
        "input_feature_row_count": len(feature_rows),
        "orphan_summary_row_count": len(orphan_rows),
        "merged_feature_count": len(selected),
        "outputs": {"features": str(features_out), "schema": str(schema_out)},
        "notes": [
            "This is a SHA256 join of precomputed compact numeric summaries.",
            "No labels, APKs, raw graphs, paths, or signatures are read or written.",
        ],
    }
    (output / "orphan_sensitive_merge_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
