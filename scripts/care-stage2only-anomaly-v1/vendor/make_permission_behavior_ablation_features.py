#!/usr/bin/env python3
"""Create label-free P22 permission-behavior feature ablations.

This script only copies an existing P22 feature CSV while removing selected
``permission_behavior__`` columns.  It never opens an APK, builds a graph, or
reads labels for feature selection.  The resulting CSV and schema can be used
with the normal fixed-threshold evaluators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


GROUPS = (
    "sms", "phone", "contacts_calendar", "location", "microphone", "camera",
    "storage", "accounts", "overlay", "boot", "install", "network",
    "device_admin", "accessibility",
)

# These are privileged-control / high-impact groups.  Common platform-use
# groups (network, storage, location, camera, microphone) are deliberately
# excluded from this narrow policy variant.
HIGH_RISK_GROUPS = {
    "sms", "phone", "contacts_calendar", "accounts", "overlay", "boot",
    "install", "device_admin", "accessibility",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def permission_fields_for_mode(mode: str) -> set[str]:
    declared = {
        f"permission_behavior__{group}__declared" for group in GROUPS
    } | {"permission_behavior__declared_group_count"}
    consistency = {
        f"permission_behavior__{group}__{kind}"
        for group in GROUPS
        for kind in ("declared_and_api", "api_without_declared")
    } | {
        "permission_behavior__declared_and_api_group_count",
        "permission_behavior__api_without_declared_group_count",
    }
    all_fields = declared | consistency | {
        "permission_behavior__api_group_count",
    }

    if mode == "all":
        return all_fields
    if mode == "declared_only":
        return declared
    if mode == "consistency_only":
        return consistency
    if mode == "high_risk_consistency":
        return {
            f"permission_behavior__{group}__{kind}"
            for group in HIGH_RISK_GROUPS
            for kind in ("declared_and_api", "api_without_declared")
        }
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=(
        "all", "declared_only", "consistency_only", "high_risk_consistency",
    ))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    features_path = args.features.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    with features_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise SystemExit("input feature CSV is empty")

    available_permission_fields = {
        name for name in fields if name.startswith("permission_behavior__")
    }
    if not available_permission_fields:
        raise SystemExit("input feature CSV contains no P22 permission-behavior fields")
    selected_permission_fields = permission_fields_for_mode(args.mode)
    selected_permission_fields &= available_permission_fields
    retained_fields = [
        name for name in fields
        if not name.startswith("permission_behavior__") or name in selected_permission_fields
    ]

    feature_out = output / "features.csv"
    with feature_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=retained_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in retained_fields})

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["feature_fields"] = [
        name for name in schema.get("feature_fields", fields)
        if name in retained_fields
    ]
    schema["permission_behavior_feature_fields"] = sorted(selected_permission_fields)
    schema["permission_behavior_ablation"] = {
        "mode": args.mode,
        "label_agnostic": True,
        "source_features_sha256": sha256_file(features_path),
        "removed_permission_field_count": len(
            available_permission_fields - selected_permission_fields
        ),
    }
    schema_out = output / "feature_schema.json"
    schema_out.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P22_PERMISSION_BEHAVIOR_FEATURE_ABLATION",
        "formal_training_enabled": False,
        "mode": args.mode,
        "input_row_count": len(rows),
        "input_column_count": len(fields),
        "output_column_count": len(retained_fields),
        "retained_permission_field_count": len(selected_permission_fields),
        "removed_permission_field_count": len(
            available_permission_fields - selected_permission_fields
        ),
        "outputs": {"features": str(feature_out), "schema": str(schema_out)},
        "notes": [
            "This is a deterministic column ablation; APKs and labels are not read.",
            "Candidate selection must use the 2017 rolling development splits only.",
        ],
    }
    (output / "ablation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
