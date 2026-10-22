#!/usr/bin/env python3
"""Append label-free Android-manifest exposure-surface features.

The stage parses only ``AndroidManifest.xml`` through Androguard's APK reader.
It never builds a call graph and never persists raw manifest XML, component
names, intent actions, or permissions.  Its purpose is to quantify exposure
and protection structure (exporting, intent filters, protection attributes),
which is distinct from APK/DEX size and archive-layout features.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
COMPONENT_TAGS = ("activity", "activity-alias", "service", "receiver", "provider")

EXPOSURE_FIELDS = [
    "manifest_exposure__parse_ok",
    "manifest_exposure__component_count",
    "manifest_exposure__component_count_log",
    "manifest_exposure__intent_filter_component_count",
    "manifest_exposure__intent_filter_component_ratio",
    "manifest_exposure__intent_filter_count_log",
    "manifest_exposure__intent_action_count_log",
    "manifest_exposure__intent_category_count_log",
    "manifest_exposure__intent_data_count_log",
    "manifest_exposure__explicit_exported_true_count",
    "manifest_exposure__explicit_exported_false_count",
    "manifest_exposure__exported_unspecified_count",
    "manifest_exposure__effective_exported_count",
    "manifest_exposure__effective_exported_ratio",
    "manifest_exposure__protected_component_count",
    "manifest_exposure__protected_component_ratio",
    "manifest_exposure__custom_process_component_count",
    "manifest_exposure__custom_process_component_ratio",
    "manifest_exposure__disabled_component_count",
    "manifest_exposure__application_debuggable",
    "manifest_exposure__application_test_only",
    "manifest_exposure__custom_permission_count",
    "manifest_exposure__signature_permission_count",
    "manifest_exposure__provider_read_write_permission_count",
]
for _kind in COMPONENT_TAGS:
    EXPOSURE_FIELDS.extend([
        f"manifest_exposure__{_kind}__count",
        f"manifest_exposure__{_kind}__effective_exported_count",
        f"manifest_exposure__{_kind}__intent_filter_count",
        f"manifest_exposure__{_kind}__protected_count",
    ])


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log1p(value: int) -> float:
    return math.log1p(max(0, int(value)))


def local_name(tag: object) -> str:
    text = str(tag or "")
    return text.rsplit("}", 1)[-1]


def android_attr(element, name: str) -> str:
    return str(element.get(ANDROID_NS + name, "") or "").strip()


def bool_attr(value: str):
    lowered = value.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def manifest_root(apk):
    root = apk.get_android_manifest_xml()
    if root is None:
        raise ValueError("AndroidManifest.xml could not be decoded")
    return root


def extract_one(row: dict[str, str]) -> dict[str, float]:
    apk = APK(str(Path(row["absolute_path"])))
    try:
        root = manifest_root(apk)
        application = next(
            (child for child in root if local_name(child.tag) == "application"), None
        )
        if application is None:
            raise ValueError("manifest application element is missing")

        package_name = str(root.get("package", "") or "").strip()
        result = {field: 0.0 for field in EXPOSURE_FIELDS}
        result["manifest_exposure__parse_ok"] = 1.0
        component_count = 0
        intent_filter_component_count = 0
        intent_filter_count = intent_action_count = intent_category_count = intent_data_count = 0
        exported_true = exported_false = exported_unspecified = effective_exported = 0
        protected_component_count = custom_process_count = disabled_component_count = 0
        type_counts = Counter()
        type_exported = Counter()
        type_filters = Counter()
        type_protected = Counter()

        for element in application:
            kind = local_name(element.tag)
            if kind not in COMPONENT_TAGS:
                continue
            component_count += 1
            type_counts[kind] += 1
            filters = [
                child for child in element if local_name(child.tag) == "intent-filter"
            ]
            filter_count = len(filters)
            type_filters[kind] += filter_count
            intent_filter_count += filter_count
            if filter_count:
                intent_filter_component_count += 1
            for intent_filter in filters:
                for item in intent_filter:
                    item_name = local_name(item.tag)
                    if item_name == "action":
                        intent_action_count += 1
                    elif item_name == "category":
                        intent_category_count += 1
                    elif item_name == "data":
                        intent_data_count += 1

            exported = bool_attr(android_attr(element, "exported"))
            if exported is True:
                exported_true += 1
            elif exported is False:
                exported_false += 1
            else:
                exported_unspecified += 1
            # For legacy Android manifests an intent-filter makes a component
            # externally reachable unless explicit exported=false is present.
            is_effectively_exported = exported is True or (exported is None and filter_count > 0)
            if is_effectively_exported:
                effective_exported += 1
                type_exported[kind] += 1

            protected = bool(android_attr(element, "permission"))
            if kind == "provider":
                protected = protected or bool(android_attr(element, "readPermission"))
                protected = protected or bool(android_attr(element, "writePermission"))
            if protected:
                protected_component_count += 1
                type_protected[kind] += 1
            process = android_attr(element, "process")
            if process and process not in {package_name, ":" + package_name}:
                custom_process_count += 1
            if bool_attr(android_attr(element, "enabled")) is False:
                disabled_component_count += 1

        custom_permission_count = signature_permission_count = 0
        for element in root:
            if local_name(element.tag) != "permission":
                continue
            name = android_attr(element, "name")
            if name and not name.startswith("android.permission."):
                custom_permission_count += 1
            protection_level = android_attr(element, "protectionLevel").casefold()
            if "signature" in protection_level:
                signature_permission_count += 1

        provider_rw_permission_count = 0
        for element in application:
            if local_name(element.tag) != "provider":
                continue
            provider_rw_permission_count += int(bool(android_attr(element, "readPermission")))
            provider_rw_permission_count += int(bool(android_attr(element, "writePermission")))

        result.update({
            "manifest_exposure__component_count": float(component_count),
            "manifest_exposure__component_count_log": log1p(component_count),
            "manifest_exposure__intent_filter_component_count": float(intent_filter_component_count),
            "manifest_exposure__intent_filter_component_ratio": intent_filter_component_count / max(1, component_count),
            "manifest_exposure__intent_filter_count_log": log1p(intent_filter_count),
            "manifest_exposure__intent_action_count_log": log1p(intent_action_count),
            "manifest_exposure__intent_category_count_log": log1p(intent_category_count),
            "manifest_exposure__intent_data_count_log": log1p(intent_data_count),
            "manifest_exposure__explicit_exported_true_count": float(exported_true),
            "manifest_exposure__explicit_exported_false_count": float(exported_false),
            "manifest_exposure__exported_unspecified_count": float(exported_unspecified),
            "manifest_exposure__effective_exported_count": float(effective_exported),
            "manifest_exposure__effective_exported_ratio": effective_exported / max(1, component_count),
            "manifest_exposure__protected_component_count": float(protected_component_count),
            "manifest_exposure__protected_component_ratio": protected_component_count / max(1, component_count),
            "manifest_exposure__custom_process_component_count": float(custom_process_count),
            "manifest_exposure__custom_process_component_ratio": custom_process_count / max(1, component_count),
            "manifest_exposure__disabled_component_count": float(disabled_component_count),
            "manifest_exposure__application_debuggable": float(bool_attr(android_attr(application, "debuggable")) is True),
            "manifest_exposure__application_test_only": float(bool_attr(android_attr(application, "testOnly")) is True),
            "manifest_exposure__custom_permission_count": float(custom_permission_count),
            "manifest_exposure__signature_permission_count": float(signature_permission_count),
            "manifest_exposure__provider_read_write_permission_count": float(provider_rw_permission_count),
        })
        for kind in COMPONENT_TAGS:
            result[f"manifest_exposure__{kind}__count"] = float(type_counts[kind])
            result[f"manifest_exposure__{kind}__effective_exported_count"] = float(type_exported[kind])
            result[f"manifest_exposure__{kind}__intent_filter_count"] = float(type_filters[kind])
            result[f"manifest_exposure__{kind}__protected_count"] = float(type_protected[kind])
        return result
    finally:
        del apk
        gc.collect()


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
    with features_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input feature CSV is empty")
    fields = list(rows[0])
    if "sha256" not in fields or "absolute_path" not in fields:
        raise SystemExit("input feature CSV must contain sha256 and absolute_path")
    if any(name in fields for name in EXPOSURE_FIELDS):
        raise SystemExit("manifest exposure fields already exist in input")

    failures = []
    every = max(1, args.progress_every)
    for index, row in enumerate(rows, 1):
        try:
            values = extract_one(row)
        except Exception as exc:
            values = {field: 0.0 for field in EXPOSURE_FIELDS}
            failures.append({
                "sha256": row.get("sha256", ""),
                "absolute_path": row.get("absolute_path", ""),
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
        row.update({field: str(values[field]) for field in EXPOSURE_FIELDS})
        if index % every == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)} failures={len(failures)}", flush=True)

    feature_out = output / "features_with_manifest_exposure.csv"
    with feature_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + EXPOSURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    failures_out = output / "manifest_exposure_failures.csv"
    with failures_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "absolute_path", "error_type", "error_message"],
        )
        writer.writeheader()
        writer.writerows(failures)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["feature_fields"] = list(schema.get("feature_fields", fields)) + EXPOSURE_FIELDS
    schema["manifest_exposure_feature_fields"] = EXPOSURE_FIELDS
    schema["manifest_exposure_extraction"] = {
        "label_agnostic": True,
        "manifest_only": True,
        "call_graph_built": False,
        "raw_manifest_or_component_names_persisted": False,
    }
    schema_out = output / "feature_schema_with_manifest_exposure.json"
    schema_out.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P23_MANIFEST_EXPOSURE_SURFACE_FEATURE_EXTRACTION",
        "formal_training_enabled": False,
        "input_row_count": len(rows),
        "output_row_count": len(rows),
        "feature_count": len(EXPOSURE_FIELDS),
        "failure_count": len(failures),
        "input_features_sha256": sha256_file(features_path),
        "outputs": {
            "features": str(feature_out),
            "schema": str(schema_out),
            "failures": str(failures_out),
        },
        "notes": [
            "Features are extracted without labels.",
            "Only manifest structure and attributes are read transiently.",
            "No call graph, DEX bytes, component names, actions, or raw permissions are persisted.",
        ],
    }
    (output / "manifest_exposure_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
