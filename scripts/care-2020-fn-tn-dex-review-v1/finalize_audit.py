#!/usr/bin/env python3
"""Post-run integrity and blinding QA for the independent 2020 review pilot."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
OUT = ROOT / "outputs/care-2020-fn-tn-dex-review-v1"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    if decision["status"] != "BLIND_STATIC_REVIEW_REQUIRED":
        raise RuntimeError("Main scan has not finished")
    source_manifest = json.loads((OUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))
    asset_rows = list(csv.DictReader((OUT / "apk_asset_audit_restricted.csv").open(encoding="utf-8", newline="")))
    changed = []
    for path, expected in source_manifest.items():
        current = Path(path)
        if current.stat().st_size != expected["bytes"] or current.stat().st_mtime_ns != expected["mtime_ns"] or hash_file(current) != expected["sha256"]:
            changed.append(path)
    for row in asset_rows:
        current = Path(row["path"])
        if current.stat().st_size != int(row["bytes"]) or current.stat().st_mtime_ns != int(row["mtime_ns"]) or hash_file(current) != row["content_sha256"]:
            changed.append(row["path"])
    packets = sorted((OUT / "blind_packets").glob("P*.json"))
    if len(packets) != 8:
        raise RuntimeError(f"Expected 8 packets, found {len(packets)}")
    descriptors = []
    for path in packets:
        packet = json.loads(path.read_text(encoding="utf-8"))
        raw = path.read_text(encoding="utf-8")
        # No hashes or data-set class folder paths may appear in the blind material.
        leaked = [row["sha256"] for row in asset_rows if row["sha256"] in raw]
        if leaked or '"apk_path"' in raw or '"kind"' in raw:
            raise RuntimeError(f"Blind packet identifier leak in {path.name}")
        a, b = packet["sides"]["A"], packet["sides"]["B"]
        roles_a, roles_b = set(a["role_call_counts"]), set(b["role_call_counts"])
        descriptors.append({"case_id": packet["case_id"], "A_dex": a["dex_count"], "B_dex": b["dex_count"],
                            "A_methods": a["method_count"], "B_methods": b["method_count"],
                            "A_sensitive_methods": a["sensitive_method_count"], "B_sensitive_methods": b["sensitive_method_count"],
                            "shared_roles": len(roles_a & roles_b), "union_roles": len(roles_a | roles_b)})
    dump("blinding_and_integrity_addendum.json", {
        "status": "PASS" if not changed else "FAIL", "formal_or_apk_assets_changed": changed,
        "baseline_assets_rehashed": len(source_manifest), "original_apks_rehashed": len(asset_rows),
        "blind_packets_checked": len(packets), "2020_labels_used_for_diagnostic_pair_selection": True,
        "2020_labels_used_for_model_or_threshold_selection": False,
        "interpretation": "Role and size differences are descriptive only; no semantic class direction established.",
        "blind_pair_descriptors": descriptors,
    })
    if changed:
        raise RuntimeError("Formal/APK asset integrity failed")
    print("BLIND_REVIEW_FINAL_QA_PASS packets=8 assets=" + str(len(source_manifest) + len(asset_rows)), flush=True)


if __name__ == "__main__":
    main()
