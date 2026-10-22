#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from common import OUTPUT, PROJECT, dump_json, ensure_output, file_record, load_json


def asset_paths() -> list[tuple[Path, str]]:
    candidates = [
        (PROJECT / "outputs/care-obcbc-source-v1/final_decision.json", "prior ownership-boundary decision"),
        (PROJECT / "outputs/care-obcbc-source-v1/feasibility_decision.json", "prior ownership-boundary feasibility"),
        (PROJECT / "outputs/care-obcbc-source-v1/ownership_audit.json", "prior heuristic owner coverage"),
        (PROJECT / "outputs/care-obcbc-source-v1/context_coverage.json", "prior boundary coverage"),
        (PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv", "formal C43 reference"),
        (PROJECT / "outputs/care-svm-c43-remove-bounded-correction-v1/source_predictions.csv", "direct C43 candidate"),
        (PROJECT / "outputs/care-svm-c43-unfixed-audit-v1/routed_error_audit.csv", "direct C43 routed audit"),
    ]
    return [(p, purpose) for p, purpose in candidates if p.exists() and p.is_file()]


def main() -> None:
    ensure_output()
    phase = sys.argv[1] if len(sys.argv) > 1 else "before"
    if phase not in {"before", "after"}:
        raise SystemExit("usage: asset_manifest.py before|after")
    records = [file_record(path, purpose) for path, purpose in asset_paths()]
    dump_json(OUTPUT / f"baseline_asset_manifest_{phase}.json", {"phase": phase, "asset_count": len(records), "assets": records})
    if phase == "after":
        before = load_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": []})
        old = {row["path"]: row for row in before.get("assets", [])}
        new = {row["path"]: row for row in records}
        changed = []
        for path in sorted(set(old) | set(new)):
            if path not in old:
                changed.append({"path": path, "change": "ADDED_TO_MANIFEST"})
            elif path not in new:
                changed.append({"path": path, "change": "MISSING_AFTER"})
            elif old[path]["sha256"] != new[path]["sha256"]:
                changed.append({"path": path, "change": "SHA256_CHANGED", "before": old[path]["sha256"], "after": new[path]["sha256"]})
        dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "asset_count": len(old), "changed": changed})
    print(f"PASEA_ASSET_MANIFEST_{phase.upper()} assets={len(records)}")


if __name__ == "__main__":
    main()

