#!/usr/bin/env python3
"""Post-freeze read-only cross-check against the existing lineage audit."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path("/home/boyuanjie/projects/tasa-droid/outputs")
out = root / "care-2020-fn-tn-blind-semantic-review-v1"
freeze = json.loads((out / "blind_review_freeze.json").read_text(encoding="utf-8"))
assert hashlib.sha256((out / "blind_review_frozen.csv").read_bytes()).hexdigest() == freeze["review_form_sha256"]
with (root / "care-2020-fn-tn-dex-review-v2/sealed_key_restricted.csv").open(encoding="utf-8", newline="") as f:
    selected = list(csv.DictReader(f))
with (root / "care-consensus-error-lineage-label-audit-v1/sample_lineage_audit.csv").open(encoding="utf-8", newline="") as f:
    old = {r["sha256"]: r for r in csv.DictReader(f)}
summary = {"old_lineage_rows_found": 0, "old_lineage_rows_missing": 0,
           "strong_cross_label_lineage_by_kind": Counter(),
           "same_package_cross_label_by_kind": Counter(),
           "exact_dex_cross_label_by_kind": Counter()}
for row in selected:
    prior = old.get(row["sha256"])
    if prior is None:
        summary["old_lineage_rows_missing"] += 1
        continue
    summary["old_lineage_rows_found"] += 1
    for name in ("strong_cross_label_lineage", "same_package_cross_label", "exact_dex_cross_label"):
        if prior[name] == "1":
            summary[name + "_by_kind"][row["diagnostic_kind"]] += 1
summary = {k: dict(v) if isinstance(v, Counter) else v for k, v in summary.items()}
(out / "post_unblind_lineage_crosscheck.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
