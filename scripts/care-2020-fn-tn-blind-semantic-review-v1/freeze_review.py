#!/usr/bin/env python3
"""Freeze reviewer observations without loading any label/key file."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path("/home/boyuanjie/projects/tasa-droid/outputs/care-2020-fn-tn-blind-semantic-review-v1")
FORM = OUT / "blind_review_frozen.csv"
FREEZE = OUT / "blind_review_freeze.json"

if FREEZE.exists():
    raise RuntimeError("Review is already frozen")
with FORM.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if [r["case_id"] for r in rows] != [f"M{i:02d}" for i in range(1, 9)]:
    raise RuntimeError("Case order or count mismatch")
if any(not r["reviewer_grade"] or not r["class_direction_before_unseal"] for r in rows):
    raise RuntimeError("Incomplete reviewer form")
packets = sorted((OUT / "blind_packets").glob("M*.json"))
if len(packets) != 8:
    raise RuntimeError("Incomplete reviewer packets")
record = {
    "status": "FROZEN_BEFORE_KEY_READ",
    "review_form_sha256": hashlib.sha256(FORM.read_bytes()).hexdigest(),
    "packet_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in packets},
    "cases": len(rows),
    "review_grade_counts": {grade: sum(r["reviewer_grade"] == grade for r in rows)
                             for grade in sorted({r["reviewer_grade"] for r in rows})},
    "label_key_was_not_loaded_by_this_script": True,
}
FREEZE.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
