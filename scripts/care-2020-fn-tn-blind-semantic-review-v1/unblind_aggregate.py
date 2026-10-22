#!/usr/bin/env python3
"""Only after a verified freeze, aggregate key-side diagnostic observations."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/boyuanjie/projects/tasa-droid/outputs")
OUT = ROOT / "care-2020-fn-tn-blind-semantic-review-v1"
INPUT = ROOT / "care-2020-fn-tn-dex-review-v2"

freeze = json.loads((OUT / "blind_review_freeze.json").read_text(encoding="utf-8"))
if freeze.get("status") != "FROZEN_BEFORE_KEY_READ":
    raise RuntimeError("Blind review has not been frozen")
if hashlib.sha256((OUT / "blind_review_frozen.csv").read_bytes()).hexdigest() != freeze["review_form_sha256"]:
    raise RuntimeError("Reviewer form changed after freeze")
for name, expected in freeze["packet_sha256"].items():
    if hashlib.sha256((OUT / "blind_packets" / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Reviewer packet changed after freeze")

# The sealed key is opened only after the preceding freeze checks.
with (INPUT / "sealed_key_restricted.csv").open(encoding="utf-8", newline="") as stream:
    key = {(r["case_id"], r["side"]): r for r in csv.DictReader(stream)}
with (OUT / "blind_review_frozen.csv").open(encoding="utf-8", newline="") as stream:
    review = list(csv.DictReader(stream))
if len(key) != 16 or len(review) != 8:
    raise RuntimeError("Cohort/key count mismatch")

paired = []
for row in review:
    case = row["case_id"]
    labels = {side: key[case, side]["diagnostic_kind"] for side in ("A", "B")}
    if sorted(labels.values()) != ["CORRECT_TN", "UNFIXED_FN"]:
        raise RuntimeError("Invalid pair composition")
    packet = json.loads((OUT / "blind_packets" / f"{case}.json").read_text(encoding="utf-8"))
    stats = {labels[side]: packet["sides"][side] for side in ("A", "B")}
    paired.append({"case_id": case, "reviewer_grade": row["reviewer_grade"],
                   "fn_app_sensitive_methods": stats["UNFIXED_FN"]["app_package_prefix_sensitive_methods"],
                   "tn_app_sensitive_methods": stats["CORRECT_TN"]["app_package_prefix_sensitive_methods"],
                   "fn_non_known_sdk_sensitive_methods": stats["UNFIXED_FN"]["non_known_sdk_sensitive_methods"],
                   "tn_non_known_sdk_sensitive_methods": stats["CORRECT_TN"]["non_known_sdk_sensitive_methods"]})
summary = {
    "status": "NO_VERIFIED_CLASS_DIRECTION_IN_BLIND_PILOT",
    "blind_review_cases": len(review),
    "verified_directional_facts": sum(r["reviewer_grade"] == "VERIFIED_DIRECTIONAL" for r in review),
    "insufficient_cases": sum(r["reviewer_grade"] == "INSUFFICIENT" for r in review),
    "fn_zero_app_prefix_sensitive": sum(r["fn_app_sensitive_methods"] == 0 for r in paired),
    "tn_zero_app_prefix_sensitive": sum(r["tn_app_sensitive_methods"] == 0 for r in paired),
    "fn_more_non_known_sdk_sensitive_than_tn": sum(r["fn_non_known_sdk_sensitive_methods"] > r["tn_non_known_sdk_sensitive_methods"] for r in paired),
    "tn_more_non_known_sdk_sensitive_than_fn": sum(r["tn_non_known_sdk_sensitive_methods"] > r["fn_non_known_sdk_sensitive_methods"] for r in paired),
    "ties_non_known_sdk_sensitive": sum(r["tn_non_known_sdk_sensitive_methods"] == r["fn_non_known_sdk_sensitive_methods"] for r in paired),
    "future_labels_used_for_diagnostic_unblinding": True,
    "future_labels_used_for_model_or_threshold_selection": False,
    "formal_asset_integrity": json.loads((OUT / "decision.json").read_text(encoding="utf-8"))["status"],
    "claim_boundary": "Eight matched case-control pairs cannot establish an absence of all static class evidence. The focused packets did not prove reliable class direction; counts and namespace heuristics remain descriptive.",
    "next_action": "If pursuing cause attribution, independently review full DEX/control/dataflow for selected cases or obtain external behavior/label provenance; do not train a new feature on these 2020 labels.",
}
(OUT / "post_unblind_aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
