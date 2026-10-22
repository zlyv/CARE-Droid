#!/usr/bin/env python3
"""Validate manual source-only static-evidence annotations without changing formal assets.

The validator checks identity, provenance and annotation completeness; it does
NOT turn an observed omitted feature into a proven cause of model error.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from run_audit import PROJECT, OUTPUT, dump, fail, norm_sha, read_csv, record, sha256_file, write_csv

ALLOWED = {
    "UNREVIEWED", "NOT_DETERMINED", "STATIC_OBSERVATION_GAP",
    "EXTRACTION_INCOMPLETE", "OBSERVED_IN_C43_BUT_AMBIGUOUS",
    "LABEL_DISPUTE", "OTHER_DOCUMENTED",
}
DOCUMENTED = ALLOWED - {"UNREVIEWED", "NOT_DETERMINED"}


def main() -> None:
    report = OUTPUT / "final_decision.json"
    mapping_path = OUTPUT / "review_case_key_SOURCE_ONLY.csv"
    if not report.is_file() or not mapping_path.is_file():
        fail("run the initial source-only audit first")
    decision = json.loads(report.read_text(encoding="utf-8"))
    if decision.get("future_rows_loaded") != 0 or decision.get("formal_integrity") != "PASS":
        fail("initial audit is not source-only with integrity PASS")
    review_source = Path(os.environ.get("ECA_REVIEW_INPUT", str(OUTPUT / "manual_review_blinded_template.csv"))).resolve()
    if not review_source.is_file():
        fail("missing completed manual review CSV: " + str(review_source))
    inputs = [record(mapping_path, "frozen source review identities"),
              record(review_source, "completed manual review annotations")]
    keyed = read_csv(mapping_path)
    reviews = read_csv(review_source)
    if len(keyed) != 52 or len(reviews) != 52:
        fail("review inventory must have exactly 52 source routed error samples")
    key = {x["case_id"]: x for x in keyed}
    if len(key) != 52:
        fail("duplicate key case IDs")
    seen = set()
    result = []
    artifact_count = 0
    for review in reviews:
        case = str(review.get("case_id", "")).strip()
        if case in seen or case not in key:
            fail("unexpected or duplicate review case ID")
        seen.add(case)
        if norm_sha(review.get("sha256")) != norm_sha(key[case]["sha256"]):
            fail("review SHA does not match frozen source case")
        assessment = str(review.get("assessment", "")).strip().upper()
        if assessment not in ALLOWED:
            fail("unknown manual assessment " + assessment)
        reviewer = str(review.get("reviewer", "")).strip()
        notes = str(review.get("notes", "")).strip()
        evidence_status = "UNREVIEWED_OR_UNDETERMINED"
        if assessment in DOCUMENTED:
            fields = ("reviewer", "source_artifact_path", "source_artifact_sha256",
                      "method_or_instruction", "observed_static_fact", "c43_field_or_extraction_mapping_checked",
                      "independent_validation")
            missing = [f for f in fields if not str(review.get(f, "")).strip()]
            if missing:
                fail("documented assessment missing evidence fields for " + case + ": " + ",".join(missing))
            artifact = Path(str(review["source_artifact_path"]).strip()).resolve()
            if not artifact.is_file():
                fail("review evidence file does not exist: " + str(artifact))
            given_hash = str(review["source_artifact_sha256"]).strip().lower()
            actual_hash = sha256_file(artifact)
            if actual_hash != given_hash:
                fail("static evidence SHA256 mismatch for " + case)
            # If reviewer cites the APK itself, it MUST match the identified source APK.
            if artifact.suffix.lower() == ".apk" and actual_hash != key[case]["sha256"]:
                fail("review APK bytes do not match source SHA for " + case)
            artifact_count += 1
            evidence_status = "ARTIFACT_HASH_AND_METADATA_VERIFIED_REVIEWER_CLAIM_UNTESTED"
        result.append({"case_id": case, "group": key[case]["group"], "stage1_side": key[case]["stage1_side"],
                       "reviewer": reviewer, "assessment": assessment, "evidence_status": evidence_status,
                       "static_fact_summary": str(review.get("observed_static_fact", "")).strip(),
                       "notes": notes})
    if seen != set(key):
        fail("incomplete manual review set")
    after = [record(Path(r["path"]), r["purpose"]) for r in inputs]
    changed = [a["path"] for a, b in zip(inputs, after) if a["sha256"] != b["sha256"] or a["size"] != b["size"]]
    if changed:
        fail("manual review or frozen source mapping modified while validating")
    write_csv("manual_review_evidence_inventory.csv", result)
    summary = {
        "status": "SOURCE_MANUAL_EVIDENCE_VALIDATED_CAUSAL_ATTRIBUTION_UNPROVEN",
        "source_only": True, "future_rows_loaded": 0,
        "cases": 52, "documented_artifacts_hash_checked": artifact_count,
        "counts_by_assessment": dict(sorted(Counter(x["assessment"] for x in result).items())),
        "counts_by_group_and_assessment": {
            group: dict(sorted(Counter(x["assessment"] for x in result if x["group"] == group).items()))
            for group in ("ROUTED_UNFIXED", "ROUTED_BROKEN")},
        "verified_root_causes": 0,
        "note": "Code verifies evidence file hashes and metadata only. Human judgments, missingness, label disputes and causal explanations remain independent hypotheses.",
        "source_assets_modified": False,
    }
    dump("manual_review_validation.json", summary)
    print("MANUAL_REVIEW_VALIDATION=" + summary["status"])
    print("DOCUMENTED_EVIDENCE_FILES=" + str(artifact_count))


if __name__ == "__main__":
    main()
