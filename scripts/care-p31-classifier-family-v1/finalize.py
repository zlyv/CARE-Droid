#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import (ARMS, FUTURE_LABELED, FUTURE_MEMBER, OUTPUT, REFERENCE_AUDIT, RESOLVED, SOURCE_MEMBER,
    SOURCE_P31, WITHIN_INPUTS, dump_json, file_record, read_csv, read_json)


def main() -> None:
    source = read_csv(OUTPUT / "source_metrics.csv"); source_fold = read_csv(OUTPUT / "source_fold_metrics.csv")
    frozen = read_csv(OUTPUT / "frozen_future_metrics.csv"); within = read_csv(OUTPUT / "within_year_metrics.csv")
    source_by = {row["arm"]: row for row in source}; rf = source_by["RF"]
    ranking = sorted(ARMS, key=lambda arm: (-float(source_by[arm]["BA"]), -float(source_by[arm]["AUC"]), -float(source_by[arm]["F1"]), arm))
    folds_by = {(row["arm"], int(row["fold"])): row for row in source_fold}
    consistency = {arm: sum(float(folds_by[(arm, fold)]["BA"]) >= float(folds_by[("RF", fold)]["BA"]) for fold in range(1, 6)) for arm in ARMS}
    paired = {row["arm"]: row for row in read_csv(OUTPUT / "source_paired_vs_rf.csv")}
    candidates = [arm for arm in ARMS if arm != "RF" and float(source_by[arm]["BA"]) > float(rf["BA"])
        and int(source_by[arm]["errors"]) < int(rf["errors"]) and consistency[arm] >= 3 and float(paired[arm]["mcnemar_exact_p"]) < .05]
    frozen_pooled = {row["arm"]: row for row in frozen if row["scope"] == "pooled"}
    within_pooled = {row["arm"]: row for row in within if row["scope"] == "pooled"}
    decision = {"status": "COMPARISON_COMPLETE", "source_metric_leader": ranking[0], "source_ranking": ranking,
        "formal_RF": {key: rf[key] for key in ("BA", "AUC", "F1", "errors", "error_recall25")},
        "source_candidates_beating_RF_with_majority_and_p_lt_0.05": candidates,
        "folds_noninferior_to_RF": consistency,
        "frozen_future_diagnostic_ranking_by_BA": sorted(ARMS, key=lambda arm: -float(frozen_pooled[arm]["BA"])),
        "same_year_diagnostic_ranking_by_BA": sorted(ARMS, key=lambda arm: -float(within_pooled[arm]["BA"])),
        "future_used_for_selection": False, "same_year_used_for_selection": False, "formal_stage1_changed": False,
        "interpretation": "Only 2017 fixed-fold OOF is the source comparison domain. Future and same-year rankings are diagnostics."}
    dump_json(OUTPUT / "final_decision.json", decision)
    resolved = read_json(RESOLVED); paths = {"source_member_oof": SOURCE_MEMBER, "source_p31": SOURCE_P31,
        "future_member_label_free": FUTURE_MEMBER, "future_diagnostic_labels": FUTURE_LABELED,
        "formal_resolved_inputs": RESOLVED, "future_reference_audit": REFERENCE_AUDIT,
        "within_2018_2019": WITHIN_INPUTS[0], "within_2020_2021": WITHIN_INPUTS[1],
        "future_p31_cache": Path(resolved["future_p31_cache"])}
    after = [file_record(path, name) for name, path in paths.items()]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json")["assets"]
    left = {row["absolute_path"]: row for row in before}; right = {row["absolute_path"]: row for row in after}
    changed = [path for path in sorted(set(left) | set(right)) if path not in left or path not in right or left[path]["sha256"] != right[path]["sha256"]]
    status = "PASS" if not changed else "FAIL"; dump_json(OUTPUT / "baseline_integrity_check.json", {"status": status, "changed": changed, "asset_count": len(after)})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed")
    print("P31_CLASSIFIER_FAMILY_FINALIZE_PASS integrity=true", flush=True)


if __name__ == "__main__":
    main()

