#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from future_common import (DIRECT_OUTPUT, FAMILY_OUTPUT, FORMAL_DIAG, OUTPUT, YEARS,
    dump_json, file_record, mcnemar, metrics, norm, read_csv, write_csv)


def aligned_rows(path, labels):
    rows = read_csv(path); result = []
    for row in rows:
        sid = norm(row["sha256"])
        if sid not in labels:
            raise RuntimeError("BLOCKED_BY_CONTRACT: label alignment missing " + sid)
        result.append({**row, "sha256": sid, "label": labels[sid]})
    return result


def main():
    labels = {norm(row["sha256"]): int(row["label"]) for row in read_csv(FORMAL_DIAG / "frozen_predictions_with_labels.csv")}
    direct = aligned_rows(OUTPUT / "predictions_label_free.csv", labels)
    bounded = aligned_rows(FAMILY_OUTPUT / "frozen/SVM/predictions_label_free.csv", labels)
    if len(direct) != 1989 or len(bounded) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future prediction count changed")
    bounded_by = {row["sha256"]: row for row in bounded}; ordered_bounded = [bounded_by[row["sha256"]] for row in direct]
    if len(bounded_by) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate future SHA")
    base = np.asarray([float(row["stage1_prob"]) for row in direct]); route = np.asarray([int(row["routed"]) for row in direct], bool)
    bounded_base = np.asarray([float(row["stage1_prob"]) for row in ordered_bounded])
    bounded_route = np.asarray([int(row["routed"]) for row in ordered_bounded], bool)
    if not np.allclose(base, bounded_base, atol=1e-12) or not np.array_equal(route, bounded_route):
        raise RuntimeError("BLOCKED_BY_CONTRACT: SVM Stage-I/router differs from bounded comparator")
    y = np.asarray([int(row["label"]) for row in direct]); year = np.asarray([int(row["year"]) for row in direct])
    direct_prob = np.asarray([float(row["final_prob"]) for row in direct])
    bounded_prob = np.asarray([float(row["stage2_prob"]) for row in ordered_bounded])
    rows = []; comparisons = []
    for scope, selected in [(str(current), year == current) for current in YEARS] + [("pooled", np.ones(len(y), bool))]:
        for arm, final in (("S0_STAGE1_SVM", base), ("B0_BOUNDED_C43", bounded_prob), ("D0_DIRECT_C43", direct_prob)):
            rows.append({"scope": scope, "arm": arm, **metrics(y[selected], base[selected], final[selected], route[selected])})
        direct_ok = (direct_prob[selected] >= .5) == y[selected]
        bounded_ok = (bounded_prob[selected] >= .5) == y[selected]
        candidate_only, bounded_only, p = mcnemar(direct_ok, bounded_ok)
        comparisons.append({"scope": scope, "direct_only_correct": candidate_only,
            "bounded_only_correct": bounded_only, "paired_net_direct_vs_bounded": candidate_only - bounded_only,
            "mcnemar_exact_p": p})
    write_csv(OUTPUT / "future_metrics.csv", rows); write_csv(OUTPUT / "direct_vs_bounded.csv", comparisons)

    source_rows = read_csv(DIRECT_OUTPUT / "source_results.csv")
    source = next(row for row in source_rows if row["arm"] == "U2_DIRECT_STAGE2")
    annual = {int(scope): next(row for row in rows if row["scope"] == str(scope) and row["arm"] == "D0_DIRECT_C43") for scope in YEARS}
    pooled = next(row for row in rows if row["scope"] == "pooled" and row["arm"] == "D0_DIRECT_C43")
    pooled_stage1 = next(row for row in rows if row["scope"] == "pooled" and row["arm"] == "S0_STAGE1_SVM")
    pooled_bounded = next(row for row in rows if row["scope"] == "pooled" and row["arm"] == "B0_BOUNDED_C43")
    positive_years = [year_value for year_value, row in annual.items() if int(row["net"]) > 0]
    negative_years = [year_value for year_value, row in annual.items() if int(row["net"]) < 0]
    status = "DIRECT_C43_TEMPORAL_NET_POSITIVE_ALL_YEARS" if len(positive_years) == 4 else "DIRECT_C43_TEMPORAL_MIXED"
    decision = {"status": status, "source_direct": {key: source[key] for key in ("BA", "AUC", "F1", "fixed", "broken", "net")},
        "future_pooled_stage1": pooled_stage1, "future_pooled_bounded": pooled_bounded,
        "future_pooled_direct": pooled, "positive_net_years": positive_years, "negative_net_years": negative_years,
        "future_used_for_selection": False, "threshold_changed": False, "model_changed": False,
        "interpretation": "post-freeze temporal diagnostic; results cannot select threshold, architecture, or route budget"}
    dump_json(OUTPUT / "final_decision.json", decision)

    before = json.loads((OUTPUT / "baseline_asset_manifest_before.json").read_text(encoding="utf-8"))
    after = []; changed = []
    for old in before["assets"]:
        current = file_record(Path(old["absolute_path"]), old["purpose"]); after.append(current)
        if current["sha256"] != old["sha256"] or current["size_bytes"] != old["size_bytes"]:
            changed.append({"path": old["absolute_path"], "before": old["sha256"], "after": current["sha256"]})
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after})
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL",
        "asset_count": len(after), "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal assets changed")

    lines = ["# CARE-Droid SVM + Direct-C43 frozen future diagnostic v1", "",
        "The 2017 Direct-C43 definition, SVM-anchored Top25% router, threshold 0.5 and training settings were frozen before labels were loaded.",
        "Future labels are used only by this summary step. No future result may change the method.", "",
        "## Frozen source reference", "",
        f"- 2017 Direct-C43: BA={float(source['BA']):.6f}, fixed={source['fixed']}, broken={source['broken']}, net={source['net']}.", "",
        "## 2017-trained frozen temporal diagnostic", "",
        "| Year | Stage-I BA | bounded BA | Direct BA | Direct gain pp | fixed | broken | net | FN fixed | FP fixed |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scope in [str(value) for value in YEARS] + ["pooled"]:
        stage1 = next(row for row in rows if row["scope"] == scope and row["arm"] == "S0_STAGE1_SVM")
        bounded_row = next(row for row in rows if row["scope"] == scope and row["arm"] == "B0_BOUNDED_C43")
        direct_row = next(row for row in rows if row["scope"] == scope and row["arm"] == "D0_DIRECT_C43")
        gain = 100 * (float(direct_row["BA"]) - float(stage1["BA"]))
        lines.append(f"| {scope} | {float(stage1['BA']):.6f} | {float(bounded_row['BA']):.6f} | {float(direct_row['BA']):.6f} | {gain:+.4f} | {direct_row['fixed']} | {direct_row['broken']} | {direct_row['net']} | {direct_row['fixed_FN']} | {direct_row['fixed_FP']} |")
    lines += ["", "## Decision boundary", "", f"**{status}**", "",
        f"Positive-net years: {positive_years}; negative-net years: {negative_years}.",
        "This diagnostic does not authorize retrospective threshold or route changes.",
        "Formal asset integrity: PASS.", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("SVM_DIRECT_C43_FUTURE_SUMMARY_DONE status=" + status)


if __name__ == "__main__":
    main()
