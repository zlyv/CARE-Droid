#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from common import COHORTS, DATASET, EXPECTED_N, FORMAL_DIAG, OUTPUT, SOURCE_P31, dump_json, norm, read_json, sha256_file, write_csv


def known_ids() -> tuple[set[str], dict[str, set[str]]]:
    by_scope: dict[str, set[str]] = {}
    with np.load(SOURCE_P31, allow_pickle=False) as z:
        by_scope["care_2017"] = {norm(x) for x in z["sha256"]}
    resolved = read_json(FORMAL_DIAG / "resolved_inputs.json")
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as z:
        ids = [norm(x) for x in z["sha256"]]
        years = np.asarray(z["year"], int)
    for year in sorted(set(years.tolist())):
        by_scope[f"care_{year}"] = {sid for sid, value in zip(ids, years) if int(value) == year}
    union: set[str] = set()
    for values in by_scope.values():
        union.update(values)
    return union, by_scope


def main():
    root = DATASET / "extracted/apk"
    if not root.is_dir():
        raise RuntimeError("BLOCKED_BY_CONTRACT: MalEval APK root missing: " + str(root))
    known, scopes = known_ids(); rows = []; counts = Counter(); filename_mismatch = []
    for cohort, (expected, label) in COHORTS.items():
        paths = sorted((root / cohort).glob("*.apk"))
        if len(paths) != expected:
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: cohort {cohort} expected={expected} actual={len(paths)}")
        for path in paths:
            actual = sha256_file(path); named = norm(path.stem)
            if named != actual:
                filename_mismatch.append({"path": str(path), "filename_sha": named, "actual_sha": actual})
            overlap_scopes = [scope for scope, values in scopes.items() if actual in values]
            rows.append({"sha256": actual, "absolute_path": str(path.resolve()), "cohort": cohort,
                         "label": label, "source_id": "MalEval2026", "metadata_year": "",
                         "filename_sha_match": int(named == actual), "care_overlap": int(actual in known),
                         "care_overlap_scopes": ";".join(overlap_scopes)})
            counts[cohort] += 1
    if len(rows) != EXPECTED_N or len({row["sha256"] for row in rows}) != EXPECTED_N:
        raise RuntimeError("BLOCKED_BY_CONTRACT: MalEval SHA uniqueness/count changed")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "dataset_manifest_label_sealed.csv", rows)
    label_free = [{key: value for key, value in row.items() if key != "label"} for row in rows]
    write_csv(OUTPUT / "dataset_manifest_label_free.csv", label_free)
    audit = {"status": "PASS", "rows": len(rows), "cohort_counts": dict(counts),
             "labels": {"benign": 0, "malradar": 1, "new": 1},
             "filename_sha_mismatch_count": len(filename_mismatch),
             "care_2017_2021_overlap_count": sum(int(row["care_overlap"]) for row in rows),
             "overlap_by_scope": {key: sum(row["sha256"] in values for row in rows) for key, values in scopes.items()},
             "primary_evaluation": "SHA-disjoint rows; full 255 retained as descriptive secondary result"}
    dump_json(OUTPUT / "dataset_audit.json", audit)
    dump_json(OUTPUT / "filename_sha_mismatches.json", filename_mismatch)
    print("MALEVAL2026_MANIFEST_PASS", flush=True)
    print(audit, flush=True)


if __name__ == "__main__":
    main()

