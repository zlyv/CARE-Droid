#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from common import COHORTS, FORMAL_DIAG, OUTPUT, SOURCE_P31, dump_json, norm, read_json, write_csv


def known_ids() -> tuple[set[str], dict[str, set[str]]]:
    by_scope: dict[str, set[str]] = {}
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        by_scope["care_2017"] = {norm(x) for x in archive["sha256"]}
    resolved = read_json(FORMAL_DIAG / "resolved_inputs.json")
    with np.load(resolved["future_p31_cache"], allow_pickle=False) as archive:
        ids = [norm(x) for x in archive["sha256"]]
        years = np.asarray(archive["year"], int)
    for year in sorted(set(years.tolist())):
        by_scope[f"care_{year}"] = {sid for sid, value in zip(ids, years) if int(value) == year}
    return set().union(*by_scope.values()), by_scope


def checkpoint_rows():
    rows = []
    for cohort in COHORTS:
        path = OUTPUT / "checkpoints" / f"extracted_{cohort}.jsonl"
        if not path.is_file():
            raise RuntimeError("BLOCKED_BY_CONTRACT: extraction checkpoint missing " + str(path))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    known, scopes = known_ids(); extracted = checkpoint_rows(); by_sha = defaultdict(list)
    for row in extracted:
        by_sha[norm(row["sha256"])].append(row)
    conflicts = []; duplicates = []; rows = []
    for sid, group in sorted(by_sha.items()):
        labels = {int(row["label"]) for row in group}
        cohorts = sorted({row["cohort"] for row in group})
        if len(labels) > 1:
            conflicts.append({"sha256": sid, "cohorts": cohorts, "labels": sorted(labels), "rows": group})
            continue
        canonical = sorted(group, key=lambda row: (row["cohort"], row["member"]))[0]
        if len(group) > 1:
            duplicates.append({"sha256": sid, "cohorts": cohorts, "copies": len(group)})
        overlap_scopes = [scope for scope, values in scopes.items() if sid in values]
        rows.append({"sha256": sid, "absolute_path": canonical["absolute_path"],
                     "cohort": canonical["cohort"], "label": int(canonical["label"]),
                     "source_id": "CICMalDroid2020_user_archive_subset", "metadata_year": "",
                     "archive_member": canonical["member"], "duplicate_copies": len(group),
                     "care_overlap": int(sid in known), "care_overlap_scopes": ";".join(overlap_scopes),
                     "evaluation_included": int(sid not in known)})
    primary = [row for row in rows if int(row["evaluation_included"])]
    if not primary or len({row["label"] for row in primary}) != 2:
        raise RuntimeError("BLOCKED_BY_CONTRACT: primary SHA-disjoint cohort lacks both binary labels")
    write_csv(OUTPUT / "dataset_manifest_all_label_sealed.csv", rows)
    write_csv(OUTPUT / "dataset_manifest_label_sealed.csv", primary)
    write_csv(OUTPUT / "dataset_manifest_label_free.csv",
              [{key: value for key, value in row.items() if key != "label"} for row in primary])
    write_csv(OUTPUT / "duplicate_sha_audit.csv", duplicates)
    dump_json(OUTPUT / "cross_label_conflicts.json", conflicts)
    audit = {"status": "PASS", "archive_member_rows": len(extracted), "unique_sha_rows": len(rows),
             "primary_sha_disjoint_rows": len(primary), "cross_label_conflicts_excluded": len(conflicts),
             "same_label_duplicate_sha": len(duplicates), "care_2017_2021_overlap_excluded": len(rows)-len(primary),
             "cohort_counts_primary": dict(Counter(row["cohort"] for row in primary)),
             "sampling": "deterministic uniform reservoir sample within cohort, capped at 500 before APK feature extraction",
             "sampling_used_labels_beyond_archive_cohort": False,
             "binary_label_definition": {"Benign": 0, "Banking": 1, "Riskware": 1},
             "strict_secondary_scope": "Benign versus Banking; Riskware excluded",
             "collection_limit": "SMS was not supplied and the incomplete Adware archive was explicitly skipped; results are not the complete CICMalDroid2020 benchmark",
             "overlap_by_scope": {key: sum(row["sha256"] in values for row in rows) for key, values in scopes.items()},
             "primary_evaluation": "SHA-unique, CARE-2017-2021-disjoint rows"}
    dump_json(OUTPUT / "dataset_audit.json", audit)
    print("CICMALDROID2020_MANIFEST_PASS", audit, flush=True)


if __name__ == "__main__":
    main()
