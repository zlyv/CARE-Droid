#!/usr/bin/env python3
"""Label-free A/B/C feature extraction from the frozen endpoint graph cache."""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_common import FEATURES, PREFLIGHT, norm, read_csv, reconstruct_frozen_graph, extract_candidate_features, write_csv, write_json


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "p90": float(np.percentile(values, 90)) if len(values) else 0.0,
        "max": float(np.max(values)) if len(values) else 0.0,
    }


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREFLIGHT / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "HOST_COMPLEMENT_CANDIDATE_PREFLIGHT_PASS":
        raise RuntimeError("BLOCKED_BY_CONTRACT: preflight did not pass")
    roles = list(manifest["role_names"]); role_count = len(roles)
    remap = np.asarray(manifest["payload_to_model_role"], dtype=int)
    cohort = {norm(row["sha256"]): row for row in read_csv(Path(manifest["cohort_file"]))}
    if len(cohort) != 1989: raise RuntimeError("BLOCKED_BY_CONTRACT: cohort changed after preflight")

    taxonomy = [{"role_id": index, "role_name": name, "taxonomy_source": manifest["local_asset_manifest"]}
                for index, name in enumerate(roles)]
    write_csv(FEATURES / "actual_role_taxonomy.csv", taxonomy)
    # Kept in reports as well because this is an interpretation-facing artifact.
    write_csv(FEATURES.parent / "reports" / "actual_role_taxonomy.csv", taxonomy)

    pair_names = [f"A_pair__{left:03d}__{right:03d}" for left in range(role_count) for right in range(left + 1, role_count)]
    transition_names = [f"C_gap{gap}__{left:03d}__{right:03d}"
                        for gap in (0, 1, 2) for left in range(role_count) for right in range(role_count)]
    output_path = FEATURES / "candidate_features_summary.csv"
    writer = handle = None
    group_names = None
    counts = defaultdict(int); cells = defaultdict(int); nonzero = defaultdict(int)
    relation_values = {key: [] for key in ("direct_relation_count", "gap1_relation_count", "gap2_relation_count")}
    host_counts = []
    structural_raw = []
    seen = set()
    try:
        with gzip.open(Path(manifest["endpoint_cache"]), "rt", encoding="utf-8") as source:
            for index, line in enumerate(source, 1):
                if not line.strip(): continue
                record = json.loads(line); sample = norm(record.get("sha256"))
                if sample not in cohort or sample in seen:
                    raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint/cohort lineage mismatch or duplicate")
                seen.add(sample)
                graph = reconstruct_frozen_graph(record.get("payload") or {}, remap)
                a, b, c, pair, transition, audit = extract_candidate_features(graph, role_count)
                if group_names is None:
                    group_names = {"A_summary": list(a), "B_summary": list(b), "C_summary": list(c),
                                   "A_pair": pair_names, "C_transition": transition_names}
                    fields = [
                        "sha256", "year", "fold", "host_count", "method_count",
                        "graph_node_count", "graph_edge_count", "ordinary_count",
                        "endpoint_count", "direct_relation_count", "gap1_relation_count",
                        "gap2_relation_count",
                    ]
                    fields += group_names["A_summary"] + group_names["B_summary"] + group_names["C_summary"] + pair_names + transition_names
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    handle = output_path.open("w", encoding="utf-8", newline="")
                    writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                meta = cohort[sample]
                row = {"sha256": sample, "year": int(float(meta["year"])), "fold": int(float(meta["fold"])),
                       "host_count": audit["host_count"], "method_count": audit["method_count"],
                       "graph_node_count": audit["graph_node_count"], "graph_edge_count": audit["graph_edge_count"],
                       "ordinary_count": audit["ordinary_node_count"], "endpoint_count": audit["endpoint_node_count"],
                       "direct_relation_count": audit["direct_relation_count"],
                       "gap1_relation_count": audit["gap1_relation_count"],
                       "gap2_relation_count": audit["gap2_relation_count"], **a, **b, **c}
                row.update({name: float(value) for name, value in zip(pair_names, pair)})
                row.update({name: float(value) for name, value in zip(transition_names, transition)})
                writer.writerow(row)
                for group, names in group_names.items():
                    values = np.asarray([row[name] for name in names], dtype=float)
                    cells[group] += len(values); nonzero[group] += int(np.count_nonzero(values)); counts[group] += int(np.any(values != 0))
                for key in relation_values: relation_values[key].append(audit[key])
                structural_raw.extend(audit["raw_structure_rows"])
                host_counts.append(audit["host_count"])
                if index % 100 == 0: print(f"candidate feature extraction {index}/1989", flush=True)
    finally:
        if handle is not None: handle.close()
    if seen != set(cohort) or len(seen) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: feature extraction did not cover exact 1989 cohort")

    schema = {
        "status": "LABEL_FREE_FROZEN_ARTIFACT_FEATURE_SCHEMA",
        "role_count": role_count, "role_names": roles,
        "candidate_groups": group_names,
        "fixed_primary_sets": {
            "A": ["A_summary"], "B": ["B_summary"], "C": ["C_summary"],
            "AB": ["A_summary", "B_summary"], "AC": ["A_summary", "C_summary"],
            "BC": ["B_summary", "C_summary"], "ABC": ["A_summary", "B_summary", "C_summary"],
        },
        "fixed_secondary_sets": {"A_pair": ["A_pair"], "C_transition": ["C_transition"]},
        "ordinary_method_definition": "node is neither a Host nor a sensitive endpoint",
        "host_definition": "caller with a direct frozen edge to a retained sensitive endpoint carrying at least one mapped role",
        "coordination_definition": "gap0 Host->Host; gap1 Host->Ordinary->Host; gap2 Host->Ordinary->Ordinary->Host",
        "coordination_dedup_key": "(src_host,dst_host,gap)", "maximum_ordinary_intermediates": 2,
        "labels_used_for_feature_construction": False,
    }
    write_json(FEATURES / "candidate_feature_schema.json", schema)

    coverage = []
    for group, names in group_names.items():
        coverage.append({"candidate": group, "feature_dimension": len(names),
                         "nonzero_feature_fraction": nonzero[group] / cells[group] if cells[group] else 0.0,
                         "APK_coverage": counts[group] / 1989, "host_coverage": float(np.mean(np.asarray(host_counts) > 0))})
    direct, gap1, gap2 = [relation_values[key] for key in relation_values]
    coverage.append({"candidate": "C_relation_audit", "feature_dimension": len(group_names["C_summary"]),
                     "total_direct_relations": int(sum(direct)), "total_gap1_relations": int(sum(gap1)), "total_gap2_relations": int(sum(gap2)),
                     **{f"direct_{key}": value for key, value in percentile_summary(direct).items()},
                     **{f"gap1_{key}": value for key, value in percentile_summary(gap1).items()},
                     **{f"gap2_{key}": value for key, value in percentile_summary(gap2).items()},
                     "gap1_APK_coverage": float(np.mean(np.asarray(gap1) > 0)), "gap2_APK_coverage": float(np.mean(np.asarray(gap2) > 0))})
    write_csv(FEATURES / "candidate_feature_coverage.csv", coverage)
    if structural_raw:
        array = np.asarray(structural_raw, dtype=float)
        names = ("ordinary_predecessor_count", "ordinary_successor_count", "host_predecessor_count", "host_successor_count",
                 "all_context_predecessor_count", "all_context_successor_count", "upstream_reach_le2", "downstream_reach_le2")
        write_csv(FEATURES / "host_structural_raw_distribution.csv", [
            {"quantity": name, **percentile_summary(array[:, index])} for index, name in enumerate(names)])
    print("HOST_COMPLEMENT_CANDIDATE_FEATURE_EXTRACTION_DONE")


if __name__ == "__main__":
    main()
