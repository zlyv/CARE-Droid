#!/usr/bin/env python3

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


REGRESSION = Path(
    "artifacts/p53-top5-correction-stratification/"
    "p53_regression_origin.csv"
)

CACHE = Path(
    "artifacts/p3-a3-mvp-tensor-cache/"
    "a3_mvp_tensor_cache.jsonl.gz"
)

LINEAGE = Path(
    "artifacts/p0-lineage/"
    "package_certificate_lineage.csv"
)

MAPPING = Path(
    "configs/protocol/"
    "semantic_mapping_v1.yaml"
)

OUT = Path(
    "artifacts/p53-coarse-error-profile"
)

WANTED_ORIGINS = {
    "COARSE_ALREADY_REGRESSED",
    "COARSE_ALREADY_RESCUED",
}


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_sha(obj):
    if not isinstance(obj, dict):
        return None

    for key in (
        "sha256",
        "sha_256",
        "sha",
        "apk_sha256",
    ):
        value = obj.get(key)
        if isinstance(value, str) and len(value) >= 32:
            return value.lower()

    for key in ("metadata", "meta", "audit"):
        value = obj.get(key)
        if isinstance(value, dict):
            got = get_sha(value)
            if got:
                return got

    return None


def find_payload(obj):
    """
    Locate A3 graph payload without assuming the outer cache wrapper.
    """
    queue = [obj]

    while queue:
        current = queue.pop(0)

        if not isinstance(current, dict):
            continue

        if (
            isinstance(current.get("node_base_features"), list)
            and isinstance(current.get("node_role_ids"), list)
            and isinstance(current.get("edge_features"), list)
        ):
            return current

        for value in current.values():
            if isinstance(value, dict):
                queue.append(value)

    return None


def load_role_names(path):
    roles = []
    in_roles = False

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "roles:":
            in_roles = True
            continue

        if in_roles and line.startswith("  - "):
            roles.append(line.split("-", 1)[1].strip())
            continue

        if in_roles and line and not line.startswith(" "):
            break

    return sorted(set(roles))


def safe_ratio(a, b):
    return a / b if b else 0.0


def profile_payload(payload, role_names):
    base = payload.get("node_base_features", [])
    role_ids = payload.get("node_role_ids", [])
    edge_features = payload.get("edge_features", [])

    n = len(base)
    e = len(edge_features)

    external = 0
    mapped_flag = 0
    entries = 0
    tier_values = []

    for row in base:
        if len(row) >= 1 and f(row[0]) > 0.5:
            external += 1

        if len(row) >= 2 and f(row[1]) > 0.5:
            mapped_flag += 1

        if len(row) >= 3:
            tier_values.append(f(row[2]))

        if len(row) >= 4 and f(row[3]) > 0.5:
            entries += 1

    mapped_by_role = sum(
        1 for ids in role_ids
        if isinstance(ids, list) and len(ids) > 0
    )

    entry_mapped = 0

    for i in range(min(len(base), len(role_ids))):
        is_entry = (
            len(base[i]) >= 4
            and f(base[i][3]) > 0.5
        )

        has_role = (
            isinstance(role_ids[i], list)
            and len(role_ids[i]) > 0
        )

        if is_entry and has_role:
            entry_mapped += 1

    role_counter = Counter()

    for ids in role_ids:
        if not isinstance(ids, list):
            continue

        for rid in ids:
            try:
                rid = int(rid)
            except Exception:
                continue

            if 0 <= rid < len(role_names):
                role_counter[role_names[rid]] += 1
            else:
                role_counter[f"ROLE_ID_{rid}"] += 1

    icc = 0
    reflection = 0
    nondirect = 0

    for row in edge_features:
        if len(row) >= 1 and f(row[0]) > 0.5:
            icc += 1

        if len(row) >= 2 and f(row[1]) > 0.5:
            reflection += 1

        if len(row) >= 3 and f(row[2]) > 0.5:
            nondirect += 1

    role_incidence = sum(role_counter.values())

    return {
        "node_count": n,
        "edge_count": e,

        "external_node_count": external,
        "mapped_flag_node_count": mapped_flag,
        "mapped_role_node_count": mapped_by_role,
        "entry_node_count": entries,
        "entry_mapped_node_count": entry_mapped,

        "distinct_role_count": len(role_counter),
        "role_incidence_count": role_incidence,

        "entry_ratio": safe_ratio(entries, n),
        "mapped_ratio": safe_ratio(mapped_by_role, n),
        "external_ratio": safe_ratio(external, n),

        "entry_mapped_ratio_of_entries":
            safe_ratio(entry_mapped, entries),

        "edges_per_node":
            safe_ratio(e, n),

        "nodes_per_entry":
            safe_ratio(n, entries),

        "edges_per_entry":
            safe_ratio(e, entries),

        "mean_tier_normalized":
            mean(tier_values) if tier_values else 0.0,

        "icc_edge_ratio":
            safe_ratio(icc, e),

        "reflection_edge_ratio":
            safe_ratio(reflection, e),

        "nondirect_edge_ratio":
            safe_ratio(nondirect, e),

        "role_counts_json":
            json.dumps(
                dict(role_counter.most_common()),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
    }


def detect_sha_column(fieldnames):
    candidates = {
        "sha256",
        "sha_256",
        "sha",
        "apk_sha256",
    }

    for name in fieldnames or []:
        if name.lower() in candidates:
            return name

    return None


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    role_names = load_role_names(MAPPING)

    print("role count:", len(role_names))

    # ---------------------------------------------------------
    # Load only case identities / probabilities.
    # ---------------------------------------------------------
    cases = {}

    with REGRESSION.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["origin"] not in WANTED_ORIGINS:
                continue

            sha = row["sha256"].lower()

            cases[sha] = {
                "sha256": sha,
                "split_name": row["split_name"],
                "year": row["year"],
                "label": int(row["label"]),
                "origin": row["origin"],
                "top5_count": int(row["top5_count"]),
                "top5_roles": row["top5_roles"],

                "compact_probability":
                    f(row["compact_probability"]),

                "coarse_probability":
                    f(row["coarse_probability"]),

                "p53_probability":
                    f(row["p53_probability"]),

                "compact_to_coarse_delta":
                    f(row["compact_to_coarse_delta"]),
            }

    target_shas = set(cases)

    print("target cases:", len(target_shas))

    # ---------------------------------------------------------
    # Stream cache. Keep only numeric profiles.
    # ---------------------------------------------------------
    found_cache = set()

    with gzip.open(
        CACHE, "rt", encoding="utf-8"
    ) as handle:

        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue

            obj = json.loads(line)

            sha = get_sha(obj)
            if sha not in target_shas:
                continue

            payload = find_payload(obj)

            if payload is None:
                print(
                    "[warning] payload not found:",
                    sha,
                )
                continue

            cases[sha].update(
                profile_payload(payload, role_names)
            )

            found_cache.add(sha)

            if len(found_cache) == len(target_shas):
                break

    print(
        "cache profiles:",
        len(found_cache),
        "/",
        len(target_shas),
    )

    # ---------------------------------------------------------
    # Stream lineage, attach metadata without knowing schema.
    # ---------------------------------------------------------
    lineage_rows = {}

    if LINEAGE.exists():
        with LINEAGE.open(
            "r", encoding="utf-8", newline=""
        ) as handle:

            reader = csv.DictReader(handle)
            sha_col = detect_sha_column(
                reader.fieldnames
            )

            if sha_col:
                for row in reader:
                    sha = row.get(
                        sha_col, ""
                    ).lower()

                    if sha in target_shas:
                        lineage_rows[sha] = row

    # ---------------------------------------------------------
    # Write full profile.
    # ---------------------------------------------------------
    rows = []

    all_profile_keys = set()

    for sha, case in cases.items():
        r = dict(case)

        meta = lineage_rows.get(sha, {})

        # Only attach useful human-readable metadata columns.
        for key, value in meta.items():
            low = key.lower()

            if any(token in low for token in (
                "package",
                "family",
                "cert",
                "year",
                "label",
                "path",
                "market",
                "source",
                "status",
            )):
                r[f"lineage__{key}"] = value

        rows.append(r)
        all_profile_keys.update(r)

    fixed = [
        "sha256",
        "year",
        "split_name",
        "label",
        "origin",
        "compact_probability",
        "coarse_probability",
        "p53_probability",
        "compact_to_coarse_delta",
        "top5_count",
        "top5_roles",

        "node_count",
        "edge_count",
        "entry_node_count",
        "mapped_role_node_count",
        "external_node_count",

        "distinct_role_count",
        "role_incidence_count",

        "entry_ratio",
        "mapped_ratio",
        "external_ratio",

        "entry_mapped_node_count",
        "entry_mapped_ratio_of_entries",

        "edges_per_node",
        "nodes_per_entry",
        "edges_per_entry",

        "mean_tier_normalized",

        "icc_edge_ratio",
        "reflection_edge_ratio",
        "nondirect_edge_ratio",

        "role_counts_json",
    ]

    extras = sorted(
        key
        for key in all_profile_keys
        if key not in fixed
    )

    fields = fixed + extras

    profile_path = OUT / "coarse_case_profiles.csv"

    with profile_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda x: (
                x["origin"],
                x["label"],
                -abs(x["compact_to_coarse_delta"]),
            ),
        ):
            writer.writerow(row)

    # ---------------------------------------------------------
    # Compare REGRESSED vs RESCUED within the same true class.
    # ---------------------------------------------------------
    numeric_features = [
        "node_count",
        "edge_count",
        "entry_node_count",
        "mapped_role_node_count",
        "external_node_count",
        "distinct_role_count",
        "role_incidence_count",

        "entry_ratio",
        "mapped_ratio",
        "external_ratio",

        "entry_mapped_node_count",
        "entry_mapped_ratio_of_entries",

        "edges_per_node",
        "nodes_per_entry",
        "edges_per_entry",

        "mean_tier_normalized",

        "icc_edge_ratio",
        "reflection_edge_ratio",
        "nondirect_edge_ratio",
    ]

    groups = defaultdict(list)

    for row in rows:
        groups[
            (row["origin"], row["label"])
        ].append(row)

    summary_rows = []

    for label in (0, 1):
        for feature in numeric_features:

            reg = [
                f(r.get(feature), math.nan)
                for r in groups.get(
                    ("COARSE_ALREADY_REGRESSED", label),
                    []
                )
            ]

            res = [
                f(r.get(feature), math.nan)
                for r in groups.get(
                    ("COARSE_ALREADY_RESCUED", label),
                    []
                )
            ]

            reg = [
                x for x in reg
                if not math.isnan(x)
            ]

            res = [
                x for x in res
                if not math.isnan(x)
            ]

            if not reg or not res:
                continue

            reg_mean = mean(reg)
            res_mean = mean(res)

            summary_rows.append({
                "label": label,
                "class_name":
                    "BENIGN" if label == 0
                    else "MALWARE",

                "feature": feature,

                "regressed_n": len(reg),
                "rescued_n": len(res),

                "regressed_mean": reg_mean,
                "rescued_mean": res_mean,

                "regressed_median": median(reg),
                "rescued_median": median(res),

                "mean_difference_regressed_minus_rescued":
                    reg_mean - res_mean,
            })

    summary_path = (
        OUT /
        "coarse_regressed_vs_rescued_summary.csv"
    )

    with summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:

        fields2 = [
            "label",
            "class_name",
            "feature",
            "regressed_n",
            "rescued_n",
            "regressed_mean",
            "rescued_mean",
            "regressed_median",
            "rescued_median",
            "mean_difference_regressed_minus_rescued",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields2,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # ---------------------------------------------------------
    # Console: strongest benign regressions.
    # ---------------------------------------------------------
    print()
    print("=== STRONGEST BENIGN COARSE REGRESSIONS ===")

    bad_benign = [
        r for r in rows
        if r["origin"] == "COARSE_ALREADY_REGRESSED"
        and r["label"] == 0
    ]

    bad_benign.sort(
        key=lambda r:
            abs(r["compact_to_coarse_delta"]),
        reverse=True,
    )

    for r in bad_benign[:12]:
        print(
            r["sha256"][:16],
            f"C={r['compact_probability']:.3f}",
            f"G={r['coarse_probability']:.3f}",
            f"d={r['compact_to_coarse_delta']:+.3f}",
            f"N={r.get('node_count','?')}",
            f"E={r.get('edge_count','?')}",
            f"entry={r.get('entry_node_count','?')}",
            f"roles={r.get('distinct_role_count','?')}",
            f"mapped_ratio={fmt(r.get('mapped_ratio','?'))}",
            f"entry_ratio={fmt(r.get('entry_ratio','?'))}",
        )

    print()
    print("=== BENIGN: REGRESSED VS RESCUED ===")

    for row in summary_rows:
        if row["label"] != 0:
            continue

        print(
            f"{row['feature']:34s} "
            f"reg={row['regressed_mean']:.4f} "
            f"rescue={row['rescued_mean']:.4f} "
            f"diff={row['mean_difference_regressed_minus_rescued']:+.4f}"
        )

    print()
    print("wrote:")
    print(" ", profile_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
