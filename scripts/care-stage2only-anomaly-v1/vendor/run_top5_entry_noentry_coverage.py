#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path


TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

MAX_HOPS = 4

CACHE_PATH = Path(
    "artifacts/p3-a3-mvp-tensor-cache/"
    "a3_mvp_tensor_cache.jsonl.gz"
)

UPSTREAM_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

P53_APK_PATH = Path(
    "artifacts/p53-top5-correction-stratification/"
    "p53_top5_apk_level.csv"
)

NOENTRY_PAIRED_PATH = Path(
    "artifacts/top5-upstream-refinement/"
    "top5_upstream_paired_apk.csv"
)

MAPPING_PATH = Path(
    "configs/protocol/semantic_mapping_v1.yaml"
)

OUTPUT = Path(
    "artifacts/top5-entry-noentry-coverage"
)


def key(x):
    return str(x).strip().casefold()


def parse_present_roles(raw):
    s = str(raw or "").strip()

    if not s:
        return []

    # JSON list if present.
    if s.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj]
        except Exception:
            pass

    parts = re.split(r"[|,; ]+", s)

    return [
        x.strip()
        for x in parts
        if x.strip() in TOP5
    ]


def load_role_names(path):
    """
    Match the role-order logic used by the existing Entry evaluation:
    collect the YAML roles list and sort it.
    """
    roles = []
    in_roles = False

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if line.strip() == "roles:":
            in_roles = True
            continue

        if in_roles and line.startswith("  - "):
            roles.append(
                line.split("-", 1)[1].strip()
            )
            continue

        if (
            in_roles
            and line
            and not line.startswith(" ")
        ):
            break

    roles = sorted(set(roles))

    if len(roles) != 31:
        raise RuntimeError(
            f"expected 31 role names, got {len(roles)}: "
            f"{roles}"
        )

    return roles


def normalize_role_ids(value):
    if value is None:
        return []

    if isinstance(value, int):
        return [int(value)]

    if isinstance(value, float):
        return [int(value)]

    if isinstance(value, list):
        out = []

        for x in value:
            try:
                out.append(int(x))
            except Exception:
                pass

        return out

    s = str(value).strip()

    if not s:
        return []

    try:
        obj = json.loads(s)

        if isinstance(obj, list):
            return [
                int(x)
                for x in obj
            ]

        return [int(obj)]

    except Exception:
        pass

    result = []

    for x in re.split(r"[,;| ]+", s):
        if not x:
            continue

        try:
            result.append(int(x))
        except Exception:
            pass

    return result


def normalize_edges(edge_index):
    if not edge_index:
        return []

    # PyG-style:
    # [[src0, src1, ...],
    #  [dst0, dst1, ...]]
    if (
        isinstance(edge_index, list)
        and len(edge_index) == 2
        and isinstance(edge_index[0], list)
        and isinstance(edge_index[1], list)
        and (
            not edge_index[0]
            or not isinstance(
                edge_index[0][0], list
            )
        )
    ):
        return list(
            zip(
                map(int, edge_index[0]),
                map(int, edge_index[1]),
            )
        )

    # Pair-style:
    # [[src,dst], ...]
    result = []

    for pair in edge_index:
        if (
            isinstance(pair, list)
            and len(pair) >= 2
        ):
            result.append(
                (int(pair[0]), int(pair[1]))
            )

    return result


def entry_reach_rates(
    payload,
    role_to_id,
    target_roles,
):
    """
    Reproduce the binary semantics of
    entry_role_reachability_features():

    for each recognized Entry:
      forward BFS <= H4
      check whether at least one node
      of each target role is reachable.

    return:
        role -> fraction of Entry nodes
                reaching >=1 role node
    """
    base = payload.get(
        "node_base_features", []
    )
    node_roles_raw = payload.get(
        "node_role_ids", []
    )
    edge_index = payload.get(
        "edge_index", []
    )

    n = len(base)

    result = {
        role: 0.0
        for role in target_roles
    }

    if n == 0:
        return result

    entries = []

    for i, row in enumerate(base):
        try:
            if float(row[3]) > 0.5:
                entries.append(i)
        except Exception:
            pass

    if not entries:
        return result

    role_sets = []

    for i in range(n):
        raw = (
            node_roles_raw[i]
            if i < len(node_roles_raw)
            else []
        )

        role_sets.append(
            set(normalize_role_ids(raw))
        )

    adjacency = [
        []
        for _ in range(n)
    ]

    for src, dst in normalize_edges(
        edge_index
    ):
        if 0 <= src < n and 0 <= dst < n:
            adjacency[src].append(dst)

    reachable_entry_count = {
        role: 0
        for role in target_roles
    }

    for entry in entries:
        distances = {entry: 0}
        frontier = [entry]

        for hop in range(
            1, MAX_HOPS + 1
        ):
            nxt = []

            for node in frontier:
                for neighbor in adjacency[node]:
                    if neighbor in distances:
                        continue

                    distances[neighbor] = hop
                    nxt.append(neighbor)

            frontier = nxt

            if not frontier:
                break

        reached_nodes = [
            node
            for node in distances
            if node != entry
        ]

        for role in target_roles:
            rid = role_to_id[role]

            if any(
                rid in role_sets[node]
                for node in reached_nodes
            ):
                reachable_entry_count[role] += 1

    for role in target_roles:
        result[role] = (
            reachable_entry_count[role]
            / len(entries)
        )

    return result


def category(entry_context, noentry_context):
    if entry_context and noentry_context:
        return "BOTH"

    if noentry_context:
        return "NOENTRY_ONLY"

    if entry_context:
        return "ENTRY_ONLY"

    return "NEITHER"


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def transition_summary(rows):
    c = Counter(
        r["transition"]
        for r in rows
    )

    corrected = c["CORRECTED"]
    regressed = c["REGRESSED"]

    return {
        "n": len(rows),
        "corrected": corrected,
        "regressed": regressed,
        "net": corrected - regressed,
        "still_correct": c["STILL_CORRECT"],
        "still_wrong": c["STILL_WRONG"],
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------
    # Role vocabulary
    # ---------------------------------------
    role_names = load_role_names(
        MAPPING_PATH
    )

    role_to_id = {
        role: i
        for i, role in enumerate(
            role_names
        )
    }

    print("=== ROLE IDS ===")

    for role in TOP5:
        if role not in role_to_id:
            raise RuntimeError(
                f"Top5 role absent: {role}"
            )

        print(
            f"{role:32s} "
            f"id={role_to_id[role]}"
        )

    # ---------------------------------------
    # Exact 1985 evaluation population
    # ---------------------------------------
    test = {}

    with P53_APK_PATH.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for r in csv.DictReader(f):
            year = str(
                r.get("year", "")
            ).strip()

            if year not in {
                "2018", "2019",
                "2020", "2021",
            }:
                continue

            sha = key(r["sha256"])

            present = parse_present_roles(
                r.get(
                    "present_top5_roles",
                    "",
                )
            )

            test[sha] = {
                "sha256": sha,
                "year": year,
                "label": int(r["label"]),
                "present_roles": present,
            }

    if len(test) != 1985:
        raise RuntimeError(
            f"expected 1985 test APKs, "
            f"got {len(test)}"
        )

    present_apks = [
        r
        for r in test.values()
        if r["present_roles"]
    ]

    print()
    print(
        "test APKs =",
        len(test),
    )
    print(
        "Top5-present APKs =",
        len(present_apks),
    )

    if len(present_apks) != 1126:
        raise RuntimeError(
            "expected 1126 Top5-present APKs"
        )

    # ---------------------------------------
    # Load No-Entry feature rows
    # ---------------------------------------
    upstream = {}

    with gzip.open(
        UPSTREAM_PATH,
        "rt",
        newline="",
        encoding="utf-8",
    ) as f:

        for r in csv.DictReader(f):
            sha = key(r["sha256"])

            if sha in test:
                upstream[sha] = r

    print(
        "No-Entry rows found =",
        len(upstream),
    )

    if len(upstream) != len(test):
        missing = set(test) - set(upstream)

        raise RuntimeError(
            f"missing upstream rows: "
            f"{len(missing)}"
        )

    # ---------------------------------------
    # Load only test rows from Entry cache
    # ---------------------------------------
    entry_cache = {}

    with gzip.open(
        CACHE_PATH,
        "rt",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)
            sha = key(r["sha256"])

            if sha in test:
                entry_cache[sha] = r

    print(
        "Entry cache rows found =",
        len(entry_cache),
    )

    if len(entry_cache) != len(test):
        missing = set(test) - set(
            entry_cache
        )

        raise RuntimeError(
            f"missing Entry cache rows: "
            f"{len(missing)}"
        )

    # ---------------------------------------
    # No-Entry correction transitions
    # ---------------------------------------
    transitions = {}

    with NOENTRY_PAIRED_PATH.open(
        newline="",
        encoding="utf-8",
    ) as f:

        for r in csv.DictReader(f):
            sha = key(r["sha256"])

            if sha in test:
                transitions[sha] = (
                    r["transition"]
                )

    print(
        "paired predictions found =",
        len(transitions),
    )

    if len(transitions) != len(test):
        raise RuntimeError(
            "paired prediction population mismatch"
        )

    # ---------------------------------------
    # Coverage analysis
    # ---------------------------------------
    instance_rows = []
    apk_rows = []

    role_summary = defaultdict(
        Counter
    )
    apk_category_count = Counter()
    year_category = defaultdict(
        Counter
    )

    for sha, info in sorted(
        test.items()
    ):
        present_roles = (
            info["present_roles"]
        )

        if not present_roles:
            continue

        payload = entry_cache[
            sha
        ]["payload"]

        reach_rates = (
            entry_reach_rates(
                payload,
                role_to_id,
                present_roles,
            )
        )

        entry_by_role = {}
        noentry_by_role = {}

        for role in present_roles:
            entry_available = (
                reach_rates[role] > 0.0
            )

            caller_log = float(
                upstream[sha].get(
                    f"up_role__{role}"
                    "__caller_log",
                    0,
                )
                or 0
            )

            noentry_available = (
                caller_log > 0.0
            )

            entry_by_role[role] = (
                entry_available
            )
            noentry_by_role[role] = (
                noentry_available
            )

            cat = category(
                entry_available,
                noentry_available,
            )

            role_summary[
                role
            ]["PRESENT"] += 1

            role_summary[
                role
            ][cat] += 1

            instance_rows.append({
                "sha256": sha,
                "year": info["year"],
                "label": info["label"],
                "role": role,
                "entry_reach_rate": (
                    reach_rates[role]
                ),
                "entry_context": int(
                    entry_available
                ),
                "noentry_caller_log": (
                    caller_log
                ),
                "noentry_context": int(
                    noentry_available
                ),
                "category": cat,
                "transition": (
                    transitions[sha]
                ),
            })

        entry_any = any(
            entry_by_role.values()
        )

        noentry_any = any(
            noentry_by_role.values()
        )

        apk_cat = category(
            entry_any,
            noentry_any,
        )

        # Important:
        # this is more informative than
        # APK-level BOTH/ONLY.
        # An APK may have:
        # role A = BOTH
        # role B = NOENTRY_ONLY.
        has_noentry_only_role = any(
            (
                not entry_by_role[role]
                and noentry_by_role[role]
            )
            for role in present_roles
        )

        has_entry_only_role = any(
            (
                entry_by_role[role]
                and not noentry_by_role[role]
            )
            for role in present_roles
        )

        entry_all_present = all(
            entry_by_role.values()
        )

        noentry_all_present = all(
            noentry_by_role.values()
        )

        apk_category_count[
            apk_cat
        ] += 1

        year_category[
            info["year"]
        ][apk_cat] += 1

        apk_rows.append({
            "sha256": sha,
            "year": info["year"],
            "label": info["label"],
            "present_role_count": (
                len(present_roles)
            ),
            "present_roles": "|".join(
                present_roles
            ),
            "entry_any_context": int(
                entry_any
            ),
            "noentry_any_context": int(
                noentry_any
            ),
            "apk_category": apk_cat,
            "entry_all_present_roles": int(
                entry_all_present
            ),
            "noentry_all_present_roles": int(
                noentry_all_present
            ),
            "has_noentry_only_role": int(
                has_noentry_only_role
            ),
            "has_entry_only_role": int(
                has_entry_only_role
            ),
            "transition": transitions[
                sha
            ],
        })

    # ---------------------------------------
    # Summary
    # ---------------------------------------
    total_instances = len(
        instance_rows
    )

    if total_instances != 3365:
        raise RuntimeError(
            f"expected 3365 Top5 role instances, "
            f"got {total_instances}"
        )

    instance_cat = Counter(
        r["category"]
        for r in instance_rows
    )

    entry_instance_n = sum(
        r["entry_context"]
        for r in instance_rows
    )

    noentry_instance_n = sum(
        r["noentry_context"]
        for r in instance_rows
    )

    entry_apk_n = sum(
        r["entry_any_context"]
        for r in apk_rows
    )

    noentry_apk_n = sum(
        r["noentry_any_context"]
        for r in apk_rows
    )

    print()
    print(
        "=================================="
    )
    print(
        "ROLE-INSTANCE COVERAGE"
    )
    print(
        "=================================="
    )

    print(
        "present instances =",
        total_instances,
    )

    print(
        "Entry context      =",
        entry_instance_n,
        f"({entry_instance_n / total_instances:.2%})",
    )

    print(
        "No-Entry context   =",
        noentry_instance_n,
        f"({noentry_instance_n / total_instances:.2%})",
    )

    print()

    for name in (
        "BOTH",
        "NOENTRY_ONLY",
        "ENTRY_ONLY",
        "NEITHER",
    ):
        n = instance_cat[name]

        print(
            f"{name:14s} "
            f"{n:5d} "
            f"{n / total_instances:7.2%}"
        )

    print()
    print(
        "=================================="
    )
    print(
        "PER ROLE"
    )
    print(
        "=================================="
    )

    for role in TOP5:
        c = role_summary[role]
        total = c["PRESENT"]

        entry_n = (
            c["BOTH"]
            + c["ENTRY_ONLY"]
        )

        noentry_n = (
            c["BOTH"]
            + c["NOENTRY_ONLY"]
        )

        print(
            f"{role:32s} "
            f"N={total:4d} "
            f"Entry={entry_n:4d} "
            f"{entry_n / total:7.2%} "
            f"NoEntry={noentry_n:4d} "
            f"{noentry_n / total:7.2%} "
            f"NOENTRY_ONLY="
            f"{c['NOENTRY_ONLY']:4d}"
        )

    print()
    print(
        "=================================="
    )
    print(
        "APK-LEVEL ANY-CONTEXT"
    )
    print(
        "=================================="
    )

    apk_total = len(apk_rows)

    print(
        "Top5-present APKs =",
        apk_total,
    )

    print(
        "Entry >=1 context =",
        entry_apk_n,
        f"({entry_apk_n / apk_total:.2%})",
    )

    print(
        "NoEntry >=1 context =",
        noentry_apk_n,
        f"({noentry_apk_n / apk_total:.2%})",
    )

    print()

    for name in (
        "BOTH",
        "NOENTRY_ONLY",
        "ENTRY_ONLY",
        "NEITHER",
    ):
        n = apk_category_count[name]

        print(
            f"{name:14s} "
            f"{n:5d} "
            f"{n / apk_total:7.2%}"
        )

    # ---------------------------------------
    # More informative expansion flag
    # ---------------------------------------
    expanded = [
        r
        for r in apk_rows
        if r[
            "has_noentry_only_role"
        ] == 1
    ]

    not_expanded = [
        r
        for r in apk_rows
        if r[
            "has_noentry_only_role"
        ] == 0
    ]

    print()
    print(
        "=================================="
    )
    print(
        "APK HAS >=1 NOENTRY-ONLY ROLE"
    )
    print(
        "=================================="
    )

    print(
        "with expanded role coverage =",
        len(expanded),
        f"({len(expanded) / apk_total:.2%})",
    )

    # ---------------------------------------
    # Correction analysis
    # ---------------------------------------
    print()
    print(
        "=================================="
    )
    print(
        "CORRECTION BY APK CATEGORY"
    )
    print(
        "=================================="
    )

    for cat in (
        "BOTH",
        "NOENTRY_ONLY",
        "ENTRY_ONLY",
        "NEITHER",
    ):
        rows = [
            r
            for r in apk_rows
            if r["apk_category"] == cat
        ]

        if not rows:
            continue

        s = transition_summary(rows)

        print(
            f"{cat:14s} "
            f"N={s['n']:4d} "
            f"corrected={s['corrected']:3d} "
            f"regressed={s['regressed']:3d} "
            f"net={s['net']:+4d}"
        )

    print()
    print(
        "=================================="
    )
    print(
        "CORRECTION BY EXPANDED ROLE FLAG"
    )
    print(
        "=================================="
    )

    for name, rows in (
        (
            "HAS_NOENTRY_ONLY_ROLE",
            expanded,
        ),
        (
            "NO_NOENTRY_ONLY_ROLE",
            not_expanded,
        ),
    ):
        s = transition_summary(rows)

        print(
            f"{name:24s} "
            f"N={s['n']:4d} "
            f"corrected={s['corrected']:3d} "
            f"regressed={s['regressed']:3d} "
            f"net={s['net']:+4d}"
        )

    # Among all corrected APKs:
    corrected = [
        r
        for r in apk_rows
        if r["transition"] == "CORRECTED"
    ]

    corrected_expanded = [
        r
        for r in corrected
        if r[
            "has_noentry_only_role"
        ] == 1
    ]

    print()
    print(
        "corrected total =",
        len(corrected),
    )

    print(
        "corrected with >=1 "
        "NOENTRY_ONLY role =",
        len(corrected_expanded),
        (
            f"({len(corrected_expanded) / len(corrected):.2%})"
            if corrected
            else ""
        ),
    )

    # ---------------------------------------
    # Per year
    # ---------------------------------------
    print()
    print(
        "=================================="
    )
    print(
        "PER YEAR APK COVERAGE"
    )
    print(
        "=================================="
    )

    for year in (
        "2018", "2019",
        "2020", "2021",
    ):
        rows = [
            r
            for r in apk_rows
            if r["year"] == year
        ]

        entry_n = sum(
            r["entry_any_context"]
            for r in rows
        )

        noentry_n = sum(
            r["noentry_any_context"]
            for r in rows
        )

        expanded_n = sum(
            r["has_noentry_only_role"]
            for r in rows
        )

        print(
            f"{year} "
            f"N={len(rows):4d} "
            f"Entry={entry_n:4d} "
            f"{entry_n / len(rows):7.2%} "
            f"NoEntry={noentry_n:4d} "
            f"{noentry_n / len(rows):7.2%} "
            f"expanded={expanded_n:4d} "
            f"{expanded_n / len(rows):7.2%}"
        )

    # ---------------------------------------
    # Persist
    # ---------------------------------------
    write_csv(
        OUTPUT
        / "top5_entry_noentry_role_instances.csv",
        instance_rows,
    )

    write_csv(
        OUTPUT
        / "top5_entry_noentry_apk.csv",
        apk_rows,
    )

    print()
    print(
        "saved to:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
