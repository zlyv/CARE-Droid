#!/usr/bin/env python3

from __future__ import annotations

import csv
import gc
import logging
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Silence Androguard DEBUG / INFO.
logging.getLogger().setLevel(logging.WARNING)

try:
    from loguru import logger as _logger
    _logger.remove()
    _logger.add(sys.stderr, level="WARNING")
except ImportError:
    pass

from androguard.misc import AnalyzeAPK

from run_result_centered_slice_dry_run_ag import (
    build_method_ir,
)

from run_sensitive_upstream_tensor_cache_ag import (
    canonicalize_call_graph,
    canonical_signature,
)

from run_endpoint_centered_tensor_cache_ag import (
    role_tiers,
)

from semantic_backbone import (
    build_neutral_transit_profiles,
)


ROOT = Path("artifacts/top5-semantic-backbone-test20")

CASES = ROOT / "semantic_effect_cases.csv"

LINEAGE = Path(
    "artifacts/p0-lineage/"
    "package_certificate_lineage.csv"
)

MAPPING = Path(
    "configs/protocol/semantic_mapping_v1.yaml"
)

PRIORITY = Path(
    "configs/protocol/sensitive_endpoint_priority_v1.yaml"
)

EDGE_SEMANTICS = Path(
    "configs/protocol/semantic_edge_mapping_v1.yaml"
)

OUTPUT = ROOT / "recovered_semantic_distance.csv"

MAX_SEMANTIC_DISTANCE = 4
RAW_HOP_CAP = 16


def load_yaml(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def split_roles(value):
    return {
        x.strip()
        for x in str(value or "").split("|")
        if x.strip()
    }


def bucket(value):
    if value is None:
        return "NONE"
    return str(int(value))


def forward_role_distances(
    start,
    node_records,
    outgoing,
    neutral_flags,
):
    """
    Search forward from one recovered meaningful caller.

    neutral internal method:
        semantic cost 0

    meaningful internal method:
        semantic cost 1

    mapped external API:
        terminal semantic evidence

    unmapped external API:
        terminal noise; ignored

    Distance therefore means:
        number of additional meaningful INTERNAL layers
        between recovered caller and mapped API.
    """

    best = {
        start: (0, 0)
    }

    queue = deque([start])

    role_min_distance = {}

    while queue:
        current = queue.popleft()

        sem_dist, raw_dist = best[current]

        if raw_dist >= RAW_HOP_CAP:
            continue

        for target in outgoing.get(current, ()):

            if target not in node_records:
                continue

            rec = node_records[target]

            next_raw = raw_dist + 1

            if next_raw > RAW_HOP_CAP:
                continue

            # External API: semantic terminal.
            if rec.get("external"):

                for role in rec.get("roles", []):
                    old = role_min_distance.get(role)

                    if (
                        old is None
                        or sem_dist < old
                    ):
                        role_min_distance[role] = (
                            sem_dist
                        )

                # Never traverse through external APIs.
                continue

            # Internal method.
            neutral = bool(
                neutral_flags.get(target, False)
            )

            step = 0 if neutral else 1

            next_sem = sem_dist + step

            if next_sem > MAX_SEMANTIC_DISTANCE:
                continue

            candidate = (
                next_sem,
                next_raw,
            )

            previous = best.get(target)

            if (
                previous is None
                or candidate < previous
            ):
                best[target] = candidate

                if step == 0:
                    queue.appendleft(target)
                else:
                    queue.append(target)

    return role_min_distance


def main():

    # ----------------------------------------
    # Load recovered caller cases.
    # ----------------------------------------
    with CASES.open(
        newline="",
        encoding="utf-8",
    ) as f:
        cases = list(csv.DictReader(f))

    print("recovered caller cases =", len(cases))

    by_sha = defaultdict(list)

    for row in cases:
        by_sha[
            row["sha256"].strip().casefold()
        ].append(row)

    # ----------------------------------------
    # Resolve APK paths.
    # ----------------------------------------
    lineage = {}

    with LINEAGE.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            sha = (
                row.get("sha256", "")
                .strip()
                .casefold()
            )

            if sha:
                lineage[sha] = row

    # ----------------------------------------
    # Configs.
    # ----------------------------------------
    mapping = load_yaml(MAPPING)
    priority = load_yaml(PRIORITY)
    edge_mapping = load_yaml(
        EDGE_SEMANTICS
    )

    rules = list(
        mapping.get("rules", [])
    )

    tiers = role_tiers(priority)

    edge_rules = list(
        edge_mapping.get("rules", [])
    )

    results = []

    # ----------------------------------------
    # Only reanalyse APKs that actually had
    # recovered meaningful callers.
    # ----------------------------------------
    for apk_index, sha in enumerate(
        sorted(by_sha),
        start=1,
    ):
        row = lineage.get(sha)

        if row is None:
            raise RuntimeError(
                f"lineage missing for {sha}"
            )

        apk = dalvik = analysis = graph = None

        try:
            apk, dalvik, analysis = AnalyzeAPK(
                row["absolute_path"]
            )

            graph = analysis.get_call_graph(
                no_isolated=False
            )

            node_records, edge_records = (
                canonicalize_call_graph(
                    graph,
                    rules,
                    tiers,
                    edge_rules,
                )
            )

            ir_audit = {}

            method_irs = build_method_ir(
                dalvik,
                analysis,
                ir_audit,
            )

            (
                neutral_flags,
                profiles,
                neutral_audit,
            ) = build_neutral_transit_profiles(
                method_irs,
                node_records,
                edge_records,
                canonical_signature,
            )

            outgoing = defaultdict(set)

            for edge in edge_records:
                source = edge["source"]
                target = edge["target"]

                if (
                    source in node_records
                    and target in node_records
                ):
                    outgoing[source].add(target)

            for case in by_sha[sha]:

                start = case[
                    "method_signature"
                ]

                if start not in node_records:
                    raise RuntimeError(
                        "Recovered method missing "
                        f"from graph: {start}"
                    )

                hotspot_roles = split_roles(
                    case.get(
                        "hotspot_roles",
                        case.get(
                            "top5_hotspot_roles",
                            "",
                        ),
                    )
                )

                role_min = (
                    forward_role_distances(
                        start,
                        node_records,
                        outgoing,
                        neutral_flags,
                    )
                )

                # --------------------------------
                # Original hotspot rediscovery.
                # This is our implementation sanity
                # check.
                # --------------------------------
                hotspot_distances = [
                    role_min[role]
                    for role in hotspot_roles
                    if role in role_min
                ]

                first_hotspot = (
                    min(hotspot_distances)
                    if hotspot_distances
                    else None
                )

                # --------------------------------
                # ANY mapped API semantic role.
                # --------------------------------
                first_any = (
                    min(role_min.values())
                    if role_min
                    else None
                )

                # --------------------------------
                # SIDE semantic context:
                # mapped role minus original
                # hotspot role(s).
                # --------------------------------
                side_role_min = {
                    role: dist
                    for role, dist
                    in role_min.items()
                    if role not in hotspot_roles
                }

                first_side = (
                    min(side_role_min.values())
                    if side_role_min
                    else None
                )

                roles_by_distance = defaultdict(
                    list
                )

                for role, dist in role_min.items():
                    roles_by_distance[int(dist)].append(
                        role
                    )

                result = {
                    "sha256": sha,

                    "year":
                        case.get("year", ""),

                    "label":
                        case.get("label", ""),

                    "method_signature":
                        start,

                    "raw_distance":
                        case.get(
                            "raw_distance", ""
                        ),

                    "semantic_distance":
                        case.get(
                            "semantic_distance", ""
                        ),

                    "saved_hops":
                        case.get(
                            "saved_hops", ""
                        ),

                    "hotspot_roles":
                        "|".join(
                            sorted(hotspot_roles)
                        ),

                    "first_hotspot_distance":
                        (
                            ""
                            if first_hotspot is None
                            else first_hotspot
                        ),

                    "first_any_mapped_distance":
                        (
                            ""
                            if first_any is None
                            else first_any
                        ),

                    "first_side_distance":
                        (
                            ""
                            if first_side is None
                            else first_side
                        ),

                    "side_roles":
                        "|".join(
                            sorted(side_role_min)
                        ),

                    "roles_d0":
                        "|".join(
                            sorted(
                                roles_by_distance[0]
                            )
                        ),

                    "roles_d1":
                        "|".join(
                            sorted(
                                roles_by_distance[1]
                            )
                        ),

                    "roles_d2":
                        "|".join(
                            sorted(
                                roles_by_distance[2]
                            )
                        ),

                    "roles_d3":
                        "|".join(
                            sorted(
                                roles_by_distance[3]
                            )
                        ),

                    "roles_d4":
                        "|".join(
                            sorted(
                                roles_by_distance[4]
                            )
                        ),
                }

                results.append(result)

            print(
                f"{apk_index}/{len(by_sha)} "
                f"APK complete "
                f"callers={len(by_sha[sha])}",
                flush=True,
            )

        finally:
            apk = None
            dalvik = None
            analysis = None
            graph = None
            gc.collect()

    # ----------------------------------------
    # Save.
    # ----------------------------------------
    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0]),
        )

        writer.writeheader()
        writer.writerows(results)

    # ----------------------------------------
    # Summary.
    # ----------------------------------------
    hotspot_counter = Counter()
    any_counter = Counter()
    side_counter = Counter()
    side_role_counter = Counter()

    apks_with_side = set()

    for row in results:

        h = row[
            "first_hotspot_distance"
        ]

        a = row[
            "first_any_mapped_distance"
        ]

        s = row[
            "first_side_distance"
        ]

        hotspot_counter[
            "NONE" if h == "" else str(int(float(h)))
        ] += 1

        any_counter[
            "NONE" if a == "" else str(int(float(a)))
        ] += 1

        side_counter[
            "NONE" if s == "" else str(int(float(s)))
        ] += 1

        if s != "":
            apks_with_side.add(
                row["sha256"]
            )

        for role in split_roles(
            row["side_roles"]
        ):
            side_role_counter[role] += 1

    print()
    print(
        "======================================"
    )
    print(
        "RECOVERED CALLER SEMANTIC DISTANCE"
    )
    print(
        "======================================"
    )

    print(
        "recovered callers =",
        len(results),
    )

    print()
    print(
        "HOTSPOT rediscovery distance "
        "(sanity check)"
    )

    for name in [
        "0", "1", "2", "3", "4", "NONE"
    ]:
        print(
            f"  d={name:4s} "
            f"N={hotspot_counter[name]:2d}"
        )

    print()
    print(
        "FIRST ANY mapped semantic API"
    )

    for name in [
        "0", "1", "2", "3", "4", "NONE"
    ]:
        print(
            f"  d={name:4s} "
            f"N={any_counter[name]:2d}"
        )

    print()
    print(
        "FIRST SIDE semantic role "
        "(hotspot removed)"
    )

    for name in [
        "0", "1", "2", "3", "4", "NONE"
    ]:
        print(
            f"  d={name:4s} "
            f"N={side_counter[name]:2d}"
        )

    side_count = (
        len(results)
        - side_counter["NONE"]
    )

    print()
    print(
        "callers with SIDE semantics <=4 =",
        side_count,
        "/",
        len(results),
        (
            f"({side_count/len(results):.2%})"
            if results
            else ""
        ),
    )

    print(
        "affected APKs with SIDE semantics <=4 =",
        len(apks_with_side),
        "/",
        len(by_sha),
    )

    print()
    print("SIDE semantic roles:")

    for role, count in (
        side_role_counter.most_common()
    ):
        print(
            f"  {role:35s} {count}"
        )

    print()
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
