#!/usr/bin/env python3

from __future__ import annotations

import csv
import gc
import logging
import sys
from collections import defaultdict, Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Silence Androguard noise.
logging.getLogger().setLevel(logging.WARNING)

try:
    from loguru import logger as _logger
    _logger.remove()
    _logger.add(sys.stderr, level="WARNING")
except ImportError:
    pass

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
    build_semantic_upstream_view,
)


TOP5 = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}


def load_yaml(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def main():

    lineage_path = Path(
        "artifacts/p0-lineage/"
        "package_certificate_lineage.csv"
    )

    mapping_path = Path(
        "configs/protocol/"
        "semantic_mapping_v1.yaml"
    )

    priority_path = Path(
        "configs/protocol/"
        "sensitive_endpoint_priority_v1.yaml"
    )

    edge_path = Path(
        "configs/protocol/"
        "semantic_edge_mapping_v1.yaml"
    )

    output = Path(
        "artifacts/top5-semantic-backbone-test20/"
        "semantic_effect_cases.csv"
    )

    mapping = load_yaml(mapping_path)
    priority = load_yaml(priority_path)
    edge_mapping = load_yaml(edge_path)

    rules = list(mapping.get("rules", []))
    tiers = role_tiers(priority)
    edge_rules = list(edge_mapping.get("rules", []))

    rows = []

    with lineage_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for r in csv.DictReader(f):

            if not r.get("absolute_path"):
                continue

            rows.append(r)

            if len(rows) >= 20:
                break

    details = []

    apk_summaries = []

    print(
        "Inspecting",
        len(rows),
        "APKs...",
        flush=True,
    )

    for index, row in enumerate(rows, start=1):

        apk = dalvik = analysis = graph = None

        try:
            from androguard.misc import AnalyzeAPK

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

            # --------------------------------------
            # Raw H4.
            # --------------------------------------
            (
                raw_nodes,
                raw_edges,
                raw_summary,
                raw_sem_distance,
                raw_distance,
            ) = build_semantic_upstream_view(
                node_records,
                edge_records,
                tiers,
                TOP5,
                semantic_hops=4,
                raw_hop_cap=4,
                k_in=8,
                endpoint_cap=128,
                node_cap=2048,
                edge_cap=4096,
                neutral_flags={},
            )

            # --------------------------------------
            # Semantic H4.
            # --------------------------------------
            (
                sem_nodes,
                sem_edges,
                sem_summary,
                sem_distance,
                sem_raw_distance,
            ) = build_semantic_upstream_view(
                node_records,
                edge_records,
                tiers,
                TOP5,
                semantic_hops=4,
                raw_hop_cap=16,
                k_in=8,
                endpoint_cap=128,
                node_cap=2048,
                edge_cap=4096,
                neutral_flags=neutral_flags,
            )

            def meaningful(nodes):
                return {
                    s
                    for s in nodes
                    if not node_records[s].get(
                        "external"
                    )
                    and not neutral_flags.get(
                        s, False
                    )
                }

            raw_meaningful = meaningful(
                raw_nodes
            )

            sem_meaningful = meaningful(
                sem_nodes
            )

            extra = sorted(
                sem_meaningful - raw_meaningful
            )

            # --------------------------------------
            # Full outgoing graph:
            #
            # Used ONLY for semantic annotation of
            # newly recovered meaningful methods.
            # We do not put raw API identities into
            # model features.
            # --------------------------------------
            full_outgoing = defaultdict(set)

            for edge in edge_records:
                source = edge["source"]
                target = edge["target"]

                if (
                    source in node_records
                    and target in node_records
                ):
                    full_outgoing[source].add(
                        target
                    )

            # --------------------------------------
            # Retained semantic path graph.
            # Used to determine WHICH Top5 endpoint
            # a recovered caller organizes.
            # --------------------------------------
            path_outgoing = defaultdict(set)

            for edge in sem_edges:
                path_outgoing[
                    edge["source"]
                ].add(edge["target"])

            def downstream_top5_roles(start):

                found = set()
                queue = [start]
                seen = {start}

                while queue:
                    current = queue.pop()

                    for target in path_outgoing.get(
                        current, ()
                    ):
                        if target in seen:
                            continue

                        seen.add(target)

                        rec = node_records[target]

                        if rec.get("external"):
                            found.update(
                                set(
                                    rec.get(
                                        "roles", []
                                    )
                                )
                                & TOP5
                            )
                        else:
                            queue.append(target)

                return found

            apk_semantic_extra = 0
            apk_non_top5_extra = 0
            apk_mapped_api_calls = 0
            apk_unmapped_api_calls = 0

            for signature in extra:

                all_external = set()
                mapped_external = set()
                mapped_roles = set()

                for target in full_outgoing.get(
                    signature, ()
                ):
                    target_record = (
                        node_records[target]
                    )

                    if not target_record.get(
                        "external"
                    ):
                        continue

                    all_external.add(target)

                    roles = set(
                        target_record.get(
                            "roles", []
                        )
                    )

                    if roles:
                        mapped_external.add(
                            target
                        )
                        mapped_roles.update(
                            roles
                        )

                unmapped_external_count = (
                    len(all_external)
                    - len(mapped_external)
                )

                hotspot_roles = (
                    downstream_top5_roles(
                        signature
                    )
                )

                non_top5_roles = (
                    mapped_roles - TOP5
                )

                saved_hops = (
                    int(
                        sem_raw_distance.get(
                            signature, 0
                        )
                    )
                    -
                    int(
                        sem_distance.get(
                            signature, 0
                        )
                    )
                )

                if mapped_roles:
                    apk_semantic_extra += 1

                if non_top5_roles:
                    apk_non_top5_extra += 1

                apk_mapped_api_calls += len(
                    mapped_external
                )

                apk_unmapped_api_calls += (
                    unmapped_external_count
                )

                details.append({
                    "sha256":
                        row.get("sha256", ""),

                    "year":
                        row.get(
                            "metadata_year", ""
                        ),

                    "label":
                        row.get(
                            "metadata_label", ""
                        ),

                    "method_signature":
                        signature,

                    "raw_distance":
                        int(
                            sem_raw_distance.get(
                                signature, 0
                            )
                        ),

                    "semantic_distance":
                        int(
                            sem_distance.get(
                                signature, 0
                            )
                        ),

                    "saved_hops":
                        saved_hops,

                    "top5_hotspot_roles":
                        "|".join(
                            sorted(hotspot_roles)
                        ),

                    "direct_mapped_roles":
                        "|".join(
                            sorted(mapped_roles)
                        ),

                    "direct_non_top5_roles":
                        "|".join(
                            sorted(non_top5_roles)
                        ),

                    "direct_external_count":
                        len(all_external),

                    "direct_mapped_api_count":
                        len(mapped_external),

                    "direct_unmapped_api_count":
                        unmapped_external_count,
                })

            apk_summaries.append({
                "sha256":
                    row.get("sha256", ""),

                "extra_meaningful":
                    len(extra),

                "extra_with_direct_semantic_api":
                    apk_semantic_extra,

                "extra_with_non_top5_semantics":
                    apk_non_top5_extra,

                "mapped_external_calls":
                    apk_mapped_api_calls,

                "unmapped_external_calls":
                    apk_unmapped_api_calls,
            })

            print(
                f"{index:2d}/20 "
                f"extra={len(extra):2d} "
                f"semantic={apk_semantic_extra:2d} "
                f"nonTop5={apk_non_top5_extra:2d}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"{index:2d}/20 FAILED "
                f"{type(exc).__name__}: "
                f"{str(exc)[:200]}",
                flush=True,
            )

        finally:
            apk = None
            dalvik = None
            analysis = None
            graph = None
            gc.collect()

    # ------------------------------------------
    # Save case-level detail.
    # ------------------------------------------
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if details:
        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(details[0]),
            )
            writer.writeheader()
            writer.writerows(details)

    # ------------------------------------------
    # Summary.
    # ------------------------------------------
    total_extra = len(details)

    with_semantic = sum(
        bool(r["direct_mapped_roles"])
        for r in details
    )

    with_non_top5 = sum(
        bool(r["direct_non_top5_roles"])
        for r in details
    )

    saved_hops = [
        int(r["saved_hops"])
        for r in details
    ]

    hotspot_counter = Counter()

    semantic_role_counter = Counter()

    for r in details:

        for role in (
            r["top5_hotspot_roles"]
            .split("|")
        ):
            if role:
                hotspot_counter[role] += 1

        for role in (
            r["direct_mapped_roles"]
            .split("|")
        ):
            if role:
                semantic_role_counter[
                    role
                ] += 1

    affected_apks = sum(
        int(r["extra_meaningful"]) > 0
        for r in apk_summaries
    )

    semantic_affected_apks = sum(
        int(
            r[
                "extra_with_direct_semantic_api"
            ]
        ) > 0
        for r in apk_summaries
    )

    non_top5_affected_apks = sum(
        int(
            r[
                "extra_with_non_top5_semantics"
            ]
        ) > 0
        for r in apk_summaries
    )

    print()
    print(
        "======================================"
    )
    print(
        "SEMANTIC BACKBONE EFFECT — TEST20"
    )
    print(
        "======================================"
    )

    print(
        "APKs gaining meaningful caller =",
        affected_apks,
        "/ 20",
    )

    print(
        "extra meaningful callers =",
        total_extra,
    )

    print(
        "extra callers with direct mapped API =",
        with_semantic,
        "/",
        total_extra,
        (
            f"({with_semantic/total_extra:.2%})"
            if total_extra
            else ""
        ),
    )

    print(
        "extra callers with non-Top5 semantic API =",
        with_non_top5,
        "/",
        total_extra,
        (
            f"({with_non_top5/total_extra:.2%})"
            if total_extra
            else ""
        ),
    )

    print(
        "APKs gaining direct semantic evidence =",
        semantic_affected_apks,
        "/ 20",
    )

    print(
        "APKs gaining broader non-Top5 semantics =",
        non_top5_affected_apks,
        "/ 20",
    )

    if saved_hops:
        print(
            "mean neutral hops bypassed =",
            f"{sum(saved_hops)/len(saved_hops):.2f}",
        )

        print(
            "max neutral hops bypassed =",
            max(saved_hops),
        )

    print()
    print("Top5 hotspots contextualized:")

    for role, n in (
        hotspot_counter.most_common()
    ):
        print(
            f"  {role:32s} {n}"
        )

    print()
    print(
        "Direct mapped semantic roles "
        "of recovered callers:"
    )

    for role, n in (
        semantic_role_counter.most_common()
    ):
        print(
            f"  {role:32s} {n}"
        )

    print()
    print("saved:", output)


if __name__ == "__main__":
    main()
