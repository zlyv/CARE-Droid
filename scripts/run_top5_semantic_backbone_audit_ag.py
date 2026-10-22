#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from collections import Counter
from pathlib import Path

import yaml

# ------------------------------------------------------------
# Keep Androguard DEBUG/INFO noise out of terminal and run.log.
# Our experiment progress is emitted explicitly with print().
# ------------------------------------------------------------
import logging

logging.getLogger().setLevel(logging.WARNING)

try:
    from loguru import logger as _loguru_logger

    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stderr,
        level="WARNING",
    )
except ImportError:
    pass



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


TOP5 = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)


def load_yaml(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def mean(xs):
    return statistics.mean(xs) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--lineage",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--mapping",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--priority",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--edge-semantics",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--semantic-hops",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--raw-hop-cap",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--K-in",
        dest="k_in",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--endpoint-cap",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--node-cap",
        type=int,
        default=2048,
    )
    ap.add_argument(
        "--edge-cap",
        type=int,
        default=4096,
    )

    args = ap.parse_args()
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = load_yaml(args.mapping)
    priority = load_yaml(args.priority)
    edge_mapping = load_yaml(
        args.edge_semantics
    )

    rules = list(
        mapping.get("rules", [])
    )

    tiers = role_tiers(priority)

    edge_rules = list(
        edge_mapping.get("rules", [])
    )

    lineage_rows = []

    with args.lineage.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            if not row.get("absolute_path"):
                continue

            lineage_rows.append(row)

            if (
                args.limit
                and len(lineage_rows) >= args.limit
            ):
                break

    print(
        "semantic-backbone audit:",
        len(lineage_rows),
        "APKs",
    )

    outputs = []

    for i, row in enumerate(
        lineage_rows,
        start=1,
    ):
        apk = dalvik = analysis = graph = None

        out = {
            "sha256": row.get(
                "sha256", ""
            ),
            "metadata_year": row.get(
                "metadata_year", ""
            ),
            "metadata_label": row.get(
                "metadata_label", ""
            ),
            "status": "FAILED",
            "error": "",
        }

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

            # ----------------------------------
            # Raw H4 baseline.
            # All internal methods cost one.
            # ----------------------------------
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
                set(TOP5),
                semantic_hops=args.semantic_hops,
                raw_hop_cap=args.semantic_hops,
                k_in=args.k_in,
                endpoint_cap=args.endpoint_cap,
                node_cap=args.node_cap,
                edge_cap=args.edge_cap,
                neutral_flags={},
            )

            # ----------------------------------
            # Wrapper-transparent semantic H4.
            # ----------------------------------
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
                set(TOP5),
                semantic_hops=args.semantic_hops,
                raw_hop_cap=args.raw_hop_cap,
                k_in=args.k_in,
                endpoint_cap=args.endpoint_cap,
                node_cap=args.node_cap,
                edge_cap=args.edge_cap,
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

            extra_meaningful = (
                sem_meaningful
                - raw_meaningful
            )

            sem_neutral = {
                s
                for s in sem_nodes
                if not node_records[s].get(
                    "external"
                )
                and neutral_flags.get(
                    s, False
                )
            }

            beyond_raw_h4 = {
                s
                for s in sem_meaningful
                if sem_raw_distance.get(s, 0)
                > args.semantic_hops
            }

            out.update({
                "status": "OK",

                "graph_internal_nodes":
                    neutral_audit[
                        "internal_node_count"
                    ],

                "decoded_internal_nodes":
                    neutral_audit[
                        "decoded_internal_count"
                    ],

                "decoded_coverage":
                    neutral_audit[
                        "decoded_coverage"
                    ],

                "neutral_transit_nodes":
                    neutral_audit[
                        "neutral_transit_count"
                    ],

                "neutral_rate_all_internal":
                    neutral_audit[
                        "neutral_rate_all_internal"
                    ],

                "neutral_rate_decoded":
                    neutral_audit[
                        "neutral_rate_decoded"
                    ],

                "raw_h4_meaningful_nodes":
                    len(raw_meaningful),

                "semantic_h4_meaningful_nodes":
                    len(sem_meaningful),

                "extra_meaningful_nodes":
                    len(extra_meaningful),

                "semantic_neutral_nodes":
                    len(sem_neutral),

                "meaningful_beyond_raw_h4":
                    len(beyond_raw_h4),

                "semantic_max_raw_distance":
                    sem_summary[
                        "max_raw_distance"
                    ],

                "semantic_max_semantic_distance":
                    sem_summary[
                        "max_semantic_distance"
                    ],

                "raw_node_count":
                    len(raw_nodes),

                "semantic_node_count":
                    len(sem_nodes),

                "raw_edge_count":
                    len(raw_edges),

                "semantic_edge_count":
                    len(sem_edges),

                "raw_node_truncated":
                    int(
                        raw_summary[
                            "node_truncated"
                        ]
                    ),

                "semantic_node_truncated":
                    int(
                        sem_summary[
                            "node_truncated"
                        ]
                    ),

                "raw_edge_truncated":
                    int(
                        raw_summary[
                            "edge_truncated"
                        ]
                    ),

                "semantic_edge_truncated":
                    int(
                        sem_summary[
                            "edge_truncated"
                        ]
                    ),
            })

        except Exception as exc:
            out["error"] = (
                f"{type(exc).__name__}: "
                f"{str(exc)[:500]}"
            )

        finally:
            outputs.append(out)

            apk = None
            dalvik = None
            analysis = None
            graph = None

            gc.collect()

        print(
            f"{i}/{len(lineage_rows)} "
            f"{out['status']} "
            f"neutral={out.get('neutral_transit_nodes', 0)} "
            f"extra_meaningful={out.get('extra_meaningful_nodes', 0)} "
            f"beyondH4={out.get('meaningful_beyond_raw_h4', 0)}",
            flush=True,
        )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(outputs[0]),
        )
        writer.writeheader()
        writer.writerows(outputs)

    ok = [
        r for r in outputs
        if r["status"] == "OK"
    ]

    print()
    print("======================================")
    print("SEMANTIC BACKBONE AUDIT")
    print("======================================")

    print(
        "OK / total =",
        len(ok),
        "/",
        len(outputs),
    )

    if not ok:
        return

    def vals(name):
        return [
            float(r[name])
            for r in ok
        ]

    print(
        "mean decoded coverage =",
        f"{mean(vals('decoded_coverage')):.2%}",
    )

    print(
        "mean neutral rate all internal =",
        f"{mean(vals('neutral_rate_all_internal')):.2%}",
    )

    print(
        "mean neutral rate decoded =",
        f"{mean(vals('neutral_rate_decoded')):.2%}",
    )

    print(
        "APK with >=1 neutral transit =",
        sum(
            float(r["neutral_transit_nodes"]) > 0
            for r in ok
        ),
        "/",
        len(ok),
    )

    print()
    print(
        "mean raw-H4 meaningful nodes =",
        f"{mean(vals('raw_h4_meaningful_nodes')):.2f}",
    )

    print(
        "mean semantic-H4 meaningful nodes =",
        f"{mean(vals('semantic_h4_meaningful_nodes')):.2f}",
    )

    print(
        "mean extra meaningful nodes =",
        f"{mean(vals('extra_meaningful_nodes')):.2f}",
    )

    print(
        "APK gaining meaningful nodes =",
        sum(
            float(r["extra_meaningful_nodes"]) > 0
            for r in ok
        ),
        "/",
        len(ok),
    )

    print(
        "APK reaching meaningful nodes beyond raw H4 =",
        sum(
            float(r["meaningful_beyond_raw_h4"]) > 0
            for r in ok
        ),
        "/",
        len(ok),
    )

    print(
        "mean meaningful nodes beyond raw H4 =",
        f"{mean(vals('meaningful_beyond_raw_h4')):.2f}",
    )

    print(
        "max raw distance reached by semantic traversal =",
        int(max(vals(
            "semantic_max_raw_distance"
        ))),
    )

    print()
    print(
        "semantic node truncations =",
        sum(
            float(r[
                "semantic_node_truncated"
            ]) > 0
            for r in ok
        ),
    )

    print(
        "semantic edge truncations =",
        sum(
            float(r[
                "semantic_edge_truncated"
            ]) > 0
            for r in ok
        ),
    )

    print()
    print("saved:", args.output)


if __name__ == "__main__":
    main()
