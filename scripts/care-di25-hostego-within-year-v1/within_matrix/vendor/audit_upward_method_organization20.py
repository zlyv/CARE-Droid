#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# 不输出 Androguard 的 DEBUG / INFO。
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
    canonical_signature,
    canonicalize_call_graph,
)

from run_endpoint_centered_tensor_cache_ag import (
    role_tiers,
)

from semantic_backbone import (
    build_neutral_transit_profiles,
    build_semantic_upstream_view,
)

from run_clean_local_representation_ag import (
    anchor_roles_for_method,
    contract_upward_structure,
)


TOP5 = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}


def load_yaml(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def has_prefix(
    name: str,
    prefixes: tuple[str, ...],
) -> bool:
    return any(
        name.startswith(prefix)
        for prefix in prefixes
    )


def instruction_profile(ir):
    """
    这里只做诊断，不决定删除。

    检查一个内部方法有没有：
      - 条件判断
      - 状态读取
      - 状态写入
      - 同步
      - 抛异常
      - 算术/数据变换
      - 多次方法调用
    """

    if ir is None:
        return {
            "decoded": 0,
            "instruction_count": 0,
            "invoke_count": 0,
            "has_branch": 0,
            "has_state_read": 0,
            "has_state_write": 0,
            "has_monitor": 0,
            "has_throw": 0,
            "has_transform": 0,
        }

    names = [
        str(inst.name).strip().casefold()
        for inst in ir.instructions
    ]

    branch = any(
        name.startswith("if-")
        or name in {
            "packed-switch",
            "sparse-switch",
        }
        for name in names
    )

    state_read = any(
        has_prefix(
            name,
            (
                "iget",
                "sget",
                "aget",
            ),
        )
        for name in names
    )

    state_write = any(
        has_prefix(
            name,
            (
                "iput",
                "sput",
                "aput",
            ),
        )
        for name in names
    )

    monitor = any(
        name in {
            "monitor-enter",
            "monitor-exit",
        }
        for name in names
    )

    throw = any(
        name.startswith("throw")
        for name in names
    )

    transform_prefixes = (
        "add-",
        "sub-",
        "mul-",
        "div-",
        "rem-",
        "and-",
        "or-",
        "xor-",
        "shl-",
        "shr-",
        "ushr-",
        "neg-",
        "not-",
        "cmp-",
        "int-to-",
        "long-to-",
        "float-to-",
        "double-to-",
    )

    transform = any(
        has_prefix(
            name,
            transform_prefixes,
        )
        for name in names
    )

    invoke_count = sum(
        name.startswith("invoke-")
        for name in names
    )

    return {
        "decoded": 1,
        "instruction_count": len(names),
        "invoke_count": invoke_count,
        "has_branch": int(branch),
        "has_state_read": int(state_read),
        "has_state_write": int(state_write),
        "has_monitor": int(monitor),
        "has_throw": int(throw),
        "has_transform": int(transform),
    }


def evidence_level(
    profile,
    out_method_degree,
    in_method_degree,
    anchor_role_count,
):
    """
    只为了观察 125 个方法的组成。

    强：
      有明显控制/状态行为，
      或者本身处在分支/汇合位置，
      或者同时组织多类敏感行为。

    一般：
      有状态读取、数据变换，
      或者调用多个其他方法。

    弱：
      暂时没有观察到上述证据。

    注意：
      弱 != 应该删除。
      这一轮绝不据此剪枝。
    """

    strong = any([
        profile["has_branch"],
        profile["has_state_write"],
        profile["has_monitor"],
        profile["has_throw"],
        out_method_degree >= 2,
        in_method_degree >= 2,
        anchor_role_count >= 2,
    ])

    if strong:
        return "strong"

    some = any([
        profile["has_state_read"],
        profile["has_transform"],
        profile["invoke_count"] >= 2,
    ])

    if some:
        return "some"

    return "weak"


def write_gzip_csv(
    path: Path,
    fieldnames,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gzip.open(
        path,
        "wt",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lineage",
        type=Path,
        default=Path(
            "artifacts/p0-lineage/"
            "package_certificate_lineage.csv"
        ),
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(
            "configs/protocol/"
            "semantic_mapping_v1.yaml"
        ),
    )

    parser.add_argument(
        "--priority",
        type=Path,
        default=Path(
            "configs/protocol/"
            "sensitive_endpoint_priority_v1.yaml"
        ),
    )

    parser.add_argument(
        "--edge-semantics",
        type=Path,
        default=Path(
            "configs/protocol/"
            "semantic_edge_mapping_v1.yaml"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/"
            "upward-method-organization-test20"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--up-hops",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--raw-hop-cap",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--k-in",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--endpoint-cap",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--node-cap",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--edge-cap",
        type=int,
        default=4096,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = load_yaml(
        args.mapping
    )

    priority = load_yaml(
        args.priority
    )

    edge_mapping = load_yaml(
        args.edge_semantics
    )

    rules = list(
        mapping.get(
            "rules",
            [],
        )
    )

    tiers = role_tiers(
        priority
    )

    edge_rules = list(
        edge_mapping.get(
            "rules",
            [],
        )
    )

    lineage_rows = []

    with args.lineage.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):

            if not row.get(
                "absolute_path"
            ):
                continue

            lineage_rows.append(row)

            if (
                args.limit > 0
                and len(lineage_rows)
                >= args.limit
            ):
                break

    method_rows = []
    apk_rows = []

    print(
        "开始检查向上保留的方法：",
        len(lineage_rows),
        "个 APK",
        flush=True,
    )

    for index, row in enumerate(
        lineage_rows,
        start=1,
    ):
        apk = None
        dalvik = None
        analysis = None
        graph = None

        sha = str(
            row.get(
                "sha256",
                "",
            )
        )

        try:
            apk, dalvik, analysis = AnalyzeAPK(
                row["absolute_path"]
            )

            graph = analysis.get_call_graph(
                no_isolated=False
            )

            (
                node_records,
                edge_records,
            ) = canonicalize_call_graph(
                graph,
                rules,
                tiers,
                edge_rules,
            )

            method_irs = build_method_ir(
                dalvik,
                analysis,
                {},
            )

            (
                neutral_flags,
                _,
                _,
            ) = build_neutral_transit_profiles(
                method_irs,
                node_records,
                edge_records,
                canonical_signature,
            )

            (
                up_nodes,
                up_edges,
                up_summary,
                up_sem_distance,
                up_raw_distance,
            ) = build_semantic_upstream_view(
                node_records,
                edge_records,
                tiers,
                TOP5,
                semantic_hops=args.up_hops,
                raw_hop_cap=args.raw_hop_cap,
                k_in=args.k_in,
                endpoint_cap=args.endpoint_cap,
                node_cap=args.node_cap,
                edge_cap=args.edge_cap,
                neutral_flags=neutral_flags,
            )

            meaningful_methods = {
                node
                for node in up_nodes
                if (
                    not node_records[
                        node
                    ].get("external")
                    and not neutral_flags.get(
                        node,
                        False,
                    )
                )
            }

            neutral_methods = {
                node
                for node in up_nodes
                if (
                    not node_records[
                        node
                    ].get("external")
                    and neutral_flags.get(
                        node,
                        False,
                    )
                )
            }

            (
                method_edges,
                anchor_edges,
                up_outgoing,
            ) = contract_upward_structure(
                meaningful_methods,
                neutral_flags,
                up_edges,
                node_records,
            )

            out_methods = defaultdict(set)
            in_methods = defaultdict(set)
            role_targets = defaultdict(set)

            for source, target in method_edges:
                out_methods[source].add(
                    target
                )
                in_methods[target].add(
                    source
                )

            for source, role in anchor_edges:
                role_targets[source].add(
                    role
                )

            ir_by_signature = {}

            for ir in method_irs:
                try:
                    signature = (
                        canonical_signature(
                            ir.signature
                        )
                    )
                except Exception:
                    signature = ""

                if signature:
                    ir_by_signature[
                        signature
                    ] = ir

            distance_counter = Counter()
            level_counter = Counter()

            for method in sorted(
                meaningful_methods
            ):
                sem_dist = int(
                    up_sem_distance.get(
                        method,
                        -1,
                    )
                )

                raw_dist = int(
                    up_raw_distance.get(
                        method,
                        -1,
                    )
                )

                anchors = (
                    anchor_roles_for_method(
                        method,
                        up_outgoing,
                        node_records,
                    )
                )

                profile = (
                    instruction_profile(
                        ir_by_signature.get(
                            method
                        )
                    )
                )

                out_degree = len(
                    out_methods.get(
                        method,
                        set(),
                    )
                )

                in_degree = len(
                    in_methods.get(
                        method,
                        set(),
                    )
                )

                anchor_count = len(
                    anchors
                )

                level = evidence_level(
                    profile,
                    out_degree,
                    in_degree,
                    anchor_count,
                )

                distance_counter[
                    sem_dist
                ] += 1

                level_counter[
                    level
                ] += 1

                method_rows.append({
                    "sha256":
                        sha,

                    "year":
                        row.get(
                            "metadata_year",
                            "",
                        ),

                    "label":
                        row.get(
                            "metadata_label",
                            "",
                        ),

                    "method_signature":
                        method,

                    "up_semantic_distance":
                        sem_dist,

                    "up_raw_distance":
                        raw_dist,

                    "decoded":
                        profile[
                            "decoded"
                        ],

                    "instruction_count":
                        profile[
                            "instruction_count"
                        ],

                    "invoke_count":
                        profile[
                            "invoke_count"
                        ],

                    "has_branch":
                        profile[
                            "has_branch"
                        ],

                    "has_state_read":
                        profile[
                            "has_state_read"
                        ],

                    "has_state_write":
                        profile[
                            "has_state_write"
                        ],

                    "has_monitor":
                        profile[
                            "has_monitor"
                        ],

                    "has_throw":
                        profile[
                            "has_throw"
                        ],

                    "has_transform":
                        profile[
                            "has_transform"
                        ],

                    "in_method_degree":
                        in_degree,

                    "out_method_degree":
                        out_degree,

                    "anchor_role_count":
                        anchor_count,

                    "anchor_roles":
                        "|".join(
                            sorted(
                                anchors
                            )
                        ),

                    "direct_anchor_role_count":
                        len(
                            role_targets.get(
                                method,
                                set(),
                            )
                        ),

                    "organization_evidence":
                        level,
                })

            total = len(
                meaningful_methods
            )

            strong = (
                level_counter[
                    "strong"
                ]
            )

            some = (
                level_counter[
                    "some"
                ]
            )

            weak = (
                level_counter[
                    "weak"
                ]
            )

            apk_rows.append({
                "sha256":
                    sha,

                "year":
                    row.get(
                        "metadata_year",
                        "",
                    ),

                "label":
                    row.get(
                        "metadata_label",
                        "",
                    ),

                "status":
                    "OK",

                "kept_method_count":
                    total,

                "neutral_transit_count":
                    len(
                        neutral_methods
                    ),

                "d1_count":
                    distance_counter[1],

                "d2_count":
                    distance_counter[2],

                "d3_count":
                    distance_counter[3],

                "d4_count":
                    distance_counter[4],

                "strong_count":
                    strong,

                "some_count":
                    some,

                "weak_count":
                    weak,

                "strong_rate":
                    (
                        strong / total
                        if total
                        else 0.0
                    ),

                "weak_rate":
                    (
                        weak / total
                        if total
                        else 0.0
                    ),

                "method_edge_count":
                    len(
                        method_edges
                    ),

                "anchor_edge_count":
                    len(
                        anchor_edges
                    ),

                "node_truncated":
                    int(
                        up_summary[
                            "node_truncated"
                        ]
                    ),

                "edge_truncated":
                    int(
                        up_summary[
                            "edge_truncated"
                        ]
                    ),

                "error":
                    "",
            })

            print(
                f"{index:2d}/"
                f"{len(lineage_rows)} "
                f"保留={total:3d} "
                f"d1={distance_counter[1]:3d} "
                f"d2={distance_counter[2]:3d} "
                f"d3={distance_counter[3]:3d} "
                f"d4={distance_counter[4]:3d} | "
                f"强={strong:3d} "
                f"一般={some:3d} "
                f"弱={weak:3d}",
                flush=True,
            )

        except Exception as exc:
            apk_rows.append({
                "sha256":
                    sha,

                "year":
                    row.get(
                        "metadata_year",
                        "",
                    ),

                "label":
                    row.get(
                        "metadata_label",
                        "",
                    ),

                "status":
                    "FAILED",

                "kept_method_count": 0,
                "neutral_transit_count": 0,
                "d1_count": 0,
                "d2_count": 0,
                "d3_count": 0,
                "d4_count": 0,
                "strong_count": 0,
                "some_count": 0,
                "weak_count": 0,
                "strong_rate": 0.0,
                "weak_rate": 0.0,
                "method_edge_count": 0,
                "anchor_edge_count": 0,
                "node_truncated": 0,
                "edge_truncated": 0,

                "error":
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:500]}",
            })

            print(
                f"{index:2d}/"
                f"{len(lineage_rows)} "
                f"FAILED "
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

    apk_path = (
        args.output_dir
        / "apk_summary.csv"
    )

    with apk_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                apk_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            apk_rows
        )

    method_path = (
        args.output_dir
        / "method_profiles.csv.gz"
    )

    method_fields = [
        "sha256",
        "year",
        "label",
        "method_signature",
        "up_semantic_distance",
        "up_raw_distance",
        "decoded",
        "instruction_count",
        "invoke_count",
        "has_branch",
        "has_state_read",
        "has_state_write",
        "has_monitor",
        "has_throw",
        "has_transform",
        "in_method_degree",
        "out_method_degree",
        "anchor_role_count",
        "anchor_roles",
        "direct_anchor_role_count",
        "organization_evidence",
    ]

    write_gzip_csv(
        method_path,
        method_fields,
        method_rows,
    )

    ok = [
        row
        for row in apk_rows
        if row["status"] == "OK"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "向上方法组成检查结果"
    )
    print(
        "======================================"
    )

    print(
        "成功 =",
        len(ok),
        "/",
        len(apk_rows),
    )

    if ok:
        total_methods = sum(
            int(
                row[
                    "kept_method_count"
                ]
            )
            for row in ok
        )

        total_strong = sum(
            int(
                row[
                    "strong_count"
                ]
            )
            for row in ok
        )

        total_some = sum(
            int(
                row[
                    "some_count"
                ]
            )
            for row in ok
        )

        total_weak = sum(
            int(
                row[
                    "weak_count"
                ]
            )
            for row in ok
        )

        print()
        print(
            "总保留内部方法 =",
            total_methods,
        )

        print(
            "强组织证据 =",
            total_strong,
        )

        print(
            "一般组织证据 =",
            total_some,
        )

        print(
            "弱组织证据 =",
            total_weak,
        )

        if total_methods:
            print(
                "强组织证据比例 =",
                f"{total_strong / total_methods:.2%}",
            )

            print(
                "弱组织证据比例 =",
                f"{total_weak / total_methods:.2%}",
            )

        print()
        print("按向上距离统计：")

        for dist in (
            1,
            2,
            3,
            4,
        ):
            total_d = sum(
                1
                for method in method_rows
                if (
                    method[
                        "up_semantic_distance"
                    ]
                    == dist
                )
            )

            strong_d = sum(
                1
                for method in method_rows
                if (
                    method[
                        "up_semantic_distance"
                    ]
                    == dist
                    and method[
                        "organization_evidence"
                    ]
                    == "strong"
                )
            )

            weak_d = sum(
                1
                for method in method_rows
                if (
                    method[
                        "up_semantic_distance"
                    ]
                    == dist
                    and method[
                        "organization_evidence"
                    ]
                    == "weak"
                )
            )

            strong_rate = (
                strong_d / total_d
                if total_d
                else 0.0
            )

            weak_rate = (
                weak_d / total_d
                if total_d
                else 0.0
            )

            print(
                f"  距离{dist}: "
                f"N={total_d:4d} "
                f"强={strong_d:4d} "
                f"({strong_rate:.1%}) "
                f"弱={weak_d:4d} "
                f"({weak_rate:.1%})"
            )

        print()
        print(
            "保留方法最多的 APK："
        )

        largest = sorted(
            ok,
            key=lambda item: int(
                item[
                    "kept_method_count"
                ]
            ),
            reverse=True,
        )[:5]

        for row in largest:
            print(
                " ",
                row["sha256"][:12],
                "总数=",
                row["kept_method_count"],
                "d1=",
                row["d1_count"],
                "d2=",
                row["d2_count"],
                "d3=",
                row["d3_count"],
                "d4=",
                row["d4_count"],
                "强=",
                row["strong_count"],
                "弱=",
                row["weak_count"],
            )

    print()
    print(
        "保存：",
        apk_path,
    )

    print(
        "保存：",
        method_path,
    )


if __name__ == "__main__":
    main()
