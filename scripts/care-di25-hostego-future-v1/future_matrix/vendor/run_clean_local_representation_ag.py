#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import logging
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ------------------------------------------------------------
# 不输出 Androguard 的 DEBUG / INFO
# ------------------------------------------------------------
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


def short_id(text: str) -> str:
    return hashlib.sha256(
        str(text).encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()[:16]


def build_outgoing(
    edge_records,
    node_records,
):
    outgoing = defaultdict(set)

    for edge in edge_records:
        source = edge.get("source")
        target = edge.get("target")

        if (
            source in node_records
            and target in node_records
        ):
            outgoing[source].add(target)

    return outgoing


def accept_state(
    frontiers,
    node,
    semantic_distance,
    raw_distance,
):
    """
    一个节点可能通过不同路径到达。

    如果旧路径在：
      有意义层数
      原始调用层数
    两方面都不比新路径差，
    那么新路径没有必要继续搜索。
    """

    candidate = (
        int(semantic_distance),
        int(raw_distance),
    )

    old_states = frontiers.get(
        node,
        [],
    )

    for old_sem, old_raw in old_states:
        if (
            old_sem <= candidate[0]
            and old_raw <= candidate[1]
        ):
            return False

    kept = []

    for old_sem, old_raw in old_states:
        if (
            candidate[0] <= old_sem
            and candidate[1] <= old_raw
        ):
            continue

        kept.append(
            (old_sem, old_raw)
        )

    kept.append(candidate)

    frontiers[node] = kept

    return True


def anchor_roles_for_method(
    source,
    up_outgoing,
    node_records,
):
    """
    一个向上保留方法最终对应哪些五类敏感角色。
    """

    found = set()

    stack = list(
        up_outgoing.get(
            source,
            (),
        )
    )

    seen = set()

    while stack:

        current = stack.pop()

        if current in seen:
            continue

        seen.add(current)

        record = node_records[current]

        if record.get("external"):

            found.update(
                set(
                    record.get(
                        "roles",
                        [],
                    )
                )
                & TOP5
            )

            continue

        stack.extend(
            up_outgoing.get(
                current,
                (),
            )
        )

    return found


def contract_upward_structure(
    meaningful_methods,
    neutral_flags,
    up_edges,
    node_records,
):
    """
    向上结构只保留：

      1. 非纯转发内部方法
      2. 五类敏感角色

    纯转发内部方法从最终结构中消失，
    但原来的连接关系不能断。
    """

    outgoing = defaultdict(set)

    for edge in up_edges:

        source = edge.get("source")
        target = edge.get("target")

        if (
            source in node_records
            and target in node_records
        ):
            outgoing[source].add(target)

    method_edges = set()
    role_edges = set()

    for source in meaningful_methods:

        stack = list(
            outgoing.get(
                source,
                (),
            )
        )

        seen = set()

        while stack:

            target = stack.pop()

            if target in seen:
                continue

            seen.add(target)

            record = node_records[target]

            # ------------------------------------
            # 五类敏感接口：
            # 最终只保存语义角色。
            # ------------------------------------
            if record.get("external"):

                roles = (
                    set(
                        record.get(
                            "roles",
                            [],
                        )
                    )
                    & TOP5
                )

                for role in roles:
                    role_edges.add(
                        (
                            source,
                            role,
                        )
                    )

                continue

            # ------------------------------------
            # 遇到下一个真正保留的内部方法：
            # 建一条方法 -> 方法关系。
            # ------------------------------------
            if target in meaningful_methods:

                method_edges.add(
                    (
                        source,
                        target,
                    )
                )

                continue

            # ------------------------------------
            # 纯转发方法：
            # 不作为节点保留，继续穿过去。
            # ------------------------------------
            if neutral_flags.get(
                target,
                False,
            ):

                stack.extend(
                    outgoing.get(
                        target,
                        (),
                    )
                )

                continue

            # 理论上不应出现其他情况。
            # 保守处理：不继续扩张。

    return (
        method_edges,
        role_edges,
        outgoing,
    )


def collect_downward_projection(
    source,
    node_records,
    full_outgoing,
    neutral_flags,
    max_semantic_distance,
    raw_hop_cap,
    state_cap,
):
    """
    从一个“向上保留方法”向调用方向找语义。

    规则：

    纯转发内部方法：
        有意义距离 +0
        可以继续穿过
        最终不保留

    其他内部方法：
        有意义距离 +1
        可以继续搜索
        但是本次向下搜索不会把它加入最终方法节点

    有语义映射的外部接口：
        不保留具体接口身份
        只记录对应语义角色

    没有语义映射的外部接口：
        直接丢弃

    最终真正增加的只有：
        方法 -> 语义角色
    """

    frontiers = {
        source: [(0, 0)]
    }

    queue = deque([
        (
            source,
            0,
            0,
        )
    ])

    # role -> 最近有意义距离
    role_best = {}

    # 仅用于检查噪声规模。
    mapped_api_best = {}
    dropped_api_best = {}

    # 仅用于检查搜索经过多少内部节点。
    passed_neutral = set()
    passed_meaningful = set()

    state_count = 0
    truncated = False

    while queue:

        (
            current,
            current_sem,
            current_raw,
        ) = queue.popleft()

        state_count += 1

        if state_count > state_cap:
            truncated = True
            break

        if current_raw >= raw_hop_cap:
            continue

        for target in full_outgoing.get(
            current,
            (),
        ):

            if target not in node_records:
                continue

            record = node_records[target]

            next_raw = current_raw + 1

            if next_raw > raw_hop_cap:
                continue

            # ====================================
            # 外部接口
            # ====================================
            if record.get("external"):

                roles = set(
                    record.get(
                        "roles",
                        [],
                    )
                )

                # -------------------------------
                # 有语义：
                # 只留下角色。
                # -------------------------------
                if roles:

                    old = mapped_api_best.get(
                        target
                    )

                    if (
                        old is None
                        or current_sem < old
                    ):
                        mapped_api_best[
                            target
                        ] = current_sem

                    for role in roles:

                        old_role = role_best.get(
                            role
                        )

                        if (
                            old_role is None
                            or current_sem
                            < old_role
                        ):
                            role_best[
                                role
                            ] = current_sem

                # -------------------------------
                # 普通接口：
                # 直接丢掉。
                # -------------------------------
                else:

                    old = (
                        dropped_api_best.get(
                            target
                        )
                    )

                    if (
                        old is None
                        or current_sem < old
                    ):
                        dropped_api_best[
                            target
                        ] = current_sem

                # 不进入外部库内部。
                continue

            # ====================================
            # 内部方法
            # ====================================
            neutral = bool(
                neutral_flags.get(
                    target,
                    False,
                )
            )

            step = (
                0
                if neutral
                else 1
            )

            next_sem = (
                current_sem
                + step
            )

            if (
                next_sem
                > max_semantic_distance
            ):
                continue

            accepted = accept_state(
                frontiers,
                target,
                next_sem,
                next_raw,
            )

            if not accepted:
                continue

            if neutral:

                passed_neutral.add(
                    target
                )

                queue.appendleft(
                    (
                        target,
                        next_sem,
                        next_raw,
                    )
                )

            else:

                passed_meaningful.add(
                    target
                )

                queue.append(
                    (
                        target,
                        next_sem,
                        next_raw,
                    )
                )

    return {
        "role_best":
            role_best,

        "mapped_api_best":
            mapped_api_best,

        "dropped_api_best":
            dropped_api_best,

        "passed_neutral":
            passed_neutral,

        "passed_meaningful":
            passed_meaningful,

        "state_count":
            state_count,

        "truncated":
            truncated,
    }


def write_gzip_csv(
    path,
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


def mean_column(
    rows,
    name,
):

    values = [
        float(row[name])
        for row in rows
    ]

    return (
        statistics.mean(values)
        if values
        else 0.0
    )


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
            "clean-local-context-test20"
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
        "--max-down",
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

    parser.add_argument(
        "--state-cap",
        type=int,
        default=50000,
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

    configured_roles = set(
        tiers
    )

    # 防止五类规则配置异常。
    if not TOP5.issubset(
        configured_roles
    ):

        missing = sorted(
            TOP5
            - configured_roles
        )

        raise RuntimeError(
            "五类敏感角色没有全部出现在"
            "优先级配置中: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # 仍然取与之前相同方式的前 20 个 APK。
    # --------------------------------------------------------
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

            lineage_rows.append(
                row
            )

            if (
                args.limit > 0
                and len(lineage_rows)
                >= args.limit
            ):
                break

    print(
        "开始生成净化局部表示：",
        len(lineage_rows),
        "个 APK",
        flush=True,
    )

    print(
        "配置语义角色数 =",
        len(configured_roles),
        flush=True,
    )

    summary_rows = []
    context_rows = []
    structure_rows = []

    for index, lineage_row in enumerate(
        lineage_rows,
        start=1,
    ):

        apk = None
        dalvik = None
        analysis = None
        graph = None

        sha = str(
            lineage_row.get(
                "sha256",
                "",
            )
        )

        base = {
            "sha256": sha,

            "year":
                lineage_row.get(
                    "metadata_year",
                    "",
                ),

            "label":
                lineage_row.get(
                    "metadata_label",
                    "",
                ),

            "status":
                "FAILED",

            "error":
                "",
        }

        try:

            (
                apk,
                dalvik,
                analysis,
            ) = AnalyzeAPK(
                lineage_row[
                    "absolute_path"
                ]
            )

            graph = (
                analysis.get_call_graph(
                    no_isolated=False
                )
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
                neutral_audit,
            ) = build_neutral_transit_profiles(
                method_irs,
                node_records,
                edge_records,
                canonical_signature,
            )

            # =================================================
            # 第一步：
            # 从五类敏感接口向上找。
            # =================================================
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

                semantic_hops=
                    args.up_hops,

                raw_hop_cap=
                    args.raw_hop_cap,

                k_in=
                    args.k_in,

                endpoint_cap=
                    args.endpoint_cap,

                node_cap=
                    args.node_cap,

                edge_cap=
                    args.edge_cap,

                neutral_flags=
                    neutral_flags,
            )

            # -------------------------------------------------
            # 真正作为最终方法节点留下来的内部方法。
            # -------------------------------------------------
            meaningful_methods = {
                node
                for node in up_nodes

                if (
                    not node_records[
                        node
                    ].get(
                        "external"
                    )

                    and not neutral_flags.get(
                        node,
                        False,
                    )
                )
            }

            # -------------------------------------------------
            # 向上过程中被穿过、但最终删除的纯转发方法。
            # -------------------------------------------------
            neutral_up_methods = {
                node
                for node in up_nodes

                if (
                    not node_records[
                        node
                    ].get(
                        "external"
                    )

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

            full_outgoing = (
                build_outgoing(
                    edge_records,
                    node_records,
                )
            )

            # -------------------------------------------------
            # 检查语义配置是否完整。
            # -------------------------------------------------
            unexpected_roles = set()

            for record in (
                node_records.values()
            ):

                unexpected_roles.update(
                    set(
                        record.get(
                            "roles",
                            [],
                        )
                    )
                    - configured_roles
                )

            if unexpected_roles:

                raise RuntimeError(
                    "发现不在优先级配置中的"
                    "语义角色: "
                    + ", ".join(
                        sorted(
                            unexpected_roles
                        )
                    )
                )

            # =================================================
            # 第二步：
            # 对每一个“向上保留方法”单独向下找语义。
            #
            # 注意：
            # 向下经过的方法不会成为新的最终方法节点。
            # =================================================
            anchors_by_method = {}
            projection_by_method = {}

            for method in sorted(
                meaningful_methods
            ):

                anchors = (
                    anchor_roles_for_method(
                        method,
                        up_outgoing,
                        node_records,
                    )
                )

                anchors_by_method[
                    method
                ] = anchors

                projection = (
                    collect_downward_projection(
                        method,
                        node_records,
                        full_outgoing,
                        neutral_flags,

                        max_semantic_distance=
                            args.max_down,

                        raw_hop_cap=
                            args.raw_hop_cap,

                        state_cap=
                            args.state_cap,
                    )
                )

                projection_by_method[
                    method
                ] = projection

                # ---------------------------------------------
                # 保存：
                # 某个向上保留方法
                #     ->
                # 哪个语义角色
                #     ->
                # 最近需要经过多少个非纯转发内部方法
                # ---------------------------------------------
                for role, distance in sorted(
                    projection[
                        "role_best"
                    ].items()
                ):

                    context_rows.append({
                        "sha256":
                            sha,

                        "method_id":
                            short_id(
                                method
                            ),

                        # 只用于诊断。
                        # 后续分类特征不能直接使用方法名。
                        "method_signature":
                            method,

                        "up_semantic_distance":
                            up_sem_distance.get(
                                method,
                                "",
                            ),

                        "up_raw_distance":
                            up_raw_distance.get(
                                method,
                                "",
                            ),

                        "anchor_roles":
                            "|".join(
                                sorted(
                                    anchors
                                )
                            ),

                        "context_role":
                            role,

                        "down_semantic_distance":
                            distance,

                        "is_top5_context":
                            int(
                                role
                                in TOP5
                            ),
                    })

            # =================================================
            # 保存收缩后的向上结构。
            # =================================================
            for source, target in sorted(
                method_edges
            ):

                structure_rows.append({
                    "sha256":
                        sha,

                    "source_method_id":
                        short_id(source),

                    "source_signature":
                        source,

                    "target_type":
                        "method",

                    "target_method_id":
                        short_id(target),

                    "target_value":
                        target,
                })

            for source, role in sorted(
                anchor_edges
            ):

                structure_rows.append({
                    "sha256":
                        sha,

                    "source_method_id":
                        short_id(source),

                    "source_signature":
                        source,

                    "target_type":
                        "role",

                    "target_method_id":
                        "",

                    "target_value":
                        role,
                })

            base.update({
                "status":
                    "OK",

                "graph_node_count":
                    len(
                        node_records
                    ),

                "graph_edge_count":
                    len(
                        edge_records
                    ),

                "decoded_coverage":
                    neutral_audit[
                        "decoded_coverage"
                    ],

                "kept_method_count":
                    len(
                        meaningful_methods
                    ),

                "contracted_neutral_up_count":
                    len(
                        neutral_up_methods
                    ),

                "contracted_method_edge_count":
                    len(
                        method_edges
                    ),

                "anchor_role_edge_count":
                    len(
                        anchor_edges
                    ),

                "up_node_truncated":
                    int(
                        up_summary[
                            "node_truncated"
                        ]
                    ),

                "up_edge_truncated":
                    int(
                        up_summary[
                            "edge_truncated"
                        ]
                    ),
            })

            # =================================================
            # 直接从同一次搜索结果生成：
            # 向下允许 0 / 1 / 2 / 3 / 4 层时的结构统计。
            # =================================================
            for limit in range(
                args.max_down + 1
            ):

                role_types = set()

                non_top5_role_types = set()

                # 最终表示真正增加的边：
                # 保留方法 -> 语义角色
                method_role_pairs = set()

                non_top5_method_role_pairs = (
                    set()
                )

                methods_with_role = set()

                methods_with_non_top5 = (
                    set()
                )

                # 五类敏感角色
                #     ->
                # 上下文语义角色
                anchor_context_types = set()

                anchor_context_occurrences = (
                    set()
                )

                # 以下只用于统计噪声。
                mapped_api_pairs = set()
                dropped_api_pairs = set()

                passed_neutral_pairs = set()
                passed_meaningful_pairs = (
                    set()
                )

                truncated_methods = 0

                for (
                    method,
                    projection,
                ) in (
                    projection_by_method.items()
                ):

                    if projection[
                        "truncated"
                    ]:
                        truncated_methods += 1

                    anchors = (
                        anchors_by_method.get(
                            method,
                            set(),
                        )
                    )

                    current_roles = {
                        role

                        for (
                            role,
                            distance,
                        ) in projection[
                            "role_best"
                        ].items()

                        if distance <= limit
                    }

                    if current_roles:

                        methods_with_role.add(
                            method
                        )

                    for role in (
                        current_roles
                    ):

                        role_types.add(
                            role
                        )

                        method_role_pairs.add(
                            (
                                method,
                                role,
                            )
                        )

                        if role not in TOP5:

                            non_top5_role_types.add(
                                role
                            )

                            (
                                non_top5_method_role_pairs
                            ).add(
                                (
                                    method,
                                    role,
                                )
                            )

                            (
                                methods_with_non_top5
                            ).add(
                                method
                            )

                        for anchor in anchors:

                            anchor_context_types.add(
                                (
                                    anchor,
                                    role,
                                )
                            )

                            (
                                anchor_context_occurrences
                            ).add(
                                (
                                    method,
                                    anchor,
                                    role,
                                )
                            )

                    for (
                        api_node,
                        distance,
                    ) in projection[
                        "mapped_api_best"
                    ].items():

                        if distance <= limit:

                            mapped_api_pairs.add(
                                (
                                    method,
                                    api_node,
                                )
                            )

                    for (
                        api_node,
                        distance,
                    ) in projection[
                        "dropped_api_best"
                    ].items():

                        if distance <= limit:

                            dropped_api_pairs.add(
                                (
                                    method,
                                    api_node,
                                )
                            )

                    # 这些方法只是“搜索经过”，
                    # 不进入最终结构。
                    for node in projection[
                        "passed_neutral"
                    ]:

                        passed_neutral_pairs.add(
                            (
                                method,
                                node,
                            )
                        )

                    for node in projection[
                        "passed_meaningful"
                    ]:

                        (
                            passed_meaningful_pairs
                        ).add(
                            (
                                method,
                                node,
                            )
                        )

                external_total = (
                    len(
                        mapped_api_pairs
                    )
                    +
                    len(
                        dropped_api_pairs
                    )
                )

                drop_rate = (
                    len(
                        dropped_api_pairs
                    )
                    / external_total

                    if external_total
                    else 0.0
                )

                prefix = (
                    f"down{limit}"
                )

                base.update({
                    f"{prefix}_role_type_count":
                        len(
                            role_types
                        ),

                    f"{prefix}_non_top5_role_type_count":
                        len(
                            non_top5_role_types
                        ),

                    f"{prefix}_method_role_edge_count":
                        len(
                            method_role_pairs
                        ),

                    f"{prefix}_non_top5_method_role_edge_count":
                        len(
                            non_top5_method_role_pairs
                        ),

                    f"{prefix}_method_with_role_count":
                        len(
                            methods_with_role
                        ),

                    f"{prefix}_method_with_non_top5_count":
                        len(
                            methods_with_non_top5
                        ),

                    f"{prefix}_anchor_context_type_count":
                        len(
                            anchor_context_types
                        ),

                    f"{prefix}_anchor_context_occurrence_count":
                        len(
                            anchor_context_occurrences
                        ),

                    f"{prefix}_mapped_api_pair_count":
                        len(
                            mapped_api_pairs
                        ),

                    f"{prefix}_dropped_api_pair_count":
                        len(
                            dropped_api_pairs
                        ),

                    f"{prefix}_external_drop_rate":
                        drop_rate,

                    f"{prefix}_passed_neutral_pair_count":
                        len(
                            passed_neutral_pairs
                        ),

                    f"{prefix}_passed_meaningful_pair_count":
                        len(
                            passed_meaningful_pairs
                        ),

                    f"{prefix}_projection_truncated_method_count":
                        truncated_methods,

                    # -----------------------------------------
                    # 关键：
                    # 不管向下搜索几层，
                    # 最终内部方法节点都只能来自向上保留结果。
                    # -----------------------------------------
                    f"{prefix}_final_method_node_count":
                        len(
                            meaningful_methods
                        ),

                    f"{prefix}_final_role_node_count":
                        len(
                            role_types
                        ),

                    f"{prefix}_final_total_node_count":
                        (
                            len(
                                meaningful_methods
                            )
                            +
                            len(
                                role_types
                            )
                        ),

                    f"{prefix}_final_total_edge_count":
                        (
                            len(
                                method_edges
                            )
                            +
                            len(
                                method_role_pairs
                            )
                        ),
                })

            summary_rows.append(
                base
            )

            d0 = base[
                "down0_method_role_edge_count"
            ]

            dmax = base[
                f"down{args.max_down}_"
                "method_role_edge_count"
            ]

            rmax = base[
                f"down{args.max_down}_"
                "role_type_count"
            ]

            nmax = base[
                f"down{args.max_down}_"
                "non_top5_role_type_count"
            ]

            dropmax = base[
                f"down{args.max_down}_"
                "external_drop_rate"
            ]

            print(
                f"{index:2d}/"
                f"{len(lineage_rows)} "
                f"向上保留方法="
                f"{len(meaningful_methods):4d} "
                f"穿过纯转发="
                f"{len(neutral_up_methods):3d} | "
                f"向下0层角色关系="
                f"{d0:4d} | "
                f"向下{args.max_down}层角色关系="
                f"{dmax:4d} "
                f"角色={rmax:2d} "
                f"五类外={nmax:2d} "
                f"普通接口丢弃="
                f"{dropmax:.1%}",
                flush=True,
            )

        except Exception as exc:

            base["error"] = (
                f"{type(exc).__name__}: "
                f"{str(exc)[:500]}"
            )

            summary_rows.append(
                base
            )

            print(
                f"{index:2d}/"
                f"{len(lineage_rows)} "
                f"FAILED "
                f"{base['error']}",
                flush=True,
            )

        finally:

            apk = None
            dalvik = None
            analysis = None
            graph = None

            gc.collect()

    # ========================================================
    # 输出 1：
    # 每个 APK 的总体统计。
    # ========================================================
    summary_path = (
        args.output_dir
        / "apk_summary.csv"
    )

    if summary_rows:

        fieldnames = []

        for row in summary_rows:

            for key in row:

                if key not in fieldnames:
                    fieldnames.append(
                        key
                    )

        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                summary_rows
            )

    # ========================================================
    # 输出 2：
    # 每个向上保留方法 -> 每个语义角色 -> 最近距离。
    #
    # 这是后面最重要的缓存。
    # ========================================================
    write_gzip_csv(
        args.output_dir
        / "method_role_context.csv.gz",

        [
            "sha256",
            "method_id",
            "method_signature",
            "up_semantic_distance",
            "up_raw_distance",
            "anchor_roles",
            "context_role",
            "down_semantic_distance",
            "is_top5_context",
        ],

        context_rows,
    )

    # ========================================================
    # 输出 3：
    # 向上收缩之后真正保留的方法结构。
    # ========================================================
    write_gzip_csv(
        args.output_dir
        / "contracted_structure_edges.csv.gz",

        [
            "sha256",
            "source_method_id",
            "source_signature",
            "target_type",
            "target_method_id",
            "target_value",
        ],

        structure_rows,
    )

    ok_rows = [
        row
        for row in summary_rows
        if row.get("status")
        == "OK"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "净化局部表示检查结果"
    )
    print(
        "======================================"
    )

    print(
        "成功 =",
        len(ok_rows),
        "/",
        len(summary_rows),
    )

    if ok_rows:

        print()
        print("【向上保留结构】")

        avg_kept_methods = mean_column(
            ok_rows,
            "kept_method_count",
        )

        avg_neutral_up = mean_column(
            ok_rows,
            "contracted_neutral_up_count",
        )

        avg_method_edges = mean_column(
            ok_rows,
            "contracted_method_edge_count",
        )

        print(
            "平均保留内部方法 =",
            f"{avg_kept_methods:.2f}",
        )

        print(
            "平均收缩纯转发方法 =",
            f"{avg_neutral_up:.2f}",
        )

        print(
            "平均保留方法之间的边 =",
            f"{avg_method_edges:.2f}",
        )

        print()
        print(
            "【向下只保留语义角色，不新增内部方法节点】"
        )

        for limit in range(
            args.max_down + 1
        ):

            prefix = f"down{limit}"

            avg_final_methods = mean_column(
                ok_rows,
                f"{prefix}_final_method_node_count",
            )

            avg_role_nodes = mean_column(
                ok_rows,
                f"{prefix}_final_role_node_count",
            )

            avg_method_role_edges = mean_column(
                ok_rows,
                f"{prefix}_method_role_edge_count",
            )

            avg_non_top5_edges = mean_column(
                ok_rows,
                f"{prefix}_non_top5_method_role_edge_count",
            )

            avg_anchor_context = mean_column(
                ok_rows,
                f"{prefix}_anchor_context_type_count",
            )

            avg_final_nodes = mean_column(
                ok_rows,
                f"{prefix}_final_total_node_count",
            )

            avg_final_edges = mean_column(
                ok_rows,
                f"{prefix}_final_total_edge_count",
            )

            total_mapped = sum(
                int(
                    row[
                        f"{prefix}_mapped_api_pair_count"
                    ]
                )
                for row in ok_rows
            )

            total_dropped = sum(
                int(
                    row[
                        f"{prefix}_dropped_api_pair_count"
                    ]
                )
                for row in ok_rows
            )

            total_external = (
                total_mapped
                + total_dropped
            )

            global_drop_rate = (
                total_dropped / total_external
                if total_external
                else 0.0
            )

            total_truncated = sum(
                int(
                    row[
                        f"{prefix}_projection_truncated_method_count"
                    ]
                )
                for row in ok_rows
            )

            print()
            print(
                f"---- 向下允许 {limit} 层"
                "非纯转发方法 ----"
            )

            print(
                "平均最终内部方法节点 =",
                f"{avg_final_methods:.2f}",
            )

            print(
                "平均语义角色节点 =",
                f"{avg_role_nodes:.2f}",
            )

            print(
                "平均方法→语义角色关系 =",
                f"{avg_method_role_edges:.2f}",
            )

            print(
                "平均五类外方法→角色关系 =",
                f"{avg_non_top5_edges:.2f}",
            )

            print(
                "平均五类敏感角色→上下文角色类型 =",
                f"{avg_anchor_context:.2f}",
            )

            print(
                "平均最终总节点 =",
                f"{avg_final_nodes:.2f}",
            )

            print(
                "平均最终总边 =",
                f"{avg_final_edges:.2f}",
            )

            print(
                "有语义外部接口关系 =",
                total_mapped,
            )

            print(
                "丢弃普通外部接口关系 =",
                total_dropped,
            )

            print(
                "普通外部接口丢弃比例 =",
                f"{global_drop_rate:.2%}",
            )

            print(
                "搜索截断的方法总数 =",
                total_truncated,
            )

    print()
    print(
        "保存：",
        summary_path,
    )

    print(
        "保存：",
        args.output_dir
        / "method_role_context.csv.gz",
    )

    print(
        "保存：",
        args.output_dir
        / "contracted_structure_edges.csv.gz",
    )


if __name__ == "__main__":
    main()
