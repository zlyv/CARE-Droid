#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

import yaml

logging.getLogger().setLevel(logging.WARNING)

try:
    from loguru import logger as _loguru_logger

    _loguru_logger.remove()
    _loguru_logger.add(sys.stderr, level="WARNING")
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
    build_semantic_upstream_view,
)

from run_clean_local_representation_ag import (
    build_outgoing,
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
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_id(text: str, n: int = 16) -> str:
    return hashlib.sha256(
        str(text).encode("utf-8", errors="ignore")
    ).hexdigest()[:n]


def get_year(row):
    return str(
        row.get("metadata_year")
        or row.get("year")
        or ""
    ).strip()


def get_label(row):
    return str(
        row.get("metadata_label")
        or row.get("label")
        or row.get("family")
        or ""
    ).strip()


def binary_group(row):
    label = get_label(row).casefold()

    if (
        "benign" in label
        or label in {"0", "goodware", "good"}
    ):
        return "Benign"

    return "Malware"


def select_rows(
    lineage_path: Path,
    year: str,
    per_class: int,
    limit: int,
):
    rows = []

    with lineage_path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            if not row.get("absolute_path"):
                continue

            if year and get_year(row) != str(year):
                continue

            rows.append(row)

    if per_class <= 0:
        return rows[:limit] if limit else rows

    benign = [
        r for r in rows
        if binary_group(r) == "Benign"
    ]

    malware = [
        r for r in rows
        if binary_group(r) == "Malware"
    ]

    selected = (
        benign[:per_class]
        + malware[:per_class]
    )

    if limit:
        selected = selected[:limit]

    return selected


def build_reverse_method_edges(method_edges):
    reverse = defaultdict(set)

    for source, target in method_edges:
        reverse[target].add(source)

    return reverse


def collect_organizers(
    carrier,
    reverse_method_edges,
    up_hops,
):
    """
    carrier 本身已经是距离 Top-5 最近的
    有意义内部承载方法之一。

    up_hops=4 表示：
      carrier + 最多3层上游有意义方法。
    """

    max_reverse_steps = max(
        0,
        int(up_hops) - 1,
    )

    best = {
        carrier: 0
    }

    queue = deque([
        (carrier, 0)
    ])

    while queue:

        current, distance = queue.popleft()

        if distance >= max_reverse_steps:
            continue

        for caller in reverse_method_edges.get(
            current,
            (),
        ):
            next_distance = distance + 1

            old = best.get(caller)

            if (
                old is not None
                and old <= next_distance
            ):
                continue

            best[caller] = next_distance

            queue.append(
                (
                    caller,
                    next_distance,
                )
            )

    return best


def add_edge(
    edge_map,
    source,
    target,
    edge_type,
    raw_hops,
    neutral_hops,
):
    key = (
        source,
        target,
        edge_type,
    )

    candidate = {
        "source": source,
        "target": target,
        "type": edge_type,
        "raw_hops": int(raw_hops),
        "neutral_hops": int(neutral_hops),
    }

    old = edge_map.get(key)

    if old is None:
        edge_map[key] = candidate
        return

    # 同一压缩边若存在多条路径，
    # 审计图保留最短原始路径。
    if (
        candidate["raw_hops"],
        candidate["neutral_hops"],
    ) < (
        old["raw_hops"],
        old["neutral_hops"],
    ):
        edge_map[key] = candidate


def accept_best_state(
    best,
    key,
    semantic_distance,
    raw_distance,
    raw_since_kept,
):
    """
    同一 (current, last_kept) 只保留
    语义距离更小，或语义距离相同而
    原始距离更小的状态。
    """

    candidate = (
        int(semantic_distance),
        int(raw_distance),
        int(raw_since_kept),
    )

    old = best.get(key)

    if old is not None and old <= candidate:
        return False

    best[key] = candidate
    return True


def collect_behavior_region(
    organizer_methods,
    carrier,
    anchor_role,
    node_records,
    full_outgoing,
    neutral_flags,
    max_down,
    raw_hop_cap,
    state_cap,
):
    """
    与旧 collect_downward_projection 的核心区别：

    旧版：
        向下经过的有意义方法不进入最终结构，
        最后压成 method -> role。

    新版：
        有意义方法继续作为 method 节点保留；
        中性方法透明化；
        外部具体 API 不保留身份，只产生
        semantic occurrence 节点。

    最终得到：
        method -> method -> semantic

    而不是：
        anchor -> role
    """

    method_nodes = set(
        organizer_methods
    )

    semantic_nodes = {}
    edge_map = {}

    passed_neutral = set()

    state_count = 0
    truncated = False

    # 状态：
    # current
    # last_kept meaningful method
    # semantic distance from this search origin
    # total raw distance from this search origin
    # raw hops since last kept method
    queue = deque()

    best = {}

    for source in sorted(
        organizer_methods
    ):
        key = (
            source,
            source,
        )

        best[key] = (
            0,
            0,
            0,
        )

        queue.append(
            (
                source,
                source,
                0,
                0,
                0,
            )
        )

    while queue:

        (
            current,
            last_kept,
            current_sem,
            current_raw,
            raw_since_kept,
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
            # 外部 API
            # ====================================
            if record.get("external"):

                roles = sorted(
                    set(
                        record.get(
                            "roles",
                            [],
                        )
                    )
                )

                if not roles:
                    continue

                for role in roles:

                    # 保留“语义出现实例”，
                    # 不把同角色所有 API 强行合成一个节点。
                    #
                    # 具体 API 签名不会进入训练表示；
                    # hash 仅用于区分图中的多个语义实例。
                    semantic_id = (
                        "S:"
                        + role
                        + ":"
                        + stable_id(target, 12)
                    )

                    semantic_nodes[
                        semantic_id
                    ] = {
                        "id": semantic_id,
                        "type": "semantic",
                        "role": role,
                        "is_anchor_role":
                            int(
                                role == anchor_role
                            ),
                    }

                    add_edge(
                        edge_map,
                        last_kept,
                        semantic_id,
                        "method_to_semantic",
                        raw_since_kept + 1,
                        raw_since_kept,
                    )

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

            if neutral:

                passed_neutral.add(
                    target
                )

                key = (
                    target,
                    last_kept,
                )

                next_raw_since = (
                    raw_since_kept + 1
                )

                if not accept_best_state(
                    best,
                    key,
                    current_sem,
                    next_raw,
                    next_raw_since,
                ):
                    continue

                # 中性节点 cost=0，
                # 优先继续穿过。
                queue.appendleft(
                    (
                        target,
                        last_kept,
                        current_sem,
                        next_raw,
                        next_raw_since,
                    )
                )

                continue

            # ------------------------------------
            # 有意义内部方法 cost=1
            # ------------------------------------
            next_sem = current_sem + 1

            if next_sem > max_down:
                continue

            method_nodes.add(
                target
            )

            if last_kept != target:

                add_edge(
                    edge_map,
                    last_kept,
                    target,
                    "method_to_method",
                    raw_since_kept + 1,
                    raw_since_kept,
                )

            key = (
                target,
                target,
            )

            if not accept_best_state(
                best,
                key,
                next_sem,
                next_raw,
                0,
            ):
                continue

            queue.append(
                (
                    target,
                    target,
                    next_sem,
                    next_raw,
                    0,
                )
            )

    # =====================================================
    # 剪掉最终到不了任何已映射语义的内部方法分支。
    #
    # 这样普通 UI / 普通业务分支不会因为和某个 organizer
    # 共用入口就永久留在行为图。
    # =====================================================
    reverse = defaultdict(set)

    for edge in edge_map.values():
        reverse[
            edge["target"]
        ].add(
            edge["source"]
        )

    keep_nodes = set(
        semantic_nodes
    )

    queue = deque(
        semantic_nodes.keys()
    )

    while queue:

        current = queue.popleft()

        for parent in reverse.get(
            current,
            (),
        ):
            if parent in keep_nodes:
                continue

            keep_nodes.add(parent)
            queue.append(parent)

    kept_method_nodes = {
        node
        for node in method_nodes
        if node in keep_nodes
    }

    kept_semantic_nodes = {
        node: info
        for node, info
        in semantic_nodes.items()
        if node in keep_nodes
    }

    kept_edges = [
        edge
        for edge in edge_map.values()
        if (
            edge["source"] in keep_nodes
            and edge["target"] in keep_nodes
        )
    ]

    roles = sorted({
        info["role"]
        for info
        in kept_semantic_nodes.values()
    })

    valid = (
        carrier in kept_method_nodes
        and anchor_role in roles
        and len(roles) >= 2
    )

    invalid_reason = ""

    if carrier not in kept_method_nodes:
        invalid_reason = "ANCHOR_CARRIER_NOT_RETAINED"

    elif anchor_role not in roles:
        invalid_reason = "ANCHOR_ROLE_NOT_RETAINED"

    elif len(roles) < 2:
        invalid_reason = "ONLY_ONE_SEMANTIC_ROLE"

    if truncated:
        # 仍保存图供审计，但明确标记。
        invalid_reason = (
            invalid_reason + "|TRUNCATED"
            if invalid_reason
            else "TRUNCATED"
        )

    # -----------------------------------------
    # 图结构统计
    # -----------------------------------------
    local_out_degree = defaultdict(int)
    local_in_degree = defaultdict(int)

    for edge in kept_edges:

        if edge["type"] == "method_to_method":
            local_out_degree[
                edge["source"]
            ] += 1

            local_in_degree[
                edge["target"]
            ] += 1

    branching_methods = {
        m
        for m in kept_method_nodes
        if local_out_degree[m] >= 2
    }

    max_method_out_degree = max(
        (
            local_out_degree[m]
            for m in kept_method_nodes
        ),
        default=0,
    )

    return {
        "valid": bool(valid),
        "invalid_reason": invalid_reason,
        "method_nodes":
            kept_method_nodes,
        "semantic_nodes":
            kept_semantic_nodes,
        "edges":
            kept_edges,
        "roles":
            roles,
        "passed_neutral":
            passed_neutral,
        "state_count":
            state_count,
        "truncated":
            bool(truncated),
        "branching_methods":
            branching_methods,
        "max_method_out_degree":
            max_method_out_degree,
    }


def dot_escape(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )


def write_dot(
    path,
    behavior,
    carrier,
    anchor_role,
    organizer_methods,
    node_records,
):
    """
    DOT 只用于人工审计。
    可以显示 method_name，
    但后续训练严禁使用这些名字。
    """

    lines = [
        "digraph Behavior {",
        "  rankdir=LR;",
        '  graph [label="'
        + dot_escape(
            f"anchor={anchor_role}"
        )
        + '", labelloc=t];',
    ]

    for method in sorted(
        behavior["method_nodes"]
    ):
        record = node_records.get(
            method,
            {},
        )

        method_name = str(
            record.get(
                "method_name",
                "",
            )
        )

        tags = []

        if method == carrier:
            tags.append("CARRIER")

        if method in organizer_methods:
            tags.append("ORG")

        label = (
            "M:"
            + stable_id(method, 10)
        )

        if method_name:
            label += "\\n" + method_name[:40]

        if tags:
            label += "\\n[" + ",".join(tags) + "]"

        lines.append(
            '  "'
            + dot_escape(method)
            + '" [shape=box,label="'
            + dot_escape(label)
            + '"];'
        )

    for (
        semantic_id,
        info,
    ) in sorted(
        behavior[
            "semantic_nodes"
        ].items()
    ):
        label = info["role"]

        if info["role"] == anchor_role:
            label += "\\n[ANCHOR ROLE]"

        lines.append(
            '  "'
            + dot_escape(semantic_id)
            + '" [shape=ellipse,label="'
            + dot_escape(label)
            + '"];'
        )

    for edge in behavior["edges"]:

        label = ""

        if edge["neutral_hops"]:
            label = (
                f' [label="raw={edge["raw_hops"]},'
                f' neutral={edge["neutral_hops"]}"]'
            )

        lines.append(
            '  "'
            + dot_escape(edge["source"])
            + '" -> "'
            + dot_escape(edge["target"])
            + '"'
            + label
            + ";"
        )

    lines.append("}")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
        "--output-dir",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--year",
        default="2017",
    )

    ap.add_argument(
        "--per-class",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--up-hops",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--max-down",
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

    ap.add_argument(
        "--state-cap",
        type=int,
        default=50000,
    )

    ap.add_argument(
        "--dot-limit",
        type=int,
        default=80,
    )

    args = ap.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dot_dir = (
        args.output_dir / "dot"
    )

    dot_dir.mkdir(
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

    rows = select_rows(
        args.lineage,
        args.year,
        args.per_class,
        args.limit,
    )

    print(
        "selected APKs =",
        len(rows),
    )

    print(
        "year =",
        args.year,
    )

    print(
        "Benign =",
        sum(
            binary_group(r)
            == "Benign"
            for r in rows
        ),
    )

    print(
        "Malware =",
        sum(
            binary_group(r)
            == "Malware"
            for r in rows
        ),
    )

    behavior_path = (
        args.output_dir
        / "behaviors.jsonl"
    )

    summary_path = (
        args.output_dir
        / "apk_summary.csv"
    )

    method_debug_path = (
        args.output_dir
        / "method_debug.tsv"
    )

    neutral_debug_path = (
        args.output_dir
        / "neutral_debug.tsv"
    )

    summaries = []

    dot_written = 0

    with (
        behavior_path.open(
            "w",
            encoding="utf-8",
        ) as behavior_f,
        method_debug_path.open(
            "w",
            encoding="utf-8",
        ) as method_debug_f,
        neutral_debug_path.open(
            "w",
            encoding="utf-8",
        ) as neutral_debug_f,
    ):

        method_debug_f.write(
            "sha256\tbehavior_id\tmethod_id\t"
            "is_carrier\tis_organizer\t"
            "method_signature\tmethod_name\n"
        )

        neutral_debug_f.write(
            "sha256\tbehavior_id\tmethod_id\t"
            "method_signature\treason\t"
            "instruction_count\tinvoke_count\t"
            "return_count\tin_degree\tout_degree\n"
        )

        for index, lineage_row in enumerate(
            rows,
            start=1,
        ):

            sha = str(
                lineage_row.get("sha256")
                or lineage_row.get("sha")
                or ""
            ).strip()

            if not sha:
                sha = stable_id(
                    lineage_row[
                        "absolute_path"
                    ],
                    64,
                )

            label = get_label(
                lineage_row
            )

            binary_label = binary_group(
                lineage_row
            )

            print(
                f"[{index}/{len(rows)}] "
                f"{sha[:12]} "
                f"{binary_label}"
            )

            apk = None
            dalvik = None
            analysis = None
            graph = None

            summary = {
                "sha256": sha,
                "year": get_year(
                    lineage_row
                ),
                "label": label,
                "binary_label":
                    binary_label,
                "status": "FAILED",
                "error": "",
                "graph_internal_nodes": 0,
                "decoded_internal_nodes": 0,
                "neutral_transit_nodes": 0,
                "neutral_rate_all_internal": 0.0,
                "top5_anchor_instances": 0,
                "valid_behavior_count": 0,
                "invalid_behavior_count": 0,
                "truncated_behavior_count": 0,
                "total_behavior_method_nodes": 0,
                "total_behavior_semantic_nodes": 0,
                "total_behavior_role_types": 0,
                "total_neutral_bypassed": 0,
                "max_behavior_method_nodes": 0,
                "max_behavior_role_types": 0,
                "max_method_out_degree": 0,
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
                    profiles,
                    neutral_audit,
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

                (
                    method_edges,
                    anchor_edges,
                    _,
                ) = contract_upward_structure(
                    meaningful_methods,
                    neutral_flags,
                    up_edges,
                    node_records,
                )

                reverse_method_edges = (
                    build_reverse_method_edges(
                        method_edges
                    )
                )

                full_outgoing = build_outgoing(
                    edge_records,
                    node_records,
                )

                anchor_instances = sorted(
                    anchor_edges
                )

                summary[
                    "top5_anchor_instances"
                ] = len(anchor_instances)

                behavior_method_total = 0
                behavior_semantic_total = 0
                behavior_role_total = 0
                behavior_neutral_total = 0

                max_behavior_methods = 0
                max_behavior_roles = 0
                max_method_out_degree = 0

                valid_count = 0
                invalid_count = 0
                truncated_count = 0

                for (
                    behavior_index,
                    (
                        carrier,
                        anchor_role,
                    ),
                ) in enumerate(
                    anchor_instances,
                    start=1,
                ):

                    organizer_distance = (
                        collect_organizers(
                            carrier,
                            reverse_method_edges,
                            args.up_hops,
                        )
                    )

                    organizer_methods = set(
                        organizer_distance
                    )

                    behavior = (
                        collect_behavior_region(
                            organizer_methods,
                            carrier,
                            anchor_role,
                            node_records,
                            full_outgoing,
                            neutral_flags,
                            args.max_down,
                            args.raw_hop_cap,
                            args.state_cap,
                        )
                    )

                    behavior_id = (
                        sha[:12]
                        + "__"
                        + stable_id(
                            carrier,
                            10,
                        )
                        + "__"
                        + anchor_role
                    )

                    if behavior["valid"]:
                        valid_count += 1
                    else:
                        invalid_count += 1

                    if behavior["truncated"]:
                        truncated_count += 1

                    method_count = len(
                        behavior[
                            "method_nodes"
                        ]
                    )

                    semantic_count = len(
                        behavior[
                            "semantic_nodes"
                        ]
                    )

                    role_count = len(
                        behavior["roles"]
                    )

                    neutral_count = len(
                        behavior[
                            "passed_neutral"
                        ]
                    )

                    behavior_method_total += (
                        method_count
                    )

                    behavior_semantic_total += (
                        semantic_count
                    )

                    behavior_role_total += (
                        role_count
                    )

                    behavior_neutral_total += (
                        neutral_count
                    )

                    max_behavior_methods = max(
                        max_behavior_methods,
                        method_count,
                    )

                    max_behavior_roles = max(
                        max_behavior_roles,
                        role_count,
                    )

                    max_method_out_degree = max(
                        max_method_out_degree,
                        behavior[
                            "max_method_out_degree"
                        ],
                    )

                    safe_nodes = []

                    for method in sorted(
                        behavior[
                            "method_nodes"
                        ]
                    ):

                        safe_nodes.append({
                            "id":
                                "M:"
                                + stable_id(
                                    method,
                                    16,
                                ),
                            "raw_id":
                                method,
                            "type":
                                "method",
                            "is_carrier":
                                int(
                                    method
                                    == carrier
                                ),
                            "is_organizer":
                                int(
                                    method
                                    in organizer_methods
                                ),
                            "organizer_distance":
                                organizer_distance.get(
                                    method,
                                    "",
                                ),
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
                        })

                    for (
                        semantic_id,
                        info,
                    ) in sorted(
                        behavior[
                            "semantic_nodes"
                        ].items()
                    ):

                        safe_nodes.append({
                            "id":
                                semantic_id,
                            "type":
                                "semantic",
                            "role":
                                info["role"],
                            "is_anchor_role":
                                info[
                                    "is_anchor_role"
                                ],
                        })

                    # 把 method 原始签名改成稳定匿名 ID。
                    raw_to_safe = {
                        method:
                            "M:"
                            + stable_id(
                                method,
                                16,
                            )
                        for method
                        in behavior[
                            "method_nodes"
                        ]
                    }

                    safe_edges = []

                    for edge in behavior["edges"]:

                        source = edge["source"]
                        target = edge["target"]

                        safe_edges.append({
                            "source":
                                raw_to_safe.get(
                                    source,
                                    source,
                                ),
                            "target":
                                raw_to_safe.get(
                                    target,
                                    target,
                                ),
                            "type":
                                edge["type"],
                            "raw_hops":
                                edge[
                                    "raw_hops"
                                ],
                            "neutral_hops":
                                edge[
                                    "neutral_hops"
                                ],
                        })

                    behavior_row = {
                        "sha256": sha,
                        "year":
                            get_year(
                                lineage_row
                            ),
                        "label":
                            label,
                        "binary_label":
                            binary_label,
                        "behavior_id":
                            behavior_id,
                        "anchor_role":
                            anchor_role,
                        "anchor_method_id":
                            raw_to_safe.get(
                                carrier,
                                "M:"
                                + stable_id(
                                    carrier,
                                    16,
                                ),
                            ),
                        "valid_behavior":
                            int(
                                behavior[
                                    "valid"
                                ]
                            ),
                        "invalid_reason":
                            behavior[
                                "invalid_reason"
                            ],
                        "method_node_count":
                            method_count,
                        "semantic_node_count":
                            semantic_count,
                        "semantic_role_count":
                            role_count,
                        "semantic_roles":
                            behavior["roles"],
                        "organizer_count":
                            len(
                                organizer_methods
                            ),
                        "branching_method_count":
                            len(
                                behavior[
                                    "branching_methods"
                                ]
                            ),
                        "max_method_out_degree":
                            behavior[
                                "max_method_out_degree"
                            ],
                        "neutral_bypassed_count":
                            neutral_count,
                        "state_count":
                            behavior[
                                "state_count"
                            ],
                        "truncated":
                            int(
                                behavior[
                                    "truncated"
                                ]
                            ),
                        "nodes":
                            safe_nodes,
                        "edges":
                            safe_edges,
                    }

                    behavior_f.write(
                        json.dumps(
                            behavior_row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # =================================
                    # 调试映射：
                    # 仅用于人工审计，
                    # 严禁后续分类器使用。
                    # =================================
                    for method in sorted(
                        behavior[
                            "method_nodes"
                        ]
                    ):

                        record = node_records.get(
                            method,
                            {},
                        )

                        method_debug_f.write(
                            "\t".join([
                                sha,
                                behavior_id,
                                "M:"
                                + stable_id(
                                    method,
                                    16,
                                ),
                                str(
                                    int(
                                        method
                                        == carrier
                                    )
                                ),
                                str(
                                    int(
                                        method
                                        in organizer_methods
                                    )
                                ),
                                method.replace(
                                    "\t",
                                    " ",
                                ),
                                str(
                                    record.get(
                                        "method_name",
                                        "",
                                    )
                                ).replace(
                                    "\t",
                                    " ",
                                ),
                            ])
                            + "\n"
                        )

                    for neutral in sorted(
                        behavior[
                            "passed_neutral"
                        ]
                    ):

                        profile = profiles.get(
                            neutral,
                            {},
                        )

                        neutral_debug_f.write(
                            "\t".join([
                                sha,
                                behavior_id,
                                "N:"
                                + stable_id(
                                    neutral,
                                    16,
                                ),
                                neutral.replace(
                                    "\t",
                                    " ",
                                ),
                                str(
                                    profile.get(
                                        "reason",
                                        "",
                                    )
                                ),
                                str(
                                    profile.get(
                                        "instruction_count",
                                        "",
                                    )
                                ),
                                str(
                                    profile.get(
                                        "invoke_count",
                                        "",
                                    )
                                ),
                                str(
                                    profile.get(
                                        "return_count",
                                        "",
                                    )
                                ),
                                str(
                                    profile.get(
                                        "in_degree",
                                        "",
                                    )
                                ),
                                str(
                                    profile.get(
                                        "out_degree",
                                        "",
                                    )
                                ),
                            ])
                            + "\n"
                        )

                    if (
                        behavior["valid"]
                        and dot_written
                        < args.dot_limit
                    ):

                        dot_path = (
                            dot_dir
                            / (
                                f"{index:02d}_"
                                f"{behavior_index:03d}_"
                                f"{binary_label}_"
                                f"{anchor_role}_"
                                f"{stable_id(carrier, 8)}"
                                ".dot"
                            )
                        )

                        write_dot(
                            dot_path,
                            behavior,
                            carrier,
                            anchor_role,
                            organizer_methods,
                            node_records,
                        )

                        dot_written += 1

                summary.update({
                    "status": "OK",

                    "graph_internal_nodes":
                        neutral_audit[
                            "internal_node_count"
                        ],

                    "decoded_internal_nodes":
                        neutral_audit[
                            "decoded_internal_count"
                        ],

                    "neutral_transit_nodes":
                        neutral_audit[
                            "neutral_transit_count"
                        ],

                    "neutral_rate_all_internal":
                        neutral_audit[
                            "neutral_rate_all_internal"
                        ],

                    "valid_behavior_count":
                        valid_count,

                    "invalid_behavior_count":
                        invalid_count,

                    "truncated_behavior_count":
                        truncated_count,

                    "total_behavior_method_nodes":
                        behavior_method_total,

                    "total_behavior_semantic_nodes":
                        behavior_semantic_total,

                    "total_behavior_role_types":
                        behavior_role_total,

                    "total_neutral_bypassed":
                        behavior_neutral_total,

                    "max_behavior_method_nodes":
                        max_behavior_methods,

                    "max_behavior_role_types":
                        max_behavior_roles,

                    "max_method_out_degree":
                        max_method_out_degree,
                })

                print(
                    "  anchors=",
                    len(anchor_instances),
                    "valid=",
                    valid_count,
                    "invalid=",
                    invalid_count,
                    "truncated=",
                    truncated_count,
                    "max_methods=",
                    max_behavior_methods,
                    "max_roles=",
                    max_behavior_roles,
                )

            except Exception as exc:

                summary["error"] = (
                    type(exc).__name__
                    + ": "
                    + str(exc)
                )

                print(
                    "  FAILED:",
                    summary["error"],
                )

            finally:

                summaries.append(
                    summary
                )

                graph = None
                analysis = None
                dalvik = None
                apk = None

                gc.collect()

    fieldnames = [
        "sha256",
        "year",
        "label",
        "binary_label",
        "status",
        "error",
        "graph_internal_nodes",
        "decoded_internal_nodes",
        "neutral_transit_nodes",
        "neutral_rate_all_internal",
        "top5_anchor_instances",
        "valid_behavior_count",
        "invalid_behavior_count",
        "truncated_behavior_count",
        "total_behavior_method_nodes",
        "total_behavior_semantic_nodes",
        "total_behavior_role_types",
        "total_neutral_bypassed",
        "max_behavior_method_nodes",
        "max_behavior_role_types",
        "max_method_out_degree",
    ]

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

        for row in summaries:
            writer.writerow(row)

    ok_rows = [
        row
        for row in summaries
        if row["status"] == "OK"
    ]

    print()
    print("===== DONE =====")
    print(
        "APK OK =",
        len(ok_rows),
        "/",
        len(summaries),
    )

    print(
        "valid behaviors =",
        sum(
            int(
                row[
                    "valid_behavior_count"
                ]
            )
            for row in ok_rows
        ),
    )

    print(
        "invalid behaviors =",
        sum(
            int(
                row[
                    "invalid_behavior_count"
                ]
            )
            for row in ok_rows
        ),
    )

    print(
        "truncated behaviors =",
        sum(
            int(
                row[
                    "truncated_behavior_count"
                ]
            )
            for row in ok_rows
        ),
    )

    print(
        "DOT written =",
        dot_written,
    )

    print(
        "output =",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
