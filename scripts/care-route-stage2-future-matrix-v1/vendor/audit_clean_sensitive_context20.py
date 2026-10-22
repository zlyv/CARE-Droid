#!/usr/bin/env python3

from __future__ import annotations

import csv
import gc
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


# ------------------------------------------------------------
# 仍然只有这五类决定“从哪里开始分析”
# ------------------------------------------------------------
TOP5 = {
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
}


LINEAGE = Path(
    "artifacts/p0-lineage/"
    "package_certificate_lineage.csv"
)

MAPPING = Path(
    "configs/protocol/"
    "semantic_mapping_v1.yaml"
)

PRIORITY = Path(
    "configs/protocol/"
    "sensitive_endpoint_priority_v1.yaml"
)

EDGE_MAPPING = Path(
    "configs/protocol/"
    "semantic_edge_mapping_v1.yaml"
)

OUTPUT = Path(
    "artifacts/top5-semantic-backbone-test20/"
    "clean_sensitive_context.csv"
)


UP_HOPS = 4
RAW_HOP_CAP = 16

K_IN = 8
ENDPOINT_CAP = 128
NODE_CAP = 2048
EDGE_CAP = 4096

# 向下只做结构检查，一次看 0~4 层。
DOWN_LIMITS = (0, 1, 2, 3, 4)

# 防止异常图把诊断拖死。
STATE_CAP = 50000


def load_yaml(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def mean(values):
    return statistics.mean(values) if values else 0.0


def build_outgoing(edge_records, node_records):
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


def add_frontier(frontiers, node, semantic_dist, raw_dist):
    """
    一个节点可能通过多条路径到达。

    只有当新状态没有被旧状态同时在
    “有意义层数”和“原始层数”上压制时才保留。
    """

    candidate = (
        int(semantic_dist),
        int(raw_dist),
    )

    existing = frontiers.get(node, [])

    # 已经有更好的状态。
    for old_sem, old_raw in existing:
        if (
            old_sem <= candidate[0]
            and old_raw <= candidate[1]
        ):
            return False

    # 删除被新状态完全压制的旧状态。
    kept = []

    for old_sem, old_raw in existing:
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


def collect_clean_downward_context(
    seeds,
    node_records,
    outgoing,
    neutral_flags,
    semantic_limit,
):
    """
    从已经通过敏感接口向上找到的“有效内部方法”出发，
    再向调用方向检查局部程序行为。

    规则：

    1. 纯转发内部方法
       -> 不保留
       -> 有意义距离 +0
       -> 可以继续穿过去

    2. 真正有逻辑的内部方法
       -> 保留
       -> 有意义距离 +1

    3. 有语义映射的外部接口
       -> 不保留具体接口名
       -> 只留下对应的语义角色
       -> 不继续穿过外部接口

    4. 没有语义映射的外部接口
       -> 直接丢弃
       -> 不继续遍历

    semantic_limit = 0 时：
       只允许穿过纯转发方法，不允许进入新的有效内部方法。
    """

    seeds = set(seeds)

    frontiers = {}
    queue = deque()

    for seed in seeds:
        if seed not in node_records:
            continue

        frontiers[seed] = [(0, 0)]
        queue.append(
            (seed, 0, 0)
        )

    # 最终留下的内部有效方法。
    kept_meaningful = set(seeds)

    # 遍历时穿过，但最终不保留。
    contracted_neutral = set()

    # 有语义映射的具体外部接口。
    # 只用于统计，最终表示不需要保留接口身份。
    mapped_external = set()

    # 最终真正留下的是这些语义角色。
    kept_roles = set()

    # 普通外部接口：全部丢弃。
    dropped_external = set()

    # 每个角色离最近保留方法的最小有意义距离。
    role_distance = {}

    state_count = 0
    truncated = False

    while queue:

        current, current_sem, current_raw = (
            queue.popleft()
        )

        state_count += 1

        if state_count > STATE_CAP:
            truncated = True
            break

        if current_raw >= RAW_HOP_CAP:
            continue

        for target in outgoing.get(
            current, ()
        ):

            rec = node_records[target]

            next_raw = current_raw + 1

            if next_raw > RAW_HOP_CAP:
                continue

            # ------------------------------------------------
            # 外部接口：
            # 有语义 -> 转换成角色
            # 无语义 -> 丢弃
            # ------------------------------------------------
            if rec.get("external"):

                roles = set(
                    rec.get("roles", [])
                )

                if roles:
                    mapped_external.add(
                        target
                    )

                    for role in roles:
                        kept_roles.add(role)

                        old = role_distance.get(
                            role
                        )

                        if (
                            old is None
                            or current_sem < old
                        ):
                            role_distance[
                                role
                            ] = current_sem

                else:
                    dropped_external.add(
                        target
                    )

                # 外部接口作为终点，
                # 不再继续向 framework 内部走。
                continue

            # ------------------------------------------------
            # 内部方法。
            # ------------------------------------------------
            neutral = bool(
                neutral_flags.get(
                    target,
                    False,
                )
            )

            if neutral:
                step = 0
            else:
                step = 1

            next_sem = (
                current_sem + step
            )

            if next_sem > semantic_limit:
                continue

            accepted = add_frontier(
                frontiers,
                target,
                next_sem,
                next_raw,
            )

            if not accepted:
                continue

            if neutral:
                contracted_neutral.add(
                    target
                )
            else:
                kept_meaningful.add(
                    target
                )

            # 0 层优先向前穿透，
            # 有意义方法正常排队。
            if neutral:
                queue.appendleft(
                    (
                        target,
                        next_sem,
                        next_raw,
                    )
                )
            else:
                queue.append(
                    (
                        target,
                        next_sem,
                        next_raw,
                    )
                )

    non_top5_roles = (
        kept_roles - TOP5
    )

    top5_roles = (
        kept_roles & TOP5
    )

    raw_external_seen = (
        len(mapped_external)
        + len(dropped_external)
    )

    if raw_external_seen:
        drop_rate = (
            len(dropped_external)
            / raw_external_seen
        )
    else:
        drop_rate = 0.0

    return {
        "kept_meaningful":
            kept_meaningful,

        "contracted_neutral":
            contracted_neutral,

        "mapped_external":
            mapped_external,

        "kept_roles":
            kept_roles,

        "top5_roles":
            top5_roles,

        "non_top5_roles":
            non_top5_roles,

        "dropped_external":
            dropped_external,

        "role_distance":
            role_distance,

        "drop_rate":
            drop_rate,

        "state_count":
            state_count,

        "truncated":
            truncated,
    }


def main():

    # --------------------------------------------------------
    # 配置
    # --------------------------------------------------------
    mapping = load_yaml(MAPPING)
    priority = load_yaml(PRIORITY)
    edge_mapping = load_yaml(
        EDGE_MAPPING
    )

    rules = list(
        mapping.get("rules", [])
    )

    tiers = role_tiers(priority)

    edge_rules = list(
        edge_mapping.get("rules", [])
    )

    # --------------------------------------------------------
    # 与之前保持一致：取同样的前 20 个 APK。
    # --------------------------------------------------------
    rows = []

    with LINEAGE.open(
        newline="",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):

            if not row.get(
                "absolute_path"
            ):
                continue

            rows.append(row)

            if len(rows) >= 20:
                break

    print(
        "开始检查 20 个 APK",
        flush=True,
    )

    outputs = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        apk = None
        dalvik = None
        analysis = None
        graph = None

        base = {
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

            "status": "FAILED",

            "error": "",
        }

        try:
            apk, dalvik, analysis = (
                AnalyzeAPK(
                    row["absolute_path"]
                )
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

            ir_audit = {}

            method_irs = build_method_ir(
                dalvik,
                analysis,
                ir_audit,
            )

            (
                neutral_flags,
                neutral_profiles,
                neutral_audit,
            ) = build_neutral_transit_profiles(
                method_irs,
                node_records,
                edge_records,
                canonical_signature,
            )

            # ------------------------------------------------
            # 第一步：
            # 从五类敏感接口向上找。
            #
            # 纯转发方法不消耗有意义层数。
            # ------------------------------------------------
            (
                upstream_nodes,
                upstream_edges,
                upstream_summary,
                upstream_sem_distance,
                upstream_raw_distance,
            ) = build_semantic_upstream_view(
                node_records,
                edge_records,
                tiers,
                TOP5,

                semantic_hops=UP_HOPS,
                raw_hop_cap=RAW_HOP_CAP,

                k_in=K_IN,
                endpoint_cap=ENDPOINT_CAP,
                node_cap=NODE_CAP,
                edge_cap=EDGE_CAP,

                neutral_flags=neutral_flags,
            )

            upstream_meaningful = {
                signature
                for signature
                in upstream_nodes

                if (
                    not node_records[
                        signature
                    ].get("external")

                    and not neutral_flags.get(
                        signature,
                        False,
                    )
                )
            }

            upstream_neutral = {
                signature
                for signature
                in upstream_nodes

                if (
                    not node_records[
                        signature
                    ].get("external")

                    and neutral_flags.get(
                        signature,
                        False,
                    )
                )
            }

            upstream_roles = set()

            for signature in upstream_nodes:

                rec = node_records[
                    signature
                ]

                if rec.get("external"):
                    upstream_roles.update(
                        rec.get(
                            "roles", []
                        )
                    )

            outgoing = build_outgoing(
                edge_records,
                node_records,
            )

            base.update({
                "status": "OK",

                "graph_node_count":
                    len(node_records),

                "graph_edge_count":
                    len(edge_records),

                "decoded_coverage":
                    neutral_audit[
                        "decoded_coverage"
                    ],

                # 向上找以后，真正留下的内部方法。
                "up_kept_meaningful":
                    len(
                        upstream_meaningful
                    ),

                # 向上过程中穿过，但不应进入最终表示。
                "up_contracted_neutral":
                    len(
                        upstream_neutral
                    ),

                "up_role_count":
                    len(
                        upstream_roles
                    ),

                "up_roles":
                    "|".join(
                        sorted(
                            upstream_roles
                        )
                    ),

                "up_node_truncated":
                    int(
                        upstream_summary[
                            "node_truncated"
                        ]
                    ),

                "up_edge_truncated":
                    int(
                        upstream_summary[
                            "edge_truncated"
                        ]
                    ),
            })

            # ------------------------------------------------
            # 第二步：
            # 从向上找到的有效内部方法，再向下做净化搜索。
            #
            # 一次检查 0/1/2/3/4 层，不选性能最佳值。
            # ------------------------------------------------
            projections = {}

            for limit in DOWN_LIMITS:

                result = (
                    collect_clean_downward_context(
                        upstream_meaningful,
                        node_records,
                        outgoing,
                        neutral_flags,
                        semantic_limit=limit,
                    )
                )

                projections[limit] = result

                prefix = f"down{limit}"

                base.update({
                    # 最终会留下的有效内部方法。
                    f"{prefix}_kept_meaningful":
                        len(
                            result[
                                "kept_meaningful"
                            ]
                        ),

                    # 只穿过去、不留下的纯转发方法。
                    f"{prefix}_contracted_neutral":
                        len(
                            result[
                                "contracted_neutral"
                            ]
                        ),

                    # 碰到的有语义外部接口数量。
                    f"{prefix}_mapped_external":
                        len(
                            result[
                                "mapped_external"
                            ]
                        ),

                    # 最终角色节点数量。
                    f"{prefix}_role_count":
                        len(
                            result[
                                "kept_roles"
                            ]
                        ),

                    # 五类之外的语义角色。
                    f"{prefix}_non_top5_role_count":
                        len(
                            result[
                                "non_top5_roles"
                            ]
                        ),

                    f"{prefix}_roles":
                        "|".join(
                            sorted(
                                result[
                                    "kept_roles"
                                ]
                            )
                        ),

                    f"{prefix}_non_top5_roles":
                        "|".join(
                            sorted(
                                result[
                                    "non_top5_roles"
                                ]
                            )
                        ),

                    # 普通外部接口，直接被丢掉。
                    f"{prefix}_dropped_external":
                        len(
                            result[
                                "dropped_external"
                            ]
                        ),

                    f"{prefix}_external_drop_rate":
                        result[
                            "drop_rate"
                        ],

                    f"{prefix}_state_count":
                        result[
                            "state_count"
                        ],

                    f"{prefix}_truncated":
                        int(
                            result[
                                "truncated"
                            ]
                        ),
                })

            outputs.append(base)

            r4 = projections[4]

            print(
                f"{index:2d}/20 "
                f"向上保留={len(upstream_meaningful):4d} "
                f"向上穿过={len(upstream_neutral):3d} | "
                f"向下4层保留={len(r4['kept_meaningful']):4d} "
                f"角色={len(r4['kept_roles']):2d} "
                f"五类外={len(r4['non_top5_roles']):2d} "
                f"丢弃普通接口={len(r4['dropped_external']):4d}",
                flush=True,
            )

        except Exception as exc:

            base["error"] = (
                f"{type(exc).__name__}: "
                f"{str(exc)[:500]}"
            )

            outputs.append(base)

            print(
                f"{index:2d}/20 FAILED "
                f"{base['error']}",
                flush=True,
            )

        finally:
            apk = None
            dalvik = None
            analysis = None
            graph = None

            gc.collect()

    # --------------------------------------------------------
    # 保存逐 APK 结果
    # --------------------------------------------------------
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in outputs:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in outputs:
            writer.writerow(row)

    ok = [
        row
        for row in outputs
        if row["status"] == "OK"
    ]

    print()
    print(
        "======================================"
    )
    print(
        "净化局部图检查结果"
    )
    print(
        "======================================"
    )

    print(
        "成功 =",
        len(ok),
        "/",
        len(outputs),
    )

    if not ok:
        print("没有成功样本")
        return

    print()
    print("【从敏感接口向上找】")

    def avg_col(name):
        values = [
            float(row[name])
            for row in ok
        ]
        return mean(values)

    print(
        "平均保留有效内部方法 =",
        f"{avg_col('up_kept_meaningful'):.2f}",
    )

    print(
        "平均穿过纯转发方法 =",
        f"{avg_col('up_contracted_neutral'):.2f}",
    )

    print()
    print("【从保留方法向下净化搜索】")

    for limit in DOWN_LIMITS:

        prefix = f"down{limit}"

        total_mapped = sum(
            int(row[f"{prefix}_mapped_external"])
            for row in ok
        )

        total_dropped = sum(
            int(row[f"{prefix}_dropped_external"])
            for row in ok
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

        apk_non_top5 = sum(
            int(
                row[
                    f"{prefix}_non_top5_role_count"
                ]
            ) > 0
            for row in ok
        )

        trunc = sum(
            int(row[f"{prefix}_truncated"])
            for row in ok
        )

        avg_kept = avg_col(
            f"{prefix}_kept_meaningful"
        )

        avg_neutral = avg_col(
            f"{prefix}_contracted_neutral"
        )

        avg_roles = avg_col(
            f"{prefix}_role_count"
        )

        avg_non_top5 = avg_col(
            f"{prefix}_non_top5_role_count"
        )

        print()
        print(
            f"---- 向下允许 {limit} 层"
            "有效内部方法 ----"
        )

        print(
            "平均最终保留内部方法 =",
            f"{avg_kept:.2f}",
        )

        print(
            "平均穿过纯转发方法 =",
            f"{avg_neutral:.2f}",
        )

        print(
            "平均最终语义角色数 =",
            f"{avg_roles:.2f}",
        )

        print(
            "平均五类外语义角色数 =",
            f"{avg_non_top5:.2f}",
        )

        print(
            "出现五类外语义的 APK =",
            apk_non_top5,
            "/",
            len(ok),
        )

        print(
            "有语义外部接口 =",
            total_mapped,
        )

        print(
            "丢弃普通外部接口 =",
            total_dropped,
        )

        print(
            "外部接口丢弃比例 =",
            f"{global_drop_rate:.2%}",
        )

        print(
            "搜索截断 APK =",
            trunc,
        )

    print()
    print(
        "逐 APK 结果已保存：",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
