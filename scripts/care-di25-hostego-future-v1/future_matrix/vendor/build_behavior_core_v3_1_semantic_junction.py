#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import build_behavior_core_v3_semantic_junction as base


ROOT = Path(__file__).resolve().parent.parent

OUTPUT_NAME = (
    "binding_cores_v3_1_semantic_junction_train_safe.jsonl"
)

AUDIT_NAME = (
    "binding_cores_v3_1_semantic_junction_audit.csv"
)


def build_v31(record):

    nodes = record.get(
        "nodes",
        [],
    )

    edges = record.get(
        "edges",
        [],
    )

    node_by_id = {
        n["id"]: n
        for n in nodes
    }

    method_ids = [
        n["id"]
        for n in nodes
        if n.get("type") == "method"
    ]

    semantic_ids = [
        n["id"]
        for n in nodes
        if n.get("type") == "semantic"
    ]

    method_set = set(
        method_ids
    )

    semantic_set = set(
        semantic_ids
    )

    adjacency = {
        m: set()
        for m in method_ids
    }

    method_edges = []
    semantic_edges = []

    self_loop_methods = set()

    # ========================================================
    # 1. 读取原始v2局部方法图
    # ========================================================

    for e in edges:

        src = e["source"]
        dst = e["target"]

        etype = e.get(
            "type"
        )

        if etype == "method_to_method":

            if (
                src not in method_set
                or dst not in method_set
            ):
                raise RuntimeError(
                    "非法 method_to_method 边"
                )

            adjacency[
                src
            ].add(
                dst
            )

            method_edges.append(
                (
                    src,
                    dst,
                )
            )

            if src == dst:
                self_loop_methods.add(
                    src
                )

        elif etype == "method_to_semantic":

            if (
                src not in method_set
                or dst not in semantic_set
            ):
                raise RuntimeError(
                    "非法 method_to_semantic 边"
                )

            semantic_edges.append(
                (
                    src,
                    dst,
                )
            )

        else:
            raise RuntimeError(
                "未知边类型: "
                + repr(
                    etype
                )
            )

    # ========================================================
    # 2. 先压缩递归环
    # ========================================================

    components = base.tarjan_scc(
        method_ids,
        adjacency,
    )

    comp_of = {}

    for cid, members in enumerate(
        components
    ):
        for m in members:
            comp_of[
                m
            ] = cid

    comp_members = {
        cid: set(members)
        for cid, members
        in enumerate(
            components
        )
    }

    comp_adj = {
        cid: set()
        for cid in comp_members
    }

    for src, dst in method_edges:

        a = comp_of[
            src
        ]

        b = comp_of[
            dst
        ]

        if a == b:
            continue

        # 使用集合：
        # SCC压缩后相同的组件关系只保留一次。
        comp_adj[
            a
        ].add(
            b
        )

    # ========================================================
    # 3. 找语义承载组件
    # ========================================================

    semantic_sources = defaultdict(
        list
    )

    for src, sid in semantic_edges:

        cid = comp_of[
            src
        ]

        semantic_sources[
            cid
        ].append(
            sid
        )

    # ========================================================
    # 4. 此时只定义“不依赖局部度数”的固定属性
    #
    # 注意：
    # 这里绝对不能先计算branch / merge。
    # ========================================================

    fixed_attr = {}

    for cid, members in comp_members.items():

        member_nodes = [
            node_by_id[m]
            for m in members
        ]

        is_root = any(
            int(
                n.get(
                    "is_binding_root",
                    0,
                )
            )
            for n in member_nodes
        )

        is_anchor = any(
            int(
                n.get(
                    "is_anchor_carrier",
                    0,
                )
            )
            for n in member_nodes
        )

        is_semantic_carrier = bool(
            semantic_sources.get(
                cid
            )
        )

        recursive = (
            len(members) > 1
            or any(
                m in self_loop_methods
                for m in members
            )
        )

        fixed_attr[
            cid
        ] = {
            "is_binding_root":
                int(
                    is_root
                ),

            "is_anchor_carrier":
                int(
                    is_anchor
                ),

            "is_semantic_carrier":
                int(
                    is_semantic_carrier
                ),

            "is_recursive_component":
                int(
                    recursive
                ),
        }

    root_components = [
        cid
        for cid, attr
        in fixed_attr.items()
        if attr[
            "is_binding_root"
        ]
    ]

    if len(
        root_components
    ) != 1:
        raise RuntimeError(
            "每张行为核心必须恰好有一个共同组织点，"
            f"实际={len(root_components)}"
        )

    if not any(
        attr[
            "is_anchor_carrier"
        ]
        for attr in fixed_attr.values()
    ):
        raise RuntimeError(
            "行为核心没有Top-5承载方法"
        )

    # ========================================================
    # 5. 固定保留组件
    #
    # 这些节点无论度数如何都不能删除：
    # - 共同组织点
    # - Top-5承载点
    # - 语义承载点
    # - 递归组织块
    # ========================================================

    protected = {
        cid
        for cid, attr
        in fixed_attr.items()
        if any(
            attr.values()
        )
    }

    # ========================================================
    # 6. 关键修正：
    #
    # 在判断分支/汇合之前，
    # 反复消除所有非保护的 1入1出 组件。
    #
    # A -> x -> B
    #
    # 规范化为
    #
    # A ------> B
    #
    # 如果有：
    #
    # A -> x1 -> B
    # A -> x2 -> B
    #
    # 两条路径都规范化以后，
    # 集合关系最终仍然只有 A -> B，
    # 不会错误制造 branch / merge。
    # ========================================================

    active = set(
        comp_members
    )

    norm_adj = {
        cid: set(
            comp_adj[
                cid
            ]
        )
        for cid in comp_members
    }

    collapsed_linear = 0

    while True:

        norm_rev = {
            cid: set()
            for cid in active
        }

        for src in active:

            for dst in norm_adj[
                src
            ]:

                if dst in active:
                    norm_rev[
                        dst
                    ].add(
                        src
                    )

        candidate = None

        for cid in sorted(
            active
        ):

            if cid in protected:
                continue

            successors = {
                x
                for x in norm_adj[
                    cid
                ]
                if x in active
            }

            predecessors = (
                norm_rev[
                    cid
                ]
            )

            if (
                len(
                    predecessors
                ) == 1
                and len(
                    successors
                ) == 1
            ):
                candidate = cid
                break

        if candidate is None:
            break

        predecessors = (
            norm_rev[
                candidate
            ]
        )

        successors = {
            x
            for x in norm_adj[
                candidate
            ]
            if x in active
        }

        pred = next(
            iter(
                predecessors
            )
        )

        succ = next(
            iter(
                successors
            )
        )

        norm_adj[
            pred
        ].discard(
            candidate
        )

        norm_adj[
            candidate
        ].clear()

        active.remove(
            candidate
        )

        # SCC组件图原本是有向无环图。
        # 正常情况下pred != succ。
        if pred != succ:
            norm_adj[
                pred
            ].add(
                succ
            )

        collapsed_linear += 1

    # ========================================================
    # 7. 在线性结构规范化完成之后，
    # 才计算真正的分支与汇合。
    # ========================================================

    norm_rev = {
        cid: set()
        for cid in active
    }

    for src in active:

        norm_adj[
            src
        ] = {
            dst
            for dst in norm_adj[
                src
            ]
            if dst in active
        }

        for dst in norm_adj[
            src
        ]:
            norm_rev[
                dst
            ].add(
                src
            )

    final_attr = {}

    for cid in active:

        final_attr[
            cid
        ] = {
            **fixed_attr[
                cid
            ],

            "is_branch_junction":
                int(
                    len(
                        norm_adj[
                            cid
                        ]
                    ) > 1
                ),

            "is_merge_junction":
                int(
                    len(
                        norm_rev[
                            cid
                        ]
                    ) > 1
                ),
        }

    # ========================================================
    # 8. 最终保留节点
    #
    # 非保护节点现在只能是：
    # - 真正分支
    # - 真正汇合
    # ========================================================

    landmarks = {
        cid
        for cid in active
        if (
            cid in protected
            or final_attr[
                cid
            ][
                "is_branch_junction"
            ]
            or final_attr[
                cid
            ][
                "is_merge_junction"
            ]
        )
    }

    leftovers = (
        active
        - landmarks
    )

    if leftovers:
        raise RuntimeError(
            "规范化后仍存在"
            "既非语义/锚点/根/递归，"
            "又非分支汇合的普通组件: "
            + repr(
                sorted(
                    leftovers
                )
            )
        )

    root = root_components[
        0
    ]

    if root not in landmarks:
        raise RuntimeError(
            "共同组织点被错误删除"
        )

    # ========================================================
    # 9. 检查从共同组织点可达
    # ========================================================

    reachable = {
        root
    }

    stack = [
        root
    ]

    while stack:

        current = stack.pop()

        for nxt in norm_adj.get(
            current,
            (),
        ):

            if nxt in reachable:
                continue

            reachable.add(
                nxt
            )

            stack.append(
                nxt
            )

    missing = (
        landmarks
        - reachable
    )

    if missing:
        raise RuntimeError(
            "规范化后存在共同组织点不可达节点: "
            + repr(
                sorted(
                    missing
                )
            )
        )

    # ========================================================
    # 10. 分配新的局部ID
    # ========================================================

    def comp_sort_key(cid):

        members = comp_members[
            cid
        ]

        return min(
            base.node_number(
                m
            )
            for m in members
        )

    ordered_landmarks = sorted(
        landmarks,
        key=comp_sort_key,
    )

    new_method_id = {
        cid: f"m{i}"
        for i, cid
        in enumerate(
            ordered_landmarks
        )
    }

    ordered_semantics = sorted(
        semantic_ids,
        key=base.node_number,
    )

    new_semantic_id = {
        sid: f"s{i}"
        for i, sid
        in enumerate(
            ordered_semantics
        )
    }

    # ========================================================
    # 11. 输出方法节点
    # ========================================================

    out_nodes = []

    for cid in ordered_landmarks:

        attr = final_attr[
            cid
        ]

        out_nodes.append({
            "id":
                new_method_id[
                    cid
                ],

            "type":
                "method",

            "is_binding_root":
                attr[
                    "is_binding_root"
                ],

            "is_anchor_carrier":
                attr[
                    "is_anchor_carrier"
                ],

            "is_semantic_carrier":
                attr[
                    "is_semantic_carrier"
                ],

            "is_branch_junction":
                attr[
                    "is_branch_junction"
                ],

            "is_merge_junction":
                attr[
                    "is_merge_junction"
                ],

            "is_recursive_component":
                attr[
                    "is_recursive_component"
                ],
        })

    # ========================================================
    # 12. 输出语义节点
    # ========================================================

    for sid in ordered_semantics:

        old = node_by_id[
            sid
        ]

        out_nodes.append({
            "id":
                new_semantic_id[
                    sid
                ],

            "type":
                "semantic",

            "role":
                old[
                    "role"
                ],

            "is_anchor_role":
                int(
                    old.get(
                        "is_anchor_role",
                        0,
                    )
                ),
        })

    # ========================================================
    # 13. 输出规范化结构边
    # ========================================================

    out_edges = []

    for src in ordered_landmarks:

        for dst in sorted(
            norm_adj[
                src
            ],
            key=comp_sort_key,
        ):

            if dst not in landmarks:
                raise RuntimeError(
                    "结构边指向非保留节点"
                )

            out_edges.append({
                "source":
                    new_method_id[
                        src
                    ],

                "target":
                    new_method_id[
                        dst
                    ],

                "type":
                    "structural",
            })

    # ========================================================
    # 14. 输出方法->语义边
    # ========================================================

    for old_src, old_sid in semantic_edges:

        cid = comp_of[
            old_src
        ]

        if cid not in landmarks:
            raise RuntimeError(
                "语义承载组件被错误删除"
            )

        out_edges.append({
            "source":
                new_method_id[
                    cid
                ],

            "target":
                new_semantic_id[
                    old_sid
                ],

            "type":
                "method_to_semantic",
        })

    # ========================================================
    # 15. 训练安全检查
    # ========================================================

    forbidden_node_fields = {
        "raw_id",
        "method",
        "method_name",
        "signature",
        "class",
        "class_name",
        "package",
        "api",
        "api_name",
        "distance_from_root",
        "local_in_degree",
        "local_out_degree",
    }

    for n in out_nodes:

        bad = (
            set(n)
            & forbidden_node_fields
        )

        if bad:
            raise RuntimeError(
                "v3.1泄漏禁止字段: "
                + repr(
                    bad
                )
            )

    for e in out_edges:

        bad = (
            set(e)
            & {
                "raw_hops",
                "neutral_hops",
            }
        )

        if bad:
            raise RuntimeError(
                "v3.1边泄漏跳数信息: "
                + repr(
                    bad
                )
            )

    roles = sorted({
        node_by_id[
            sid
        ][
            "role"
        ]
        for sid in semantic_ids
    })

    if len(
        roles
    ) < 2:
        raise RuntimeError(
            "v3.1丢失多语义有效性"
        )

    if not any(
        int(
            node_by_id[
                sid
            ].get(
                "is_anchor_role",
                0,
            )
        )
        for sid in semantic_ids
    ):
        raise RuntimeError(
            "v3.1丢失锚点语义"
        )

    output = {
        "sha256":
            record[
                "sha256"
            ],

        "year":
            record.get(
                "year"
            ),

        "label":
            record.get(
                "label"
            ),

        "binary_label":
            record.get(
                "binary_label"
            ),

        "anchor_role":
            record.get(
                "anchor_role"
            ),

        "core_valid":
            1,

        "representation":
            "semantic_junction_v3_1",

        "semantic_roles":
            roles,

        "nodes":
            out_nodes,

        "edges":
            out_edges,
    }

    audit = {
        "sha256":
            record[
                "sha256"
            ],

        "year":
            record.get(
                "year"
            ),

        "anchor_role":
            record.get(
                "anchor_role"
            ),

        "v2_method_nodes":
            len(
                method_ids
            ),

        "v3_method_nodes":
            len(
                ordered_landmarks
            ),

        "removed_method_nodes":
            len(
                method_ids
            )
            - len(
                ordered_landmarks
            ),

        "method_keep_ratio":
            (
                len(
                    ordered_landmarks
                )
                / max(
                    1,
                    len(
                        method_ids
                    ),
                )
            ),

        "semantic_nodes":
            len(
                semantic_ids
            ),

        "semantic_roles":
            len(
                roles
            ),

        "v3_structural_edges":
            sum(
                len(
                    norm_adj[
                        c
                    ]
                )
                for c
                in landmarks
            ),

        "v3_semantic_edges":
            len(
                semantic_edges
            ),

        "binding_root_nodes":
            sum(
                final_attr[
                    c
                ][
                    "is_binding_root"
                ]
                for c
                in landmarks
            ),

        "anchor_carrier_nodes":
            sum(
                final_attr[
                    c
                ][
                    "is_anchor_carrier"
                ]
                for c
                in landmarks
            ),

        "semantic_carrier_nodes":
            sum(
                final_attr[
                    c
                ][
                    "is_semantic_carrier"
                ]
                for c
                in landmarks
            ),

        "branch_junction_nodes":
            sum(
                final_attr[
                    c
                ][
                    "is_branch_junction"
                ]
                for c
                in landmarks
            ),

        "merge_junction_nodes":
            sum(
                final_attr[
                    c
                ][
                    "is_merge_junction"
                ]
                for c
                in landmarks
            ),

        "recursive_component_nodes":
            sum(
                final_attr[
                    c
                ][
                    "is_recursive_component"
                ]
                for c
                in landmarks
            ),

        "collapsed_linear_components":
            collapsed_linear,
    }

    return (
        output,
        audit,
    )


# 给稳定性脚本统一接口
build_v3 = build_v31


def main():

    # 复用原v3的年份目录和输入文件。
    base.build_v3 = build_v31

    base.OUTPUT_NAME = (
        OUTPUT_NAME
    )

    base.AUDIT_NAME = (
        AUDIT_NAME
    )

    summary = []

    for year, directory in base.DATASETS.items():

        summary.append(
            base.process_year(
                year,
                directory,
            )
        )

    summary_path = (
        ROOT
        / "artifacts/"
        "behavior-core-v3-1-semantic-junction-summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary
        )

    print()
    print("=" * 72)
    print("V3.1 COMPLETE")
    print("=" * 72)
    print(
        "summary =",
        summary_path,
    )


if __name__ == "__main__":
    main()
