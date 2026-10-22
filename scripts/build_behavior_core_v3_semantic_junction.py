#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    2017:
        ROOT
        / "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551",

    2018:
        ROOT
        / "artifacts/top5-semantic-behavior-graph-v1-2018-triggered",

    2019:
        ROOT
        / "artifacts/top5-semantic-behavior-graph-v1-2019-triggered",

    2020:
        ROOT
        / "artifacts/top5-semantic-behavior-graph-v1-2020-triggered",

    2021:
        ROOT
        / "artifacts/top5-semantic-behavior-graph-v1-2021-triggered",
}

INPUT_NAME = (
    "binding_cores_v2_train_safe_source_supported.jsonl"
)

OUTPUT_NAME = (
    "binding_cores_v3_semantic_junction_train_safe.jsonl"
)

AUDIT_NAME = (
    "binding_cores_v3_semantic_junction_audit.csv"
)


def node_number(x):
    s = str(x)

    digits = "".join(
        c
        for c in s
        if c.isdigit()
    )

    if digits:
        return int(digits)

    return 10**9


# ============================================================
# 强连通分量
#
# 先把递归环压缩成一个内部组织节点。
# 不把原环大小写入训练图。
# ============================================================

def tarjan_scc(
    nodes,
    adjacency,
):
    index = 0

    stack = []

    on_stack = set()

    indices = {}

    lowlink = {}

    result = []

    def visit(v):
        nonlocal index

        indices[v] = index
        lowlink[v] = index

        index += 1

        stack.append(v)
        on_stack.add(v)

        for w in adjacency.get(
            v,
            (),
        ):
            if w not in indices:
                visit(w)

                lowlink[v] = min(
                    lowlink[v],
                    lowlink[w],
                )

            elif w in on_stack:
                lowlink[v] = min(
                    lowlink[v],
                    indices[w],
                )

        if lowlink[v] == indices[v]:

            component = []

            while True:

                w = stack.pop()
                on_stack.remove(w)

                component.append(w)

                if w == v:
                    break

            result.append(
                component
            )

    for v in nodes:
        if v not in indices:
            visit(v)

    return result


def build_v3(record):

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

    # --------------------------------------------------------
    # 只认两种v2边
    # --------------------------------------------------------

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
    # 1. 压缩递归环
    # ========================================================

    components = tarjan_scc(
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

    comp_rev = {
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

        comp_adj[
            a
        ].add(
            b
        )

        comp_rev[
            b
        ].add(
            a
        )

    # ========================================================
    # 2. 确定每个压缩组件的语义属性
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

    comp_attr = {}

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

        indeg = len(
            comp_rev[
                cid
            ]
        )

        outdeg = len(
            comp_adj[
                cid
            ]
        )

        branch = (
            outdeg > 1
        )

        merge = (
            indeg > 1
        )

        comp_attr[
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

            "is_branch_junction":
                int(
                    branch
                ),

            "is_merge_junction":
                int(
                    merge
                ),

            "is_recursive_component":
                int(
                    recursive
                ),
        }

    root_components = [
        cid
        for cid, attr
        in comp_attr.items()
        if attr[
            "is_binding_root"
        ]
    ]

    anchor_components = [
        cid
        for cid, attr
        in comp_attr.items()
        if attr[
            "is_anchor_carrier"
        ]
    ]

    if len(
        root_components
    ) != 1:
        raise RuntimeError(
            "每张v2行为核心必须恰好有一个共同组织点，"
            f"实际={len(root_components)}"
        )

    if not anchor_components:
        raise RuntimeError(
            "行为核心没有Top-5承载方法"
        )

    # ========================================================
    # 3. v3保留节点
    #
    # 只有：
    # - 最近共同组织点
    # - Top-5承载点
    # - 语义承载点
    # - 分支点
    # - 汇合点
    # - 递归组织块
    #
    # 其余方法全部作为匿名内部过渡结构被压缩。
    # ========================================================

    landmarks = set()

    for cid, attr in comp_attr.items():

        if any(
            attr.values()
        ):
            landmarks.add(
                cid
            )

    if not landmarks:
        raise RuntimeError(
            "v3没有任何内部组织节点"
        )

    # ========================================================
    # 4. 收缩普通内部方法链
    #
    # 从每一个保留节点出发，
    # 穿过任意数量的非保留内部组件，
    # 遇到下一个保留节点立即停止。
    #
    # 不保存跨越了几个方法。
    # ========================================================

    contracted_edges = set()

    dead_ends = []

    for source in landmarks:

        stack = list(
            comp_adj[
                source
            ]
        )

        visited = set()

        reached_landmark = False

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(
                current
            )

            if current in landmarks:

                if current != source:
                    contracted_edges.add(
                        (
                            source,
                            current,
                        )
                    )

                    reached_landmark = True

                continue

            successors = (
                comp_adj[
                    current
                ]
            )

            if not successors:

                dead_ends.append(
                    current
                )

                continue

            for nxt in successors:
                stack.append(
                    nxt
                )

    # v2已经是语义有效最小核心。
    # 一个不承载语义的普通方法不应成为最终死端。
    if dead_ends:
        raise RuntimeError(
            "发现无法收缩的普通方法死端: "
            + repr(
                sorted(
                    set(
                        dead_ends
                    )
                )
            )
        )

    # ========================================================
    # 5. 保证从共同组织点仍能到达全部v3内部节点
    # ========================================================

    root = root_components[
        0
    ]

    contracted_adj = {
        cid: set()
        for cid in landmarks
    }

    for src, dst in contracted_edges:
        contracted_adj[
            src
        ].add(
            dst
        )

    reachable = {
        root
    }

    stack = [
        root
    ]

    while stack:

        current = stack.pop()

        for nxt in contracted_adj.get(
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

    missing_landmarks = (
        landmarks
        - reachable
    )

    if missing_landmarks:
        raise RuntimeError(
            "收缩后存在共同组织点不可达节点: "
            + repr(
                sorted(
                    missing_landmarks
                )
            )
        )

    # ========================================================
    # 6. 分配新的局部ID
    #
    # 不保留任何原始方法ID。
    # ========================================================

    def comp_sort_key(cid):

        members = comp_members[
            cid
        ]

        return min(
            node_number(m)
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
        key=node_number,
    )

    new_semantic_id = {
        sid: f"s{i}"
        for i, sid
        in enumerate(
            ordered_semantics
        )
    }

    # ========================================================
    # 7. 输出训练安全节点
    # ========================================================

    out_nodes = []

    for cid in ordered_landmarks:

        attr = comp_attr[
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
                old["role"],

            "is_anchor_role":
                int(
                    old.get(
                        "is_anchor_role",
                        0,
                    )
                ),
        })

    # ========================================================
    # 8. 输出边
    #
    # 不保存 raw_hops / neutral_hops。
    # ========================================================

    out_edges = []

    for src, dst in sorted(
        contracted_edges,
        key=lambda x: (
            comp_sort_key(
                x[0]
            ),
            comp_sort_key(
                x[1]
            ),
        ),
    ):

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

    for old_src, old_sid in semantic_edges:

        cid = comp_of[
            old_src
        ]

        if cid not in landmarks:
            raise RuntimeError(
                "语义承载方法竟被收缩掉"
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
    # 9. 训练安全检查
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
                "v3泄漏禁止字段: "
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
                "v3边泄漏跳数信息: "
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

    if len(roles) < 2:
        raise RuntimeError(
            "v3丢失多语义有效性"
        )

    anchor_semantic_count = sum(
        int(
            node_by_id[
                sid
            ].get(
                "is_anchor_role",
                0,
            )
        )
        for sid in semantic_ids
    )

    if anchor_semantic_count < 1:
        raise RuntimeError(
            "v3丢失锚点语义"
        )

    # ========================================================
    # 训练JSON只保存模型真正需要的信息。
    #
    # 不保存：
    # binding_distance
    # method counts
    # degree
    # path length
    # raw hop count
    # neutral hop count
    # ========================================================

    output = {
        "sha256":
            record["sha256"],

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
            "semantic_junction_v3",

        "semantic_roles":
            roles,

        "nodes":
            out_nodes,

        "edges":
            out_edges,
    }

    # 单独返回诊断数据，不写入训练图。
    audit = {
        "sha256":
            record["sha256"],

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
            len(
                contracted_edges
            ),

        "v3_semantic_edges":
            len(
                semantic_edges
            ),

        "binding_root_nodes":
            sum(
                x[
                    "is_binding_root"
                ]
                for x in comp_attr.values()
                if x in [
                    comp_attr[c]
                    for c in landmarks
                ]
            ),

        "anchor_carrier_nodes":
            sum(
                comp_attr[
                    c
                ][
                    "is_anchor_carrier"
                ]
                for c in landmarks
            ),

        "semantic_carrier_nodes":
            sum(
                comp_attr[
                    c
                ][
                    "is_semantic_carrier"
                ]
                for c in landmarks
            ),

        "branch_junction_nodes":
            sum(
                comp_attr[
                    c
                ][
                    "is_branch_junction"
                ]
                for c in landmarks
            ),

        "merge_junction_nodes":
            sum(
                comp_attr[
                    c
                ][
                    "is_merge_junction"
                ]
                for c in landmarks
            ),

        "recursive_component_nodes":
            sum(
                comp_attr[
                    c
                ][
                    "is_recursive_component"
                ]
                for c in landmarks
            ),
    }

    return (
        output,
        audit,
    )


def percentile(
    values,
    q,
):
    if not values:
        return 0.0

    xs = sorted(
        values
    )

    if len(xs) == 1:
        return float(
            xs[0]
        )

    pos = (
        len(xs) - 1
    ) * q

    lo = int(pos)

    hi = min(
        lo + 1,
        len(xs) - 1,
    )

    frac = (
        pos - lo
    )

    return float(
        xs[lo]
        * (
            1.0 - frac
        )
        + xs[hi]
        * frac
    )


def process_year(
    year,
    directory,
):

    src = (
        directory
        / INPUT_NAME
    )

    dst = (
        directory
        / OUTPUT_NAME
    )

    audit_path = (
        directory
        / AUDIT_NAME
    )

    if not src.exists():
        raise FileNotFoundError(
            src
        )

    audit_rows = []

    total = 0

    with src.open(
        "r",
        encoding="utf-8",
    ) as fin, dst.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for line_no, line in enumerate(
            fin,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

                output, audit = build_v3(
                    record
                )

            except Exception as exc:

                raise RuntimeError(
                    f"{src}:{line_no}: {exc}"
                ) from exc

            fout.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )

            audit_rows.append(
                audit
            )

            total += 1

    if audit_rows:

        with audit_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    audit_rows[
                        0
                    ].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                audit_rows
            )

    old_counts = [
        r[
            "v2_method_nodes"
        ]
        for r in audit_rows
    ]

    new_counts = [
        r[
            "v3_method_nodes"
        ]
        for r in audit_rows
    ]

    removed = [
        r[
            "removed_method_nodes"
        ]
        for r in audit_rows
    ]

    recursive_graphs = sum(
        1
        for r in audit_rows
        if r[
            "recursive_component_nodes"
        ] > 0
    )

    changed_graphs = sum(
        1
        for r in audit_rows
        if r[
            "removed_method_nodes"
        ] > 0
    )

    old_total = sum(
        old_counts
    )

    new_total = sum(
        new_counts
    )

    print()
    print(
        "=" * 72
    )
    print(
        f"YEAR {year}"
    )
    print(
        "=" * 72
    )

    print(
        "graphs =",
        total,
    )

    print(
        "graphs structurally compressed =",
        changed_graphs,
        "/",
        total,
    )

    print(
        "graphs containing recursive component =",
        recursive_graphs,
    )

    print(
        "v2 method total =",
        old_total,
    )

    print(
        "v3 method total =",
        new_total,
    )

    print(
        "removed methods =",
        old_total
        - new_total,
    )

    print(
        "overall keep ratio =",
        (
            new_total
            / max(
                1,
                old_total
            )
        ),
    )

    print(
        "v2 method mean =",
        statistics.mean(
            old_counts
        )
        if old_counts
        else 0,
    )

    print(
        "v3 method mean =",
        statistics.mean(
            new_counts
        )
        if new_counts
        else 0,
    )

    print(
        "v3 method p50 =",
        percentile(
            new_counts,
            0.50,
        )
    )

    print(
        "v3 method p90 =",
        percentile(
            new_counts,
            0.90,
        )
    )

    print(
        "v3 method p95 =",
        percentile(
            new_counts,
            0.95,
        )
    )

    print(
        "v3 method max =",
        max(
            new_counts
        )
        if new_counts
        else 0,
    )

    print(
        "output =",
        dst,
    )

    return {
        "year":
            year,

        "graphs":
            total,

        "compressed_graphs":
            changed_graphs,

        "recursive_graphs":
            recursive_graphs,

        "v2_method_total":
            old_total,

        "v3_method_total":
            new_total,

        "keep_ratio":
            (
                new_total
                / max(
                    1,
                    old_total
                )
            ),

        "v2_method_mean":
            statistics.mean(
                old_counts
            )
            if old_counts
            else 0,

        "v3_method_mean":
            statistics.mean(
                new_counts
            )
            if new_counts
            else 0,

        "v3_method_p50":
            percentile(
                new_counts,
                0.50,
            ),

        "v3_method_p90":
            percentile(
                new_counts,
                0.90,
            ),

        "v3_method_p95":
            percentile(
                new_counts,
                0.95,
            ),

        "v3_method_max":
            max(
                new_counts
            )
            if new_counts
            else 0,
    }


def main():

    summary = []

    for year, directory in DATASETS.items():

        summary.append(
            process_year(
                year,
                directory,
            )
        )

    summary_path = (
        ROOT
        / "artifacts/"
        "behavior-core-v3-semantic-junction-summary.csv"
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
    print(
        "=" * 72
    )

    print(
        "V3 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "summary =",
        summary_path,
    )


if __name__ == "__main__":
    main()
