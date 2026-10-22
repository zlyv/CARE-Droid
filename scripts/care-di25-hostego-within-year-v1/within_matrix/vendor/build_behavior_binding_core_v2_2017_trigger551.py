#!/usr/bin/env python3

import csv
import json
from collections import defaultdict, deque, Counter
from pathlib import Path


ROOT = Path(
    "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551"
)

BEHAVIORS = ROOT / "behaviors.jsonl"
BINDINGS = ROOT / "binding_specificity.csv"

OUT_JSONL = ROOT / "binding_cores_v2_train_safe.jsonl"
OUT_CSV = ROOT / "binding_core_v2_summary.csv"


def quantile(xs, q):
    xs = sorted(xs)

    if not xs:
        return 0.0

    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo

    return (
        xs[lo] * (1.0 - frac)
        + xs[hi] * frac
    )


def reachable_from(root, outgoing):
    seen = {root}
    queue = deque([root])

    while queue:
        current = queue.popleft()

        for target in outgoing.get(
            current,
            (),
        ):
            if target in seen:
                continue

            seen.add(target)
            queue.append(target)

    return seen


binding_rows = list(
    csv.DictReader(
        BINDINGS.open(
            encoding="utf-8",
        )
    )
)

binding_by_behavior = {
    row["behavior_id"]: row
    for row in binding_rows
}


behaviors = [
    json.loads(line)
    for line in BEHAVIORS.open(
        encoding="utf-8",
    )
]

behaviors = [
    x
    for x in behaviors
    if x["valid_behavior"]
]


summary_rows = []
safe_rows = []


for x in behaviors:

    behavior_id = x["behavior_id"]

    binding = binding_by_behavior.get(
        behavior_id
    )

    if binding is None:
        continue

    root = binding["binding_method_id"]
    carrier = x["anchor_method_id"]
    anchor_role = x["anchor_role"]

    node_by_id = {
        node["id"]: node
        for node in x["nodes"]
    }

    outgoing = defaultdict(set)

    for edge in x["edges"]:
        outgoing[
            edge["source"]
        ].add(
            edge["target"]
        )

    # ==========================================
    # v2:
    # 最近共同组织点以下的所有可达节点。
    # ==========================================
    reachable = reachable_from(
        root,
        outgoing,
    )

    # ------------------------------------------
    # 再做一次语义反向剪枝：
    # 只留下最终能够到达某个语义终点的节点。
    # ------------------------------------------
    semantic_targets = {
        node_id
        for node_id in reachable
        if (
            node_id in node_by_id
            and node_by_id[
                node_id
            ].get("type")
            == "semantic"
        )
    }

    reverse = defaultdict(set)

    for edge in x["edges"]:

        s = edge["source"]
        t = edge["target"]

        if (
            s in reachable
            and t in reachable
        ):
            reverse[t].add(s)

    keep = set(
        semantic_targets
    )

    queue = deque(
        semantic_targets
    )

    while queue:

        current = queue.popleft()

        for parent in reverse.get(
            current,
            (),
        ):

            if parent == root:
                if parent not in keep:
                    keep.add(parent)
                continue

            if parent not in reachable:
                continue

            if parent in keep:
                continue

            keep.add(parent)
            queue.append(parent)

    if root in reachable:
        keep.add(root)

    kept_edges = [
        edge
        for edge in x["edges"]
        if (
            edge["source"] in keep
            and edge["target"] in keep
        )
    ]

    method_ids = {
        node_id
        for node_id in keep
        if (
            node_id in node_by_id
            and node_by_id[
                node_id
            ].get("type")
            == "method"
        )
    }

    semantic_ids = {
        node_id
        for node_id in keep
        if (
            node_id in node_by_id
            and node_by_id[
                node_id
            ].get("type")
            == "semantic"
        )
    }

    roles = sorted({
        node_by_id[node_id]["role"]
        for node_id in semantic_ids
    })

    # ==========================================
    # 方法局部结构。
    # ==========================================
    method_out = defaultdict(set)
    method_in = defaultdict(set)

    for edge in kept_edges:

        if edge["type"] != "method_to_method":
            continue

        s = edge["source"]
        t = edge["target"]

        method_out[s].add(t)
        method_in[t].add(s)

    # ==========================================
    # 从核心根节点计算局部方法距离。
    # ==========================================
    root_distance = {
        root: 0
    }

    queue = deque([root])

    while queue:

        current = queue.popleft()

        for target in method_out.get(
            current,
            (),
        ):

            next_distance = (
                root_distance[current]
                + 1
            )

            old = root_distance.get(
                target
            )

            if (
                old is not None
                and old <= next_distance
            ):
                continue

            root_distance[
                target
            ] = next_distance

            queue.append(target)

    # ==========================================
    # 关键：
    # 重新编号成图内局部 ID。
    #
    # 不允许训练文件继续携带：
    #   方法签名
    #   方法签名哈希
    #   外部 API 哈希
    # ==========================================
    method_order = sorted(
        method_ids,
        key=lambda m: (
            root_distance.get(
                m,
                999999,
            ),
            m,
        )
    )

    semantic_order = sorted(
        semantic_ids,
        key=lambda s: (
            node_by_id[s]["role"],
            s,
        )
    )

    local_id = {}

    for i, method in enumerate(
        method_order
    ):
        local_id[method] = f"m{i}"

    for i, semantic in enumerate(
        semantic_order
    ):
        local_id[semantic] = f"s{i}"

    safe_nodes = []

    for method in method_order:

        safe_nodes.append({
            "id":
                local_id[method],

            "type":
                "method",

            "is_anchor_carrier":
                int(
                    method == carrier
                ),

            "is_binding_root":
                int(
                    method == root
                ),

            "distance_from_root":
                root_distance.get(
                    method,
                    -1,
                ),

            "local_in_degree":
                len(
                    method_in.get(
                        method,
                        (),
                    )
                ),

            "local_out_degree":
                len(
                    method_out.get(
                        method,
                        (),
                    )
                ),
        })

    for semantic in semantic_order:

        role = node_by_id[
            semantic
        ]["role"]

        safe_nodes.append({
            "id":
                local_id[semantic],

            "type":
                "semantic",

            "role":
                role,

            "is_anchor_role":
                int(
                    role == anchor_role
                ),
        })

    safe_edges = []

    for edge in kept_edges:

        s = edge["source"]
        t = edge["target"]

        if (
            s not in local_id
            or t not in local_id
        ):
            continue

        safe_edges.append({
            "source":
                local_id[s],

            "target":
                local_id[t],

            "type":
                edge["type"],

            "raw_hops":
                int(
                    edge.get(
                        "raw_hops",
                        1,
                    )
                ),

            "neutral_hops":
                int(
                    edge.get(
                        "neutral_hops",
                        0,
                    )
                ),
        })

    v1_methods = int(
        x["method_node_count"]
    )

    v2_methods = len(
        method_ids
    )

    shrink = (
        1.0
        - v2_methods / v1_methods
        if v1_methods
        else 0.0
    )

    core_valid = (
        root in method_ids
        and carrier in method_ids
        and anchor_role in roles
        and len(roles) >= 2
    )

    branching = sum(
        len(
            method_out.get(
                m,
                (),
            )
        ) >= 2
        for m in method_ids
    )

    safe_rows.append({
        # 这些是样本元数据，不作为图节点特征。
        "sha256":
            x["sha256"],

        "year":
            x["year"],

        "label":
            x["label"],

        "binary_label":
            x["binary_label"],

        "anchor_role":
            anchor_role,

        "core_valid":
            int(core_valid),

        # 结构信息。
        "binding_distance":
            int(
                binding[
                    "binding_distance"
                ]
            ),

        "method_node_count":
            v2_methods,

        "semantic_node_count":
            len(
                semantic_ids
            ),

        "semantic_role_count":
            len(roles),

        "semantic_roles":
            roles,

        "branching_method_count":
            branching,

        "nodes":
            safe_nodes,

        "edges":
            safe_edges,
    })

    summary_rows.append({
        "sha256":
            x["sha256"],

        "binary_label":
            x["binary_label"],

        "anchor_role":
            anchor_role,

        "binding_distance":
            int(
                binding[
                    "binding_distance"
                ]
            ),

        "v1_method_nodes":
            v1_methods,

        "v2_method_nodes":
            v2_methods,

        "removed_method_nodes":
            v1_methods
            - v2_methods,

        "method_shrink_rate":
            shrink,

        "v1_role_types":
            int(
                x[
                    "semantic_role_count"
                ]
            ),

        "v2_role_types":
            len(roles),

        "v2_semantic_nodes":
            len(
                semantic_ids
            ),

        "v2_branching_methods":
            branching,

        "core_valid":
            int(core_valid),
    })


with OUT_JSONL.open(
    "w",
    encoding="utf-8",
) as f:

    for row in safe_rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


fieldnames = list(
    summary_rows[0].keys()
)

with OUT_CSV.open(
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


print("===== BINDING CORE V2 =====")
print(
    "input valid behaviors =",
    len(behaviors),
)

print(
    "output cores =",
    len(safe_rows),
)

print(
    "core valid =",
    sum(
        r["core_valid"]
        for r in summary_rows
    ),
)

print(
    "core invalid =",
    sum(
        not r["core_valid"]
        for r in summary_rows
    ),
)


v1_sizes = [
    r["v1_method_nodes"]
    for r in summary_rows
]

v2_sizes = [
    r["v2_method_nodes"]
    for r in summary_rows
]

shrink = [
    r["method_shrink_rate"]
    for r in summary_rows
]


print()
print("===== METHOD SIZE =====")

for name, vals in [
    ("V1", v1_sizes),
    ("V2", v2_sizes),
]:

    print(
        name,
        "mean=",
        f"{sum(vals)/len(vals):.2f}",
        "p50=",
        f"{quantile(vals,.50):.1f}",
        "p90=",
        f"{quantile(vals,.90):.1f}",
        "p95=",
        f"{quantile(vals,.95):.1f}",
        "max=",
        max(vals),
    )


print()
print(
    "mean shrink =",
    f"{sum(shrink)/len(shrink):.2%}",
)

print(
    "median shrink =",
    f"{quantile(shrink,.50):.2%}",
)


print()
print("===== BY BINDING DISTANCE =====")

for distance in sorted({
    r["binding_distance"]
    for r in summary_rows
}):

    xs = [
        r
        for r in summary_rows
        if (
            r[
                "binding_distance"
            ]
            == distance
        )
    ]

    print(
        "distance=",
        distance,
        "n=",
        len(xs),
        "v1_mean=",
        f"{sum(r['v1_method_nodes'] for r in xs)/len(xs):.2f}",
        "v2_mean=",
        f"{sum(r['v2_method_nodes'] for r in xs)/len(xs):.2f}",
        "shrink=",
        f"{sum(r['method_shrink_rate'] for r in xs)/len(xs):.2%}",
    )


print()
print("===== TOP 15 LARGEST V2 CORES =====")

largest = sorted(
    summary_rows,
    key=lambda r: (
        r["v2_method_nodes"],
        r["v2_role_types"],
    ),
    reverse=True,
)[:15]

for i, r in enumerate(
    largest,
    1,
):
    print(
        i,
        r["sha256"][:12],
        r["binary_label"],
        r["anchor_role"],
        "distance=",
        r["binding_distance"],
        "v1=",
        r["v1_method_nodes"],
        "v2=",
        r["v2_method_nodes"],
        "roles=",
        r["v2_role_types"],
        "branch=",
        r["v2_branching_methods"],
    )


print()
print("===== TRAIN-SAFE CHECK =====")

bad_keys = Counter()

for graph in safe_rows:

    for node in graph["nodes"]:

        for key in node:

            if key in {
                "raw_id",
                "method_signature",
                "method_name",
            }:
                bad_keys[key] += 1

print(
    "forbidden node fields =",
    dict(bad_keys),
)

print()
print("safe jsonl =", OUT_JSONL)
print("summary    =", OUT_CSV)
