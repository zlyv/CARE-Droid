#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(
    "artifacts/top5-semantic-behavior-graph-v1-2021-triggered"
)

INPUT = ROOT / "behaviors.jsonl"
OUTPUT = ROOT / "binding_specificity.csv"


def reachable(start, outgoing, method_nodes, semantic_role):
    seen = {start}
    q = deque([start])

    methods = set()
    roles = set()

    while q:
        cur = q.popleft()

        if cur in method_nodes:
            methods.add(cur)

        if cur in semantic_role:
            roles.add(semantic_role[cur])

        for nxt in outgoing.get(cur, ()):
            if nxt in seen:
                continue

            seen.add(nxt)
            q.append(nxt)

    return methods, roles


rows = [
    json.loads(line)
    for line in INPUT.open(encoding="utf-8")
]

valid = [
    x for x in rows
    if x["valid_behavior"]
]

results = []

for x in valid:

    carrier = x["anchor_method_id"]
    anchor_role = x["anchor_role"]

    method_nodes = {
        n["id"]
        for n in x["nodes"]
        if n["type"] == "method"
    }

    semantic_role = {
        n["id"]: n["role"]
        for n in x["nodes"]
        if n["type"] == "semantic"
    }

    outgoing = defaultdict(set)
    reverse_method = defaultdict(set)

    for e in x["edges"]:

        s = e["source"]
        t = e["target"]

        outgoing[s].add(t)

        if e["type"] == "method_to_method":
            reverse_method[t].add(s)

    # =========================================
    # 计算每个上游方法距离 carrier 多远。
    # =========================================
    dist_to_carrier = {
        carrier: 0
    }

    q = deque([carrier])

    while q:

        cur = q.popleft()

        for parent in reverse_method.get(cur, ()):

            nd = dist_to_carrier[cur] + 1

            old = dist_to_carrier.get(parent)

            if old is not None and old <= nd:
                continue

            dist_to_carrier[parent] = nd
            q.append(parent)

    candidates = []

    for method, distance in dist_to_carrier.items():

        methods_reached, roles_reached = reachable(
            method,
            outgoing,
            method_nodes,
            semantic_role,
        )

        other_roles = (
            roles_reached
            - {anchor_role}
        )

        if not other_roles:
            continue

        local_out_degree = len(
            outgoing.get(method, ())
        )

        candidates.append({
            "method": method,
            "distance": distance,
            "scope_methods":
                len(methods_reached),
            "out_degree":
                local_out_degree,
            "other_roles":
                sorted(other_roles),
        })

    if candidates:

        # 最先按距离最小：
        # 离 carrier 越近越局部。
        #
        # 同距离时取覆盖方法更少的，
        # 避免优先选择巨大公共组织点。
        chosen = sorted(
            candidates,
            key=lambda z: (
                z["distance"],
                z["scope_methods"],
                z["out_degree"],
                z["method"],
            )
        )[0]

        binding_found = 1
        binding_distance = chosen["distance"]
        binding_scope_methods = (
            chosen["scope_methods"]
        )
        binding_out_degree = (
            chosen["out_degree"]
        )
        binding_other_roles = "|".join(
            chosen["other_roles"]
        )
        binding_method = chosen["method"]

    else:

        binding_found = 0
        binding_distance = -1
        binding_scope_methods = 0
        binding_out_degree = 0
        binding_other_roles = ""
        binding_method = ""

    results.append({
        "sha256":
            x["sha256"],

        "binary_label":
            x["binary_label"],

        "behavior_id":
            x["behavior_id"],

        "anchor_role":
            anchor_role,

        "anchor_method_id":
            carrier,

        "method_node_count":
            x["method_node_count"],

        "semantic_role_count":
            x["semantic_role_count"],

        "branching_method_count":
            x["branching_method_count"],

        "binding_found":
            binding_found,

        "binding_distance":
            binding_distance,

        "binding_method_id":
            binding_method,

        "binding_scope_methods":
            binding_scope_methods,

        "binding_out_degree":
            binding_out_degree,

        "binding_other_roles":
            binding_other_roles,
    })


fieldnames = list(results[0])

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    w = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    w.writeheader()
    w.writerows(results)


print("===== MINIMAL COMMON ORGANIZER =====")

print("valid behaviors =", len(results))

print(
    "binding found =",
    sum(
        r["binding_found"]
        for r in results
    ),
)

print(
    "binding missing =",
    sum(
        not r["binding_found"]
        for r in results
    ),
)

print()
print("===== BINDING DISTANCE =====")

dist_counts = Counter(
    r["binding_distance"]
    for r in results
)

for d in sorted(dist_counts):
    print(
        f"distance={d:2d}",
        "count=",
        dist_counts[d],
        "rate=",
        f"{dist_counts[d]/len(results):.2%}",
    )


print()
print("===== DISTANCE BY ANCHOR =====")

anchors = sorted({
    r["anchor_role"]
    for r in results
})

for anchor in anchors:

    xs = [
        r
        for r in results
        if r["anchor_role"] == anchor
    ]

    c = Counter(
        r["binding_distance"]
        for r in xs
    )

    print()
    print(anchor, "n=", len(xs))

    for d in sorted(c):
        print(
            "   ",
            d,
            c[d],
            f"{c[d]/len(xs):.2%}",
        )


print()
print("===== MOST SUSPICIOUS LARGE-SCOPE BINDINGS =====")

suspicious = sorted(
    [
        r
        for r in results
        if r["binding_found"]
    ],
    key=lambda r: (
        r["binding_distance"],
        r["binding_scope_methods"],
        r["binding_out_degree"],
    ),
    reverse=True,
)[:20]

for i, r in enumerate(
    suspicious,
    1,
):

    print("-" * 90)

    print(
        i,
        r["behavior_id"],
    )

    print(
        "label=",
        r["binary_label"],
        "| anchor=",
        r["anchor_role"],
    )

    print(
        "methods=",
        r["method_node_count"],
        "| binding_distance=",
        r["binding_distance"],
        "| binding_scope=",
        r["binding_scope_methods"],
        "| binding_out_degree=",
        r["binding_out_degree"],
    )

    print(
        "other_roles=",
        r["binding_other_roles"],
    )


# ==========================================================
# 跨 APK 重复锚点承载方法
# ==========================================================

print()
print("===== CROSS-APK REUSED ANCHOR CARRIERS =====")

groups = defaultdict(list)

for x in valid:

    key = (
        x["anchor_method_id"],
        x["anchor_role"],
    )

    groups[key].append(x)


reuse = []

for key, xs in groups.items():

    shas = {
        x["sha256"]
        for x in xs
    }

    if len(shas) < 2:
        continue

    labels = {
        x["binary_label"]
        for x in xs
    }

    reuse.append(
        (
            int(len(labels) > 1),
            len(shas),
            key,
            xs,
        )
    )


reuse.sort(
    reverse=True,
    key=lambda z: (
        z[0],
        z[1],
    ),
)


for (
    cross_label,
    apk_count,
    (
        carrier,
        anchor,
    ),
    xs,
) in reuse[:25]:

    print("-" * 90)

    print(
        "carrier=",
        carrier,
        "| anchor=",
        anchor,
        "| APKs=",
        apk_count,
        "| cross_label=",
        bool(cross_label),
    )

    for x in xs:

        print(
            "   ",
            x["sha256"][:12],
            x["binary_label"],
            "methods=",
            x["method_node_count"],
            "roles=",
            ",".join(
                x["semantic_roles"]
            ),
        )


print()
print("output =", OUTPUT)
