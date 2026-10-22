#!/usr/bin/env python3

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

from build_behavior_core_v3_semantic_junction import build_v3


SRC = Path(
    "artifacts/top5-semantic-behavior-graph-v1-2017-trigger551/"
    "binding_cores_v2_train_safe_source_supported.jsonl"
)

DEPTHS = (1, 2, 4, 8)


def perturb(record, depth):
    r = copy.deepcopy(record)

    old_edges = r["edges"]
    new_edges = []

    next_id = 1000000

    for edge_index, e in enumerate(old_edges):

        if e.get("type") != "method_to_method":
            new_edges.append(e)
            continue

        src = e["source"]
        dst = e["target"]

        previous = src

        for j in range(depth):

            wid = f"m{next_id}"
            next_id += 1

            r["nodes"].append({
                "id": wid,
                "type": "method",
                "is_anchor_carrier": 0,
                "is_binding_root": 0,

                # build_v3并不依赖这些旧统计，
                # 这里只是保持v2输入格式完整。
                "distance_from_root": 0,
                "local_in_degree": 1,
                "local_out_degree": 1,
            })

            new_edges.append({
                "source": previous,
                "target": wid,
                "type": "method_to_method",
                "raw_hops": 1,
                "neutral_hops": 0,
            })

            previous = wid

        new_edges.append({
            "source": previous,
            "target": dst,
            "type": "method_to_method",
            "raw_hops": 1,
            "neutral_hops": 0,
        })

    r["edges"] = new_edges

    return r


def canonical(v3):
    """
    build_v3已经重新分配局部m/s编号。
    这里比较真正的训练表示。
    """
    return {
        "anchor_role": v3["anchor_role"],
        "semantic_roles": v3["semantic_roles"],
        "nodes": v3["nodes"],
        "edges": v3["edges"],
    }


records = []

with SRC.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))


print("graphs =", len(records))

tested = 0

results = {
    d: {
        "tested": 0,
        "same": 0,
        "different": 0,
    }
    for d in DEPTHS
}

examples = []


for i, record in enumerate(records):

    mm_edges = [
        e
        for e in record.get("edges", [])
        if e.get("type") == "method_to_method"
    ]

    # 没有内部方法边的图无法插包装层，
    # 不纳入这个扰动实验。
    if not mm_edges:
        continue

    base, _ = build_v3(record)
    base = canonical(base)

    tested += 1

    for depth in DEPTHS:

        altered = perturb(
            record,
            depth,
        )

        after, _ = build_v3(
            altered
        )

        after = canonical(
            after
        )

        results[depth]["tested"] += 1

        if after == base:
            results[depth]["same"] += 1
        else:
            results[depth]["different"] += 1

            if len(examples) < 10:
                examples.append({
                    "index": i,
                    "sha256": record.get("sha256"),
                    "anchor_role": record.get("anchor_role"),
                    "depth": depth,
                })


print()
print("eligible graphs =", tested)

print()
print("=" * 72)
print("WRAPPER INVARIANCE")
print("=" * 72)

for depth in DEPTHS:
    x = results[depth]

    rate = (
        x["same"] / x["tested"]
        if x["tested"]
        else 0.0
    )

    print(
        f"depth={depth}: "
        f"same={x['same']}/{x['tested']} "
        f"rate={rate:.6f} "
        f"different={x['different']}"
    )

if examples:
    print()
    print("different examples:")
    for x in examples:
        print(x)
