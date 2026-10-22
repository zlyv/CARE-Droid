#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter

from context import nearest_boundaries
from ownership import CorpusStats, assign_graph, class_fingerprint, class_method_groups


def main() -> None:
    # 0 FIRST -> 1 THIRD -> 2 THIRD Host and 3 FIRST -> 1 THIRD.
    ownership = ["FIRST_PARTY", "THIRD_PARTY", "THIRD_PARTY", "FIRST_PARTY", "UNKNOWN"]
    reverse = {1: [0, 3], 2: [1], 4: [2]}
    rows = nearest_boundaries(2, ownership, reverse)
    assert {(row["source"], row["target"], row["distance"]) for row in rows} == {(0, 1, 2), (3, 1, 2)}
    assert nearest_boundaries(0, ownership, reverse) == []
    assert nearest_boundaries(4, ownership, reverse) == []
    assert Counter(row["source"] for row in rows) == Counter({0: 1, 3: 1})
    graph = {
        "classes": {
            "lapp/main;": {"super": "Ljava/lang/Object;", "interfaces": []},
            "lvendor/sdk;": {"super": "Ljava/lang/Object;", "interfaces": []},
        },
        "methods": [
            {"class": "lapp/main;", "name": "onCreate", "descriptor": "()V", "code_present": True},
            {"class": "lvendor/sdk;", "name": "a", "descriptor": "()V", "code_present": True},
            {"class": "lvendor/sdk;", "name": "b", "descriptor": "()V", "code_present": True},
            {"class": "lvendor/sdk;", "name": "c", "descriptor": "()V", "code_present": True},
            {"class": "lvendor/sdk;", "name": "d", "descriptor": "()V", "code_present": True},
        ],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
    }
    groups = class_method_groups(graph)
    main_fp, main_shape = class_fingerprint("lapp/main;", groups["lapp/main;"], graph)
    sdk_fp, sdk_shape = class_fingerprint("lvendor/sdk;", groups["lvendor/sdk;"], graph)
    corpus = CorpusStats(
        Counter({main_fp: 1, sdk_fp: 5}), Counter({main_shape: 1, sdk_shape: 5}),
        Counter({
            "1f9b2e184e71ca195f2684dab5e062e005d0ef90403c1a7e585c89be066c1dc2": 1,
            "11c5a2b0ebbe58f11a15f79ab7f424b41bba69d72f18e8a5e3872b92a94cfb4f": 5,
        }), 10,
    )
    # Use computed name hashes to avoid depending on hand-written constants.
    from common import stable_hash
    corpus.class_name_count = Counter({stable_hash("lapp/main;"): 1, stable_hash("lvendor/sdk;"): 5})
    method_ownership, records, _ = assign_graph(graph, {
        "package_namespace": "app", "component_descriptors": {"lapp/main;"},
        "component_namespaces": ["app"],
    }, corpus)
    assert method_ownership[0] == "FIRST_PARTY"
    assert set(method_ownership[1:]) == {"THIRD_PARTY"}
    assert all("class_name" not in record and record["raw_name_emitted"] == 0 for record in records)
    print("OBCBC_SELFTEST_PASS static_only=true labels_used=false future_rows_loaded=0")


if __name__ == "__main__":
    main()
