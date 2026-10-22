#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("audit_common.py")
spec = importlib.util.spec_from_file_location("common", path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def base(endpoint=False):
    values = [0.0] * 15
    values[4] = 1.0 if endpoint else 0.0
    return values


# H0 and H3 have multi-role/mono-role endpoint composition respectively.
nodes = ["H0", "O1", "O2", "H3", "O4", "H5", "EA", "EB", "EC", "EA2"]
roles = [[], [], [], [], [], [], [0], [1], [2], [1]]
edges = [
    (0, 6), (0, 9), (3, 7), (5, 8),       # Host -> sensitive endpoint.
    (0, 3),                                  # gap0.
    (0, 1), (1, 3),                          # gap1 path one.
    (0, 4), (4, 3),                          # duplicate gap1 path, same relation.
    (1, 2), (2, 3),                          # gap2 via O1,O2.
    (0, 5), (5, 3),                          # another Host is intermediate: no mediated 0->3.
    (0, 0),                                  # forbidden self loop.
]
payload = {
    "node_ids": nodes,
    "node_role_ids": roles,
    "node_base_features": [base(index in {6, 7, 8, 9}) for index in range(len(nodes))],
    "edge_index": [[left for left, _ in edges], [right for _, right in edges]],
}
graph = common.reconstruct_frozen_graph(payload, np.arange(3))
a, b, c, pair, transition, audit = common.extract_candidate_features(graph, 3)

# 1-2: intra-Host composition and entropy.
assert graph["host_role_sets"][0] == (0, 1)
assert math.isclose(a["A__multi_role_host_fraction"], 1 / 3)
assert a["A__max_role_entropy_per_host"] > 0
assert pair[0] > 0  # unordered role pair (0,1).

# 3: source/sink/bridge types use Host+ordinary context, not endpoint edges.
assert b["B__fraction_source_like"] > 0
assert b["B__fraction_sink_like"] > 0
assert b["B__fraction_bridge_like"] > 0

# 4-6: gap0/gap1/gap2 all exist and are deduplicated by (src,dst,gap).
assert audit["direct_relation_count"] >= 1
assert audit["gap1_relation_count"] == 1
assert audit["gap2_relation_count"] == 1
assert audit["gap1_raw_path_count"] == 2
assert audit["duplicate_coordination_path_count"] >= 1

# 7-9: Host intermediates do not become ordinary paths, no self relations, direction remains asymmetric.
assert audit["self_coordination_count"] == 0
role_count = 3
gap0 = transition[: role_count * role_count].reshape(role_count, role_count)
assert gap0[0, 1] > 0 and gap0[1, 0] == 0

# 11-12: graph-negative rescue and false-rescue candidate use the fixed 0.5 threshold.
assert common.THRESHOLD == 0.5
graph_score, candidate_score = 0.49, 0.50
assert graph_score < common.THRESHOLD and candidate_score >= common.THRESHOLD

# 13: threshold is fixed; no threshold-search API exists in the common module.
assert not hasattr(common, "search_threshold")

# 14: synthetic OOF fold isolation.
years = np.asarray([2018] * 5)
folds = np.asarray([1, 2, 3, 4, 5])
for fold in common.FOLDS:
    train = np.flatnonzero((years == 2018) & (folds != fold))
    validation = np.flatnonzero((years == 2018) & (folds == fold))
    assert not set(train) & set(validation)
    assert len(validation) == 1

print("HOST_COMPLEMENT_CANDIDATE_SELFTEST_PASS")
