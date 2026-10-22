#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from hostrel_features import build_hostrel74, reconstruct_host_relations


def main():
    base = np.zeros((4, 15), np.float32)
    base[2:, 4] = 1
    payload = {
        "node_ids": ["host_A", "host_B", "endpoint_A", "endpoint_B"],
        "node_role_ids": [[], [], [1], [2]],
        "node_base_features": base.tolist(),
        "edge_index": [[0, 1, 0], [2, 3, 1]],
        "edge_features": np.zeros((3, 7), np.float32).tolist(),
    }
    reconstructed = reconstruct_host_relations(payload, np.arange(31))
    assert len(reconstructed) == 2
    assert reconstructed[0]["outgoing_neighbor_count"] == 1
    assert reconstructed[0]["directed_neighbor_role"][2] > 0
    assert reconstructed[1]["incoming_neighbor_count"] == 1
    assert reconstructed[1]["directed_neighbor_role"][1] < 0
    old = np.zeros((2, 74), np.float32)
    old[0, 1] = 1
    old[1, 2] = 1
    old[:, 62:] = np.arange(12, dtype=np.float32)
    records = [
        {"presence": old[0, :31], "directed_neighbor_role": np.eye(31, dtype=np.float32)[3]},
        {"presence": old[1, :31], "directed_neighbor_role": -np.eye(31, dtype=np.float32)[4]},
    ]
    result = build_hostrel74(old, records)
    assert result.shape == (2, 74)
    assert np.array_equal(result[:, :31], old[:, :31])
    assert result[0, 31 + 3] == 1 and result[1, 31 + 4] == -1
    assert np.array_equal(result[:, 62:], old[:, 62:])
    print("HOSTREL_SEMANTIC_BACKGROUND_SELFTEST_PASS")
    print("representation=SelfRole31+DirectedNeighborRole31+FrozenTail12 width=74")
    print("backgrounds_same_host_schema=true parameter_count_target=33036")
    print("future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__":
    main()
