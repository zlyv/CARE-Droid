#!/usr/bin/env python3
"""Synthetic directed Host-Ego and role-coordinate contract."""

import numpy as np
from extract_source_local_assets_v2 import reconstruct, NONTRIVIAL_EGO


def main():
    base = np.zeros((4, 15), dtype=float)
    base[1, 4] = 1.0; base[3, 4] = 1.0
    payload = {
        "node_ids": ["host/A", "endpoint/X", "host/B", "endpoint/Y"],
        "node_role_ids": [[], [0], [], [0]],
        "node_base_features": base.tolist(),
        "edge_index": [[0, 2, 0], [1, 3, 2]],
    }
    # Payload role 0 maps to declared/model role 7.
    mapping = np.arange(31, dtype=np.int16); mapping[0] = 7; mapping[7] = 0
    result = reconstruct(payload, mapping)
    assert result["host_ids"].tolist() == ["host/A", "host/B"]
    assert result["host_primary_role"].tolist() == [7, 7]
    assert result["ego_status"].tolist() == [NONTRIVIAL_EGO, NONTRIVIAL_EGO]
    assert result["same_role_edge_count"] == 1
    assert result["cross_role_edge_count"] == 0
    assert not np.array_equal(result["host_values"][0], result["ego_values"][0])
    assert result["outgoing_host_positions"].tolist() == [1]
    assert result["incoming_host_positions"].tolist() == [0]

    # A valid APK with no retained sensitive endpoint has an empty graph.  It
    # must survive extraction as a zero-Host shard so downstream deterministic
    # missing-region handling can retain the APK in the exact cohort.
    empty = reconstruct({
        "node_ids": [],
        "node_role_ids": [],
        "node_base_features": [],
        "edge_index": [[], []],
    }, mapping)
    assert empty["host_values"].shape == (0, 74)
    assert empty["ego_values"].shape == (0, 74)
    assert empty["host_ids"].shape == (0,)
    assert empty["role_node_counts"].shape == (31,)
    assert empty["role_edge_counts"].shape == (31, 31)
    print("SOURCE_ASSET_RECONSTRUCTION_SELFTEST_PASS")


if __name__ == "__main__":
    main()
