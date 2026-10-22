import numpy as np

from scripts.run_entry_path_consistency_2017_eval import (
    entry_path_consistency_vector,
)


def test_entry_path_consistency_tracks_layers_and_role_transitions():
    payload = {
        "node_ids": ["entry", "normal", "network", "callee"],
        "node_role_ids": [[0], [31], [1], []],
        "node_base_features": [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.75, 0.0],
            [1.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.25, 0.0],
        ],
        "edge_index": [[0, 1, 2], [1, 2, 3]],
        "edge_features": [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        "edge_type_ids": [0, 0, 0],
    }

    vector, audit = entry_path_consistency_vector(payload)

    assert vector.ndim == 1
    assert vector.shape == (1256,)
    assert np.isfinite(vector).all()
    assert audit["entry_count"] == 1
    assert audit["reachable_node_count"] == 4
    assert audit["path_edge_count"] == 3
    assert audit["nonzero_role_transition_count"] >= 2
    assert audit["semantic_transition_ratio"] > 0.0
