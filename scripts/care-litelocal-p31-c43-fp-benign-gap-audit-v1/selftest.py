#!/usr/bin/env python3
import numpy as np
from feature_core import FEATURE_GROUPS, centroid_score, extract_features


def main():
    base = np.zeros((4, 15), float); base[1, 4] = base[2, 4] = 1
    base[:, 5] = 1; base[1, 8] = 1; base[2, 9] = 1
    payload = {"node_base_features": base.tolist(), "node_role_ids": [[], [0], [1], []],
               "edge_index": [[0, 0, 3], [1, 2, 0]], "edge_features": [[0]*7, [0]*7, [0]*7]}
    feature, audit = extract_features(payload, np.arange(31))
    assert set(FEATURE_GROUPS["ALL_SEMANTIC"]) == set(feature)
    assert audit["endpoint_count"] == 2 and audit["host_count"] == 1
    train = np.asarray([[0., 0.], [.1, .1], [1., 1.], [.9, .9]])
    score = centroid_score(train, np.asarray([0, 0, 1, 1]), np.asarray([[0., 0.], [1., 1.]]))
    assert score[0] > 0 and score[1] < 0
    print("P31_C43_FP_BENIGN_GAP_SELFTEST_PASS")
    print("training_invoked=false neural_inference_invoked=false future_rows_loaded=0 ego_consumed=false")


if __name__ == "__main__": main()
