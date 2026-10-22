#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import numpy as np

ARMS = (
    "FULL",
    "N0_ROW_ORDER",
    "I1_BACKGROUND_HOST_ALIGNMENT",
    "I2_RELATION_HOST_ALIGNMENT",
    "I3_BACKGROUND_RELATION_JOINT_TO_CORE",
    "I4_BACKGROUND_RELATION_MUTUAL_ALIGNMENT",
    "I5_ROLEGRAPH_LOCAL_QUERY_ALIGNMENT",
    "I6_ROLEGRAPH_WHOLE_ALIGNMENT",
    "I7_GLOBAL24_LOCAL_ALIGNMENT",
)


def stable_seed(*parts) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & 0x7FFFFFFF


def row_permutation(n: int, sample_id: str, arm: str, repeat: int) -> np.ndarray:
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(sample_id, arm, repeat))
    result = rng.permutation(n)
    if np.array_equal(result, np.arange(n)):
        result = np.roll(result, 1)
    return result.astype(np.int64)


def matched_donor(base_probability, host_count, repeat: int) -> np.ndarray:
    """One-to-one, label-free donor permutation within Stage-I direction.

    Ordering by base probability then host count makes small cyclic shifts locally
    matched.  It preserves every donor marginal exactly and never consults labels.
    """
    base = np.asarray(base_probability, float)
    host = np.log1p(np.asarray(host_count, float))
    result = np.arange(len(base), dtype=np.int64)
    for direction in (0, 1):
        index = np.flatnonzero((base >= 0.5).astype(int) == direction)
        if len(index) <= 1:
            continue
        order = index[np.lexsort((host[index], base[index]))]
        shift = 1 + (int(repeat) % (len(order) - 1))
        result[order] = np.roll(order, shift)
    return result


def intervene_bag(bag, sample_id: str, arm: str, repeat: int):
    x = np.asarray(bag, np.float32)
    if x.ndim != 2 or x.shape[1] != 142:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected N x 142 formal A0 Host bag")
    out = x.copy(); n = len(x)
    if arm == "FULL":
        return out
    if arm == "N0_ROW_ORDER":
        return out[row_permutation(n, sample_id, arm, repeat)]
    if arm == "I1_BACKGROUND_HOST_ALIGNMENT":
        p = row_permutation(n, sample_id, arm, repeat); out[:, 107:111] = x[p, 107:111]
        return out
    if arm == "I2_RELATION_HOST_ALIGNMENT":
        p = row_permutation(n, sample_id, arm, repeat); out[:, 111:142] = x[p, 111:142]
        return out
    if arm == "I3_BACKGROUND_RELATION_JOINT_TO_CORE":
        p = row_permutation(n, sample_id, arm, repeat); out[:, 107:142] = x[p, 107:142]
        return out
    if arm == "I4_BACKGROUND_RELATION_MUTUAL_ALIGNMENT":
        pb = row_permutation(n, sample_id, arm+"_B", repeat)
        pr = row_permutation(n, sample_id, arm+"_R", repeat)
        out[:, 107:111] = x[pb, 107:111]; out[:, 111:142] = x[pr, 111:142]
        return out
    raise KeyError(arm)


def intervention_semantics():
    return {
        "FULL": "unaltered frozen Direct-C43 replay",
        "N0_ROW_ORDER": "permute complete Host rows; exact set-order invariance control",
        "I1_BACKGROUND_HOST_ALIGNMENT": "permute Background4 among Hosts inside the same APK",
        "I2_RELATION_HOST_ALIGNMENT": "permute Relation31 among Hosts inside the same APK",
        "I3_BACKGROUND_RELATION_JOINT_TO_CORE": "jointly permute Background4+Relation31, preserving their mutual pairing but breaking pairing with Compact43",
        "I4_BACKGROUND_RELATION_MUTUAL_ALIGNMENT": "independently permute Background4 and Relation31, additionally breaking their mutual pairing",
        "I5_ROLEGRAPH_LOCAL_QUERY_ALIGNMENT": "use a matched donor RoleGraph only for Host selection, then restore the APK's own RoleToken in the readout",
        "I6_ROLEGRAPH_WHOLE_ALIGNMENT": "replace the whole RoleGraph by a label-free matched donor graph",
        "I7_GLOBAL24_LOCAL_ALIGNMENT": "replace Global24 by a label-free matched donor while keeping local and RoleGraph evidence",
    }

