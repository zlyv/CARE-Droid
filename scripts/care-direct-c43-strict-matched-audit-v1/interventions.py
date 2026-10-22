#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import numpy as np

ARMS = (
    "FULL",
    "N0_ROW_ORDER",
    "I5_ROLEGRAPH_LOCAL_QUERY_ALIGNMENT",
    "I6_ROLEGRAPH_WHOLE_ALIGNMENT",
    "I7_GLOBAL24_LOCAL_ALIGNMENT",
)

BASE_CALIPER = 0.10
LOG_HOST_CALIPER = 0.70
P31_HELLINGER_CALIPER = 0.45


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


def strict_matched_donor(base_probability, host_count, p31, repeat: int):
    """Strict label-free nearest donors within Stage-I direction.

    A candidate must pass frozen calipers on Stage-I probability, log Host count,
    and Hellinger distance between P31 compositions.  Donors may be reused; this
    favors comparability over marginal permutation.  The kth nearest candidate is
    selected across repeats.  A target with no candidate receives -1 and is
    explicitly excluded from the strict matched analysis.
    """
    base = np.asarray(base_probability, float)
    host = np.log1p(np.asarray(host_count, float))
    p31 = np.asarray(p31, float)
    if p31.shape != (len(base), 31):
        raise RuntimeError("BLOCKED_BY_CONTRACT: strict matcher expected N x 31 P31")
    mass = np.maximum(p31.sum(axis=1, keepdims=True), 1e-12)
    root = np.sqrt(np.maximum(p31 / mass, 0.0))
    result = np.full(len(base), -1, dtype=np.int64)
    diagnostics = []
    direction = (base >= 0.5).astype(int)
    for i in range(len(base)):
        candidates = np.flatnonzero(direction == direction[i])
        candidates = candidates[candidates != i]
        if not len(candidates):
            diagnostics.append(None); continue
        bd = np.abs(base[candidates] - base[i])
        hd = np.abs(host[candidates] - host[i])
        pd = np.linalg.norm(root[candidates] - root[i], axis=1) / np.sqrt(2.0)
        keep = (bd <= BASE_CALIPER) & (hd <= LOG_HOST_CALIPER) & (pd <= P31_HELLINGER_CALIPER)
        candidates, bd, hd, pd = candidates[keep], bd[keep], hd[keep], pd[keep]
        if not len(candidates):
            diagnostics.append(None); continue
        cost = bd / BASE_CALIPER + hd / LOG_HOST_CALIPER + pd / P31_HELLINGER_CALIPER
        order = np.lexsort((candidates, cost))
        pick = int(order[int(repeat) % min(len(order), 12)])
        result[i] = int(candidates[pick])
        diagnostics.append({"base_abs_difference": float(bd[pick]),
            "log_host_abs_difference": float(hd[pick]),
            "p31_hellinger_distance": float(pd[pick]),
            "candidate_count": int(len(candidates)), "match_cost": float(cost[pick])})
    return result, diagnostics


def intervene_bag(bag, sample_id: str, arm: str, repeat: int):
    x = np.asarray(bag, np.float32)
    if x.ndim != 2 or x.shape[1] != 142:
        raise RuntimeError("BLOCKED_BY_CONTRACT: expected N x 142 formal A0 Host bag")
    out = x.copy(); n = len(x)
    if arm == "FULL":
        return out
    if arm == "N0_ROW_ORDER":
        return out[row_permutation(n, sample_id, arm, repeat)]
    raise KeyError(arm)


def intervention_semantics():
    return {
        "FULL": "unaltered frozen Direct-C43 replay",
        "N0_ROW_ORDER": "permute complete Host rows; exact set-order invariance control",
        "I5_ROLEGRAPH_LOCAL_QUERY_ALIGNMENT": "use a strict P31/base/Host-matched donor RoleGraph only for Host selection, then restore the APK's own RoleToken in the readout",
        "I6_ROLEGRAPH_WHOLE_ALIGNMENT": "replace the whole RoleGraph by a strict label-free matched donor graph",
        "I7_GLOBAL24_LOCAL_ALIGNMENT": "replace Global24 by the same strict label-free matched donor while keeping local and RoleGraph evidence",
    }

