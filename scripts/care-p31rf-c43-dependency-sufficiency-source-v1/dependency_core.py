#!/usr/bin/env python3
"""Fold-safe, source-only dependency representations for CARE-Droid."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROLE_N = 31
PROJECTION_DIM = 31
SINK_ROLES = (
    "dynamic_code_loading",
    "network_transmission_or_upload",
    "command_execution",
    "sms_send_receive",
)


def canonical_dependency_names(role_names):
    names = []
    for source in role_names:
        for sink in SINK_ROLES:
            names.append(f"dep_verified_pair__{source}__to__{sink}")
    names.extend(f"dep_verified_same__{sink}" for sink in SINK_ROLES)
    return tuple(names)


def verified_matrix(ids, rows, role_names):
    """Use a predeclared 31 x 4 + 4 schema; absent columns are structural zero."""
    by_sha = {str(row.get("sha256", "")).strip().casefold(): row for row in rows}
    names = canonical_dependency_names(role_names)
    matrix = np.zeros((len(ids), len(names)), dtype=np.float64)
    status = []
    for i, sha in enumerate(ids):
        row = by_sha.get(sha, {})
        status.append(str(row.get("analysis_status", "MISSING")))
        for j, name in enumerate(names):
            try:
                matrix[i, j] = max(0.0, float(row.get(name, 0) or 0))
            except (TypeError, ValueError):
                matrix[i, j] = 0.0
    matrix = np.log1p(matrix)
    if matrix.shape[1]:
        matrix = normalize(matrix, norm="l2", axis=1)
    return matrix, names, status


def role_cooccurrence_matrix(bags):
    """Within-Host unordered role co-occurrence; no call/dependency claim."""
    pairs = tuple((a, b) for a in range(ROLE_N) for b in range(a, ROLE_N))
    index = {pair: i for i, pair in enumerate(pairs)}
    matrix = np.zeros((len(bags), len(pairs)), dtype=np.float64)
    for apk_i, bag in enumerate(bags):
        for host in bag:
            roles = sorted(set(int(v) for v in host.get("role_ids", ())))
            for pos, left in enumerate(roles):
                matrix[apk_i, index[(left, left)]] += 1.0
                for right in roles[pos + 1:]:
                    matrix[apk_i, index[(left, right)]] += 1.0
    matrix = np.log1p(matrix)
    if matrix.shape[1]:
        matrix = normalize(matrix, norm="l2", axis=1)
    return matrix, pairs


class TrainOnlyProjector:
    def __init__(self, dim=PROJECTION_DIM, seed=0):
        self.dim = int(dim)
        self.seed = int(seed)
        self.svd = None
        self.direct = False
        self.active = None
        self.components = 0
        self.schema_hash = ""

    def fit(self, matrix, names):
        matrix = np.asarray(matrix, dtype=np.float64)
        self.active = np.flatnonzero(np.any(np.abs(matrix) > 0, axis=0))
        active_names = [str(names[i]) for i in self.active]
        self.schema_hash = hashlib.sha256("\n".join(active_names).encode()).hexdigest()
        selected = matrix[:, self.active]
        if 0 < selected.shape[1] <= self.dim:
            self.direct = True
            self.components = selected.shape[1]
            return self
        maximum = min(self.dim, max(selected.shape[0] - 1, 0), max(selected.shape[1] - 1, 0))
        if maximum >= 1 and np.any(selected):
            self.svd = TruncatedSVD(n_components=maximum, random_state=self.seed)
            self.svd.fit(selected)
            self.components = maximum
        return self

    def transform(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        out = np.zeros((len(matrix), self.dim), dtype=np.float64)
        if self.direct:
            out[:, :self.components] = matrix[:, self.active]
        elif self.svd is not None:
            out[:, :self.components] = self.svd.transform(matrix[:, self.active])
        if not np.isfinite(out).all():
            raise RuntimeError("NaN/Inf in dependency projection")
        return out


def stratified_shuffle(matrix, strata, seed):
    """Destroy APK alignment while preserving each Stage-I prediction stratum."""
    matrix = np.asarray(matrix)
    out = matrix.copy()
    rng = np.random.default_rng(int(seed))
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        out[idx] = matrix[rng.permutation(idx)]
    return out

