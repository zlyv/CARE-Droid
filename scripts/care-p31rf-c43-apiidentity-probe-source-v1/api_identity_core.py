#!/usr/bin/env python3
"""Fold-safe exact-sensitive-API identity features for CARE-Droid.

The endpoint cache deliberately stores stable hashes rather than raw method
signatures.  A hash is therefore an exact identity token, but not a readable
API name.  This module never opens an APK and never uses validation frequency
to create a vocabulary.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

ROLE_N = 31
API_DIM = 31
MIN_APK_DF = 2
MAX_VOCAB = 2048


def norm_sha(value):
    return str(value or "").strip().casefold()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_endpoint_cache(path):
    result = {}
    started = time.perf_counter()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "OK":
                continue
            sha = norm_sha(row.get("sha256"))
            if sha in result:
                raise RuntimeError("duplicate endpoint-cache SHA: " + sha)
            result[sha] = row
    return result, time.perf_counter() - started


def payload_arrays(payload):
    ids = list(payload.get("node_ids") or [])
    roles_raw = list(payload.get("node_role_ids") or [])
    base = np.asarray(payload.get("node_base_features") or [], dtype=np.float32)
    edges = payload.get("edge_index") or [[], []]
    if base.size == 0:
        base = np.zeros((0, 15), dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 15:
        raise RuntimeError("endpoint node_base_features is not N x 15")
    if len(ids) != len(roles_raw) or len(ids) != len(base) or len(edges) != 2:
        raise RuntimeError("endpoint payload arrays are misaligned")
    source = np.asarray(edges[0], dtype=np.int64)
    target = np.asarray(edges[1], dtype=np.int64)
    if len(source) != len(target):
        raise RuntimeError("endpoint edge arrays are misaligned")
    roles = []
    for values in roles_raw:
        roles.append(tuple(sorted({int(v) for v in (values or []) if 0 <= int(v) < ROLE_N})))
    return ids, roles, base, source, target


def reconstruct_hosts(payload):
    """Return C43-compatible sorted hosts with exact sensitive endpoint IDs."""
    ids, roles, base, source, target = payload_arrays(payload)
    endpoints = set(map(int, np.flatnonzero(base[:, 4] > .5)))
    calls = {}
    for left, right in zip(source, target):
        left = int(left); right = int(right)
        if right in endpoints:
            calls.setdefault(left, set()).add(right)
    host_rows = []
    for host in sorted(calls):
        merged = sorted({role for endpoint in calls[host] for role in roles[endpoint]})
        if not merged:
            continue
        role = np.zeros(ROLE_N, dtype=np.float32)
        role[merged] = 1.0
        api_ids = tuple(sorted({str(ids[endpoint]) for endpoint in calls[host]}))
        host_rows.append({
            "host_node_index": int(host),
            "host_node_id": str(ids[host]),
            "role": role,
            "role_ids": tuple(merged),
            "api_ids": api_ids,
        })
    return host_rows


def build_api_bags(rows, endpoint_cache):
    bags = []
    audits = []
    failures = []
    started = time.perf_counter()
    for row in rows:
        sha = norm_sha(row["sha256"])
        try:
            hosts = reconstruct_hosts(endpoint_cache[sha]["payload"])
            source = endpoint_cache[sha]
            unique = set(api for host in hosts for api in host["api_ids"])
            calls = sum(len(host["api_ids"]) for host in hosts)
            audits.append({
                "sha256": sha,
                "host_count": len(hosts),
                "api_occurrence_count": calls,
                "unique_api_count": len(unique),
                "graph_node_count": int(source.get("node_count", 0)),
                "graph_edge_count": int(source.get("edge_count", 0)),
            })
            bags.append(hosts)
        except Exception as exc:
            failures.append({"sha256": sha, "error_type": type(exc).__name__, "error": str(exc)[:300]})
            bags.append([])
            audits.append({"sha256": sha, "host_count": 0, "api_occurrence_count": 0,
                           "unique_api_count": 0, "graph_node_count": 0, "graph_edge_count": 0})
    elapsed = time.perf_counter() - started
    return bags, audits, failures, elapsed


def vocabulary_from_train(train_bags, min_apk_df=MIN_APK_DF, max_vocab=MAX_VOCAB):
    apk_df = Counter()
    for bag in train_bags:
        apk_df.update(set(api for host in bag for api in host["api_ids"]))
    kept = [(api, count) for api, count in apk_df.items() if count >= int(min_apk_df)]
    kept.sort(key=lambda item: (-item[1], item[0]))
    kept = kept[:int(max_vocab)]
    return {api: index for index, (api, _) in enumerate(kept)}, dict(kept), apk_df


def host_sparse(bags, vocabulary, l2_normalize=True):
    rows = []; cols = []; data = []; owner = []; unknown = []; total = []
    cursor = 0
    for apk_index, bag in enumerate(bags):
        for host in bag:
            counts = Counter(host["api_ids"])
            known = 0
            for api, count in counts.items():
                if api in vocabulary:
                    rows.append(cursor); cols.append(vocabulary[api]); data.append(float(count)); known += count
            owner.append(apk_index); total.append(sum(counts.values())); unknown.append(sum(counts.values()) - known)
            cursor += 1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(cursor, len(vocabulary)), dtype=np.float64)
    if l2_normalize and matrix.shape[0] and matrix.shape[1]:
        matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    return matrix, np.asarray(owner, int), np.asarray(unknown, int), np.asarray(total, int)


class FoldAPIProjector:
    def __init__(self, dim=API_DIM, seed=0):
        self.dim = int(dim); self.seed = int(seed); self.vocabulary = {}; self.frequency = {}
        self.svd = None; self.components = 0; self.vocab_hash = ""

    def fit(self, train_bags):
        self.vocabulary, self.frequency, _ = vocabulary_from_train(train_bags)
        ordered = sorted(self.vocabulary, key=self.vocabulary.get)
        self.vocab_hash = hashlib.sha256("\n".join(ordered).encode()).hexdigest()
        x, _, _, _ = host_sparse(train_bags, self.vocabulary, l2_normalize=True)
        maximum = min(self.dim, max(x.shape[0] - 1, 0), max(x.shape[1] - 1, 0))
        if maximum >= 1 and x.nnz:
            self.svd = TruncatedSVD(n_components=maximum, random_state=self.seed)
            self.svd.fit(x)
            self.components = maximum
        return self

    def transform(self, bags, raw_count=False):
        x, owner, unknown, total = host_sparse(bags, self.vocabulary, l2_normalize=True)
        host_embedding = np.zeros((len(owner), self.dim), dtype=np.float64)
        if self.svd is not None and len(owner):
            host_embedding[:, :self.components] = self.svd.transform(x)
        apk = np.zeros((len(bags), self.dim), dtype=np.float64)
        host_count = np.zeros(len(bags), dtype=np.int64)
        unk = np.zeros(len(bags), dtype=np.int64); api_total = np.zeros(len(bags), dtype=np.int64)
        role = np.zeros((len(bags), ROLE_N), dtype=np.float64)
        host_role_rows = []
        offset = 0
        for apk_index, bag in enumerate(bags):
            count = len(bag); host_count[apk_index] = count
            if count:
                block = host_embedding[offset:offset + count]
                apk[apk_index] = block.sum(axis=0) if raw_count else block.mean(axis=0)
                role[apk_index] = np.mean([host["role"] for host in bag], axis=0)
                unk[apk_index] = int(unknown[offset:offset + count].sum())
                api_total[apk_index] = int(total[offset:offset + count].sum())
                for local_index, host in enumerate(bag):
                    host_role_rows.append((apk_index, local_index, block[local_index]))
            offset += count
        if offset != len(owner):
            raise RuntimeError("Host/API transform accounting mismatch")
        unknown_rate = np.divide(unk, api_total, out=np.zeros(len(bags), float), where=api_total > 0)
        return {"role": role, "api": apk, "host_api": host_embedding, "host_rows": host_role_rows,
                "host_count": host_count, "unknown": unk, "api_total": api_total,
                "unknown_rate": unknown_rate}


def jaccard(left, right):
    left = set(left); right = set(right)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)

