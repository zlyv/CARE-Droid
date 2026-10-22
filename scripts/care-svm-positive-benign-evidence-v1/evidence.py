#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from common import CONTEXT_FAMILIES, norm


def context_matrix(ids, inventory_rows):
    by_sha = {norm(row["sha256"]): row for row in inventory_rows}
    missing = [sid for sid in ids if sid not in by_sha]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: function inventory missing SHA " + missing[0])
    counts = np.asarray([[float(by_sha[sid].get(name, 0) or 0) for name in CONTEXT_FAMILIES] for sid in ids], float)
    # Presence, not volume: repeated calls cannot create arbitrarily strong evidence.
    return (counts > 0).astype(float), counts


def role_arrays(ids, p31_path):
    with np.load(p31_path, allow_pickle=False) as archive:
        stored = [norm(value) for value in archive["sha256"]]
        prevalence = np.asarray(archive["role_prevalence"], float)
    position = {sid: index for index, sid in enumerate(stored)}
    missing = [sid for sid in ids if sid not in position]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 missing SHA " + missing[0])
    role = prevalence[[position[sid] for sid in ids]]
    role = np.maximum(role, 0.0)
    mass = role.sum(axis=1, keepdims=True)
    weight = np.divide(role, mass, out=np.zeros_like(role), where=mass > 0)
    return role, weight, (role > 0).astype(float)


def fit_positive_lift(context_presence, role_active, benign_index, min_support=3):
    """Fit positive-only context->Role lift on benign training APKs.

    No absence term exists.  A context family with insufficient support, or a
    conditional probability not above benign background, contributes exactly 0.
    """
    x = np.asarray(context_presence, float)[benign_index]
    r = np.asarray(role_active, float)[benign_index]
    n = len(x)
    if n < 2:
        raise RuntimeError("BLOCKED_BY_CONTRACT: insufficient benign training rows")
    background = (r.sum(axis=0) + 1.0) / (n + 2.0)
    lift = np.zeros((x.shape[1], r.shape[1]), float)
    support = x.sum(axis=0).astype(int)
    for family in range(x.shape[1]):
        if support[family] < int(min_support):
            continue
        conditional = (r[x[:, family] > 0].sum(axis=0) + 1.0) / (support[family] + 2.0)
        lift[family] = np.maximum((conditional - background) / np.maximum(1.0 - background, 1e-8), 0.0)
    return {"lift": lift, "background": background, "family_support": support,
            "benign_train_n": n, "min_support": int(min_support)}


def transform_certificate(model, context_presence, role_weight):
    x = np.asarray(context_presence, float); weight = np.asarray(role_weight, float)
    lift = np.asarray(model["lift"], float)
    active_count = x.sum(axis=1, keepdims=True)
    per_role = np.divide(x @ lift, active_count, out=np.zeros((len(x), lift.shape[1])), where=active_count > 0)
    weighted_support = np.sum(weight * per_role, axis=1)
    explained_mass = np.sum(weight * (per_role > 0).astype(float), axis=1)
    peak = np.max(weight * per_role, axis=1)
    # Both strength and coverage must be positive.  Low anomaly never enters.
    certificate = weighted_support * explained_mass
    return {"certificate": certificate, "weighted_support": weighted_support,
            "explained_mass": explained_mass, "peak": peak,
            "active_context_count": x.sum(axis=1)}


def oof_certificates(y, fold, context_presence, role_weight, role_active):
    y = np.asarray(y, int); fold = np.asarray(fold, int)
    fields = {name: np.zeros(len(y), float) for name in
              ("certificate", "weighted_support", "explained_mass", "peak", "active_context_count")}
    audits = []
    for current in sorted(set(fold.tolist())):
        train = fold != current; benign = train & (y == 0); validation = fold == current
        model = fit_positive_lift(context_presence, role_active, benign)
        values = transform_certificate(model, context_presence[validation], role_weight[validation])
        for name in fields:
            fields[name][validation] = values[name]
        audits.append({"fold": int(current), "benign_train_n": int(benign.sum()),
            "known_context_role_edges": int(np.sum(model["lift"] > 0)),
            "supported_context_families": int(np.sum(model["family_support"] >= model["min_support"]))})
    return fields, audits

