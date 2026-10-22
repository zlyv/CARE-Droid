#!/usr/bin/env python3
"""Fold-local role/class K=1 prototypes with one shared comparison scale."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


ROLE_COUNT = 31


def pairwise_rms(values, centers):
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    return np.sqrt(np.maximum(np.mean(np.square(values[:, None, :] - centers[None, :, :]), axis=2), 0.0))


def robust_scale(values, fallback=1.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 1e-8)]
    return max(float(np.median(values)) if len(values) else float(fallback), 1e-6)


def cosine(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def primary_roles(raw_bag):
    values = np.asarray(raw_bag, dtype=np.float32)[:, :ROLE_COUNT]
    role = np.argmax(values, axis=1).astype(np.int64)
    role[np.max(values, axis=1) <= 0.0] = -1
    return role


def sample_by_apk(bags, roles, labels, label, per_apk, seed):
    rng = np.random.RandomState(int(seed) + 101 * int(label))
    output = []
    output_roles = []
    for index in np.flatnonzero(np.asarray(labels) == int(label)):
        bag = np.asarray(bags[int(index)], dtype=np.float32)
        role = np.asarray(roles[int(index)], dtype=np.int64)
        if len(bag) > int(per_apk):
            chosen = np.sort(rng.choice(len(bag), size=int(per_apk), replace=False))
            bag = bag[chosen]; role = role[chosen]
        output.append(bag); output_roles.append(role)
    if not output:
        raise ValueError("training class has no regions")
    return np.concatenate(output), np.concatenate(output_roles)


@dataclass(frozen=True)
class SharedRolePrototypeState:
    global_centers: np.ndarray
    global_tau: float
    role_centers: np.ndarray
    role_tau: np.ndarray
    support: np.ndarray
    use_role: np.ndarray
    min_support: int
    quality_rows: tuple[dict, ...]


def fit_role_prototypes(scaled_bags, raw_bags, labels, per_apk, min_support, seed):
    labels = np.asarray(labels, dtype=np.int64)
    roles = [primary_roles(bag) for bag in raw_bags]
    sampled = {}
    sampled_roles = {}
    for label in (0, 1):
        sampled[label], sampled_roles[label] = sample_by_apk(
            scaled_bags, roles, labels, label, per_apk, seed
        )
    dim = sampled[0].shape[1]
    global_centers = np.stack([np.mean(sampled[label], axis=0) for label in (0, 1)]).astype(np.float32)
    global_distances = np.concatenate([
        pairwise_rms(sampled[label], global_centers[label:label + 1])[:, 0]
        for label in (0, 1)
    ])
    global_tau = robust_scale(global_distances, 1.0)
    role_centers = np.zeros((2, ROLE_COUNT, dim), dtype=np.float32)
    role_tau = np.full(ROLE_COUNT, global_tau, dtype=np.float32)
    support = np.zeros((2, ROLE_COUNT), dtype=np.int64)
    use_role = np.zeros((2, ROLE_COUNT), dtype=bool)
    selected = {}
    for label in (0, 1):
        for role in range(ROLE_COUNT):
            values = sampled[label][sampled_roles[label] == role]
            selected[label, role] = values
            support[label, role] = len(values)
            if len(values) >= int(min_support):
                role_centers[label, role] = np.mean(values, axis=0)
                use_role[label, role] = True
            else:
                # Independent class-side fallback: never borrow the opposite class.
                role_centers[label, role] = global_centers[label]
    quality = []
    for role in range(ROLE_COUNT):
        fallback_ben = not bool(use_role[0, role])
        fallback_mal = not bool(use_role[1, role])
        if not fallback_ben and not fallback_mal:
            pooled = np.concatenate([
                pairwise_rms(selected[label, role], role_centers[label, role:role + 1])[:, 0]
                for label in (0, 1)
            ])
            role_tau[role] = robust_scale(pooled, global_tau)
        else:
            # One region must compare both class similarities on the same scale.
            role_tau[role] = global_tau
        quality.append({
            "role": role,
            "num_ben_regions": int(support[0, role]),
            "num_mal_regions": int(support[1, role]),
            "tau_role": float(role_tau[role]),
            "tau_global": float(global_tau),
            "fallback_ben": int(fallback_ben),
            "fallback_mal": int(fallback_mal),
            "fallback": int(fallback_ben or fallback_mal),
            "prototype_distance": float(pairwise_rms(
                role_centers[0, role:role + 1], role_centers[1, role:role + 1]
            )[0, 0]),
            "within_ben_dispersion": float(np.mean(pairwise_rms(
                selected[0, role], role_centers[0, role:role + 1]
            )[:, 0])) if len(selected[0, role]) else float("nan"),
            "within_mal_dispersion": float(np.mean(pairwise_rms(
                selected[1, role], role_centers[1, role:role + 1]
            )[:, 0])) if len(selected[1, role]) else float("nan"),
            "prototype_cosine": cosine(role_centers[0, role], role_centers[1, role]),
        })
    return SharedRolePrototypeState(
        global_centers, float(global_tau), role_centers, role_tau,
        support, use_role, int(min_support), tuple(quality),
    )


def evidence_columns(scaled_bag, raw_bag, state):
    values = np.asarray(scaled_bag, dtype=np.float32)
    roles = primary_roles(raw_bag)
    output = np.zeros((len(values), 9), dtype=np.float32)
    for index, role in enumerate(roles):
        if role < 0:
            centers = state.global_centers
            tau = state.global_tau
            support_ben = support_mal = 0
            fallback_ben = fallback_mal = 1
        else:
            centers = state.role_centers[:, role]
            tau = float(state.role_tau[role])
            support_ben = int(state.support[0, role])
            support_mal = int(state.support[1, role])
            fallback_ben = int(not state.use_role[0, role])
            fallback_mal = int(not state.use_role[1, role])
        distance = pairwise_rms(values[index:index + 1], centers)[0]
        similarity = np.exp(-np.clip(distance / max(tau, 1e-6), 0.0, 30.0))
        s_ben, s_mal = map(float, similarity)
        output[index] = (
            s_ben, s_mal, s_mal - s_ben, max(s_ben, s_mal),
            float(role), float(support_ben), float(support_mal),
            float(fallback_ben), float(fallback_mal),
        )
    return output
