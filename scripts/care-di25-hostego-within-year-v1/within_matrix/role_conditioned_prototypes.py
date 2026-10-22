#!/usr/bin/env python3
"""Role-conditioned, training-partition-only prototype evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ROLE_COUNT = 31


def pairwise_rms(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    squared = np.mean(np.square(values[:, None, :] - centers[None, :, :]), axis=2)
    return np.sqrt(np.maximum(squared, 0.0))


def rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(
        np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    ))))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 1e-12 else 0.0


def robust_scale(distance: np.ndarray, fallback: float = 1.0) -> float:
    value = np.asarray(distance, dtype=np.float64)
    value = value[np.isfinite(value) & (value > 1e-8)]
    if not len(value):
        return max(float(fallback), 1e-6)
    return max(float(np.median(value)), 1e-6)


def primary_roles(raw_bag: np.ndarray) -> np.ndarray:
    """Return the deterministic primary role encoded by the first 31 columns.

    A host can have several equally weighted roles.  To preserve one region per
    host without changing the region extractor, ties use the smallest role id.
    An all-zero synthetic region is assigned role -1 and always uses fallback.
    """
    value = np.asarray(raw_bag, dtype=np.float32)
    role_value = value[:, :ROLE_COUNT]
    role = np.argmax(role_value, axis=1).astype(np.int64)
    role[np.max(role_value, axis=1) <= 0.0] = -1
    return role


def _sample_by_apk(
    bags: list[np.ndarray],
    roles: list[np.ndarray],
    labels: np.ndarray,
    label: int,
    per_apk: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed) + 101 * int(label))
    values: list[np.ndarray] = []
    role_values: list[np.ndarray] = []
    for index in np.flatnonzero(np.asarray(labels, dtype=np.int64) == int(label)):
        bag = np.asarray(bags[int(index)], dtype=np.float32)
        role = np.asarray(roles[int(index)], dtype=np.int64)
        if len(bag) != len(role) or not len(bag):
            raise ValueError("bad role/bag alignment")
        if len(bag) > int(per_apk):
            chosen = np.sort(rng.choice(len(bag), size=int(per_apk), replace=False))
            bag = bag[chosen]
            role = role[chosen]
        values.append(bag)
        role_values.append(role)
    if not values:
        raise ValueError("class has no regions")
    return np.concatenate(values), np.concatenate(role_values)


@dataclass(frozen=True)
class RolePrototypeState:
    global_centers: np.ndarray
    global_scales: np.ndarray
    role_centers: np.ndarray
    role_scales: np.ndarray
    support: np.ndarray
    use_role: np.ndarray
    min_support: int
    quality_rows: tuple[dict, ...]


def fit_role_prototypes(
    scaled_bags: list[np.ndarray],
    raw_bags: list[np.ndarray],
    labels: np.ndarray,
    per_apk: int,
    min_support: int,
    seed: int,
) -> RolePrototypeState:
    """Fit K=1 prototypes per role and class from one training partition."""
    labels = np.asarray(labels, dtype=np.int64)
    roles = [primary_roles(bag) for bag in raw_bags]
    sampled = {}
    sampled_roles = {}
    for label in (0, 1):
        sampled[label], sampled_roles[label] = _sample_by_apk(
            scaled_bags, roles, labels, label, per_apk, seed
        )

    dim = int(sampled[0].shape[1])
    global_centers = np.stack([
        np.mean(sampled[label], axis=0) for label in (0, 1)
    ]).astype(np.float32)
    global_scales = np.asarray([
        robust_scale(pairwise_rms(sampled[label], global_centers[label:label + 1])[:, 0])
        for label in (0, 1)
    ], dtype=np.float32)

    role_centers = np.zeros((2, ROLE_COUNT, dim), dtype=np.float32)
    role_scales = np.ones((2, ROLE_COUNT), dtype=np.float32)
    support = np.zeros((2, ROLE_COUNT), dtype=np.int64)
    use_role = np.zeros((2, ROLE_COUNT), dtype=bool)
    selected_by_class_role: dict[tuple[int, int], np.ndarray] = {}
    for label in (0, 1):
        for role in range(ROLE_COUNT):
            selected = sampled[label][sampled_roles[label] == role]
            selected_by_class_role[(label, role)] = selected
            support[label, role] = int(len(selected))
            if len(selected) >= int(min_support):
                center = np.mean(selected, axis=0)
                scale = robust_scale(
                    pairwise_rms(selected, center.reshape(1, -1))[:, 0],
                    global_scales[label],
                )
                role_centers[label, role] = center
                role_scales[label, role] = scale
                use_role[label, role] = True
            else:
                role_centers[label, role] = global_centers[label]
                role_scales[label, role] = global_scales[label]

    quality_rows = []
    for role in range(ROLE_COUNT):
        benign_center = role_centers[0, role]
        malware_center = role_centers[1, role]
        benign_values = selected_by_class_role[(0, role)]
        malware_values = selected_by_class_role[(1, role)]
        benign_dispersion = (
            float(np.mean(pairwise_rms(benign_values, benign_center[None, :])[:, 0]))
            if len(benign_values) else float("nan")
        )
        malware_dispersion = (
            float(np.mean(pairwise_rms(malware_values, malware_center[None, :])[:, 0]))
            if len(malware_values) else float("nan")
        )
        quality_rows.append({
            "role": role,
            "num_ben_regions": int(support[0, role]),
            "num_mal_regions": int(support[1, role]),
            "prototype_distance": rms_distance(benign_center, malware_center),
            "prototype_cosine": cosine(benign_center, malware_center),
            "within_ben_dispersion": benign_dispersion,
            "within_mal_dispersion": malware_dispersion,
            "fallback_ben": int(not use_role[0, role]),
            "fallback_mal": int(not use_role[1, role]),
            "fallback": int(not (use_role[0, role] and use_role[1, role])),
        })
    return RolePrototypeState(
        global_centers=global_centers,
        global_scales=global_scales,
        role_centers=role_centers,
        role_scales=role_scales,
        support=support,
        use_role=use_role,
        min_support=int(min_support),
        quality_rows=tuple(quality_rows),
    )


def evidence_columns(
    scaled_bag: np.ndarray, raw_bag: np.ndarray, state: RolePrototypeState
) -> np.ndarray:
    """Return S_ben, S_mal, margin, Q plus auditable role metadata."""
    values = np.asarray(scaled_bag, dtype=np.float32)
    role = primary_roles(raw_bag)
    output = np.zeros((len(values), 9), dtype=np.float32)
    for index, item in enumerate(role):
        if item < 0:
            centers = state.global_centers
            scales = state.global_scales
            support_ben = support_mal = 0
            fallback_ben = fallback_mal = 1
        else:
            centers = state.role_centers[:, item]
            scales = state.role_scales[:, item]
            support_ben = int(state.support[0, item])
            support_mal = int(state.support[1, item])
            fallback_ben = int(not state.use_role[0, item])
            fallback_mal = int(not state.use_role[1, item])
        distance = pairwise_rms(values[index:index + 1], centers)[0]
        similarity = np.exp(-np.clip(distance / np.maximum(scales, 1e-6), 0.0, 30.0))
        s_ben, s_mal = float(similarity[0]), float(similarity[1])
        output[index] = (
            s_ben,
            s_mal,
            s_mal - s_ben,
            max(s_ben, s_mal),
            float(item),
            float(support_ben),
            float(support_mal),
            float(fallback_ben),
            float(fallback_mal),
        )
    return output

