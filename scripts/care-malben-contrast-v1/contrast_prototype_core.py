#!/usr/bin/env python3
"""Training-fold-only benign/malware prototype contrast helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class ContrastState:
    benign_centers: np.ndarray
    malware_centers: np.ndarray
    benign_scale: float
    malware_scale: float
    sampled_benign_n: int
    sampled_malware_n: int


def sample_rows_per_apk(
    bags: list[np.ndarray], labels: np.ndarray, label: int, limit: int, seed: int
) -> np.ndarray:
    """Give each APK the same maximum contribution to its class reference."""
    rng = np.random.RandomState(int(seed) + 31 * int(label))
    parts: list[np.ndarray] = []
    for index in np.flatnonzero(np.asarray(labels, dtype=np.int64) == int(label)):
        bag = np.asarray(bags[int(index)], dtype=np.float32)
        if bag.ndim != 2 or not len(bag):
            raise ValueError("each APK must contain at least one 2-D region row")
        if len(bag) > int(limit):
            chosen = np.sort(rng.choice(len(bag), size=int(limit), replace=False))
            bag = bag[chosen]
        parts.append(bag)
    if not parts:
        raise ValueError("class reference has no APK regions")
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)


def pairwise_rms(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    squared = np.mean(np.square(values[:, None, :] - centers[None, :, :]), axis=2)
    return np.sqrt(np.maximum(squared, 0.0))


def nearest_distance(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.min(pairwise_rms(values, centers), axis=1)


def robust_scale(distance: np.ndarray) -> float:
    value = np.asarray(distance, dtype=np.float64)
    value = value[np.isfinite(value) & (value > 1e-8)]
    if not len(value):
        return 1.0
    return max(float(np.median(value)), 1e-6)


def _fit_centers(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    k = min(int(count), len(values))
    if k < 1:
        raise ValueError("prototype count must be positive")
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        batch_size=min(2048, max(256, len(values))),
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.01,
    ).fit(values)
    return np.asarray(model.cluster_centers_, dtype=np.float32)


def fit_contrast_state(
    bags: list[np.ndarray],
    labels: np.ndarray,
    regions_per_apk: int,
    benign_prototypes: int,
    malware_prototypes: int,
    seed: int,
) -> ContrastState:
    """Fit both references from the current training partition and nowhere else."""
    benign = sample_rows_per_apk(bags, labels, 0, regions_per_apk, seed)
    malware = sample_rows_per_apk(bags, labels, 1, regions_per_apk, seed)
    benign_centers = _fit_centers(benign, benign_prototypes, int(seed) + 101)
    malware_centers = _fit_centers(malware, malware_prototypes, int(seed) + 202)
    return ContrastState(
        benign_centers=benign_centers,
        malware_centers=malware_centers,
        benign_scale=robust_scale(nearest_distance(benign, benign_centers)),
        malware_scale=robust_scale(nearest_distance(malware, malware_centers)),
        sampled_benign_n=int(len(benign)),
        sampled_malware_n=int(len(malware)),
    )


def contrast_columns(bag: np.ndarray, state: ContrastState) -> np.ndarray:
    """Return benign similarity, malware similarity, signed margin and strength."""
    bag = np.asarray(bag, dtype=np.float32)
    d_ben = nearest_distance(bag, state.benign_centers)
    d_mal = nearest_distance(bag, state.malware_centers)
    s_ben = np.exp(-np.clip(d_ben / state.benign_scale, 0.0, 30.0))
    s_mal = np.exp(-np.clip(d_mal / state.malware_scale, 0.0, 30.0))
    margin = s_mal - s_ben
    strength = np.maximum(s_mal, s_ben)
    return np.stack([s_ben, s_mal, margin, strength], axis=1).astype(np.float32)

