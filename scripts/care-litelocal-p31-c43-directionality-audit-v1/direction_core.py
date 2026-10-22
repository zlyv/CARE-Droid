#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


REPRESENTATIONS = ("CORE43", "RELATION31", "JOINT74")


def equal_apk_scaler(bags: list[np.ndarray]):
    nonempty = [np.asarray(bag, float) for bag in bags if len(bag)]
    if not nonempty:
        raise RuntimeError("no nonempty training Host bag")
    mean = np.mean([bag.mean(axis=0) for bag in nonempty], axis=0)
    second = np.mean([(bag * bag).mean(axis=0) for bag in nonempty], axis=0)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    std[std < 1e-6] = 1.0
    return mean, std


def transform_bags(bags: list[np.ndarray], mean, std):
    return [((np.asarray(bag, float) - mean) / std).astype(np.float32)
            for bag in bags]


def primary_roles(role_bag: np.ndarray):
    role_bag = np.asarray(role_bag, float)
    valid = role_bag.sum(axis=1) > 0
    role = np.argmax(role_bag, axis=1).astype(np.int16)
    role[~valid] = -1
    return role


def build_references(feature_bags, role_bags, labels):
    """One centroid per training APK and primary role, preventing large-APK dominance."""
    by_class_role = {(label, role): [] for label in (0, 1) for role in range(31)}
    global_by_class = {0: [], 1: []}
    for feature, role_values, label in zip(feature_bags, role_bags, labels):
        if not len(feature):
            continue
        label = int(label)
        global_by_class[label].append(np.mean(feature, axis=0))
        role = primary_roles(role_values)
        for value in np.unique(role[role >= 0]):
            by_class_role[(label, int(value))].append(np.mean(feature[role == value], axis=0))
    result = {}
    for key, values in by_class_role.items():
        result[key] = np.asarray(values, np.float32) if values else np.empty((0, feature_bags[0].shape[1]), np.float32)
    global_result = {
        label: np.asarray(values, np.float32)
        for label, values in global_by_class.items()
    }
    if any(len(global_result[label]) == 0 for label in (0, 1)):
        raise RuntimeError("both class-conditional Host backgrounds are required")
    return result, global_result


def knn_rms(query, reference, k=5, chunk=512):
    query = np.asarray(query, np.float32)
    reference = np.asarray(reference, np.float32)
    if not len(query):
        return np.empty(0, float)
    if not len(reference):
        raise RuntimeError("empty direction reference")
    use_k = min(int(k), len(reference))
    ref_norm = np.mean(reference * reference, axis=1)
    output = np.empty(len(query), float)
    for begin in range(0, len(query), chunk):
        value = query[begin:begin + chunk]
        distance = (np.mean(value * value, axis=1)[:, None] + ref_norm[None, :] -
                    2.0 * np.matmul(value, reference.T) / value.shape[1])
        distance = np.sqrt(np.maximum(distance, 0.0))
        nearest = np.partition(distance, use_k - 1, axis=1)[:, :use_k]
        output[begin:begin + len(value)] = np.mean(nearest, axis=1)
    return output


def class_distances(feature, role_values, references, global_references, min_role_support=5):
    roles = primary_roles(role_values)
    output = {0: np.empty(len(feature), float), 1: np.empty(len(feature), float)}
    support = {0: np.empty(len(feature), int), 1: np.empty(len(feature), int)}
    for label in (0, 1):
        for role in np.unique(roles):
            selected = np.flatnonzero(roles == role)
            candidate = references.get((label, int(role)), np.empty((0, feature.shape[1])))
            if role < 0 or len(candidate) < min_role_support:
                candidate = global_references[label]
            output[label][selected] = knn_rms(feature[selected], candidate)
            support[label][selected] = len(candidate)
    return output[0], output[1], support[0], support[1]


def aggregate_host_values(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"mean": np.nan, "median": np.nan, "max": np.nan,
                "top3_mean": np.nan, "fraction_positive": np.nan}
    top = np.sort(values)[-min(3, len(values)):]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "top3_mean": float(np.mean(top)),
        "fraction_positive": float(np.mean(values > 0)),
    }
