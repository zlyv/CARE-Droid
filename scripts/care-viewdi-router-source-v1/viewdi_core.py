#!/usr/bin/env python3
"""Pure helpers for the strict-source leave-one-view-out DI audit."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


VIEW_GROUPS = {
    "manifest": {
        "manifest_permission_count", "manifest_permission_count_log",
        "manifest_activity_count", "manifest_service_count",
        "manifest_receiver_count", "manifest_provider_count",
        "manifest_component_count", "manifest_component_count_log",
        "min_sdk", "target_sdk",
    },
    "code_call": {"method_count", "graph_node_count", "graph_edge_count"},
    "sensitive": {
        "context_role__boot_persistence", "mapped_api_node_count",
        "sensitive_node_count", "sensitive_edge_count",
        "manifest_context_match_count",
    },
    "graph": {
        "internal_node_count", "internal_edge_count", "external_node_count",
        "external_edge_count", "unmapped_external_node_count",
        "mapped_external_ratio",
    },
}
MEMBERS = tuple("drop_" + name for name in VIEW_GROUPS)


def feature_groups(names: list[str]) -> dict[str, np.ndarray]:
    if len(names) != 24 or len(set(names)) != 24:
        raise RuntimeError(f"Global24 contract failed: dimension={len(names)}")
    actual = set(names)
    expected = set().union(*VIEW_GROUPS.values())
    if actual != expected:
        raise RuntimeError(
            "Global24 semantic grouping mismatch: "
            + repr({"missing": sorted(expected-actual), "unexpected": sorted(actual-expected)})
        )
    result = {}
    all_index = np.arange(24, dtype=np.int64)
    for dropped, fields in VIEW_GROUPS.items():
        drop_index = {i for i, name in enumerate(names) if name in fields}
        keep = np.asarray([i for i in all_index if i not in drop_index], dtype=np.int64)
        if len(keep) != 24-len(fields):
            raise RuntimeError("bad leave-one-view-out group: " + dropped)
        result["drop_" + dropped] = keep
    return result


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0-1e-8)
    return np.log(p/(1.0-p))


def decision_instability(stage1_probability: np.ndarray,
                         member_probability: np.ndarray) -> dict[str, np.ndarray]:
    base = np.asarray(stage1_probability, dtype=np.float64)
    members = np.asarray(member_probability, dtype=np.float64)
    if base.ndim != 1 or members.shape != (len(base), 4):
        raise ValueError("expected p_stage1[n] and four member probabilities[n,4]")
    if not np.isfinite(base).all() or not np.isfinite(members).all():
        raise ValueError("non-finite probabilities")
    decision = (base >= 0.5).astype(np.float64)
    support_by_member = decision[:, None]*members + (1.0-decision[:, None])*(1.0-members)
    support = support_by_member.mean(axis=1)
    opposition = 1.0-support
    return {
        "stage1_decision": decision.astype(np.int64),
        "support_by_member": support_by_member,
        "support": support,
        "score": opposition,
        "p_bar": members.mean(axis=1),
    }


def exact_top_fraction(score: np.ndarray, budget: float = 0.25) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("score must be a finite vector")
    count = max(1, min(len(score), int(round(float(budget)*len(score)))))
    order = np.argsort(score, kind="mergesort")
    route = np.zeros(len(score), dtype=bool)
    route[order[-count:]] = True
    return route


def groupwise_top_fraction(score: np.ndarray, group: np.ndarray,
                           budget: float = 0.25) -> np.ndarray:
    score, group = np.asarray(score), np.asarray(group)
    if score.shape != group.shape:
        raise ValueError("score/group mismatch")
    route = np.zeros(len(score), dtype=bool)
    for value in sorted(np.unique(group)):
        index = np.flatnonzero(group == value)
        route[index] = exact_top_fraction(score[index], budget)
    return route


def route_metrics(name: str, labels: np.ndarray, p_stage1: np.ndarray,
                  route: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(p_stage1, dtype=np.float64)
    routed = np.asarray(route, dtype=bool)
    pred = p >= 0.5
    error = pred != y
    confidence = np.maximum(p, 1.0-p)
    high_error = error & (confidence >= 0.8)
    captured = routed & error
    return {
        "method": name,
        "n": int(len(y)),
        "routed_n": int(routed.sum()),
        "routed_rate": float(routed.mean()),
        "base_error_n": int(error.sum()),
        "captured_error_n": int(captured.sum()),
        "error_recall": float(captured.sum()/error.sum()) if error.any() else 0.0,
        "routing_precision": float(captured.sum()/routed.sum()) if routed.any() else 0.0,
        "highconf_error_n": int(high_error.sum()),
        "captured_highconf_error_n": int((routed & high_error).sum()),
        "highconf_error_recall": float((routed & high_error).sum()/high_error.sum()) if high_error.any() else 0.0,
        "score_mean": float(np.mean(score)),
        "score_q75": float(np.quantile(score, 0.75)),
        "score_max": float(np.max(score)),
    }


def exact_binomial_two_sided(b: int, c: int) -> float:
    """Exact two-sided sign/McNemar p-value for discordant counts."""
    n = int(b)+int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    lower = sum(math.comb(n, i) for i in range(k+1))/(2.0**n)
    return float(min(1.0, 2.0*lower))

