#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from common import SCS_OUTPUT

GROUPS = ("TRANSFER", "PERSIST", "EXECUTE", "VISIBLE", "SENSITIVE_ACTION")
COUNT_FIELDS = ("graph_method_count", "graph_edge_count", "host_count", "effect_method_count", "effect_tag_count")


def load_record(sha256: str) -> dict[str, Any]:
    path = SCS_OUTPUT / "cache/features/source" / f"{sha256}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def entropy(values: list[float]) -> float:
    array = np.maximum(np.asarray(values, float), 0.0); total = float(array.sum())
    if total <= 0:
        return 0.0
    probability = array / total; nonzero = probability[probability > 0]
    return float(-(nonzero * np.log(nonzero)).sum() / math.log(len(array))) if len(array) > 1 else 0.0


def hhi(values: list[float]) -> float:
    array = np.maximum(np.asarray(values, float), 0.0); total = float(array.sum())
    return float(np.square(array / total).sum()) if total > 0 else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def normalized(record: dict[str, Any], variant: str) -> dict[str, float]:
    raw = record["variants"][variant]; output: dict[str, float] = {}
    reaches = [float(raw[f"attn_{group}_reach"]) for group in GROUPS]
    means = [float(raw[f"mean_{group}_reach"]) for group in GROUPS]
    output["group_entropy"] = entropy(reaches); output["group_hhi"] = hhi(reaches)
    output["mean_group_entropy"] = entropy(means); output["mean_group_hhi"] = hhi(means)
    output["attn_mean_any_contrast"] = float(raw["attn_any_effect"] - raw["mean_any_effect"])
    total_reach = sum(reaches)
    for group, reach, mean_reach in zip(GROUPS, reaches, means):
        output[f"{group}_share"] = ratio(reach, total_reach)
        output[f"{group}_attn_mean_contrast"] = reach - mean_reach
        output[f"{group}_inverse_distance_norm"] = ratio(float(raw[f"attn_{group}_inverse_distance"]), reach)
        for depth in ("d0", "d1", "d2", "d3_4"):
            output[f"{group}_{depth}_share"] = ratio(float(raw[f"attn_{group}_{depth}"]), reach)
        for boundary in ("SAME_CLASS", "SAME_PACKAGE", "CROSS_PACKAGE"):
            output[f"{group}_{boundary}_share"] = ratio(float(raw[f"attn_{group}_{boundary}"]), reach)
        unique = float(raw[f"attn_{group}_unique_target"]); shared = float(raw[f"attn_{group}_shared_target"])
        output[f"{group}_unique_share"] = ratio(unique, unique + shared)
        output[f"{group}_shared_share"] = ratio(shared, unique + shared)
    transfer, persist, execute, visible, sensitive = reaches
    output["outward_vs_visible_balance"] = float((transfer + execute - visible) / (transfer + execute + visible + 1e-9))
    output["outward_vs_local_balance"] = float((transfer + execute - persist - visible) / (transfer + execute + persist + visible + 1e-9))
    critical = [float(raw[f"critical_{group}"]) for group in GROUPS]
    output["critical_group_fraction"] = float(sum(critical) / len(GROUPS)); output["critical_group_entropy"] = entropy(critical)
    output["slice_node_ratio"] = float(raw["slice_node_ratio"]); output["effect_group_diversity"] = float(raw["effect_group_diversity"])
    for group in GROUPS:
        values = [float(raw[f"role_{role:02d}_{group}"]) for role in range(31)]
        output[f"{group}_role_entropy"] = entropy(values); output[f"{group}_role_hhi"] = hhi(values)
    return output


def raw_semantic(record: dict[str, Any], variant: str = "REAL") -> dict[str, float]:
    return {key: float(value) for key, value in record["variants"][variant].items()}


def counts(record: dict[str, Any]) -> dict[str, float]:
    audit = record["audit"]
    return {key: float(audit[key]) for key in COUNT_FIELDS}


def matrix(dicts: list[dict[str, float]]) -> tuple[np.ndarray, list[str]]:
    fields = sorted(dicts[0]); values = np.asarray([[float(item.get(field, 0.0)) for field in fields] for item in dicts], float)
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite feature matrix")
    return values, fields
