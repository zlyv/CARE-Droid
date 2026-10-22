#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

import numpy as np

from contracts import CompatibilityModel


MALWARE_FEATURES = ("vm_attn", "vm_critical", "vm_mean", "ui", "mapping_coverage")
BENIGN_FEATURES = ("fb_attn", "fb_critical", "fb_mean", "confinement", "ui")
DECL_MALWARE_FEATURES = ("coarse_u", "coarse_unknown")
DECL_BENIGN_FEATURES = ("coarse_j", "coarse_unknown")
ORIGIN_FEATURES = ("mapping_coverage", "origin_mixed", "origin_unassigned", "origin_multi_component")
COUNT_FEATURES = ("component_count", "token_count", "host_count", "assigned_host_count", "graph_node_count", "graph_edge_count")


class RolePermutedCompatibility:
    def __init__(self, base: CompatibilityModel, seed: int):
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(31)
        if np.any(permutation == np.arange(31)):
            permutation = np.roll(permutation, 1)
        self.base = base; self.permutation = permutation

    def lookup(self, tokens, role):
        return self.base.lookup(tokens, int(self.permutation[int(role)]))


def prepare_records(records: list[dict[str, Any]], p31: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for record, vector in zip(records, np.asarray(p31, float)):
        row = copy.deepcopy(record)
        row["active_roles"] = np.flatnonzero(vector > 0).astype(int).tolist()
        output.append(row)
    return output


def _component_role(model, component, role):
    value = model.lookup(component.get("tokens", []), int(role))
    if not value.get("known"):
        return {"known": False, "match": 0.0, "violation": 0.0, "specificity": 0.0}
    probability = float(value["compatibility"]); background = float(value.get("background", 0.0))
    match = max(0.0, probability - background) / max(1.0 - background, 1e-6)
    violation = max(0.0, background - probability) / max(background, 1e-6)
    return {
        "known": True, "match": float(match), "violation": float(violation),
        "specificity": float(value.get("specificity", 0.0)),
    }


def coarse_scores(record, vector, model):
    tokens = sorted({token for component in record.get("components", []) for token in component.get("tokens", [])} | set(record.get("uses_features", [])))
    vector = np.asarray(vector, float)
    known_mass = unknown_mass = u = j = 0.0
    for role, mass in enumerate(vector):
        if mass <= 0:
            continue
        value = model.lookup(tokens, role)
        if not value.get("known"):
            unknown_mass += mass; continue
        known_mass += mass
        compatibility = float(value["compatibility"])
        specificity = float(value.get("specificity", 0.0))
        u += mass * (1.0 - compatibility)
        j += mass * compatibility * specificity
    denominator = max(known_mass, 1e-8)
    return {
        "coarse_u": float(u / denominator) if known_mass else 0.0,
        "coarse_j": float(j / denominator) if known_mass else 0.0,
        "coarse_unknown": float(unknown_mass / max(float(vector.sum()), 1e-8)),
    }


def evidence_for_apk(record, vector, model, attention, critical_index):
    components = {int(row["component_id"]): row for row in record.get("components", [])}
    hosts = record.get("hosts", [])
    attention = np.asarray(attention, float)
    if len(attention) != max(len(hosts), 1):
        raise RuntimeError(f"Host attention mismatch {record['sha256']}: {len(attention)} != {len(hosts)}")
    if not hosts:
        hosts = [{"roles": [], "reachable_component_ids": []}]
    attention = attention / max(float(attention.sum()), 1e-12)
    host_rows = []
    for position, host in enumerate(hosts):
        reachable = [components[value] for value in host.get("reachable_component_ids", []) if value in components]
        vm_values = []; fb_values = []; uncertain_roles = 0; mixed_roles = 0
        for role in host.get("roles", []):
            values = [_component_role(model, component, role) for component in reachable]
            known = [value for value in values if value["known"]]
            matched = [value for value in known if value["match"] > 0]
            mismatched = [value for value in known if value["violation"] > 0]
            if matched and mismatched:
                mixed_roles += 1; uncertain_roles += 1
                continue
            if matched:
                confinement = len(matched) / max(len(reachable), 1)
                fb_values.append(max(value["match"] * max(value["specificity"], 1e-6) for value in matched) * confinement)
            elif mismatched:
                vm_values.append(max(value["violation"] for value in mismatched))
            else:
                uncertain_roles += 1
        role_n = max(len(host.get("roles", [])), 1)
        vm = float(np.mean(vm_values)) if vm_values else 0.0
        fb = float(np.mean(fb_values)) if fb_values else 0.0
        uncertain = min(1.0, uncertain_roles / role_n)
        confinement = len([v for v in fb_values if v > 0]) / role_n
        host_rows.append({
            "host_position": position, "vm": vm, "fb": fb,
            "uncertain": uncertain, "confinement": confinement,
            "state": "mixed" if mixed_roles else ("matched" if fb > 0 and vm == 0 else ("mismatched" if vm > 0 and fb == 0 else "unassigned")),
            "reachable_component_count": len(reachable),
            "roles": host.get("roles", []),
        })
    vm = np.asarray([row["vm"] for row in host_rows], float)
    fb = np.asarray([row["fb"] for row in host_rows], float)
    uncertain = np.asarray([row["uncertain"] for row in host_rows], float)
    confinement = np.asarray([row["confinement"] for row in host_rows], float)
    assigned = np.asarray([row["reachable_component_count"] > 0 for row in host_rows], float)
    states = Counter(row["state"] for row in host_rows)
    coverage = float(np.sum(attention * assigned))
    conflict = min(float(np.sum(attention * vm)), float(np.sum(attention * fb)))
    ui = min(1.0, .5 * float(np.sum(attention * uncertain)) + .3 * (1.0 - coverage) + .2 * conflict)
    critical_index = int(np.clip(critical_index, 0, len(host_rows) - 1))
    result = {
        "vm_attn": float(np.sum(attention * vm)), "vm_critical": float(vm[critical_index]), "vm_mean": float(vm.mean()),
        "fb_attn": float(np.sum(attention * fb)), "fb_critical": float(fb[critical_index]), "fb_mean": float(fb.mean()),
        "confinement": float(np.sum(attention * confinement)), "ui": ui,
        "mapping_coverage": coverage,
        "origin_mixed": states["mixed"] / max(len(host_rows), 1),
        "origin_unassigned": states["unassigned"] / max(len(host_rows), 1),
        "origin_multi_component": sum(row["reachable_component_count"] > 1 for row in host_rows) / max(len(host_rows), 1),
        "component_count": float(len(components)),
        "token_count": float(len({token for component in components.values() for token in component.get("tokens", [])})),
        "host_count": float(len(record.get("hosts", []))),
        "assigned_host_count": float(sum(row["reachable_component_count"] > 0 for row in host_rows)),
        "graph_node_count": float(record.get("graph_node_count", 0)),
        "graph_edge_count": float(record.get("graph_edge_count", 0)),
        "critical_host_state": host_rows[critical_index]["state"],
        "critical_host_roles": "|".join(map(str, host_rows[critical_index]["roles"])),
        "matched_hosts": int(states["matched"]), "mismatched_hosts": int(states["mismatched"]),
        "mixed_hosts": int(states["mixed"]), "unassigned_hosts": int(states["unassigned"]),
        "host_rows": host_rows,
    }
    result.update(coarse_scores(record, vector, model))
    return result


def feature_matrix(rows, names):
    matrix = np.asarray([[float(row.get(name, 0.0)) for name in names] for row in rows], float)
    if not np.isfinite(matrix).all():
        raise RuntimeError("NaN/Inf in CCBC feature matrix")
    return matrix


def permission_view(record):
    row = copy.deepcopy(record)
    weak = list(row.get("weak_permissions", []))
    row["components"] = [{
        "component_id": 0, "type": "permission_null", "tokens": weak,
        "reachable_roles": list(row.get("active_roles", [])),
    }]
    return row


def shuffle_contracts(records, stage1_probability, seed, p31=None):
    """Within-stratum contract permutation; topology/Host scopes stay fixed."""
    output = copy.deepcopy(records)
    rng = np.random.default_rng(seed)
    component_count = np.asarray([len(row.get("components", [])) for row in records], int)
    host_count = np.asarray([len(row.get("hosts", [])) for row in records], int)
    pred = np.asarray(stage1_probability) >= .5
    if p31 is None:
        p31 = np.zeros((len(records), 31), float)
    p31 = np.asarray(p31, float)
    if p31.shape != (len(records), 31):
        raise RuntimeError(f"P31 shuffle stratum shape mismatch: {p31.shape}")
    mass = p31.sum(axis=1)
    normalized = p31 / np.maximum(mass[:, None], 1e-12)
    entropy = -np.sum(np.where(normalized > 0, normalized * np.log(np.maximum(normalized, 1e-12)), 0.0), axis=1)
    projection = normalized @ np.linspace(-1.0, 1.0, 31) + .05 * entropy
    component_bin = np.digitize(component_count, [0, 1, 3, 6], right=True)
    host_edges = np.quantile(host_count, [0, .25, .5, .75, 1])
    host_bin = np.digitize(host_count, np.unique(host_edges)[1:-1], right=True)
    strata = {}
    for index, key in enumerate(zip(pred.tolist(), component_bin.tolist(), host_bin.tolist())):
        strata.setdefault(key, []).append(index)
    for indices in strata.values():
        indices = sorted(indices, key=lambda index: (float(projection[index]), records[index].get("sha256", "")))
        donors = np.asarray(indices, int).copy()
        if len(donors) > 1:
            # Adjacent cyclic assignment keeps coarse P31 composition close while ensuring a derangement.
            offset = 1 if len(donors) < 4 else int(rng.integers(1, min(4, len(donors))))
            donors = np.roll(donors, offset)
        for target, donor in zip(indices, donors):
            donor_components = records[int(donor)].get("components", [])
            for position, component in enumerate(output[target].get("components", [])):
                if donor_components:
                    source = donor_components[position % len(donor_components)]
                    component["tokens"] = list(source.get("tokens", []))
                    component["weak_tokens"] = list(source.get("weak_tokens", []))
            output[target]["uses_features"] = list(records[int(donor)].get("uses_features", []))
            output[target]["weak_permissions"] = list(records[int(donor)].get("weak_permissions", []))
    return output

