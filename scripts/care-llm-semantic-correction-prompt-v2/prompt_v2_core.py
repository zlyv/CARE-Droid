#!/usr/bin/env python3
"""Pure, label-free Prompt V2 feature and rendering helpers."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

import numpy as np

MODEL_ROLE_ORDER = [
    "command_execution", "dynamic_code_loading", "native_library_loading",
    "reflective_invocation", "package_installation", "package_uninstallation",
    "device_administration", "accessibility_control", "sms_send_receive",
    "call_control_or_recording", "account_or_credential_access",
    "contacts_or_calendar_access", "device_identifier_access", "location_access",
    "microphone_access", "camera_access", "clipboard_access",
    "network_transmission_or_upload", "sensitive_file_or_content_write",
    "boot_persistence", "task_or_alarm_scheduling", "background_service_control",
    "overlay_or_ui_deception", "icon_or_component_hiding",
    "process_or_wakelock_control", "cryptography", "encoding_or_compression",
    "url_or_http_setup", "file_read", "database_access", "other",
]

PAYLOAD_TO_MODEL_ROLE = [
    7, 10, 21, 19, 9, 15, 16, 0, 11, 25, 29, 6, 12, 1, 26, 28,
    23, 13, 14, 2, 17, 30, 22, 4, 5, 24, 3, 18, 8, 20, 27,
]

FORBIDDEN_METADATA_KEYS = {
    "label", "ground_truth", "fixed", "broken", "correct", "outcome",
    "is_malware", "is_benign", "year", "fold",
}

SYSTEM_PROMPT = """You are an Android static-behavior semantic auditor.

Your task is NOT to independently classify an APK as benign or malicious.

You are given a current Stage-I decision, a proposed bounded correction, and
scale-normalized static evidence. Evaluate whether the supplied evidence
semantically supports the PROPOSED CORRECTION DIRECTION.

Important rules:
- Use only the supplied evidence.
- Do not assume behavior that is not shown.
- Do not infer from APK identity, dataset, year, size, or ground-truth label.
- All percentages are within-APK composition values, not population rates.
- Role presence alone is not sufficient evidence of maliciousness.
- Common legitimate uses of reflection, networking, cryptography, databases,
  dynamic loading, background services, etc. must be considered.
- Reconstructed direct Host-primary-role relations are static graph
  relationships, not proven runtime control flow or data flow.
- An unavailable direct relation means this bounded view did not observe one;
  it does not mean the APK has no behavioral relationship.
- Be conservative when evidence is ambiguous or insufficient.

Return JSON only:
{
  "semantic_support": <number from 0 to 1>,
  "semantic_conflict": <number from 0 to 1>,
  "benign_plausibility": <number from 0 to 1>,
  "evidence_sufficiency": <number from 0 to 1>,
  "behavior_summary": "<short sentence>",
  "reason_codes": ["CODE1", "CODE2"]
}

semantic_support means how strongly the evidence supports the proposed
correction, NOT the probability that the APK is malicious.
"""


def norm_sha(value) -> str:
    return str(value or "").strip().lower()


def pred_name(value) -> str:
    return "malicious" if int(value) == 1 else "benign"


def percent(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def reconstruct_primary_graph(primary, outgoing_offsets, outgoing_positions):
    primary = np.asarray(primary, dtype=np.int64)
    offsets = np.asarray(outgoing_offsets, dtype=np.int64)
    positions = np.asarray(outgoing_positions, dtype=np.int64)
    n = len(primary)
    if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(positions):
        raise RuntimeError("Host outgoing-offset contract failed")
    if np.any(np.diff(offsets) < 0):
        raise RuntimeError("Host outgoing offsets are not monotonic")
    edges = set()
    for source in range(n):
        for target in positions[offsets[source]:offsets[source + 1]]:
            target = int(target)
            if not 0 <= target < n:
                raise RuntimeError("Host neighbor position is out of range")
            if source == target:
                raise RuntimeError("Unexpected Host self-neighbor")
            edges.add((source, target))
    matrix = np.zeros((31, 31), dtype=np.int64)
    for source, target in sorted(edges):
        left, right = int(primary[source]), int(primary[target])
        if not (0 <= left < 31 and 0 <= right < 31):
            raise RuntimeError("Host primary role is outside model role order")
        matrix[left, right] += 1
    return matrix, edges


def normalized_entropy(counts) -> float:
    value = np.asarray(counts, dtype=np.float64)
    value = value[value > 0]
    if len(value) <= 1:
        return 0.0
    probability = value / value.sum()
    return float(-np.sum(probability * np.log(probability)) / math.log(len(probability)))


def summarize_evidence(role_node_counts, relation_matrix, summary: dict) -> dict:
    roles = np.asarray(role_node_counts, dtype=np.float64)
    relation = np.asarray(relation_matrix, dtype=np.float64)
    if roles.shape != (31,) or relation.shape != (31, 31):
        raise RuntimeError("Role tensor shape contract failed")
    if np.any(roles < 0) or np.any(relation < 0):
        raise RuntimeError("Negative role count")

    role_total = float(roles.sum())
    role_share = roles / role_total if role_total > 0 else np.zeros(31, dtype=np.float64)
    role_items = [
        (MODEL_ROLE_ORDER[index], float(role_share[index]))
        for index in range(30) if role_share[index] > 0
    ]
    role_items.sort(key=lambda item: (-item[1], item[0]))
    shown_roles = role_items[:10]
    remaining_role_mass = max(0.0, 1.0 - sum(value for _, value in shown_roles))

    diagonal = float(np.trace(relation))
    total = float(relation.sum())
    cross_total = total - diagonal
    semantic_cross = relation.copy()
    np.fill_diagonal(semantic_cross, 0)
    semantic_cross[30, :] = 0
    semantic_cross[:, 30] = 0
    semantic_cross_total = float(semantic_cross.sum())
    relation_items = []
    if semantic_cross_total > 0:
        for left, right in np.argwhere(semantic_cross > 0):
            relation_items.append((
                MODEL_ROLE_ORDER[int(left)], MODEL_ROLE_ORDER[int(right)],
                float(semantic_cross[left, right] / semantic_cross_total),
            ))
    relation_items.sort(key=lambda item: (-item[2], item[0], item[1]))

    reachable = int(summary.get("entry_reachable_selected_endpoint_count", 0) or 0)
    unreachable = int(summary.get("entry_unreachable_selected_endpoint_count", 0) or 0)
    entry_total = reachable + unreachable

    provenance_raw = summary.get("active_node_provenance_counts", {}) or {}
    provenance = {
        str(key).strip().lower(): max(0, int(value or 0))
        for key, value in provenance_raw.items()
    }
    provenance_total = sum(provenance.values())
    provenance_share = {
        key: value / provenance_total for key, value in provenance.items()
    } if provenance_total else {}

    return {
        "top_roles": shown_roles,
        "remaining_role_mass": remaining_role_mass,
        "role_profile_available": role_total > 0,
        "top_relations": relation_items[:10],
        "direct_relation_available": total > 0,
        "semantic_cross_relation_available": semantic_cross_total > 0,
        "same_role_ratio": diagonal / total if total else None,
        "cross_role_ratio": cross_total / total if total else None,
        "semantic_cross_relation_entropy": normalized_entropy(semantic_cross.reshape(-1)),
        "entry_available": entry_total > 0,
        "entry_reachable_ratio": reachable / entry_total if entry_total else None,
        "entry_unreachable_ratio": unreachable / entry_total if entry_total else None,
        "provenance_available": provenance_total > 0,
        "provenance_share": provenance_share,
        # Audit-only values; these are never rendered into the prompt.
        "audit_role_mass_total": int(role_total),
        "audit_direct_edge_total": int(total),
        "audit_semantic_cross_edge_total": int(semantic_cross_total),
        "audit_relation_type_n": int(np.count_nonzero(relation)),
    }


def render_user_prompt(stage1_pred: int, candidate_pred: int, evidence: dict) -> str:
    lines = [
        f"CURRENT STAGE-I DECISION: {pred_name(stage1_pred)}",
        f"PROPOSED CORRECTION: {pred_name(stage1_pred)} -> {pred_name(candidate_pred)}",
        "", "SENSITIVE ROLE COMPOSITION (within-APK role-mass share):",
    ]
    if evidence["role_profile_available"]:
        for role, share in evidence["top_roles"]:
            lines.append(f"- {role}: {percent(share)}")
        lines.append(f"- remaining_role_mass: {percent(evidence['remaining_role_mass'])}")
    else:
        lines.append("- unavailable in this bounded static view")

    lines.extend(["", "RECONSTRUCTED DIRECT HOST-PRIMARY-ROLE RELATIONS:"])
    if evidence["semantic_cross_relation_available"]:
        for left, right, share in evidence["top_relations"]:
            lines.append(f"- {left} -> {right}: {percent(share)} of observed semantic cross-role relations")
    else:
        lines.append("- no direct semantic cross-role relation was observed in this bounded view")

    lines.extend(["", "DIRECT RELATION COMPOSITION:"])
    if evidence["direct_relation_available"]:
        lines.append(f"- same_role_ratio: {percent(evidence['same_role_ratio'])}")
        lines.append(f"- cross_role_ratio: {percent(evidence['cross_role_ratio'])}")
    else:
        lines.append("- direct_relation_available: false")
        lines.append("- observation_scope: no direct Host-to-Host relation was observed")

    lines.extend(["", "ENTRY REACHABILITY OF SELECTED SENSITIVE ENDPOINTS:"])
    if evidence["entry_available"]:
        lines.append(f"- reachable_ratio: {percent(evidence['entry_reachable_ratio'])}")
        lines.append(f"- unreachable_ratio: {percent(evidence['entry_unreachable_ratio'])}")
    else:
        lines.append("- entry_reachability_available: false")

    lines.extend(["", "ACTIVE NODE PROVENANCE COMPOSITION:"])
    if evidence["provenance_available"]:
        for key in ("bundled_non_package", "external_or_framework", "package_aligned"):
            lines.append(f"- {key}_ratio: {percent(evidence['provenance_share'].get(key, 0.0))}")
    else:
        lines.append("- provenance_available: false")

    lines.extend([
        "", "QUESTION:",
        "Does this scale-normalized static behavioral evidence semantically support the proposed correction direction?",
    ])
    return "\n".join(lines)


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    canonical = json.dumps(
        {"system_prompt": system_prompt, "user_prompt": user_prompt},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
