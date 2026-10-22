"""Deterministic, bounded Android behavior-graph representation.

This module deliberately keeps the full Androguard/NetworkX graph in memory.
The public result is a compact audit summary that contains no raw DEX, method
signature list, or serialized graph.  The same builder can later be called
online by the graph model trainer.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class GraphPolicy:
    H: int = 2
    K_out: int = 8
    P_cap: int = 4
    E_cap: int = 2000
    max_states: int = 20000


def _safe_call(function, default=None):
    try:
        return function()
    except Exception:
        return default


def _object_method(node):
    try:
        getter = getattr(node, "get_method", None)
        if callable(getter):
            method = getter()
            if method is not None:
                return method
    except Exception:
        pass
    return node


def method_signature(node) -> str:
    method = _object_method(node)
    parts = []
    for name in ("get_class_name", "get_name", "get_descriptor"):
        try:
            getter = getattr(method, name, None)
            parts.append(str(getter()) if callable(getter) else "")
        except Exception:
            parts.append("")
    if parts[0] or parts[1]:
        return f"{parts[0]}->{parts[1]}{parts[2]}"
    return str(method)


def canonical_signature(value: Any) -> str:
    return WHITESPACE_RE.sub("", str(value or "")).casefold()


def class_and_method(signature: str) -> tuple[str, str]:
    value = canonical_signature(signature)
    if "->" not in value:
        return value, ""
    class_name, tail = value.split("->", 1)
    return class_name, tail.split("(", 1)[0]


def is_external(node) -> bool:
    method = _object_method(node)
    try:
        getter = getattr(method, "is_external", None)
        if callable(getter) and bool(getter()):
            return True
    except Exception:
        pass
    class_name, _ = class_and_method(method_signature(node))
    return class_name.startswith(
        (
            "landroid/",
            "landroidx/",
            "ljava/",
            "Ljava/".casefold(),
            "Ljavax/".casefold(),
            "lkotlin/",
            "lorg/json/",
        )
    )


def api_rule_matches(rule: dict, class_name: str, method_name: str) -> bool:
    if rule.get("rule_kind", "api") != "api":
        return False
    class_prefixes = [canonical_signature(x) for x in rule.get("class_prefixes", [])]
    class_contains = [canonical_signature(x) for x in rule.get("class_contains", [])]
    method_names = [canonical_signature(x) for x in rule.get("method_names", [])]
    method_prefixes = [canonical_signature(x) for x in rule.get("method_prefixes", [])]
    class_ok = any(class_name.startswith(x) for x in class_prefixes)
    class_ok = class_ok or any(x in class_name for x in class_contains)
    method_ok = method_name in method_names
    method_ok = method_ok or any(method_name.startswith(x) for x in method_prefixes)
    return bool(class_ok and method_ok and (class_prefixes or class_contains))


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on"}


def _edge_value(data: Any, names: Iterable[str]):
    if not hasattr(data, "get"):
        return None
    for name in names:
        if name in data:
            return data.get(name)
    return None


def semantic_edge_match(target_signature: str, edge_rules: list[dict]) -> tuple[str, bool, bool, bool]:
    class_name, method_name = class_and_method(target_signature)
    for rule in edge_rules:
        class_prefixes = [canonical_signature(x) for x in rule.get("class_prefixes", [])]
        method_names = [canonical_signature(x) for x in rule.get("method_names", [])]
        class_ok = any(class_name.startswith(item) for item in class_prefixes)
        method_ok = method_name in method_names
        if not (class_ok and method_ok):
            continue
        flag = str(rule.get("flag", "")).strip().casefold()
        return (
            canonical_signature(rule.get("edge_type", "semantic_edge")).upper(),
            flag == "icc_any",
            flag == "reflection_any",
            True,
        )
    return "DIRECT_CALL", False, False, False


def edge_descriptor(
    data: Any,
    target_signature: str = "",
    edge_rules: list[dict] | None = None,
) -> tuple[str, bool, bool, bool, str]:
    """Return edge type, ICC flag, reflection flag, availability, and source.

    Androguard's default call graph exposes direct calls but does not always
    expose ICC/reflection annotations.  Missing annotations are therefore
    recorded as unavailable instead of being silently guessed.
    """

    edge_rules = edge_rules or []
    if not hasattr(data, "get"):
        edge_type, icc_any, reflection_any, matched = semantic_edge_match(
            target_signature, edge_rules
        )
        return edge_type, icc_any, reflection_any, False, "SEMANTIC_RULE" if matched else "NONE"
    edge_type = _edge_value(data, ("edge_type", "type", "kind", "call_type"))
    edge_type = canonical_signature(edge_type or "direct_call").upper()
    icc_value = _edge_value(data, ("icc_any", "icc", "is_icc", "inter_component"))
    reflection_value = _edge_value(
        data, ("reflection_any", "reflection", "is_reflection")
    )
    flag_available = icc_value is not None or reflection_value is not None
    if icc_value is None and "icc" in edge_type.casefold():
        icc_value = True
        flag_available = True
    if reflection_value is None and "reflect" in edge_type.casefold():
        reflection_value = True
        flag_available = True
    if flag_available:
        return (
            edge_type,
            _boolish(icc_value),
            _boolish(reflection_value),
            True,
            "CALL_GRAPH",
        )
    semantic_type, semantic_icc, semantic_reflection, matched = semantic_edge_match(
        target_signature, edge_rules
    )
    if matched:
        return semantic_type, semantic_icc, semantic_reflection, True, "SEMANTIC_RULE"
    return edge_type, False, False, False, "NONE"


def _manifest_attr(element, suffix: str) -> str:
    for key, value in getattr(element, "attrib", {}).items():
        if str(key).casefold().endswith(suffix.casefold()):
            return str(value)
    return ""


def _descriptor_from_name(name: str, package_name: str = "") -> str:
    value = str(name or "").strip()
    if not value:
        return ""
    if value.startswith("L") and value.endswith(";"):
        return canonical_signature(value)
    if value.startswith("."):
        value = f"{package_name}{value}"
    elif "." not in value and package_name:
        value = f"{package_name}.{value}"
    return canonical_signature("L" + value.replace(".", "/") + ";")


def _component_names(apk, package_name: str):
    components = []
    for component_type, getter_name in (
        ("activity", "get_activities"),
        ("service", "get_services"),
        ("receiver", "get_receivers"),
        ("provider", "get_providers"),
    ):
        values = _safe_call(getattr(apk, getter_name, lambda: []), []) or []
        for value in values:
            descriptor = _descriptor_from_name(value, package_name)
            if descriptor:
                components.append({"type": component_type, "descriptor": descriptor})
    return components


def _manifest_aliases(apk, package_name: str):
    aliases = []
    try:
        manifest = apk.get_android_manifest_xml()
        for element in manifest.iter():
            tag = str(getattr(element, "tag", "")).casefold()
            if not tag.endswith("activity-alias"):
                continue
            alias = _descriptor_from_name(_manifest_attr(element, "name"), package_name)
            target = _descriptor_from_name(
                _manifest_attr(element, "targetactivity"), package_name
            )
            if alias and target:
                aliases.append((alias, target))
    except Exception:
        pass
    return aliases


LIFECYCLE_METHODS = {
    "activity": {
        "oncreate", "onstart", "onresume", "onnewintent", "onactivityresult",
        "onpause", "onstop",
    },
    "service": {
        "oncreate", "onstartcommand", "onbind", "onunbind", "ondestroy",
        "onhandleintent",
    },
    "receiver": {"onreceive"},
    "provider": {"oncreate", "query", "insert", "update", "delete", "openfile", "call"},
}


def bind_manifest_entries(apk, node_records: dict[str, dict]) -> dict:
    package_name = str(_safe_call(apk.get_package, "") or "")
    components = _component_names(apk, package_name)
    entries = set()
    bound_components = 0
    component_present_no_entry = 0
    for component in components:
        candidate_methods = []
        for signature, record in node_records.items():
            if record["class_name"] != component["descriptor"]:
                continue
            if record["method_name"] in LIFECYCLE_METHODS.get(component["type"], set()):
                candidate_methods.append(signature)
        if candidate_methods:
            bound_components += 1
            entries.update(candidate_methods)
        else:
            component_present_no_entry += 1

    alias_bound = 0
    aliases = _manifest_aliases(apk, package_name)
    for _, target in aliases:
        target_methods = {
            signature
            for signature, record in node_records.items()
            if record["class_name"] == target
            and record["method_name"] in LIFECYCLE_METHODS["activity"]
        }
        if target_methods:
            alias_bound += 1
            entries.update(target_methods)

    return {
        "component_count": len(components),
        "bound_component_count": bound_components,
        "component_present_no_entry_count": component_present_no_entry,
        "alias_count": len(aliases),
        "alias_bound_count": alias_bound,
        "entry_method_count": len(entries),
        "entry_signatures": entries,
    }


def manifest_context_role_counts(apk, rules: list[dict]) -> Counter:
    result = Counter()
    try:
        manifest = apk.get_android_manifest_xml()
    except Exception:
        return result
    for rule in rules:
        if rule.get("rule_kind") != "manifest_context":
            continue
        wanted = {canonical_signature(x) for x in rule.get("context_values", [])}
        if not wanted:
            continue
        for element in manifest.iter():
            if not str(getattr(element, "tag", "")).casefold().endswith("action"):
                continue
            value = _manifest_attr(element, "name")
            if canonical_signature(value) in wanted:
                result[rule["role"]] += 1
    return result


def _iter_edges(graph):
    try:
        return graph.edges(data=True)
    except TypeError:
        return ((left, right, {}) for left, right in graph.edges)


def _path_hash(nodes: tuple[str, ...], edges: tuple[dict, ...]) -> str:
    payload = "|".join(nodes) + "||" + "|".join(edge["edge_type"] for edge in edges)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enumerate_sensitive_paths(
    adjacency: dict[str, list[dict]],
    entry_signatures: set[str],
    sensitive_targets: set[str],
    policy: GraphPolicy,
    include_path_nodes: bool = False,
) -> dict:
    candidates: dict[str, dict[str, dict]] = defaultdict(dict)
    overflow: set[str] = set()
    states_explored = 0
    state_cap_hit = False
    target_cap = max(1, policy.P_cap)

    for entry in sorted(entry_signatures):
        if entry not in adjacency and entry not in sensitive_targets:
            continue
        queue = deque([(entry, (entry,), tuple())])
        while queue:
            if states_explored >= policy.max_states:
                state_cap_hit = True
                break
            current, nodes, edges = queue.popleft()
            states_explored += 1
            if current in sensitive_targets:
                record = {
                    "target": current,
                    "hop": len(edges),
                    "icc_any": any(edge["icc_any"] for edge in edges),
                    "reflection_any": any(edge["reflection_any"] for edge in edges),
                    "path_hash": _path_hash(nodes, edges),
                    "nodes": nodes,
                    "edges": edges,
                }
                bucket = candidates[current]
                key = record["path_hash"]
                if key not in bucket:
                    if len(bucket) < target_cap + 1:
                        bucket[key] = record
                    else:
                        overflow.add(current)
                continue
            if len(edges) >= max(0, policy.H):
                continue
            for edge in adjacency.get(current, [])[: max(1, policy.K_out)]:
                target = edge["target"]
                if target in nodes:
                    continue
                queue.append((target, nodes + (target,), edges + (edge,)))
        if state_cap_hit:
            break

    retained = []
    candidate_path_count = 0
    target_count = 0
    per_target = {}
    for target in sorted(candidates):
        bucket = candidates[target]
        target_count += 1
        candidate_count = len(bucket) + (1 if target in overflow else 0)
        candidate_path_count += candidate_count
        ordered = sorted(
            bucket.values(),
            key=lambda item: (
                item["hop"],
                -int(item["icc_any"]),
                -int(item["reflection_any"]),
                item["path_hash"],
            ),
        )
        chosen = ordered[:target_cap]
        retained.extend(chosen)
        per_target[target] = {
            "candidate_path_count_capped": candidate_count,
            "retained_path_count": len(chosen),
            "flags_truncated": target in overflow or candidate_count > target_cap,
        }

    retained.sort(
        key=lambda item: (
            item["hop"],
            -int(item["icc_any"]),
            -int(item["reflection_any"]),
            item["path_hash"],
        )
    )
    global_truncated = len(retained) > max(1, policy.E_cap)
    retained = retained[: max(1, policy.E_cap)]
    retained_edge_keys = {
        (
            edge["source"],
            edge["target"],
            edge["edge_type"],
            edge["icc_any"],
            edge["reflection_any"],
        )
        for path in retained
        for edge in path["edges"]
    }
    result = {
        "candidate_path_count_capped": candidate_path_count,
        "retained_path_count": len(retained),
        "retained_edge_count": len(retained_edge_keys),
        "sensitive_target_count_reached": target_count,
        "flags_truncated": global_truncated or bool(overflow) or state_cap_hit,
        "target_summaries": list(per_target.values()),
        "path_hashes": [item["path_hash"] for item in retained[:64]],
        "states_explored": states_explored,
        "state_cap_hit": state_cap_hit,
    }
    if include_path_nodes:
        result["path_node_signatures"] = sorted(
            {
                signature
                for path in retained
                for signature in path["nodes"]
            }
        )
    return result


def build_compact_graph_summary(
    apk,
    graph,
    rules: list[dict],
    priority: dict,
    policy: GraphPolicy,
    edge_rules: list[dict] | None = None,
    include_internal: bool = False,
) -> dict:
    """Build a bounded summary from one in-memory call graph."""

    raw_nodes = list(graph.nodes)
    node_records: dict[str, dict] = {}
    node_object_to_signature = {}
    api_rules = [rule for rule in rules if rule.get("rule_kind", "api") == "api"]
    tiers = priority.get("tiers", {}) if isinstance(priority, dict) else {}
    role_tier = {}
    for tier, value in tiers.items():
        for role in (value or {}).get("roles", []):
            try:
                role_tier[role] = int(tier)
            except Exception:
                role_tier[role] = 4

    for node in raw_nodes:
        signature = canonical_signature(method_signature(node))
        if not signature:
            continue
        node_object_to_signature[node] = signature
        class_name, method_name = class_and_method(signature)
        external = is_external(node)
        roles = set()
        if external:
            roles = {
                rule["role"]
                for rule in api_rules
                if api_rule_matches(rule, class_name, method_name)
            }
        current = node_records.get(signature)
        if current is None:
            node_records[signature] = {
                "class_name": class_name,
                "method_name": method_name,
                "external": external,
                "roles": set(roles),
            }
        else:
            current["external"] = current["external"] or external
            current["roles"].update(roles)

    edge_map = {}
    edge_flag_metadata_seen = False
    semantic_edge_rule_seen = False
    edge_source_counts = Counter()
    raw_edge_count = 0
    for left, right, data in _iter_edges(graph):
        raw_edge_count += 1
        source = node_object_to_signature.get(left) or canonical_signature(method_signature(left))
        target = node_object_to_signature.get(right) or canonical_signature(method_signature(right))
        if not source or not target:
            continue
        edge_type, icc_any, reflection_any, available, edge_source = edge_descriptor(
            data, target, edge_rules
        )
        edge_flag_metadata_seen = edge_flag_metadata_seen or available
        semantic_edge_rule_seen = semantic_edge_rule_seen or edge_source == "SEMANTIC_RULE"
        edge_source_counts[edge_source] += 1
        key = (source, target, edge_type, icc_any, reflection_any)
        edge_map[key] = {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "icc_any": icc_any,
            "reflection_any": reflection_any,
        }

    adjacency = defaultdict(list)
    role_edge_counts = Counter()
    for edge in edge_map.values():
        adjacency[edge["source"]].append(edge)
        roles = node_records.get(edge["source"], {}).get("roles", set()) | node_records.get(
            edge["target"], {}
        ).get("roles", set())
        for role in roles:
            role_edge_counts[role] += 1
    for source in adjacency:
        adjacency[source].sort(
            key=lambda item: (
                item["target"],
                item["edge_type"],
                -int(item["icc_any"]),
                -int(item["reflection_any"]),
            )
        )

    edge_type_counts = Counter(edge["edge_type"] for edge in edge_map.values())

    for record in node_records.values():
        record["roles"] = sorted(record["roles"])
        record["tier"] = min(
            (role_tier.get(role, 4) for role in record["roles"]), default=4
        )

    entry_info = bind_manifest_entries(apk, node_records)
    context_roles = manifest_context_role_counts(apk, rules)
    sensitive_targets = {
        signature for signature, record in node_records.items() if record["roles"]
    }
    path_summary = enumerate_sensitive_paths(
        adjacency,
        entry_info["entry_signatures"],
        sensitive_targets,
        policy,
        include_path_nodes=include_internal,
    )
    external_count = sum(1 for item in node_records.values() if item["external"])
    mapped_count = sum(1 for item in node_records.values() if item["roles"])
    role_node_counts = Counter(
        role for item in node_records.values() for role in item["roles"]
    )
    summary = {
        "raw_graph_node_count": len(raw_nodes),
        "raw_graph_edge_count": raw_edge_count,
        "canonical_node_count": len(node_records),
        "canonical_edge_count": len(edge_map),
        "internal_node_count": len(node_records) - external_count,
        "external_node_count": external_count,
        "mapped_api_node_count": mapped_count,
        "unmapped_external_node_count": max(0, external_count - mapped_count),
        "sensitive_node_count": mapped_count,
        "role_node_counts": dict(sorted(role_node_counts.items())),
        "role_edge_counts": dict(sorted(role_edge_counts.items())),
        "manifest_context_role_counts": dict(sorted(context_roles.items())),
        "manifest_context_match_count": sum(context_roles.values()),
        "entry_binding": {
            key: value
            for key, value in entry_info.items()
            if key != "entry_signatures"
        },
        "edge_flag_status": (
            "PROVIDED_BY_CALL_GRAPH"
            if edge_flag_metadata_seen and not semantic_edge_rule_seen
            else "PARTIAL_BY_SEMANTIC_RULES"
            if semantic_edge_rule_seen
            else "NOT_AVAILABLE_FROM_CALL_GRAPH"
        ),
        "edge_source_counts": dict(sorted(edge_source_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "path_policy": {
            "H": policy.H,
            "K_out": policy.K_out,
            "P_cap": policy.P_cap,
            "E_cap": policy.E_cap,
            "max_states": policy.max_states,
        },
        "path_summary": path_summary,
        "persistence_policy": "IN_MEMORY_GRAPH_ONLY_COMPACT_SUMMARY_OUTPUT",
    }
    if include_internal:
        summary["_internal"] = {
            "node_records": node_records,
            "edge_records": list(edge_map.values()),
            "entry_signatures": entry_info["entry_signatures"],
            "path_node_signatures": set(path_summary.pop("path_node_signatures", [])),
        }
    return summary
