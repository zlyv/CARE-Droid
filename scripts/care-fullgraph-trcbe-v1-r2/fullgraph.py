#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from common import canonical, stable_id
from taxonomy import classify

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+", re.I)


def safe_call(callable_object, default=None):
    try:
        return callable_object()
    except Exception:
        return default


def interfaces(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return [canonical(item) for item in re.findall(r"L[^;]+;", str(value or ""))]


def instruction_records(method) -> list[tuple[int, str, str]]:
    getter = getattr(method, "get_instructions_idx", None)
    if callable(getter):
        iterator = getter()
    else:
        offset = 0
        fallback = []
        for instruction in safe_call(getattr(method, "get_instructions", lambda: ()), ()) or ():
            fallback.append((offset, instruction))
            offset += int(safe_call(getattr(instruction, "get_length", lambda: 0), 0) or 0)
        iterator = fallback
    result = []
    for offset, instruction in iterator:
        name = str(safe_call(getattr(instruction, "get_name", lambda: ""), "") or "")
        output = str(safe_call(getattr(instruction, "get_output", lambda: ""), "") or "")
        result.append((int(offset), name, output))
    return result


def parse_method_reference(output: str) -> str | None:
    matches = METHOD_REFERENCE.findall(canonical(output))
    return canonical(matches[-1]) if matches else None


def split_reference(reference: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(L[^;]+;)->([^\s(]+)(\([^)]*\).+)", canonical(reference), re.I)
    return match.groups() if match else None


def type_closure(name: str, classes: dict[str, dict[str, Any]], memo: dict[str, set[str]], stack: set[str] | None = None) -> set[str]:
    if name in memo:
        return memo[name]
    stack = set() if stack is None else set(stack)
    if name in stack:
        return {name}
    stack.add(name)
    result = {name}
    record = classes.get(name)
    if record:
        for parent in [record.get("super", "")] + list(record.get("interfaces", [])):
            if parent:
                result.add(parent)
                if parent in classes:
                    result.update(type_closure(parent, classes, memo, stack))
    memo[name] = result
    return result


def build_full_graph(apk_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX

    apk = APK(str(apk_path))
    classes: dict[str, dict[str, Any]] = {}
    raw_methods: dict[str, dict[str, Any]] = {}
    dex_count = 0
    instruction_count = 0
    invoke_count = 0
    parsed_invoke_count = 0
    duplicate_methods = 0
    for raw_dex in apk.get_all_dex():
        dex_count += 1
        dex = DEX(raw_dex)
        for class_object in dex.get_classes():
            class_name = canonical(class_object.get_name())
            classes[class_name] = {
                "name": class_name,
                "super": canonical(class_object.get_superclassname()),
                "interfaces": interfaces(class_object.get_interfaces()),
            }
            for method in class_object.get_methods():
                method_name = canonical(method.get_name())
                descriptor = canonical(method.get_descriptor())
                signature = canonical(f"{class_name}->{method_name}{descriptor}")
                if signature in raw_methods:
                    duplicate_methods += 1
                    continue
                invokes = []
                code_present = safe_call(getattr(method, "get_code", lambda: None)) is not None
                if code_present:
                    for offset, instruction_name, instruction_output in instruction_records(method):
                        instruction_count += 1
                        if instruction_name.casefold().startswith("invoke-"):
                            invoke_count += 1
                            reference = parse_method_reference(instruction_output)
                            if reference:
                                parsed_invoke_count += 1
                                invokes.append({"offset": offset, "opcode": instruction_name.casefold(), "reference": reference})
                raw_methods[signature] = {
                    "class": class_name,
                    "name": method_name,
                    "descriptor": descriptor,
                    "signature": signature,
                    "node_id": stable_id(signature),
                    "code_present": bool(code_present),
                    "invokes": invokes,
                }
        dex = None

    signatures = sorted(raw_methods)
    index_by_signature = {signature: index for index, signature in enumerate(signatures)}
    methods = [{key: value for key, value in raw_methods[signature].items() if key != "invokes"} for signature in signatures]
    class_methods: dict[tuple[str, str, str], int] = {}
    for index, method in enumerate(methods):
        class_methods[(method["class"], method["name"], method["descriptor"])] = index
    closure_memo: dict[str, set[str]] = {}
    implementations: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, method in enumerate(methods):
        for parent in type_closure(method["class"], classes, closure_memo):
            implementations[(parent, method["name"], method["descriptor"])].append(index)

    edges: set[tuple[int, int]] = set()
    exact_edge_count = 0
    cha_edge_count = 0
    external_invoke_count = 0
    unresolved_app_invoke_count = 0
    for caller_signature in signatures:
        caller_index = index_by_signature[caller_signature]
        for invoke in raw_methods[caller_signature]["invokes"]:
            target = split_reference(invoke["reference"])
            if not target:
                continue
            target_class, target_name, target_descriptor = target
            exact = class_methods.get((target_class, target_name, target_descriptor))
            added = False
            if exact is not None:
                if (caller_index, exact) not in edges:
                    exact_edge_count += 1
                edges.add((caller_index, exact))
                added = True
            if invoke["opcode"].startswith(("invoke-virtual", "invoke-interface")):
                for implementation in implementations.get((target_class, target_name, target_descriptor), []):
                    if (caller_index, implementation) not in edges:
                        cha_edge_count += 1
                    edges.add((caller_index, implementation))
                    added = True
            if not added:
                if target_class in classes:
                    unresolved_app_invoke_count += 1
                else:
                    external_invoke_count += 1

    edge_list = sorted(edges)
    outgoing = {index: 0 for index in range(len(methods))}
    incoming = {index: 0 for index in range(len(methods))}
    for source, target in edge_list:
        outgoing[source] += 1
        incoming[target] += 1
    return {
        "schema_version": "1.0.0",
        "graph_kind": "FULL_APP_DEFINED_METHOD_GRAPH",
        "edge_direction": "caller_to_callee",
        "algorithm": "DEX direct-reference app-internal edges plus conservative app-class-hierarchy expansion for virtual/interface invokes",
        "limitations": [
            "reflection and dynamically loaded code are not resolved",
            "framework internals are excluded; app callback overrides are roots",
            "native calls are excluded",
            "virtual dispatch is a conservative app-local class-hierarchy over-approximation",
        ],
        "dex_count": dex_count,
        "classes": classes,
        "methods": methods,
        "edges": edge_list,
        "stats": {
            "class_count": len(classes),
            "method_count": len(methods),
            "code_method_count": sum(method["code_present"] for method in methods),
            "instruction_count": instruction_count,
            "invoke_count": invoke_count,
            "parsed_invoke_count": parsed_invoke_count,
            "invoke_parse_rate": parsed_invoke_count / max(1, invoke_count),
            "edge_count": len(edge_list),
            "exact_edge_count": exact_edge_count,
            "cha_edge_count": cha_edge_count,
            "external_invoke_count": external_invoke_count,
            "unresolved_app_invoke_count": unresolved_app_invoke_count,
            "duplicate_method_count": duplicate_methods,
            "isolated_method_count": sum(outgoing[index] == 0 and incoming[index] == 0 for index in range(len(methods))),
            "wall_seconds": time.monotonic() - started,
        },
    }


def identify_roots(graph: dict[str, Any]) -> list[dict[str, Any]]:
    classes = graph["classes"]
    memo: dict[str, set[str]] = {}
    roots = []
    for node_index, method in enumerate(graph["methods"]):
        rule = classify(type_closure(method["class"], classes, memo), method["name"], method["descriptor"])
        if rule:
            roots.append({
                "node_index": node_index,
                "signature": method["signature"],
                "node_id": int(method["node_id"]),
                "family": rule["family"],
                "category": rule["category"],
                "confidence": rule["confidence"],
            })
    return roots


def adjacency(graph: dict[str, Any]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for source, target in graph["edges"]:
        result[int(source)].append(int(target))
    return dict(result)


def multi_source_distances(graph_adj: dict[int, list[int]], sources: set[int]) -> dict[int, int]:
    distances = {int(source): 0 for source in sources}
    queue = deque(sorted(distances))
    while queue:
        current = queue.popleft()
        for target in graph_adj.get(current, []):
            if target not in distances:
                distances[target] = distances[current] + 1
                queue.append(target)
    return distances


def shortest_path(graph_adj: dict[int, list[int]], sources: set[int], target: int) -> list[int]:
    previous = {int(source): None for source in sources}
    queue = deque(sorted(previous))
    while queue:
        current = queue.popleft()
        if current == target:
            break
        for child in graph_adj.get(current, []):
            if child not in previous:
                previous[child] = current
                queue.append(child)
    if target not in previous:
        return []
    path = []
    current: int | None = target
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def classify_host(host_index: int, user: dict[int, int], autonomous: dict[int, int], neutral: dict[int, int]) -> tuple[str, int | None, int | None, int | None]:
    user_distance = user.get(host_index)
    autonomous_distance = autonomous.get(host_index)
    neutral_distance = neutral.get(host_index)
    if user_distance is not None and autonomous_distance is None:
        return "USER_ONLY", user_distance, None, neutral_distance
    if autonomous_distance is not None and user_distance is None:
        return "AUTONOMOUS_ONLY", None, autonomous_distance, neutral_distance
    if user_distance is not None and autonomous_distance is not None:
        return "BOTH", user_distance, autonomous_distance, neutral_distance
    if neutral_distance is not None:
        return "NEUTRAL_ONLY", None, None, neutral_distance
    return "UNREACHABLE", None, None, None
