#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from common import canonical
from taxonomy import classify

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^^\s(]+\([^)]*\)[^\s,}\]]+", re.I)


def safe_call(callable_object, default=None):
    try:
        return callable_object()
    except Exception:
        return default


def type_closure(name: str, classes: dict[str, dict[str, Any]], memo: dict[str, set[str]], stack=None) -> set[str]:
    if name in memo:
        return memo[name]
    stack = set() if stack is None else set(stack)
    if name in stack:
        return {name}
    stack.add(name)
    result = {name}
    record = classes.get(name)
    if record:
        parents = [record.get("super", "")] + list(record.get("interfaces", []))
        for parent in parents:
            parent = canonical(parent)
            if parent:
                result.add(parent)
                if parent in classes:
                    result.update(type_closure(parent, classes, memo, stack))
    memo[name] = result
    return result


def identify_roots(graph: dict[str, Any]) -> list[dict[str, Any]]:
    classes = {canonical(key): value for key, value in graph["classes"].items()}
    memo: dict[str, set[str]] = {}
    roots = []
    for index, method in enumerate(graph["methods"]):
        rule = classify(type_closure(canonical(method["class"]), classes, memo), method["name"], method["descriptor"])
        if rule:
            roots.append({"node": index, "category": rule["category"], "family": rule["family"]})
    return roots


def multi_source_distances(adjacency: dict[int, list[int]], sources: set[int]) -> dict[int, int]:
    distance = {int(source): 0 for source in sources}
    queue = deque(sorted(distance))
    while queue:
        current = queue.popleft()
        for target in adjacency.get(current, []):
            if target not in distance:
                distance[target] = distance[current] + 1
                queue.append(target)
    return distance


def graph_context(graph: dict[str, Any]) -> dict[str, Any]:
    forward: dict[int, list[int]] = defaultdict(list)
    reverse: dict[int, list[int]] = defaultdict(list)
    for source, target in graph["edges"]:
        source, target = int(source), int(target)
        forward[source].append(target)
        reverse[target].append(source)
    roots = identify_roots(graph)
    user = {row["node"] for row in roots if row["category"].startswith("USER_")}
    auto = {row["node"] for row in roots if row["category"].startswith("AUTONOMOUS_")}
    neutral = {row["node"] for row in roots if row["category"].startswith("NEUTRAL_")}
    return {
        "forward": dict(forward), "reverse": dict(reverse), "roots": roots,
        "user_distance": multi_source_distances(forward, user),
        "auto_distance": multi_source_distances(forward, auto),
        "neutral_distance": multi_source_distances(forward, neutral),
    }


def root_category(node: int, ctx: dict[str, Any]) -> tuple[str, int | None, int | None]:
    user = ctx["user_distance"].get(node)
    auto = ctx["auto_distance"].get(node)
    if user is not None and auto is None:
        return "USER_ONLY", user, None
    if auto is not None and user is None:
        return "AUTONOMOUS_ONLY", None, auto
    if user is not None and auto is not None:
        return "BOTH", user, auto
    return "UNREACHABLE_OR_NEUTRAL", None, None


def nearest_boundaries(host: int, ownership: list[str], reverse: dict[int, list[int]], max_depth: int = 64) -> list[dict[str, int]]:
    if ownership[host] != "THIRD_PARTY":
        return []
    queue = deque([(host, 0)])
    visited = {host}
    found = {}
    while queue:
        current, distance = queue.popleft()
        if distance >= max_depth:
            continue
        for parent in reverse.get(current, []):
            if ownership[parent] == "FIRST_PARTY" and ownership[current] == "THIRD_PARTY":
                found[(parent, current)] = min(found.get((parent, current), 10**9), distance + 1)
                continue
            if ownership[parent] == "THIRD_PARTY" and parent not in visited:
                visited.add(parent)
                queue.append((parent, distance + 1))
    return [{"source": source, "target": target, "distance": distance} for (source, target), distance in sorted(found.items(), key=lambda item: (item[1], item[0]))]


def instruction_context(apk_path: Path, source_signatures: set[str]) -> dict[str, dict[str, Any]]:
    if not source_signatures:
        return {}
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    result = {}
    apk = APK(str(apk_path))
    for raw_dex in apk.get_all_dex():
        dex = DEX(raw_dex)
        for clazz in dex.get_classes():
            class_name = canonical(clazz.get_name())
            for method in clazz.get_methods():
                signature = canonical(f"{class_name}->{method.get_name()}{method.get_descriptor()}")
                if signature not in source_signatures:
                    continue
                branch_count = 0
                switch_count = 0
                invoke_count = 0
                instruction_count = 0
                code = safe_call(method.get_code)
                if code is not None:
                    for instruction in safe_call(method.get_instructions, ()) or ():
                        instruction_count += 1
                        name = str(safe_call(instruction.get_name, "") or "").casefold()
                        if name.startswith("if-"):
                            branch_count += 1
                        elif "switch" in name:
                            switch_count += 1
                        elif name.startswith("invoke-"):
                            invoke_count += 1
                result[signature] = {
                    "method_code_available": int(code is not None),
                    "instruction_count": instruction_count,
                    "branch_count": branch_count,
                    "switch_count": switch_count,
                    "invoke_count": invoke_count,
                    "branch_context_proxy": int(branch_count + switch_count > 0),
                    "exact_control_dependency": 0,
                }
    return result


def resource_callbacks(apk_path: Path) -> dict[str, Any]:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    apk = APK(str(apk_path))
    callbacks = set()
    layout_files = [name for name in apk.get_files() if name.startswith("res/layout") and name.endswith(".xml")]
    parsed = 0
    try:
        from androguard.core.axml import AXMLPrinter
        for name in layout_files:
            try:
                root = AXMLPrinter(apk.get_file(name)).get_xml_obj()
                parsed += 1
                for element in root.iter():
                    for key, value in element.attrib.items():
                        if str(key).endswith("}onClick") or str(key) == "android:onClick":
                            if str(value).strip():
                                callbacks.add(canonical(value))
            except Exception:
                continue
    except Exception:
        pass
    return {"layout_files": len(layout_files), "parsed_layout_files": parsed, "callback_names": callbacks}
