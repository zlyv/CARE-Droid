#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import canonical, stable_hash


def disable_androguard_logging() -> None:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass


def descriptor(package: str, name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    if raw.startswith("L") and raw.endswith(";"):
        return canonical(raw)
    if raw.startswith("."):
        raw = package + raw
    elif "." not in raw:
        raw = package + "." + raw
    return canonical("L" + raw.replace(".", "/") + ";")


def namespace_of_class(class_name: str) -> str:
    value = canonical(class_name)
    if value.startswith("l") and value.endswith(";"):
        value = value[1:-1]
    return value.rsplit("/", 1)[0] if "/" in value else value


def manifest_metadata(apk_path: Path) -> dict[str, Any]:
    disable_androguard_logging()
    from androguard.core.apk import APK
    apk = APK(str(apk_path))
    package = canonical(apk.get_package()).replace(".", "/")
    components = []
    for kind, getter in (
        ("activity", apk.get_activities), ("service", apk.get_services),
        ("receiver", apk.get_receivers), ("provider", apk.get_providers),
    ):
        for name in getter() or []:
            value = descriptor(str(apk.get_package() or ""), str(name))
            if value:
                components.append({"kind": kind, "descriptor": value})
    component_namespaces = sorted({namespace_of_class(row["descriptor"]) for row in components})
    return {
        "package_hash": stable_hash(package),
        "package_namespace": package,
        "component_descriptors": {row["descriptor"] for row in components},
        "component_namespaces": component_namespaces,
        "component_count": len(components),
    }


def class_method_groups(graph: dict[str, Any]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, method in enumerate(graph["methods"]):
        grouped[canonical(method["class"])].append((index, method))
    return grouped


def class_fingerprint(class_name: str, methods: list[tuple[int, dict[str, Any]]], graph: dict[str, Any]) -> tuple[str, str]:
    degree_cache = graph.get("_obcbc_degree_cache")
    if degree_cache is None:
        indegree = Counter()
        outdegree = Counter()
        for source, target in graph["edges"]:
            outdegree[int(source)] += 1
            indegree[int(target)] += 1
        degree_cache = (indegree, outdegree)
        graph["_obcbc_degree_cache"] = degree_cache
    indegree, outdegree = degree_cache
    semantic = sorted(f"{canonical(method['name'])}{canonical(method['descriptor'])}:{int(bool(method.get('code_present')))}" for _, method in methods)
    shape = sorted(f"{canonical(method['descriptor'])}:{min(9,outdegree[index])}:{min(9,indegree[index])}" for index, method in methods)
    record = graph.get("classes", {}).get(class_name, {})
    framework_parent = canonical(record.get("super", ""))
    if not framework_parent.startswith(("landroid/", "ljava/", "ljavax/", "lkotlin/")):
        framework_parent = "APP_PARENT"
    structure = hashlib.sha256(json.dumps([framework_parent, semantic], separators=(",", ":")).encode()).hexdigest()
    shape_hash = hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()
    return structure, shape_hash


@dataclass
class CorpusStats:
    structure_count: Counter
    shape_count: Counter
    class_name_count: Counter
    class_count: int


def build_corpus_stats(graph_paths: dict[str, Path], read_graph) -> CorpusStats:
    structure_count: Counter = Counter()
    shape_count: Counter = Counter()
    class_name_count: Counter = Counter()
    class_count = 0
    for number, (sid, path) in enumerate(sorted(graph_paths.items()), 1):
        graph = read_graph(path)
        grouped = class_method_groups(graph)
        seen_structure, seen_shape, seen_name = set(), set(), set()
        for class_name, methods in grouped.items():
            structure, shape = class_fingerprint(class_name, methods, graph)
            seen_structure.add(structure)
            seen_shape.add(shape)
            seen_name.add(stable_hash(class_name))
            class_count += 1
        structure_count.update(seen_structure)
        shape_count.update(seen_shape)
        class_name_count.update(seen_name)
        if number % 10 == 0 or number == len(graph_paths):
            print(f"OBCBC_CORPUS_FINGERPRINT {number}/{len(graph_paths)} classes={class_count}", flush=True)
    return CorpusStats(structure_count, shape_count, class_name_count, class_count)


def in_namespace(class_name: str, namespace: str) -> bool:
    raw = canonical(class_name)
    if raw.startswith("l") and raw.endswith(";"):
        raw = raw[1:-1]
    prefix = canonical(namespace).strip("/")
    return bool(prefix and (raw == prefix or raw.startswith(prefix + "/")))


def assign_graph(graph: dict[str, Any], manifest: dict[str, Any], corpus: CorpusStats) -> tuple[list[str], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped = class_method_groups(graph)
    class_records = []
    class_result: dict[str, dict[str, Any]] = {}
    package_ns = manifest["package_namespace"]
    component_set = manifest["component_descriptors"]
    component_namespaces = manifest["component_namespaces"]
    for class_name, methods in grouped.items():
        structure, shape = class_fingerprint(class_name, methods, graph)
        name_hash = stable_hash(class_name)
        method_count = len(methods)
        exact_component = class_name in component_set
        package_match = in_namespace(class_name, package_ns)
        component_namespace_match = any(in_namespace(class_name, prefix) for prefix in component_namespaces)
        structure_reuse = int(corpus.structure_count[structure])
        shape_reuse = int(corpus.shape_count[shape])
        name_reuse = int(corpus.class_name_count[name_hash])
        strong_reuse = method_count >= 4 and structure_reuse >= 3 and (name_reuse >= 3 or shape_reuse >= 5)
        app_specific = structure_reuse == 1 and name_reuse == 1
        first_evidence = []
        third_evidence = []
        if exact_component:
            first_evidence.append("MANIFEST_COMPONENT_EXACT")
        if (package_match or component_namespace_match) and app_specific and method_count >= 2:
            first_evidence.append("NAMESPACE_PLUS_APP_SPECIFIC_STRUCTURE")
        if strong_reuse and not package_match:
            third_evidence.append("CROSS_APK_STRUCTURE_REUSE_OUTSIDE_APP_NAMESPACE")
        if structure_reuse >= 5 and name_reuse >= 5 and method_count >= 3:
            third_evidence.append("CROSS_APK_NAME_AND_STRUCTURE_REUSE")
        if first_evidence and not third_evidence:
            ownership = "FIRST_PARTY"
        elif third_evidence and not first_evidence:
            ownership = "THIRD_PARTY"
        else:
            ownership = "UNKNOWN"
        record = {
            "class_id": name_hash,
            "ownership": ownership,
            "method_count": method_count,
            "manifest_component_exact": int(exact_component),
            "manifest_namespace_match": int(package_match),
            "component_namespace_match": int(component_namespace_match),
            "structure_reuse_apks": structure_reuse,
            "shape_reuse_apks": shape_reuse,
            "class_name_reuse_apks": name_reuse,
            "structure_id": structure,
            "shape_id": shape,
            "evidence": "|".join(first_evidence + third_evidence) or "INSUFFICIENT_OR_CONFLICTING",
            "raw_name_emitted": 0,
        }
        class_result[class_name] = record
        class_records.append(record)
    method_ownership = [class_result[canonical(method["class"])]["ownership"] for method in graph["methods"]]
    return method_ownership, class_records, class_result
