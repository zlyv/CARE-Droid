#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import canonical, dump_json, stable_id, write_gzip_json

try:
    import resource
except ImportError:  # The production server is Linux; keep local Windows self-tests importable.
    resource = None

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+", re.I)
SINK_TAGS = ("NETWORK", "FILE_WRITE", "PROCESS_EXEC", "DYNAMIC_LOAD", "SMS_CALL",
             "COMPONENT_STATE", "AUTH_ACCOUNT", "MEDIA_CAPTURE", "UI_OUTPUT", "IPC")


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
        offset, fallback = 0, []
        for instruction in safe_call(getattr(method, "get_instructions", lambda: ()), ()) or ():
            fallback.append((offset, instruction))
            offset += int(safe_call(getattr(instruction, "get_length", lambda: 0), 0) or 0)
        iterator = fallback
    output = []
    for offset, instruction in iterator:
        output.append((int(offset), str(safe_call(instruction.get_name, "") or ""),
                       str(safe_call(instruction.get_output, "") or "")))
    return output


def parse_reference(output: str) -> str | None:
    matches = METHOD_REFERENCE.findall(canonical(output))
    return canonical(matches[-1]) if matches else None


def split_reference(reference: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(L[^;]+;)->([^\s(]+)(\([^)]*\).+)", canonical(reference), re.I)
    return match.groups() if match else None


def classify_external(reference: str) -> list[str]:
    value = canonical(reference)
    tags = set()
    if any(token in value for token in ("ljava/net/", "ljavax/net/", "lokhttp", "lorg/apache/http/", "landroid/net/http/")):
        tags.add("NETWORK")
    if any(token in value for token in ("fileoutputstream;->", "randomaccessfile;->write", "outputstream;->write", "contentresolver;->insert", "contentresolver;->update")):
        tags.add("FILE_WRITE")
    if "runtime;->exec" in value or "processbuilder;->start" in value:
        tags.add("PROCESS_EXEC")
    if any(token in value for token in ("dexclassloader;", "pathclassloader;", "inmemorydexclassloader;", "classloader;->loadclass", "system;->load(", "system;->loadlibrary")):
        tags.add("DYNAMIC_LOAD")
    if any(token in value for token in ("smsmanager;->send", "telephonymanager;->dial", "intent;->setdata(landroid/net/uri")):
        tags.add("SMS_CALL")
    if "packagemanager;->setcomponentenabledsetting" in value:
        tags.add("COMPONENT_STATE")
    if "accountmanager;" in value:
        tags.add("AUTH_ACCOUNT")
    if any(token in value for token in ("audiorecord;", "mediarecorder;", "hardware/camera;", "camera2/")):
        tags.add("MEDIA_CAPTURE")
    if any(token in value for token in ("textview;->settext", "imageview;->setimage", "toast;->show", "mediaplayer;->start")):
        tags.add("UI_OUTPUT")
    if any(token in value for token in ("context;->sendbroadcast", "context;->startservice", "context;->bindservice", "contentresolver;->call")):
        tags.add("IPC")
    return sorted(tags)


def type_closure(name: str, classes: dict[str, dict[str, Any]], memo: dict[str, set[str]], stack=None) -> set[str]:
    if name in memo:
        return memo[name]
    stack = set() if stack is None else set(stack)
    if name in stack:
        return {name}
    stack.add(name); result = {name}; record = classes.get(name)
    if record:
        for parent in [record.get("super", "")] + list(record.get("interfaces", [])):
            if parent:
                result.add(parent)
                if parent in classes:
                    result.update(type_closure(parent, classes, memo, stack))
    memo[name] = result
    return result


def build(apk_path: Path, sha256: str) -> dict[str, Any]:
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
    raw: dict[str, dict[str, Any]] = {}
    stats = defaultdict(int)
    for raw_dex in apk.get_all_dex():
        stats["dex_count"] += 1
        dex = DEX(raw_dex)
        for cls in dex.get_classes():
            class_name = canonical(cls.get_name())
            classes[class_name] = {"name": class_name, "super": canonical(cls.get_superclassname()),
                                   "interfaces": interfaces(cls.get_interfaces())}
            for method in cls.get_methods():
                name, descriptor = canonical(method.get_name()), canonical(method.get_descriptor())
                signature = canonical(f"{class_name}->{name}{descriptor}")
                if signature in raw:
                    stats["duplicate_method_count"] += 1; continue
                invokes, external_tags = [], set()
                code = safe_call(getattr(method, "get_code", lambda: None)) is not None
                if code:
                    stats["code_method_count"] += 1
                    for offset, opcode, output in instruction_records(method):
                        stats["instruction_count"] += 1
                        if opcode.casefold().startswith("invoke-"):
                            stats["invoke_count"] += 1
                            reference = parse_reference(output)
                            if reference:
                                stats["parsed_invoke_count"] += 1
                                invokes.append({"offset": offset, "opcode": opcode.casefold(), "reference": reference})
                                external_tags.update(classify_external(reference))
                raw[signature] = {"class": class_name, "name": name, "descriptor": descriptor,
                                  "signature": signature, "node_id": stable_id(signature), "code_present": bool(code),
                                  "invokes": invokes, "external_tags": sorted(external_tags)}
        dex = None

    signatures = sorted(raw)
    index = {signature: number for number, signature in enumerate(signatures)}
    methods = [{key: value for key, value in raw[signature].items() if key != "invokes"} for signature in signatures]
    exact_index = {(m["class"], m["name"], m["descriptor"]): i for i, m in enumerate(methods)}
    memo: dict[str, set[str]] = {}
    implementations: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, method in enumerate(methods):
        for parent in type_closure(method["class"], classes, memo):
            implementations[(parent, method["name"], method["descriptor"])].append(i)

    edges: set[tuple[int, int, int, int]] = set()
    for signature in signatures:
        caller = index[signature]
        for invoke in raw[signature]["invokes"]:
            target = split_reference(invoke["reference"])
            if not target:
                continue
            target_class, target_name, target_desc = target
            candidates = []
            direct = exact_index.get((target_class, target_name, target_desc))
            if direct is not None:
                candidates.append(direct); stats["exact_edge_count"] += 1
            if invoke["opcode"].startswith(("invoke-virtual", "invoke-interface")):
                candidates.extend(implementations.get((target_class, target_name, target_desc), []))
            candidates = sorted(set(candidates))
            if not candidates:
                stats["external_or_unresolved_invoke_count"] += 1
            for callee in candidates:
                callsite = stable_id(f"{signature}@{invoke['offset']}->{invoke['reference']}")
                edges.add((caller, callee, int(invoke["offset"]), callsite))
    edge_rows = [list(value) for value in sorted(edges)]
    return {"schema_version": "2.0.0", "sha256": sha256, "graph_kind": "CALLSITE_AWARE_APP_METHOD_GRAPH",
            "edge_direction": "caller_to_callee", "classes": classes, "methods": methods, "edges": edge_rows,
            "sink_tags": list(SINK_TAGS), "limitations": ["reflection unresolved", "native code excluded",
            "dynamically loaded code excluded", "framework bodies excluded", "app-local CHA is conservative"],
            "stats": {**dict(stats), "class_count": len(classes), "method_count": len(methods),
                      "edge_count": len(edge_rows), "wall_seconds": time.monotonic() - started}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", type=Path, required=True); parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    try:
        graph = build(args.apk, args.sha256); write_gzip_json(args.output, graph)
        peak_rss = (int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    if resource is not None else None)
        status = {"sha256": args.sha256, "status": "OK", "wall_seconds": time.monotonic() - started,
                  "method_count": graph["stats"]["method_count"], "edge_count": graph["stats"]["edge_count"],
                  "peak_rss_kib": peak_rss}
    except Exception as exc:
        status = {"sha256": args.sha256, "status": "ERROR", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc(limit=20),
                  "wall_seconds": time.monotonic() - started}
    dump_json(args.status, status)
    print("CMW_BUILD_ONE " + str(status), flush=True)


if __name__ == "__main__":
    main()
