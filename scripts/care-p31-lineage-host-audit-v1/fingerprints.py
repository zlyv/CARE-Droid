#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any

from common import norm, sketch

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+", re.I)


def safe(callable_object, default=None):
    try:
        return callable_object()
    except Exception:
        return default


def certificate_hashes(apk) -> list[str]:
    result = set()
    for getter_name in ("get_certificates_der_v3", "get_certificates_der_v2", "get_certificates_der_v1"):
        getter = getattr(apk, getter_name, None)
        if not callable(getter):
            continue
        for value in safe(getter, []) or []:
            try:
                result.add(hashlib.sha256(bytes(value)).hexdigest())
            except Exception:
                pass
    return sorted(result)


def manifest_tokens(apk) -> list[str]:
    result = []
    for kind, getter_name in (("activity", "get_activities"), ("service", "get_services"),
                              ("receiver", "get_receivers"), ("provider", "get_providers")):
        names = safe(getattr(apk, getter_name, lambda: []), []) or []
        result.append(f"component_count:{kind}:{min(len(names), 32)}")
        for name in names:
            filters = safe(lambda k=kind, n=name: apk.get_intent_filters(k, n), {}) or {}
            for key in ("action", "category"):
                for value in filters.get(key, []) or []:
                    result.append(f"{kind}:{key}:{norm(value)}")
            for data in filters.get("data", []) or []:
                if isinstance(data, dict):
                    for key in ("scheme", "mimeType", "host", "port", "path", "pathPrefix", "pathPattern"):
                        if data.get(key):
                            result.append(f"{kind}:data:{key}:{norm(data[key])}")
    for permission in safe(apk.get_permissions, []) or []:
        result.append("permission:" + norm(permission))
    return result


def instruction_records(method):
    getter = getattr(method, "get_instructions_idx", None)
    if callable(getter):
        iterator = getter()
    else:
        iterator = enumerate(safe(getattr(method, "get_instructions", lambda: ()), ()) or ())
    for _, instruction in iterator:
        name = norm(safe(getattr(instruction, "get_name", lambda: ""), ""))
        output = norm(safe(getattr(instruction, "get_output", lambda: ""), ""))
        yield name, output


def fingerprint_apk(apk_path: Path, sid: str) -> dict[str, Any]:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX

    apk = APK(str(apk_path))
    package = norm(safe(apk.get_package, ""))
    dex_blobs = list(apk.get_all_dex())
    dex_hashes = sorted(hashlib.sha256(bytes(blob)).hexdigest() for blob in dex_blobs)
    class_names = set()
    method_records = []
    class_tokens = []
    for raw_dex in dex_blobs:
        dex = DEX(raw_dex)
        for class_object in dex.get_classes():
            class_name = norm(class_object.get_name())
            class_names.add(class_name)
            super_name = norm(safe(class_object.get_superclassname, ""))
            class_tokens.append(f"{class_name}|{super_name}")
            for method in class_object.get_methods():
                name = norm(method.get_name())
                descriptor = norm(method.get_descriptor())
                signature = f"{class_name}->{name}{descriptor}"
                opcodes, refs = [], []
                if safe(getattr(method, "get_code", lambda: None)) is not None:
                    for opcode, output in instruction_records(method):
                        opcodes.append(opcode)
                        if opcode.startswith("invoke-"):
                            matches = METHOD_REFERENCE.findall(output)
                            if matches:
                                refs.append(norm(matches[-1]))
                method_records.append((signature, descriptor, opcodes, refs))
        dex = None
    method_exact = [record[0] for record in method_records]
    method_shape = [record[1] + "|" + ",".join(record[2]) for record in method_records]
    internal_edges, external_refs = [], []
    for signature, _, _, refs in method_records:
        for reference in refs:
            target_class = reference.split(";->", 1)[0] + ";" if ";->" in reference else ""
            if target_class in class_names:
                internal_edges.append(signature + "=>" + reference)
            else:
                external_refs.append(reference)
    with zipfile.ZipFile(apk_path, "r") as archive:
        zip_names = sorted(name.casefold() for name in archive.namelist())
    result = {
        "schema_version": "1.0.0", "sha256": sid, "status": "OK",
        "package_sha256": hashlib.sha256(package.encode("utf-8")).hexdigest() if package else None,
        "certificate_sha256": certificate_hashes(apk), "dex_sha256": dex_hashes,
        "apk_size": apk_path.stat().st_size, "dex_count": len(dex_blobs),
        "class_count": len(class_names), "method_count": len(method_records),
        "sketches": {
            "manifest": sketch(manifest_tokens(apk)),
            "classes": sketch(class_tokens),
            "method_exact": sketch(method_exact),
            "method_shape": sketch(method_shape),
            "internal_edges": sketch(internal_edges),
            "external_api": sketch(external_refs),
            "zip_layout": sketch(zip_names),
        },
    }
    return result


def host_direct_invokes(apk_path: Path, wanted_signatures: set[str]) -> dict[str, list[str]]:
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    wanted = {norm(value) for value in wanted_signatures}
    result = {value: [] for value in wanted}
    apk = APK(str(apk_path))
    for raw_dex in apk.get_all_dex():
        dex = DEX(raw_dex)
        for class_object in dex.get_classes():
            class_name = norm(class_object.get_name())
            for method in class_object.get_methods():
                signature = f"{class_name}->{norm(method.get_name())}{norm(method.get_descriptor())}"
                if signature not in wanted:
                    continue
                references = []
                for opcode, output in instruction_records(method):
                    if opcode.startswith("invoke-"):
                        matches = METHOD_REFERENCE.findall(output)
                        if matches:
                            references.append(norm(matches[-1]))
                result[signature] = sorted(set(references))
        dex = None
    return result
