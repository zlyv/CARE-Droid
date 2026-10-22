#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

from common import dump_json, norm

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+", re.I)
SDK_PREFIXES = (
    "lcom/google/", "lcom/facebook/", "lcom/amazon/", "lcom/unity3d/", "lcom/flurry/",
    "lcom/adjust/", "lcom/appsflyer/", "lcom/mopub/", "lcom/startapp/", "lcom/ironsource/",
    "lcom/tencent/", "lcom/baidu/", "lcom/alibaba/", "lcom/umeng/", "lcom/bytedance/",
    "lcom/applovin/", "lcom/chartboost/", "lcom/adcolony/", "lcom/inmobi/",
)
PACKER_MARKERS = (
    "libjiagu", "libsecexe", "libsecmain", "libnqshield", "libprotectclass", "libshell",
    "ijiami", "bangcle", "secneo", "qihoo", "reinforce", "stubapp", "legu",
)
DYNAMIC_MARKERS = (
    "ldalvik/system/dexclassloader;", "ldalvik/system/pathclassloader;",
    "ldalvik/system/inmemorydexclassloader;", "ldalvik/system/dexfile;->loaddex(",
    "ljava/lang/classloader;->defineclass(",
)
REFLECTION_MARKERS = (
    "ljava/lang/class;->forname(", "ljava/lang/reflect/method;->invoke(",
    "ljava/lang/reflect/constructor;->newinstance(", "->getdeclaredmethod(",
    "->getdeclaredfield(", "ljava/lang/reflect/proxy;->newproxyinstance(",
)
NATIVE_LOAD_MARKERS = (
    "ljava/lang/system;->loadlibrary(", "ljava/lang/system;->load(",
    "ljava/lang/runtime;->loadlibrary(", "ljava/lang/runtime;->load(",
)
PAYLOAD_SUFFIXES = (".dex", ".jar", ".apk", ".odex", ".vdex", ".zip")


def safe_call(callable_object, default=None):
    try:
        return callable_object()
    except Exception:
        return default


def references(method) -> list[str]:
    result = []
    for instruction in safe_call(getattr(method, "get_instructions", lambda: ()), ()) or ():
        name = norm(safe_call(getattr(instruction, "get_name", lambda: ""), ""))
        if not name.startswith("invoke-"):
            continue
        output = norm(safe_call(getattr(instruction, "get_output", lambda: ""), ""))
        matches = METHOD_REFERENCE.findall(output)
        if matches:
            result.append(norm(matches[-1]))
    return result


def component_names(apk, package: str) -> list[str]:
    values = []
    for getter in ("get_activities", "get_services", "get_receivers", "get_providers"):
        for value in safe_call(getattr(apk, getter, lambda: []), []) or []:
            value = str(value or "")
            if value.startswith("."):
                value = package + value
            elif "." not in value and package:
                value = package + "." + value
            values.append(value)
    return values


def scan(apk_path: Path, sid: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX

    with zipfile.ZipFile(apk_path) as archive:
        infos = archive.infolist()
        names = [info.filename.casefold() for info in infos]
        so_entries = [info for info in infos if info.filename.casefold().endswith(".so")]
        payload_entries = [info for info in infos if info.filename.casefold().endswith(PAYLOAD_SUFFIXES)
                           and not re.fullmatch(r"classes\d*\.dex", Path(info.filename).name.casefold())]
        packer_entry_hits = sorted({marker for name in names for marker in PACKER_MARKERS if marker in name})
        total_uncompressed = sum(info.file_size for info in infos)
        payload_bytes = sum(info.file_size for info in payload_entries)

    apk = APK(str(apk_path)); package = norm(apk.get_package()).replace(".", "/")
    package_descriptor = "l" + package + "/" if package else ""
    class_names = set(); app_classes = set(); sdk_classes = set(); method_names = []
    method_count = native_methods = abstract_methods = instruction_count = 0
    reflection_calls = dynamic_calls = native_load_calls = 0
    dex_count = 0
    for raw_dex in apk.get_all_dex():
        dex_count += 1; dex = DEX(raw_dex)
        for cls in dex.get_classes():
            class_name = norm(cls.get_name()); class_names.add(class_name)
            if package_descriptor and class_name.startswith(package_descriptor):
                app_classes.add(class_name)
            if any(class_name.startswith(prefix) for prefix in SDK_PREFIXES):
                sdk_classes.add(class_name)
            for method in cls.get_methods():
                method_count += 1; name = norm(method.get_name()); method_names.append(name)
                flags = norm(safe_call(getattr(method, "get_access_flags_string", lambda: ""), ""))
                native_methods += int("native" in flags)
                abstract_methods += int("abstract" in flags)
                refs = references(method); instruction_count += len(list(safe_call(getattr(method, "get_instructions", lambda: ()), ()) or ()))
                reflection_calls += sum(any(marker in ref for marker in REFLECTION_MARKERS) for ref in refs)
                dynamic_calls += sum(any(marker in ref for marker in DYNAMIC_MARKERS) for ref in refs)
                native_load_calls += sum(any(marker in ref for marker in NATIVE_LOAD_MARKERS) for ref in refs)
        dex = None

    components = component_names(apk, package.replace("/", "."))
    component_descriptors = {"l" + name.replace(".", "/").casefold() + ";" for name in components}
    mapped_components = sum(descriptor in class_names for descriptor in component_descriptors)
    identifiers = [Path(name.strip("l;")).name for name in app_classes] + [name for name in method_names if name not in {"<init>", "<clinit>"}]
    short_identifier_fraction = sum(len(value) <= 2 for value in identifiers) / max(len(identifiers), 1)
    non_ascii_identifier_fraction = sum(any(ord(ch) > 127 for ch in value) for value in identifiers) / max(len(identifiers), 1)
    class_count = len(class_names); app_class_count = len(app_classes); sdk_class_count = len(sdk_classes)
    known_packer = bool(packer_entry_hits or any(any(marker in name for marker in PACKER_MARKERS) for name in class_names))

    result = {
        "sha256": sid, "status": "OK", "apk_path": str(apk_path.resolve()),
        "apk_size_bytes": apk_path.stat().st_size, "zip_uncompressed_bytes": total_uncompressed,
        "dex_count": dex_count, "class_count": class_count, "method_count": method_count,
        "instruction_count": instruction_count, "native_method_count": native_methods,
        "native_method_fraction": native_methods / max(method_count, 1), "abstract_method_count": abstract_methods,
        "native_library_count": len(so_entries), "native_library_bytes": sum(info.file_size for info in so_entries),
        "reflection_call_count": reflection_calls, "reflection_call_rate": reflection_calls / max(method_count, 1),
        "dynamic_loading_call_count": dynamic_calls, "dynamic_loading_call_rate": dynamic_calls / max(method_count, 1),
        "native_load_call_count": native_load_calls,
        "embedded_payload_count": len(payload_entries), "embedded_payload_bytes": payload_bytes,
        "embedded_payload_fraction": payload_bytes / max(total_uncompressed, 1),
        "known_packer": int(known_packer), "packer_markers": "|".join(packer_entry_hits),
        "app_class_count": app_class_count, "app_class_fraction": app_class_count / max(class_count, 1),
        "known_sdk_class_count": sdk_class_count, "known_sdk_class_fraction": sdk_class_count / max(class_count, 1),
        "component_count": len(component_descriptors), "mapped_component_count": mapped_components,
        "component_mapping_rate": mapped_components / max(len(component_descriptors), 1),
        "short_identifier_fraction": short_identifier_fraction,
        "non_ascii_identifier_fraction": non_ascii_identifier_fraction,
        "has_native": int(native_methods > 0 or len(so_entries) > 0 or native_load_calls > 0),
        "has_reflection": int(reflection_calls > 0), "has_dynamic_loading": int(dynamic_calls > 0),
        "has_embedded_payload": int(len(payload_entries) > 0), "has_multidex": int(dex_count > 1),
        "high_sdk_dominance": int(sdk_class_count / max(class_count, 1) >= .50),
        "low_component_mapping": int(len(component_descriptors) > 0 and mapped_components / len(component_descriptors) < .50),
        "high_short_identifier": int(short_identifier_fraction >= .50),
        "visibility_gap_any": int(native_methods > 0 or len(so_entries) > 0 or native_load_calls > 0 or reflection_calls > 0 or dynamic_calls > 0 or len(payload_entries) > 0 or known_packer),
        "wall_seconds": time.monotonic() - started,
        "limitations": [
            "known SDK namespace coverage is a conservative heuristic, not library ownership ground truth",
            "reflection and dynamic-loading counts indicate static visibility gaps, not maliciousness",
            "short identifiers and packer markers are audit signals, not class labels",
            "embedded payload presence is not proof that the payload executes",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--apk", required=True); parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--status", required=True); args = parser.parse_args()
    output, status_path = Path(args.output), Path(args.status); started = time.monotonic()
    try:
        result = scan(Path(args.apk), norm(args.sha256)); dump_json(output, result)
        status = {"sha256": norm(args.sha256), "status": "OK", "output": str(output.resolve()),
                  "wall_seconds": result["wall_seconds"], "visibility_gap_any": result["visibility_gap_any"]}
    except Exception as exc:
        status = {"sha256": norm(args.sha256), "status": "ERROR", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started}
    dump_json(status_path, status); print("HCOA_SCAN_ONE " + json.dumps(status, sort_keys=True), flush=True)
    if status["status"] != "OK": raise SystemExit(1)


if __name__ == "__main__":
    main()
