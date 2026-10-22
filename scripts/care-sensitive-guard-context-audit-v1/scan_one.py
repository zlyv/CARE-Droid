#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from common import canonical, dump_json, stable_id, write_gzip_json
from guard_core import classify_method, match_role, parse_method_reference


def safe_call(callable_object, default=None):
    try:
        return callable_object()
    except Exception:
        return default


def instruction_records(method) -> list[dict[str, Any]]:
    getter = getattr(method, "get_instructions_idx", None)
    if callable(getter):
        iterator = getter()
    else:
        offset, fallback = 0, []
        for instruction in safe_call(getattr(method, "get_instructions", lambda: ()), ()) or ():
            fallback.append((offset, instruction))
            offset += int(safe_call(getattr(instruction, "get_length", lambda: 0), 0) or 0)
        iterator = fallback
    result = []
    for offset, instruction in iterator:
        result.append({"offset": int(offset),
                       "name": str(safe_call(getattr(instruction, "get_name", lambda: ""), "") or "").casefold(),
                       "output": str(safe_call(getattr(instruction, "get_output", lambda: ""), "") or "")})
    return result


def scan(apk_path: Path, sid: str, mapping_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    try:
        from loguru import logger
        logger.remove()
    except Exception:
        pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX

    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    apk = APK(str(apk_path))
    events = []
    method_count = instruction_count = invoke_count = dex_count = 0
    context_counts = Counter()
    role_counts = Counter()
    for raw_dex in apk.get_all_dex():
        dex_count += 1
        dex = DEX(raw_dex)
        for class_object in dex.get_classes():
            class_name = canonical(class_object.get_name())
            for method in class_object.get_methods():
                method_count += 1
                method_name = canonical(method.get_name())
                descriptor = canonical(method.get_descriptor())
                signature = canonical(f"{class_name}->{method_name}{descriptor}")
                instructions = instruction_records(method)
                instruction_count += len(instructions)
                references = []
                sensitive = []
                for index, row in enumerate(instructions):
                    if not row["name"].startswith("invoke-"):
                        continue
                    invoke_count += 1
                    reference = parse_method_reference(row["output"])
                    if not reference:
                        continue
                    references.append(reference)
                    role = match_role(reference, mapping)
                    if role:
                        sensitive.append((index, row, reference, role))
                for index, row, reference, role in sensitive:
                    condition = classify_method(method_name, instructions, index, references)
                    event = {
                        "host_signature": signature,
                        "host_node_id": stable_id(signature),
                        "host_method_name": method_name,
                        "invoke_offset": int(row["offset"]),
                        "invoke_opcode": row["name"],
                        "sensitive_reference": reference,
                        **role,
                        **condition,
                    }
                    events.append(event)
                    context_counts[event["context"]] += 1
                    role_counts[event["role"]] += 1
        dex = None
    return {
        "schema_version": "1.0.0", "sha256": sid, "apk_path": str(apk_path.resolve()),
        "status": "OK", "labels_used": False, "future_rows_loaded": 0,
        "dex_count": dex_count, "method_count": method_count, "instruction_count": instruction_count,
        "invoke_count": invoke_count, "sensitive_event_count": len(events),
        "condition_bearing_event_count": sum(bool(row["condition_bearing"]) for row in events),
        "directional_callback_event_count": sum(bool(row["directional_callback"]) for row in events),
        "context_counts": dict(context_counts), "role_counts": dict(role_counts),
        "events": events, "wall_seconds": time.monotonic() - started,
        "limitations": [
            "callback names are framework-stable static context, not proof that a runtime event occurred",
            "co-located permission, consent, scheduling and branch facts are not claimed as control dependence",
            "reflection, native code and dynamically loaded code are outside this observation",
            "exact API identities are retained for audit only; the candidate representation uses Role31 and context categories",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    output, status_path = Path(args.output), Path(args.status)
    started = time.monotonic()
    try:
        result = scan(Path(args.apk), args.sha256.casefold(), Path(args.mapping))
        write_gzip_json(output, result)
        status = {"sha256": args.sha256.casefold(), "status": "OK", "output": str(output.resolve()),
                  "sensitive_event_count": result["sensitive_event_count"],
                  "condition_bearing_event_count": result["condition_bearing_event_count"],
                  "directional_callback_event_count": result["directional_callback_event_count"],
                  "wall_seconds": result["wall_seconds"]}
    except Exception as exc:
        status = {"sha256": args.sha256.casefold(), "status": "ERROR", "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc(),
                  "wall_seconds": time.monotonic() - started}
    dump_json(status_path, status)
    print("SGCA_SCAN_ONE " + json.dumps(status, sort_keys=True), flush=True)
    if status["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
