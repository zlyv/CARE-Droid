#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

from common import canonical, dump_json, stable_id, write_gzip_json
from wrapper_core import analyze_forwarder, match_role, parse_method_reference


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
        result.append({
            "offset": int(offset),
            "name": str(safe_call(getattr(instruction, "get_name", lambda: ""), "") or "").casefold(),
            "output": str(safe_call(getattr(instruction, "get_output", lambda: ""), "") or ""),
        })
    return result


def code_sizes(method) -> tuple[int, int]:
    code = safe_call(getattr(method, "get_code", lambda: None), None)
    if code is None:
        return 0, 0
    registers = int(safe_call(getattr(code, "get_registers_size", lambda: 0), 0) or 0)
    ins_size = int(safe_call(getattr(code, "get_ins_size", lambda: 0), 0) or 0)
    return registers, ins_size


def access_string(method) -> str:
    value = safe_call(getattr(method, "get_access_flags_string", lambda: ""), "")
    if value:
        return str(value)
    flags = int(safe_call(getattr(method, "get_access_flags", lambda: 0), 0) or 0)
    names = []
    for bit, name in ((0x8, "static"), (0x20, "synchronized"), (0x100, "native"), (0x400, "abstract")):
        if flags & bit:
            names.append(name)
    return " ".join(names)


def effective_callers(signature: str, verified: dict[str, dict[str, Any]], callers: dict[str, set[str]]) -> list[str]:
    result = set()
    visited = {signature}
    queue = deque([(signature, 0)])
    while queue:
        current, depth = queue.popleft()
        for caller in sorted(callers.get(current, set())):
            if caller in visited:
                continue
            visited.add(caller)
            if caller in verified and depth < 3:
                queue.append((caller, depth + 1))
            else:
                result.add(caller)
    return sorted(result)


def scan(apk_path: Path, sid: str, mapping_path: Path, max_chain: int) -> dict[str, Any]:
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
    methods: dict[str, dict[str, Any]] = {}
    callers: dict[str, set[str]] = defaultdict(set)
    dex_count = 0
    instruction_count = 0
    invoke_count = 0
    for raw_dex in apk.get_all_dex():
        dex_count += 1
        dex = DEX(raw_dex)
        for class_object in dex.get_classes():
            class_name = canonical(class_object.get_name())
            for method in class_object.get_methods():
                method_name = canonical(method.get_name())
                descriptor = canonical(method.get_descriptor())
                signature = canonical(f"{class_name}->{method_name}{descriptor}")
                if signature in methods:
                    continue
                instructions = instruction_records(method)
                instruction_count += len(instructions)
                references = []
                for row in instructions:
                    if row["name"].startswith("invoke-"):
                        invoke_count += 1
                        reference = parse_method_reference(row["output"])
                        if reference:
                            references.append(reference)
                            callers[reference].add(signature)
                registers_size, ins_size = code_sizes(method)
                record = {
                    "class": class_name,
                    "name": method_name,
                    "descriptor": descriptor,
                    "signature": signature,
                    "node_id": stable_id(signature),
                    "access": access_string(method),
                    "registers_size": registers_size,
                    "ins_size": ins_size,
                    "instructions": instructions,
                    "invoke_references": references,
                }
                record["forwarder"] = analyze_forwarder(record)
                methods[signature] = record
        dex = None

    rejection = Counter()
    candidates = {}
    declared_only = []
    for signature, method in methods.items():
        proof = method["forwarder"]
        if not proof.get("candidate"):
            for reason in proof.get("reasons", []):
                rejection[reason] += 1
            continue
        candidates[signature] = proof
        direct_role = match_role(proof["target_reference"], mapping)
        if proof["tier"] != "VERIFIED_STRICT":
            if direct_role:
                declared_only.append({"signature": signature, "node_id": method["node_id"], "target_reference": proof["target_reference"], **direct_role})

    verified: dict[str, dict[str, Any]] = {}
    for signature, proof in candidates.items():
        if proof["tier"] != "VERIFIED_STRICT":
            continue
        role = match_role(proof["target_reference"], mapping)
        if role:
            verified[signature] = {
                "signature": signature, "node_id": methods[signature]["node_id"],
                "depth": 1, "wrapper_chain": [signature],
                "sensitive_endpoint": proof["target_reference"],
                "invoke_opcode": proof["invoke_opcode"], "proof_flags": proof["proof_flags"], **role,
            }

    for depth in range(2, max_chain + 1):
        additions = {}
        for signature, proof in candidates.items():
            if signature in verified or proof["tier"] != "VERIFIED_STRICT":
                continue
            child = verified.get(proof["target_reference"])
            if child and child["depth"] == depth - 1:
                additions[signature] = {
                    "signature": signature, "node_id": methods[signature]["node_id"],
                    "depth": depth, "wrapper_chain": [signature] + child["wrapper_chain"],
                    "sensitive_endpoint": child["sensitive_endpoint"],
                    "invoke_opcode": proof["invoke_opcode"], "proof_flags": proof["proof_flags"],
                    "rule_id": child["rule_id"], "role": child["role"], "confidence": child["confidence"],
                }
        verified.update(additions)

    for signature, record in verified.items():
        direct_callers = sorted(callers.get(signature, set()))
        record["direct_callers"] = direct_callers
        record["effective_host_candidates"] = effective_callers(signature, verified, callers)
        record["semantic_view_changes_host"] = bool(record["effective_host_candidates"])

    return {
        "schema_version": "1.0.0",
        "sha256": sid,
        "apk_path": str(apk_path.resolve()),
        "status": "OK",
        "labels_used": False,
        "future_rows_loaded": 0,
        "method_count": len(methods),
        "instruction_count": instruction_count,
        "invoke_count": invoke_count,
        "dex_count": dex_count,
        "strict_candidate_count": sum(proof["tier"] == "VERIFIED_STRICT" for proof in candidates.values()),
        "declared_target_only_count": len(declared_only),
        "verified_wrapper_count": len(verified),
        "host_changing_wrapper_count": sum(row["semantic_view_changes_host"] for row in verified.values()),
        "verified_wrappers": sorted(verified.values(), key=lambda row: (row["depth"], row["signature"])),
        "declared_target_only": sorted(declared_only, key=lambda row: row["signature"]),
        "rejection_reason_counts": dict(rejection.most_common()),
        "wall_seconds": time.monotonic() - started,
        "limitations": [
            "VERIFIED_STRICT proves a restricted DEX-body forwarding contract, not whole-program semantic equivalence",
            "reflection, native code, dynamic loading and unresolved dynamic dispatch are not normalized",
            "the APK is not rewritten; only an auditable semantic-view candidate is emitted",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--max-chain", type=int, default=2)
    args = parser.parse_args()
    output, status_path = Path(args.output), Path(args.status)
    started = time.monotonic()
    try:
        result = scan(Path(args.apk), args.sha256.casefold(), Path(args.mapping), args.max_chain)
        write_gzip_json(output, result)
        status = {
            "sha256": args.sha256.casefold(), "status": "OK", "output": str(output.resolve()),
            "verified_wrapper_count": result["verified_wrapper_count"],
            "declared_target_only_count": result["declared_target_only_count"],
            "wall_seconds": result["wall_seconds"],
        }
    except Exception as exc:
        status = {
            "sha256": args.sha256.casefold(), "status": "ERROR",
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started,
        }
    dump_json(status_path, status)
    print("TW_SCAN_ONE " + json.dumps(status, sort_keys=True), flush=True)
    if status["status"] != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
