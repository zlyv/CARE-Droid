#!/usr/bin/env python3
"""Run a label-independent result-centred dependency-slice feasibility audit.

The analyser keeps no raw DEX, method signatures, field identifiers,
registers, or instruction text.  Per-sink outputs contain only a stable hash,
semantic roles, dependency evidence types, and explicit resolution status.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_semantic_mapping_audit_ag import (  # noqa: E402
    api_rule_matches,
    class_and_method,
    load_yaml,
)
from run_static_coverage_audit_ag import read_csv, suppress_logs, write_csv  # noqa: E402
from tasa_droid.p2.result_centered_slicing import (  # noqa: E402
    InstructionRecord,
    MethodIR,
    SlicePolicy,
    canonical_reference,
    slice_program,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_call(function, default=None):
    try:
        return function()
    except Exception:
        return default


def has_code(method) -> bool:
    return callable(getattr(method, "get_code", None)) and (
        safe_call(getattr(method, "get_code", lambda: None)) is not None
    )


def unwrap_encoded_method(candidate, vm=None):
    """Return an Androguard EncodedMethod from VM method-ID wrappers.

    ``DalvikVMFormat.get_methods()`` returns ``MethodIdItem`` objects in the
    Androguard version used by the analysis environment.  The bytecode APIs
    (``get_code()``, ``get_instructions_idx()``) live on the wrapped encoded
    method and are reached through ``get_method()``.
    """

    if has_code(candidate):
        return candidate

    getter = getattr(candidate, "get_method", None)
    if callable(getter):
        decoded = safe_call(getter)
        if decoded is not None and has_code(decoded):
            return decoded

    # Some Androguard releases expose a MethodIdItem from get_methods(),
    # without exposing a usable get_method() on that wrapper.  Resolve it
    # through the VM's descriptor lookup before giving up.
    if vm is not None:
        class_name = safe_call(
            getattr(candidate, "get_class_name", lambda: ""), ""
        )
        method_name = safe_call(
            getattr(candidate, "get_name", lambda: ""), ""
        )
        descriptor = safe_call(
            getattr(candidate, "get_descriptor", lambda: ""), ""
        )
        descriptor_getter = getattr(vm, "get_method_descriptor", None)
        if callable(descriptor_getter) and (class_name or method_name):
            decoded = safe_call(
                lambda: descriptor_getter(class_name, method_name, descriptor)
            )
            if isinstance(decoded, (list, tuple)):
                decoded = decoded[0] if decoded else None
            if decoded is not None and has_code(decoded):
                return decoded

        index_getter = getattr(candidate, "get_method_idx", None)
        by_index = getattr(vm, "get_method_by_idx", None)
        if callable(index_getter) and callable(by_index):
            index = safe_call(index_getter)
            decoded = safe_call(lambda: by_index(index))
            if decoded is not None and has_code(decoded):
                return decoded
    return candidate


def encoded_method_signature(method) -> str:
    method = unwrap_encoded_method(method)
    class_name = safe_call(getattr(method, "get_class_name", lambda: ""), "")
    name = safe_call(getattr(method, "get_name", lambda: ""), "")
    descriptor = safe_call(getattr(method, "get_descriptor", lambda: ""), "")
    return canonical_reference(f"{class_name}->{name}{descriptor}")


def parameter_registers(method) -> tuple[str, ...]:
    method = unwrap_encoded_method(method)
    code = safe_call(getattr(method, "get_code", lambda: None))
    if code is None:
        return tuple()
    register_count = int(
        safe_call(getattr(code, "get_registers_size", lambda: 0), 0) or 0
    )
    incoming_count = int(
        safe_call(getattr(code, "get_ins_size", lambda: 0), 0) or 0
    )
    start = max(0, register_count - incoming_count)
    return tuple(f"v{index}" for index in range(start, register_count))


def instruction_records(method) -> tuple[InstructionRecord, ...]:
    method = unwrap_encoded_method(method)
    records = []
    getter = getattr(method, "get_instructions_idx", None)
    if callable(getter):
        iterator = getter()
    else:
        offset = 0
        fallback = []
        for instruction in getattr(method, "get_instructions", lambda: ())():
            fallback.append((offset, instruction))
            offset += int(
                safe_call(getattr(instruction, "get_length", lambda: 0), 0) or 0
            )
        iterator = fallback
    for offset, instruction in iterator:
        records.append(InstructionRecord(
            offset=int(offset),
            name=str(
                safe_call(getattr(instruction, "get_name", lambda: ""), "") or ""
            ),
            output=str(
                safe_call(getattr(instruction, "get_output", lambda: ""), "") or ""
            ),
        ))
    return tuple(records)


def build_method_ir(dalvik, analysis=None, audit=None) -> list[MethodIR]:
    audit = audit if audit is not None else {}
    vms = dalvik if isinstance(dalvik, (list, tuple)) else [dalvik]
    candidates = []

    # Analysis.get_methods() is the stable route in Androguard releases where
    # DalvikVMFormat.get_methods() exposes MethodIdItem metadata rather than
    # bytecode-bearing EncodedMethod objects.
    analysis_getter = getattr(analysis, "get_methods", None)
    if callable(analysis_getter):
        candidates.extend(
            (candidate, None, "ANALYSIS")
            for candidate in (safe_call(analysis_getter, []) or [])
        )

    for vm in vms:
        for candidate in safe_call(
            getattr(vm, "get_methods", lambda: ()), []
        ) or []:
            candidates.append((candidate, vm, "DALVIK_VM"))

    methods_by_signature = {}
    decoded_count = 0
    instruction_count = 0
    invoke_count = 0
    source_counts = Counter()
    for candidate, vm, source in candidates:
        method = unwrap_encoded_method(candidate, vm)
        if not has_code(method):
            continue
        decoded_count += 1
        source_counts[source] += 1
        signature = encoded_method_signature(method)
        if not signature:
            continue
        instructions = instruction_records(method)
        instruction_count += len(instructions)
        invoke_count += sum(
            instruction.name.casefold().startswith("invoke-")
            for instruction in instructions
        )
        methods_by_signature[signature] = MethodIR(
            signature=signature,
            parameter_registers=parameter_registers(method),
            instructions=instructions,
        )

    audit.update({
        "method_candidate_count": len(candidates),
        "decoded_method_candidate_count": decoded_count,
        "unique_method_count": len(methods_by_signature),
        "instruction_count": instruction_count,
        "invoke_instruction_count": invoke_count,
        "decoded_method_source_counts": dict(source_counts),
    })
    methods = list(methods_by_signature.values())
    return methods


def mapping_function(rules):
    api_rules = [rule for rule in rules if rule.get("rule_kind", "api") == "api"]

    def map_roles(reference: str) -> set[str]:
        class_name, method_name = class_and_method(reference)
        return {
            str(rule["role"])
            for rule in api_rules
            if api_rule_matches(rule, class_name, method_name)
        }

    return map_roles


def select_rows(rows: list[dict], sha256_values: list[str], limit: int) -> list[dict]:
    by_sha = {
        str(row.get("sha256", "")).casefold(): row for row in rows
        if row.get("sha256") and row.get("absolute_path")
    }
    if sha256_values:
        selected = []
        for value in sha256_values:
            key = value.casefold()
            if key not in by_sha:
                raise SystemExit(f"requested SHA256 is absent from lineage: {value}")
            selected.append(by_sha[key])
        return selected
    return [by_sha[key] for key in sorted(by_sha)[:limit]]


def analyse_one(row: dict, map_roles, policy: SlicePolicy) -> tuple[dict, list[dict]]:
    apk = dalvik = analysis = None
    try:
        from androguard.misc import AnalyzeAPK

        apk, dalvik, analysis = AnalyzeAPK(row["absolute_path"])
        extraction_audit = {}
        methods = build_method_ir(dalvik, analysis, extraction_audit)
        if not methods:
            raise RuntimeError(
                "no bytecode-bearing methods decoded: "
                f"candidates={extraction_audit.get('method_candidate_count', 0)} "
                f"decoded={extraction_audit.get('decoded_method_candidate_count', 0)}"
            )
        result = slice_program(methods, map_roles, policy)
        units = []
        for unit in result["units"]:
            units.append({
                "sink_id": unit["sink_id"],
                "sink_role": unit["sink_role"],
                "roles": unit["roles"],
                "order": unit["order"],
                "dependency_status": unit["dependency_status"],
                "evidence": unit["evidence"],
            })
        resolved = [
            unit for unit in units
            if unit["dependency_status"] == "RESOLVED_DEPENDENCY"
        ]
        summary = {
            "sha256": str(row.get("sha256", "")).casefold(),
            "status": "OK",
            "method_count": result["method_count"],
            "method_candidate_count": extraction_audit["method_candidate_count"],
            "decoded_method_candidate_count": extraction_audit[
                "decoded_method_candidate_count"
            ],
            "instruction_count": extraction_audit["instruction_count"],
            "invoke_instruction_count": extraction_audit[
                "invoke_instruction_count"
            ],
            "sink_unit_count": len(units),
            "resolved_dependency_unit_count": len(resolved),
            "unresolved_singleton_unit_count": len(units) - len(resolved),
            "multi_role_unit_count": sum(unit["order"] >= 2 for unit in units),
            "high_order_unit_count": sum(unit["order"] >= 3 for unit in units),
            "field_with_resolved_role_count": result["field_with_resolved_role_count"],
            "fixed_point_iterations": result["fixed_point_iterations"],
            "fixed_point_converged": result["fixed_point_converged"],
            "error_type": "",
            "error_message": "",
            "analysis_characteristics": result["analysis_characteristics"],
        }
        return summary, units
    except Exception as exc:
        return ({
            "sha256": str(row.get("sha256", "")).casefold(),
            "status": "ERROR",
            "method_count": 0,
            "method_candidate_count": 0,
            "decoded_method_candidate_count": 0,
            "instruction_count": 0,
            "invoke_instruction_count": 0,
            "sink_unit_count": 0,
            "resolved_dependency_unit_count": 0,
            "unresolved_singleton_unit_count": 0,
            "multi_role_unit_count": 0,
            "high_order_unit_count": 0,
            "field_with_resolved_role_count": 0,
            "fixed_point_iterations": 0,
            "fixed_point_converged": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:500],
            "analysis_characteristics": {},
        }, [])
    finally:
        analysis = None
        dalvik = None
        apk = None
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sha256", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("limit must be positive")

    suppress_logs()
    lineage_path = args.lineage.expanduser().resolve(strict=True)
    mapping_path = args.mapping.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = select_rows(read_csv(lineage_path), args.sha256, args.limit)
    mapping = load_yaml(mapping_path)
    map_roles = mapping_function(list(mapping.get("rules", [])))
    policy = SlicePolicy(max_iterations=args.max_iterations)
    free_guard = int(max(0.1, args.min_free_gb) * 1024 ** 3)

    summaries = []
    unit_rows = []
    print(f"result-centred slice dry-run started: {len(selected)} APKs", flush=True)
    for index, row in enumerate(selected, start=1):
        if shutil.disk_usage(output).free < free_guard:
            raise SystemExit("free-space guard reached before dry-run completion")
        summary, units = analyse_one(row, map_roles, policy)
        summaries.append(summary)
        unit_rows.append({
            "sha256": summary["sha256"],
            "status": summary["status"],
            "units": units,
        })
        if index % max(1, args.progress_every) == 0 or index == len(selected):
            print(
                f"processed {index}/{len(selected)} status={summary['status']} "
                f"sinks={summary['sink_unit_count']} "
                f"resolved={summary['resolved_dependency_unit_count']}",
                flush=True,
            )

    summary_fields = [
        "sha256", "status", "method_count", "method_candidate_count",
        "decoded_method_candidate_count", "instruction_count",
        "invoke_instruction_count", "sink_unit_count",
        "resolved_dependency_unit_count", "unresolved_singleton_unit_count",
        "multi_role_unit_count", "high_order_unit_count",
        "field_with_resolved_role_count", "fixed_point_iterations",
        "fixed_point_converged", "error_type", "error_message",
    ]
    summary_path = output / "result_centered_slice_summary.csv"
    failure_path = output / "result_centered_slice_failures.csv"
    units_path = output / "result_centered_behavior_units.jsonl"
    write_csv(summary_path, summaries, summary_fields)
    write_csv(
        failure_path,
        [row for row in summaries if row["status"] != "OK"],
        summary_fields,
    )
    with units_path.open("w", encoding="utf-8") as handle:
        for row in unit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    ok = [row for row in summaries if row["status"] == "OK"]
    total_sinks = sum(row["sink_unit_count"] for row in ok)
    resolved = sum(row["resolved_dependency_unit_count"] for row in ok)
    evidence_counts = Counter()
    sink_role_counts = Counter()
    status_counts = Counter()
    for row in unit_rows:
        for unit in row["units"]:
            sink_role_counts[unit["sink_role"]] += 1
            status_counts[unit["dependency_status"]] += 1
            evidence_counts.update(unit["evidence"])
    characteristics = next(
        (row["analysis_characteristics"] for row in summaries
         if row["analysis_characteristics"]),
        {},
    )
    audit = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P48_RESULT_CENTERED_DEPENDENCY_SLICE_DRY_RUN",
        "formal_training_enabled": False,
        "label_independent_analysis": True,
        "sample_selection": (
            "explicit_sha256" if args.sha256 else "sha256_sorted_label_independent"
        ),
        "lineage_csv": str(lineage_path),
        "lineage_sha256": sha256_file(lineage_path),
        "mapping_file": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "requested_apk_count": len(selected),
        "status_counts": dict(Counter(row["status"] for row in summaries)),
        "sink_roles": sorted({
            "network_transmission_or_upload", "sms_send_receive",
            "command_execution", "dynamic_code_loading",
        }),
        "aggregate": {
            "sink_unit_count": total_sinks,
            "resolved_dependency_unit_count": resolved,
            "resolved_dependency_ratio": resolved / total_sinks if total_sinks else 0.0,
            "multi_role_unit_count": sum(row["multi_role_unit_count"] for row in ok),
            "high_order_unit_count": sum(row["high_order_unit_count"] for row in ok),
            "sink_role_counts": dict(sink_role_counts),
            "dependency_status_counts": dict(status_counts),
            "evidence_counts": dict(evidence_counts),
        },
        "policy": {
            "max_fixed_point_iterations": policy.max_iterations,
            "max_units_per_method": policy.max_units_per_method,
            "max_tokens_per_value": policy.max_tokens_per_value,
        },
        "analysis_characteristics": characteristics,
        "privacy": {
            "raw_dex_persisted": False,
            "raw_method_signatures_persisted": False,
            "raw_field_identifiers_persisted": False,
            "registers_or_instruction_text_persisted": False,
            "sink_identifier": "truncated_sha256",
        },
        "outputs": {
            "summary": str(summary_path),
            "behavior_units": str(units_path),
            "failures": str(failure_path),
        },
        "blocking_reasons": [
            "path_sensitive_control_flow_not_implemented",
            "points_to_alias_analysis_not_implemented",
            "intent_bundle_propagation_not_implemented",
            "dependency_slice_requires_manual_precision_validation",
            "behavior_hypergraph_classifier_not_implemented",
        ],
        "notes": [
            "Only dependency evidence can create a resolved multi-role behavior unit.",
            "Call reachability, same-method membership, and same-class membership are never dependency evidence.",
            "Unresolved sinks are retained as singleton units rather than discarded.",
            "This stage audits extraction feasibility and does not train or evaluate a classifier.",
        ],
    }
    audit_path = output / "result_centered_slice_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
