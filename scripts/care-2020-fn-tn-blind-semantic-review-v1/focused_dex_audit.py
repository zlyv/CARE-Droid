#!/usr/bin/env python3
"""Read-only, label-blind focused DEX context for eight diagnostic pairs.

The organizer script uses a sealed alias→APK mapping internally, but never
emits APK identifiers, labels, dataset paths, or embedded string literals in
the reviewer packets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
INPUT = ROOT / "outputs/care-2020-fn-tn-dex-review-v2"
OUTPUT = ROOT / "outputs/care-2020-fn-tn-blind-semantic-review-v1"
MAPPING = ROOT / "scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml"
sys.path.insert(0, str(ROOT / "scripts/care-dex-transparent-wrapper-audit-v1"))
from wrapper_core import match_role, parse_method_reference

SDK_PREFIXES = (
    "lcom/google/", "lcom/facebook/", "lcom/amazon/", "lcom/unity3d/",
    "lcom/flurry/", "lcom/adjust/", "lcom/appsflyer/", "lcom/mopub/",
    "lcom/startapp/", "lcom/ironsource/", "lcom/tencent/", "lcom/baidu/",
    "lcom/alibaba/", "lcom/umeng/", "lcom/bytedance/", "lcom/applovin/",
    "lcom/chartboost/", "lcom/adcolony/", "lcom/inmobi/", "lcom/yandex/",
    "lokhttp3/", "lretrofit2/", "landroid/support/", "landroidx/",
)
HIGH_PRIORITY_ROLES = {
    "device_identifier_access", "contacts_access", "sms_access", "sms_send",
    "network_transmission_or_upload", "location_access", "audio_capture",
    "camera_access", "call_log_access", "background_service_control",
    "task_or_alarm_scheduling", "overlay_or_ui_deception",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def masked_signature(value: str, app_prefix: str) -> str:
    # Namespace remains as an audit hint while the exact application id is masked.
    return value.replace(app_prefix, "L<application>/") if app_prefix else value


def scan(apk_path: Path, expected_sha: str, mapping: dict) -> dict:
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    try:
        from loguru import logger
        logger.remove()
    except ImportError:
        pass
    started = time.monotonic()
    if digest(apk_path) != expected_sha:
        raise RuntimeError("APK content SHA mismatch")
    with zipfile.ZipFile(apk_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("APK ZIP CRC failure")
    apk = APK(str(apk_path))
    package = str(apk.get_package() or "").replace(".", "/")
    app_prefix = "L" + package + "/" if package else ""
    records = []
    method_count = 0
    app_methods = 0
    all_sensitive = 0
    app_sensitive = 0
    for dex_number, raw in enumerate(apk.get_all_dex(), 1):
        dex = DEX(raw)
        for klass in dex.get_classes():
            class_name = str(klass.get_name())
            lower = class_name.casefold()
            app_namespace = bool(app_prefix and class_name.startswith(app_prefix))
            sdk_heuristic = any(lower.startswith(prefix) for prefix in SDK_PREFIXES)
            for method in klass.get_methods():
                method_count += 1
                app_methods += int(app_namespace)
                if method.get_code() is None:
                    continue
                signature = class_name + "->" + str(method.get_name()) + str(method.get_descriptor())
                instructions = list(method.get_instructions_idx())
                sensitive = []
                branch_count = 0
                for k, (offset, inst) in enumerate(instructions):
                    opcode = str(inst.get_name())
                    branch_count += int(opcode.startswith("if-") or opcode.startswith("switch"))
                    if not opcode.startswith("invoke-"):
                        continue
                    target = parse_method_reference(str(inst.get_output()))
                    role = match_role(target, mapping) if target else None
                    if not role:
                        continue
                    all_sensitive += 1
                    app_sensitive += int(app_namespace)
                    # No const-string values are exported. Local opcode/context
                    # order can be inspected, but it is not a proven dataflow.
                    window = [str(node.get_name()) for _, node in instructions[max(0, k-5):min(len(instructions), k+6)]]
                    sensitive.append({"dex_offset": int(offset), "role": role["role"],
                                      "rule_id": role["rule_id"],
                                      "api": target, "local_opcode_window": window})
                if sensitive:
                    roles = sorted({r["role"] for r in sensitive})
                    priority = 3 * sum(r in HIGH_PRIORITY_ROLES for r in roles) + 2 * int(len(roles) > 1) + int(app_namespace)
                    records.append({"dex_number": dex_number,
                                    "method": masked_signature(signature, app_prefix),
                                    "app_package_prefix": app_namespace,
                                    "known_sdk_prefix_heuristic": sdk_heuristic,
                                    "instruction_count": len(instructions), "branch_count": branch_count,
                                    "roles": roles, "sensitive_calls_total": len(sensitive),
                                    "sensitive_call_examples": sensitive[:4], "priority": priority})
    app_records = [r for r in records if r["app_package_prefix"]]
    non_sdk_records = [r for r in records if not r["known_sdk_prefix_heuristic"]]
    app_records.sort(key=lambda r: (-r["priority"], -len(r["roles"]), r["method"]))
    non_sdk_records.sort(key=lambda r: (-r["priority"], -len(r["roles"]), r["method"]))
    chosen = app_records[:25]
    used = {(r["dex_number"], r["method"]) for r in chosen}
    chosen += [r for r in non_sdk_records if (r["dex_number"], r["method"]) not in used][:10]
    for row in chosen:
        row.pop("priority")
    return {
        "method_count": method_count, "app_package_prefix_method_count": app_methods,
        "all_role_mapped_invokes": all_sensitive, "app_package_prefix_role_mapped_invokes": app_sensitive,
        "candidate_sensitive_methods": len(records), "app_package_prefix_sensitive_methods": len(app_records),
        "non_known_sdk_sensitive_methods": len(non_sdk_records),
        "focused_methods": chosen,
        "app_namespace_is_owner_ground_truth": False,
        "limitations": [
            "App package prefix is a heuristic, not code ownership proof; repackaged SDKs may use it.",
            "Known SDK prefix list is incomplete.",
            "Branch opcodes in the same method do not prove control dependence on a sensitive call.",
            "Role co-occurrence and opcode proximity do not prove source-to-sink dataflow or execution.",
            "Unknown code and truncated examples must not be treated as absence of behavior.",
        ],
        "wall_seconds": round(time.monotonic() - started, 3),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("Independent output directory already exists")
    OUTPUT.mkdir(parents=True)
    packet_dir = OUTPUT / "blind_packets"
    packet_dir.mkdir()
    with (INPUT / "sealed_key_restricted.csv").open(encoding="utf-8", newline="") as stream:
        organizer = list(csv.DictReader(stream))
    if len(organizer) != 16:
        raise RuntimeError("Expected 16 alias rows")
    alias = {(r["case_id"], r["side"]): r for r in organizer}
    source_hashes = {str(p): digest(p) for p in (MAPPING, INPUT / "sealed_key_restricted.csv",
                                               INPUT / "matched_pairs_restricted.csv")}
    write_json(OUTPUT / "reused_asset_manifest_before.json", source_hashes)
    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    all_paths = {}
    for row in organizer:
        path = Path(row["apk_path"])
        all_paths[row["sha256"]] = path
    apk_hash_before = {sid: digest(path) for sid, path in all_paths.items()}
    if any(sid != value for sid, value in apk_hash_before.items()):
        raise RuntimeError("Before APK hash mismatch")
    write_json(OUTPUT / "apk_hash_before_restricted.json", apk_hash_before)
    reviewed = 0
    for case_number in range(1, 9):
        case = f"M{case_number:02d}"
        packet = {"case_id": case, "sides": {},
                  "semantic_boundary": "This is a code-inspection aid, not verified execution or class evidence."}
        for side in ("A", "B"):
            r = alias[case, side]
            print(f"FOCUSED_STATIC_SCAN {reviewed+1}/16 case={case} side={side}", flush=True)
            packet["sides"][side] = scan(Path(r["apk_path"]), r["sha256"], mapping)
            reviewed += 1
        write_json(packet_dir / f"{case}.json", packet)
    source_after = {str(p): digest(p) for p in (MAPPING, INPUT / "sealed_key_restricted.csv",
                                              INPUT / "matched_pairs_restricted.csv")}
    apk_after = {sid: digest(path) for sid, path in all_paths.items()}
    changed = [p for p in source_hashes if source_hashes[p] != source_after[p]]
    changed_apk = [sid for sid in apk_hash_before if apk_hash_before[sid] != apk_after[sid]]
    leaks = []
    for path in packet_dir.glob("M*.json"):
        value = path.read_text(encoding="utf-8")
        if any(sid in value for sid in all_paths) or '"apk_path"' in value or '"diagnostic_kind"' in value:
            leaks.append(path.name)
    decision = {"status": "FOCUSED_BLIND_REVIEW_READY" if not (changed or changed_apk or leaks) else "FAILED",
                "cases": 8, "apks": 16, "changed_reused_assets": changed,
                "changed_original_apks": changed_apk, "blind_packet_leaks": leaks,
                "new_model_trained": False, "dynamic_execution": False,
                "labels_used_for_diagnostic_pair_sampling": True,
                "labels_visible_in_reviewer_packets": False,
                "claim_boundary": "Candidate method contexts are heuristic; class direction requires blind review and later unsealing."}
    write_json(OUTPUT / "decision.json", decision)
    if decision["status"] == "FAILED":
        raise RuntimeError("Integrity or blinding failed")
    print("FOCUSED_DEX_AUDIT_DONE " + json.dumps(decision), flush=True)


if __name__ == "__main__":
    main()
