#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import platform
import re
import shutil
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from icde_common import (
    OUTPUT, PROJECT, bool_text, dump_json, file_record, first_existing,
    norm_sha, read_csv, run_command, sha256_file,
)


FORMAL_CANDIDATES = (
    "artifacts/care-droid-formal-v1/formal_predictions.csv",
    "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
)
P31_CANDIDATES = (
    "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",
)
APK_MAP_CANDIDATES = (
    "outputs/care-ccbc-source-v3/apk_paths.json",
    "outputs/care-ccbc-source-v2/apk_paths.json",
    "outputs/care-ccbc-source-v1/apk_paths.json",
)

TOOL_NAMES = (
    "adb", "emulator", "sdkmanager", "avdmanager", "apkanalyzer", "aapt", "aapt2",
    "qemu-system-x86_64", "docker", "podman", "virsh", "VBoxManage",
    "appium", "droidbot", "maestro", "frida", "frida-ps", "objection",
    "strace", "tcpdump", "tshark", "mitmproxy", "iptables", "nft", "unshare",
    "firejail", "bwrap", "rg",
)

PROJECT_SEARCH_TERMS = {
    "lifecycle": r"adb\s+.*(install|uninstall|pm clear|force-stop)|pm\s+clear|force-stop|snapshot",
    "ui": r"uiautomator|appium|droidbot|maestro|monkey|input\s+(tap|swipe|text)|action[_ -]?replay",
    "ui_record": r"(record|save|write).{0,40}(ui[_ -]?action|event[_ -]?sequence)|action.{0,40}timestamp",
    "ui_replay": r"replay.{0,40}(action|event)|action[_ -]?replay|replay[_ -]?script",
    "ui_delete_action": r"(delete|remove|omit|ablat).{0,40}(action|event)|action.{0,40}(delete|remove|omit|ablat)",
    "dynamic": r"frida|xposed|lsposed|strace|logcat|binder|dynamic[_ -]?(api|event|trace)|instrumentation",
    "dynamic_timestamp": r"(event|trace|hook).{0,60}(timestamp|monotonic|elapsedRealtime)|timestamp.{0,60}(event|trace|hook)",
    "dynamic_attribution": r"(target[_ -]?(pid|package|process)|package[_ -]?attribution|process[_ -]?attribution)",
    "isolation": r"network[_ -]?none|--network[= ]none|unshare\s+-n|firejail.*--net|iptables|nft\s+|airplane_mode|host[-_ ]only",
    "role_mapping": r"Role31|ROLE_NAMES|role[_ -]?mapping|semantic[_ -]?role",
}


def discover_formal() -> Path | None:
    explicit = os.environ.get("ICDE_FORMAL_PREDICTIONS")
    candidates = ([Path(explicit)] if explicit else []) + [PROJECT / rel for rel in FORMAL_CANDIDATES]
    return first_existing(candidates)


def discover_p31() -> Path | None:
    explicit = os.environ.get("ICDE_P31")
    candidates = ([Path(explicit)] if explicit else []) + [PROJECT / rel for rel in P31_CANDIDATES]
    return first_existing(candidates)


def load_apk_map() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    candidates: list[Path] = []
    explicit = os.environ.get("ICDE_APK_INDEX")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(PROJECT / rel for rel in APK_MAP_CANDIDATES)
    candidates.append(PROJECT / "artifacts/care-p31rf-haig-source-v1/lineage/source_2017.csv")
    mapping: dict[str, Path] = {}
    audit: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        found = 0
        try:
            if candidate.suffix.lower() == ".json":
                value = json.loads(candidate.read_text(encoding="utf-8"))
                rows = value.items() if isinstance(value, dict) else []
                for key, raw_path in rows:
                    sid = norm_sha(key)
                    path = Path(str(raw_path))
                    if sid and path.is_file():
                        mapping.setdefault(sid, path.resolve()); found += 1
            elif candidate.suffix.lower() == ".csv":
                for row in read_csv(candidate):
                    sid = next((norm_sha(row.get(k)) for k in ("sha256", "sha", "sample_id", "id") if norm_sha(row.get(k))), "")
                    raw = next((str(row.get(k, "")).strip() for k in ("absolute_path", "apk_path", "path", "file_path", "apk", "source_path") if str(row.get(k, "")).strip().lower().endswith(".apk")), "")
                    if not sid or not raw:
                        continue
                    path = Path(raw)
                    if not path.is_absolute():
                        path = PROJECT / path
                    if path.is_file():
                        mapping.setdefault(sid, path.resolve()); found += 1
            audit.append({"path": str(candidate.resolve()), "status": "READ", "matched_rows": found})
        except Exception as exc:
            audit.append({"path": str(candidate.resolve()), "status": "ERROR", "error": repr(exc)})

    roots = [Path(value) for value in os.environ.get("ICDE_APK_ROOTS", "").split(os.pathsep) if value]
    for root in roots:
        found = 0
        if root.is_dir():
            for path in root.rglob("*.apk"):
                sid = norm_sha(path.stem)
                if sid and sid not in mapping:
                    mapping[sid] = path.resolve(); found += 1
        audit.append({"path": str(root), "status": "SCANNED_ROOT" if root.is_dir() else "ROOT_MISSING", "matched_rows": found})
    return mapping, audit


def parse_formal(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"status": "MISSING"}
    raw_rows = read_csv(path)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, row in enumerate(raw_rows):
        try:
            sid = norm_sha(row.get("sha256"))
            label = int(float(row.get("label", row.get("y", "nan"))))
            fold = int(float(row.get("fold", "nan")))
            prob = float(row.get("stage1_prob", row.get("Global24__prob", "nan")))
            routed = int(float(row.get("routed", row.get("A25__routed", 0))))
            if not sid:
                raise ValueError("invalid sha256")
            pred = int(prob >= 0.5)
            rows.append({"sha256": sid, "label": label, "fold": fold, "stage1_prob": prob, "stage1_pred": pred, "routed": routed})
        except Exception as exc:
            errors.append(f"row={i + 2}: {exc}")
    total_errors = sum(r["stage1_pred"] != r["label"] for r in rows)
    routed_rows = [r for r in rows if r["routed"]]
    captured = sum(r["stage1_pred"] != r["label"] for r in routed_rows)
    summary = {
        "status": "PARSED" if not errors else "PARTIAL",
        "path": str(path), "row_count": len(rows), "parse_errors": errors[:20],
        "unique_sha": len({r["sha256"] for r in rows}),
        "fold_sizes": dict(sorted(Counter(r["fold"] for r in rows).items())),
        "class_counts": dict(sorted(Counter(r["label"] for r in rows).items())),
        "stage1_errors": total_errors, "routed": len(routed_rows),
        "routed_errors": captured, "routed_correct": len(routed_rows) - captured,
        "routed_FN": sum(r["routed"] and r["label"] == 1 and r["stage1_pred"] == 0 for r in rows),
        "routed_FP": sum(r["routed"] and r["label"] == 0 and r["stage1_pred"] == 1 for r in rows),
        "routed_TN": sum(r["routed"] and r["label"] == 0 and r["stage1_pred"] == 0 for r in rows),
        "routed_TP": sum(r["routed"] and r["label"] == 1 and r["stage1_pred"] == 1 for r in rows),
    }
    return rows, summary


def validate_apks(rows: list[dict[str, Any]], mapping: dict[str, Path]) -> dict[str, Any]:
    wanted = {r["sha256"] for r in rows}
    records: list[dict[str, Any]] = []
    counters = Counter()
    for index, sid in enumerate(sorted(wanted), 1):
        path = mapping.get(sid)
        rec: dict[str, Any] = {"sha256": sid, "path": str(path) if path else None}
        if path is None or not path.is_file():
            rec["status"] = "MISSING"; counters["missing"] += 1; records.append(rec); continue
        try:
            actual = sha256_file(path)
            rec.update({"size_bytes": path.stat().st_size, "computed_sha256": actual, "sha_match": actual == sid})
            counters["sha_match" if actual == sid else "sha_mismatch"] += 1
            with zipfile.ZipFile(path, "r") as archive:
                names = set(archive.namelist())
                rec["zip_readable"] = True
                rec["has_android_manifest"] = "AndroidManifest.xml" in names
                rec["dex_count"] = sum(bool(re.fullmatch(r"classes(?:\d+)?\.dex", Path(name).name)) for name in names)
            if rec["has_android_manifest"] and rec["dex_count"] > 0 and rec["sha_match"]:
                rec["status"] = "STATIC_INTEGRITY_PASS"; counters["static_integrity_pass"] += 1
            else:
                rec["status"] = "STATIC_INTEGRITY_FAIL"; counters["static_integrity_fail"] += 1
        except Exception as exc:
            rec.update({"status": "READ_ERROR", "error": repr(exc)}); counters["read_error"] += 1
        records.append(rec)
        if index % 50 == 0:
            print(f"ICDE_APK_STATIC_AUDIT {index}/{len(wanted)}", flush=True)
    routed = {r["sha256"] for r in rows if r["routed"]}
    routed_errors = {r["sha256"] for r in rows if r["routed"] and r["stage1_pred"] != r["label"]}
    available = {r["sha256"] for r in records if r.get("status") == "STATIC_INTEGRITY_PASS"}
    return {
        "formal_sha_count": len(wanted), "apk_paths_found": sum(bool(r.get("path")) for r in records),
        "counters": dict(counters), "routed_apk_available": len(routed & available),
        "routed_error_apk_available": len(routed_errors & available),
        "routed_correct_apk_available": len((routed - routed_errors) & available),
        "installability_verified": False,
        "installability_note": "Stage A does not install untrusted APKs. ZIP/Manifest/DEX integrity is verified; actual package-manager installability belongs to the isolated pilot.",
        "records": records,
    }


def find_sdk_tools() -> tuple[dict[str, str | None], list[str]]:
    sdk_roots: list[Path] = []
    for key in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        value = os.environ.get(key)
        if value:
            sdk_roots.append(Path(value))
    home = Path.home()
    sdk_roots.extend((home / "Android/Sdk", home / "android-sdk", Path("/opt/android-sdk"), Path("/usr/local/android-sdk")))
    tools: dict[str, str | None] = {}
    for name in TOOL_NAMES:
        found = shutil.which(name)
        if not found:
            relative = {
                "adb": "platform-tools/adb", "emulator": "emulator/emulator",
                "sdkmanager": "cmdline-tools/latest/bin/sdkmanager", "avdmanager": "cmdline-tools/latest/bin/avdmanager",
                "apkanalyzer": "cmdline-tools/latest/bin/apkanalyzer",
            }.get(name)
            if relative:
                for root in sdk_roots:
                    candidate = root / relative
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        found = str(candidate); break
        tools[name] = str(Path(found).resolve()) if found else None
    roots = sorted({str(root.resolve()) for root in sdk_roots if root.is_dir()})
    return tools, roots


def command_inventory(tools: dict[str, str | None]) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    safe = [
        ["uname", "-a"], ["lscpu"], ["free", "-b"], ["df", "-B1", str(PROJECT)], ["id"],
    ]
    for argv in safe:
        if shutil.which(argv[0]):
            commands[" ".join(argv)] = run_command(argv, timeout=20)
    if tools.get("adb"):
        commands["adb devices -l"] = run_command([tools["adb"], "devices", "-l"], timeout=20)
        commands["adb version"] = run_command([tools["adb"], "version"], timeout=20)
    if tools.get("emulator"):
        commands["emulator -list-avds"] = run_command([tools["emulator"], "-list-avds"], timeout=20)
        commands["emulator -accel-check"] = run_command([tools["emulator"], "-accel-check"], timeout=20)
    if tools.get("avdmanager"):
        commands["avdmanager list avd"] = run_command([tools["avdmanager"], "list", "avd"], timeout=30)
    if tools.get("docker"):
        commands["docker info"] = run_command([tools["docker"], "info", "--format", "{{json .}}"], timeout=20)
        commands["docker images"] = run_command([tools["docker"], "images", "--format", "{{json .}}"], timeout=20)
        commands["docker ps -a"] = run_command([tools["docker"], "ps", "-a", "--format", "{{json .}}"], timeout=20)
    if tools.get("podman"):
        commands["podman info"] = run_command([tools["podman"], "info", "--format", "json"], timeout=20)
    for name in ("appium", "droidbot", "maestro", "frida", "frida-ps", "objection", "strace", "tcpdump", "tshark", "mitmproxy"):
        if tools.get(name):
            commands[f"{name} version"] = run_command([tools[name], "--version"], timeout=15)
    return commands


def parse_adb_devices(command: dict[str, Any] | None) -> dict[str, list[str]]:
    result = {"emulators": [], "physical_or_other": [], "offline": []}
    if not command or command.get("returncode") != 0:
        return result
    for line in command.get("stdout", "").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 2:
            continue
        serial, state = fields[0], fields[1]
        if state != "device":
            result["offline"].append(serial)
        elif serial.startswith("emulator-"):
            result["emulators"].append(serial)
        else:
            result["physical_or_other"].append(serial)
    return result


def inspect_running_emulators(adb: str | None, devices: dict[str, list[str]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not adb:
        return output
    for serial in devices["emulators"]:
        queries = {
            "properties": [adb, "-s", serial, "shell", "getprop"],
            "identity": [adb, "-s", serial, "shell", "id"],
            "routes": [adb, "-s", serial, "shell", "ip", "route"],
            "airplane_mode": [adb, "-s", serial, "shell", "settings", "get", "global", "airplane_mode_on"],
            "http_proxy": [adb, "-s", serial, "shell", "settings", "get", "global", "http_proxy"],
            "avd_name": [adb, "-s", serial, "emu", "avd", "name"],
            "snapshot_list": [adb, "-s", serial, "emu", "avd", "snapshot", "list"],
            "uiautomator_help": [adb, "-s", serial, "shell", "uiautomator", "help"],
        }
        output[serial] = {name: run_command(argv, timeout=20) for name, argv in queries.items()}
    return output


def project_code_evidence() -> dict[str, Any]:
    rg = shutil.which("rg")
    roots = [path for path in (PROJECT / "scripts", PROJECT / "src", PROJECT / "vendor") if path.exists()]
    evidence: dict[str, Any] = {}
    if not rg:
        return {"status": "RG_MISSING", "matches": {}}
    for name, pattern in PROJECT_SEARCH_TERMS.items():
        argv = [rg, "-l", "-i", "--glob", "*.py", "--glob", "*.sh", "--glob", "*.json", "--glob", "*.yaml", "--glob", "*.yml", pattern]
        argv.extend(str(root) for root in roots)
        record = run_command(argv, timeout=60)
        paths = [line.strip() for line in record.get("stdout", "").splitlines() if line.strip()][:200]
        evidence[name] = {"match_count_capped": len(paths), "paths": paths, "command_status": record.get("status"), "returncode": record.get("returncode")}
    return {"status": "SEARCHED", "matches": evidence}


def avd_inventory() -> dict[str, Any]:
    root = Path.home() / ".android/avd"
    configs = []
    if root.is_dir():
        for ini in sorted(root.glob("*.ini")):
            configs.append({"path": str(ini), "size_bytes": ini.stat().st_size})
    snapshots = []
    for path in root.glob("*.avd/snapshots") if root.is_dir() else []:
        snapshots.append({"path": str(path), "entries": len(list(path.iterdir())) if path.is_dir() else 0})
    return {"root": str(root), "root_exists": root.is_dir(), "avd_configs": configs, "snapshot_directories": snapshots}


def read_proof(path_value: str, required: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
    if not path_value:
        return False, {"status": "NOT_SUPPLIED"}
    path = Path(path_value)
    if not path.is_file():
        return False, {"status": "MISSING", "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        missing = [key for key in required if value.get(key) is not True]
        return not missing, {
            "status": "VALIDATED_FIELDS" if not missing else "INCOMPLETE_FIELDS",
            "path": str(path.resolve()), "sha256": sha256_file(path), "missing_true_fields": missing,
        }
    except Exception as exc:
        return False, {"status": "READ_ERROR", "path": str(path), "error": repr(exc)}


def timing_evidence() -> tuple[bool, dict[str, Any]]:
    explicit = os.environ.get("ICDE_EXISTING_TIMING_JSON", "")
    if not explicit:
        return False, {"status": "NOT_SUPPLIED", "note": "No ICDE/dynamic pilot timing was inferred from unrelated static timing files."}
    path = Path(explicit)
    if not path.is_file():
        return False, {"status": "MISSING", "path": str(path)}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        required = ("single_apk_single_run_seconds", "C0_C1_minimum_seconds", "C1a_increment_seconds")
        valid = all(isinstance(value.get(key), (int, float)) and float(value[key]) > 0 for key in required)
        return valid, {"status": "VALID" if valid else "INVALID", "path": str(path.resolve()), "sha256": sha256_file(path), "values": {key: value.get(key) for key in required}}
    except Exception as exc:
        return False, {"status": "READ_ERROR", "path": str(path), "error": repr(exc)}


def capability_assessment(tools: dict[str, str | None], commands: dict[str, Any], devices: dict[str, list[str]], emulator_details: dict[str, Any], code: dict[str, Any], apk: dict[str, Any], formal: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    kvm = Path("/dev/kvm")
    kvm_available = kvm.exists() and os.access(kvm, os.R_OK | os.W_OK)
    avd_out = commands.get("emulator -list-avds", {}).get("stdout", "")
    configured_avd = bool([line for line in avd_out.splitlines() if line.strip()])
    emulator_connected = bool(devices["emulators"])
    code_matches = code.get("matches", {})
    has_lifecycle_code = bool(code_matches.get("lifecycle", {}).get("paths"))
    has_ui_code = bool(code_matches.get("ui", {}).get("paths"))
    has_ui_record = bool(code_matches.get("ui_record", {}).get("paths"))
    has_ui_replay = bool(code_matches.get("ui_replay", {}).get("paths"))
    has_ui_delete = bool(code_matches.get("ui_delete_action", {}).get("paths"))
    has_dynamic_code = bool(code_matches.get("dynamic", {}).get("paths"))
    has_dynamic_timestamp = bool(code_matches.get("dynamic_timestamp", {}).get("paths"))
    has_dynamic_attribution = bool(code_matches.get("dynamic_attribution", {}).get("paths"))
    has_isolation_code = bool(code_matches.get("isolation", {}).get("paths"))
    has_role_code = bool(code_matches.get("role_mapping", {}).get("paths"))
    ui_host_tool = any(tools.get(name) for name in ("appium", "droidbot", "maestro"))
    dynamic_host_tool = any(tools.get(name) for name in ("frida", "frida-ps", "objection"))
    emulator_uia = any(detail.get("uiautomator_help", {}).get("returncode") == 0 for detail in emulator_details.values())
    emulator_root = any("uid=0" in detail.get("identity", {}).get("stdout", "") for detail in emulator_details.values())
    snapshot_query = any(detail.get("snapshot_list", {}).get("returncode") == 0 for detail in emulator_details.values())
    airplane_on = any(detail.get("airplane_mode", {}).get("stdout", "").strip() == "1" for detail in emulator_details.values())

    isolation_proof, isolation_record = read_proof(
        os.environ.get("ICDE_SANDBOX_PROOF", ""),
        ("host_lan_blocked", "egress_disabled_or_controlled", "no_sensitive_host_mounts", "clean_reset_available"),
    )
    isolation_reason = "validated sandbox proof fields" if isolation_proof else "no complete sandbox/network-isolation proof"
    if has_isolation_code and not isolation_proof:
        isolation_reason += "; project has candidate code but configuration was not treated as runtime proof"

    action_replay_proof = bool(ui_host_tool and has_ui_code and has_ui_record and has_ui_replay)
    action_deletion_proof = bool(action_replay_proof and has_ui_delete)
    collector_proof = bool(dynamic_host_tool and has_dynamic_code and has_dynamic_timestamp and has_dynamic_attribution)
    role_event_proof = bool(collector_proof and has_role_code)
    clean_reset_proof = bool((snapshot_query or configured_avd) and has_lifecycle_code)
    lifecycle_proof = bool((emulator_connected or configured_avd) and tools.get("adb") and has_lifecycle_code)
    environment_available = bool(tools.get("emulator") and tools.get("adb") and (kvm_available or emulator_connected) and (configured_avd or emulator_connected))
    cost_ok, cost_record = timing_evidence()

    hard = {
        "H1_routed_apk_files_available": apk.get("routed_apk_available", 0) > 0,
        "H2_formal_alignment": formal.get("row_count") == 799 and formal.get("unique_sha") == 799 and apk.get("routed_apk_available") == formal.get("routed"),
        "H3_safe_isolated_android_environment": environment_available and isolation_proof and not devices["physical_or_other"],
        "H4_install_start_stop_clear_automation": lifecycle_proof,
        "H5_consistent_clean_reset": clean_reset_proof,
        "H6_timed_ui_record_and_replay": action_replay_proof and emulator_uia,
        "H7_action_deletion_counterfactual": action_deletion_proof,
        "H8_timestamped_sensitive_event_collection": collector_proof,
        "H9_target_apk_attribution": collector_proof,
        "H10_role31_reliable_subset_mapping": role_event_proof,
        "H11_matched_C0_C1_environment": clean_reset_proof and action_replay_proof,
        "H12_minimal_pilot_cost_acceptable": cost_ok,
        "H13_safety_boundary_satisfied": isolation_proof and not devices["physical_or_other"],
        "H14_formal_assets_read_only": True,
    }
    capability = {
        "apk_static_integrity": apk.get("counters", {}),
        "android_execution_environment_available": environment_available,
        "kvm_path_exists": kvm.exists(), "kvm_read_write_access": kvm_available,
        "configured_avd": configured_avd, "running_emulators": devices["emulators"],
        "physical_or_other_devices_detected": devices["physical_or_other"],
        "emulator_root_observed": emulator_root, "snapshot_query_succeeded": snapshot_query,
        "airplane_mode_observed": airplane_on,
        "lifecycle_automation_evidence": lifecycle_proof,
        "clean_reset_evidence": clean_reset_proof,
        "ui_host_tool": ui_host_tool, "device_uiautomator_observed": emulator_uia,
        "timed_action_record_replay_evidence": action_replay_proof,
        "delete_one_action_replay_evidence": action_deletion_proof,
        "dynamic_host_tool": dynamic_host_tool,
        "target_attributed_timestamped_collector_evidence": collector_proof,
        "role_subset_mapping_evidence": role_event_proof,
        "network_isolation_proof": isolation_proof, "network_isolation_reason": isolation_reason,
        "network_isolation_proof_record": isolation_record,
        "existing_dynamic_timing_evidence": cost_record,
        "safety_note": "Physical/non-emulator devices are never selected or queried beyond adb inventory. No APK was installed or executed in Stage A.",
    }
    return capability, hard


def decide(hard: dict[str, bool], capability: dict[str, Any], formal: dict[str, Any], apk: dict[str, Any]) -> dict[str, Any]:
    failed = [name for name, passed in hard.items() if not passed]
    external = []
    if not hard["H1_routed_apk_files_available"]:
        external.append("Provide a readable 2017 APK index/root matching formal SHA256 values.")
    if capability.get("android_execution_environment_available") and not capability.get("network_isolation_proof"):
        external.append("Provide a verifiable sandbox launcher/network policy that blocks host-LAN access and controls egress.")
    if capability.get("physical_or_other_devices_detected"):
        external.append("Disconnect/exclude physical devices; ICDE will use emulator-only allowlisting.")

    if all(hard.values()):
        status = "FEASIBLE"
        rationale = "All 14 Stage-A hard requirements have affirmative runtime or repository evidence."
    else:
        decisive_infra = [
            hard["H3_safe_isolated_android_environment"], hard["H6_timed_ui_record_and_replay"],
            hard["H7_action_deletion_counterfactual"], hard["H8_timestamped_sensitive_event_collection"],
            hard["H9_target_apk_attribution"], hard["H10_role31_reliable_subset_mapping"],
        ]
        if not all(decisive_infra):
            status = "NOT_FEASIBLE"
            rationale = "The current environment lacks decisive safe intervention/collection capabilities; tool names alone were not accepted as proof."
        elif external:
            status = "BLOCKED"
            rationale = "Core execution and collection capabilities exist, but a required permission, APK resource, or independently verifiable isolation proof is missing."
        else:
            status = "CONDITIONALLY_FEASIBLE"
            rationale = "Only bounded in-project glue or a cost measurement is missing; Stage B remains prohibited until re-audit passes all hard conditions."
    return {
        "status": status, "stage": "A", "continue_to_stage_B": status == "FEASIBLE",
        "rationale": rationale, "hard_requirements": hard, "failed_requirements": failed,
        "external_unlocks": external, "future_rows_loaded": 0,
        "apk_rows": formal.get("row_count", 0), "routed_apk_available": apk.get("routed_apk_available", 0),
        "important_boundary": "FEASIBLE does not mean APK execution occurred; Stage B may proceed only through the separately reviewed safe pilot runner.",
    }


def manifest_assets(formal_path: Path | None, p31_path: Path | None, apk: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if formal_path:
        records.append(file_record(formal_path, "frozen formal predictions"))
    if p31_path:
        records.append(file_record(p31_path, "frozen P31 source cache"))
    for row in apk.get("records", []):
        path = row.get("path")
        if path and Path(path).is_file():
            records.append(file_record(Path(path), "read-only source APK"))
    return records


def write_reports(inventory: dict[str, Any], apk: dict[str, Any], decision: dict[str, Any], before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
    before_map = {row["absolute_path"]: row for row in before}
    after_map = {row["absolute_path"]: row for row in after}
    changed = []
    for path in sorted(set(before_map) | set(after_map)):
        a, b = before_map.get(path), after_map.get(path)
        if a != b:
            changed.append({"path": path, "before": a, "after": b})
    integrity = {"status": "PASS" if not changed else "FAIL", "asset_count": len(before), "changed": changed}
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": before, "git_used": False})
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", {"assets": after, "git_used": False})
    dump_json(OUTPUT / "baseline_integrity_check.json", integrity)
    dump_json(OUTPUT / "environment_inventory.json", inventory)
    dump_json(OUTPUT / "apk_asset_audit.json", apk)
    dump_json(OUTPUT / "feasibility_decision.json", decision)
    if integrity["status"] != "PASS":
        final_status = "BLOCKED"
    elif decision["status"] == "FEASIBLE":
        final_status = "CONDITIONALLY_FEASIBLE"
    else:
        final_status = decision["status"]
    final_decision = {
        "status": final_status, "stage_stopped": "A", "stage_B_started": False,
        "reason": ("Stage A passed; Stage B/C have not yet been executed." if decision["status"] == "FEASIBLE" else decision["rationale"]) if integrity["status"] == "PASS" else "Formal asset integrity changed during read-only audit.",
        "stage_A_feasibility": decision["status"],
        "hard_requirements": decision["hard_requirements"], "failed_requirements": decision["failed_requirements"],
        "safety_isolation_verified": decision["hard_requirements"].get("H13_safety_boundary_satisfied", False),
        "formal_asset_integrity": integrity["status"], "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "decision.json", final_decision)

    hard_lines = "\n".join(f"- {bool_text(ok)} `{name}`" for name, ok in decision["hard_requirements"].items())
    failed_lines = "\n".join(f"- `{name}`" for name in decision["failed_requirements"]) or "- 无"
    tools = inventory.get("tools", {})
    present = ", ".join(sorted(name for name, path in tools.items() if path)) or "无"
    cap = inventory["capability_assessment"]
    report = f"""# CARE-Droid ICDE 阶段A可行性报告

## 最终状态

**{final_status}** — 停止于阶段A。阶段B未启动，未安装或执行任何APK。

## 实际检查

- 正式预测与fold/SHA/routed集合对齐。
- 对可定位APK执行只读SHA256、ZIP、AndroidManifest.xml与DEX静态完整性检查。
- 实际查询Android SDK/adb/emulator/AVD/KVM/Docker与主机资源。
- 只对已连接的`emulator-*`执行只读属性、网络、快照与uiautomator查询；未操作物理设备。
- 搜索已有安装清理、UI记录重放、动态采集、隔离和Role映射代码证据。
- 未运行Git命令；未写入正式资产；future_rows_loaded=0。

## APK资产

- 正式行数：{inventory['formal_predictions'].get('row_count', 0)}
- routed：{inventory['formal_predictions'].get('routed', 0)}；routed error：{inventory['formal_predictions'].get('routed_errors', 0)}；routed correct：{inventory['formal_predictions'].get('routed_correct', 0)}
- routed APK静态可用：{apk.get('routed_apk_available', 0)}
- routed error APK静态可用：{apk.get('routed_error_apk_available', 0)}
- routed correct APK静态可用：{apk.get('routed_correct_apk_available', 0)}
- 实际安装能力：阶段A未执行不可信APK，因此仅在既有运行证据充分时判定。

## Android与动态基础设施

- 检测到的主机工具：{present}
- KVM读写权限：{cap.get('kvm_read_write_access')}
- 配置AVD：{cap.get('configured_avd')}
- 运行中emulator：{cap.get('running_emulators')}
- 非emulator设备：{cap.get('physical_or_other_devices_detected')}
- 网络/宿主LAN隔离证据：{cap.get('network_isolation_proof')}（{cap.get('network_isolation_reason')}）
- UI动作记录与重放证据：{cap.get('timed_action_record_replay_evidence')}
- 删除单动作后重放证据：{cap.get('delete_one_action_replay_evidence')}
- 带时间戳、目标APK归属的敏感事件采集证据：{cap.get('target_attributed_timestamped_collector_evidence')}
- Role31可靠子集映射证据：{cap.get('role_subset_mapping_evidence')}

## 14项硬性条件

{hard_lines}

## 未通过项

{failed_lines}

## 成本

阶段A没有执行APK。若不存在可信历史运行计时，单次启动、C0/C1/C1-a和200个routed APK总成本均标记为unavailable，不用假定值替代实测。

## 安全边界

- 未在宿主机或个人设备运行APK。
- 未安装、启动、清除或卸载任何APK。
- 未登录账户、未提供个人数据、未创建网络出口。
- 物理/未知adb设备不会被选择。
- 未经可验证的网络隔离与干净快照能力，不允许进入阶段B。

## 继续判定

{decision['rationale']}

正式资产完整性：**{integrity['status']}**。阶段B只在14项硬性条件全部PASS且安全pilot runner经过单独审查后允许启动。
"""
    (OUTPUT / "feasibility_audit.md").write_text(report, encoding="utf-8")
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(
        "# Reproduction commands\n\n```bash\ncd /home/boyuanjie/projects/tasa-droid\n"
        "bash scripts/care-icde-source-v1/launch_tmux.sh\n"
        "tail -F outputs/care-icde-source-v1.console.log\n```\n\n"
        "Optional explicit inputs: `ICDE_APK_INDEX`, `ICDE_APK_ROOTS`, `ICDE_FORMAL_PREDICTIONS`, "
        "`ICDE_P31`, and a pre-existing independently verified `ICDE_SANDBOX_PROOF`. "
        "Supplying a proof path does not modify the sandbox and is recorded in the audit.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    formal_path = discover_formal()
    p31_path = discover_p31()
    formal_rows, formal_summary = parse_formal(formal_path)
    apk_map, apk_index_audit = load_apk_map()
    apk = validate_apks(formal_rows, apk_map) if formal_rows else {"records": [], "counters": {}, "routed_apk_available": 0}
    apk["index_audit"] = apk_index_audit

    before = manifest_assets(formal_path, p31_path, apk)
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": before, "git_used": False})

    tools, sdk_roots = find_sdk_tools()
    commands = command_inventory(tools)
    devices = parse_adb_devices(commands.get("adb devices -l"))
    emulator_details = inspect_running_emulators(tools.get("adb"), devices)
    code = project_code_evidence()
    avds = avd_inventory()
    capability, hard = capability_assessment(tools, commands, devices, emulator_details, code, apk, formal_summary)
    decision = decide(hard, capability, formal_summary, apk)
    inventory = {
        "project": str(PROJECT), "output": str(OUTPUT), "platform": platform.platform(),
        "python": sys.version, "tools": tools, "android_sdk_roots": sdk_roots,
        "commands": commands, "adb_devices": devices, "running_emulator_details": emulator_details,
        "avd_inventory": avds, "project_code_evidence": code,
        "formal_predictions": formal_summary, "capability_assessment": capability,
        "cost_feasibility": capability.get("existing_dynamic_timing_evidence", {}),
        "future_rows_loaded": 0,
    }
    after = manifest_assets(formal_path, p31_path, apk)
    write_reports(inventory, apk, decision, before, after)
    print(f"ICDE_STAGE_A_DECISION={decision['status']}")
    print(f"ICDE_STAGE_A_FAILED_REQUIREMENTS={len(decision['failed_requirements'])}")
    print(f"ICDE_APK_ROWS={formal_summary.get('row_count', 0)} ROUTED_APKS={apk.get('routed_apk_available', 0)}")
    print(f"ICDE_STAGE_B_ALLOWED={str(decision['continue_to_stage_B']).lower()}")
    if decision["continue_to_stage_B"]:
        print("ICDE_STAGE_A_PASS: Stage B requires the separately reviewed safe pilot runner; no APK was started by this audit package.")
    else:
        print("ICDE_STOPPED_AT_STAGE_A")


if __name__ == "__main__":
    main()

