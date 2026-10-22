#!/usr/bin/env python3
"""Read-only, blinded 2020 Direct-C43 FN/TN DEX review pilot.

This is a diagnostic case-control packet generator, not a classifier or a
claim that DEX facts establish malware/benign intent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
OUT = ROOT / "outputs/care-2020-fn-tn-dex-review-v1"
REFERENCE = ROOT / "outputs/care_crossyear_layerwise_diagnosis_v2/per_apk_error_contract.csv"
EMBEDDING = ROOT / "outputs/care_crossyear_layerwise_diagnosis_v2/embeddings/T3_C1/validation.npz"
APK_ROOT = Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/apks_n200_2019_2022/year2020")
MAPPING = ROOT / "scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml"
OLD_WRAPPER = ROOT / "scripts/care-dex-transparent-wrapper-audit-v1"
PILOT_PAIRS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locate_apk(sid: str) -> Path:
    candidates = [APK_ROOT / cohort / (sid.upper() + ".apk") for cohort in ("benign", "malware")]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise RuntimeError(f"APK resolution count {len(present)} for {sid}")
    return present[0]


def scan_static(apk_path: Path, mapping: dict) -> dict:
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    sys.path.insert(0, str(OLD_WRAPPER))
    from wrapper_core import match_role, parse_method_reference

    started = time.monotonic()
    with zipfile.ZipFile(apk_path) as archive:
        names = archive.namelist()
        dex_names = sorted(n for n in names if re.fullmatch(r"classes\d*\.dex", Path(n).name))
        native_count = sum(n.lower().endswith(".so") for n in names)
        if not dex_names:
            raise RuntimeError("No classes*.dex")
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError("ZIP CRC error in " + bad)

    apk = APK(str(apk_path))
    role_counts: Counter[str] = Counter()
    method_records: list[dict] = []
    total_methods = 0
    for dex_index, raw_dex in enumerate(apk.get_all_dex(), 1):
        dex = DEX(raw_dex)
        for klass in dex.get_classes():
            class_name = str(klass.get_name())
            for method in klass.get_methods():
                total_methods += 1
                code = method.get_code()
                if code is None:
                    continue
                signature = class_name + "->" + str(method.get_name()) + str(method.get_descriptor())
                instructions = list(method.get_instructions_idx())
                sensitive = []
                for offset, inst in instructions:
                    opcode = str(inst.get_name())
                    if not opcode.startswith("invoke-"):
                        continue
                    ref = parse_method_reference(str(inst.get_output()))
                    role = match_role(ref, mapping) if ref else None
                    if role:
                        role_counts[role["role"]] += 1
                        sensitive.append({"offset": int(offset), "opcode": opcode,
                                          "role": role["role"], "rule_id": role["rule_id"], "reference": ref})
                if sensitive:
                    method_records.append({
                        "dex": dex_index, "method": signature,
                        "instruction_count": len(instructions),
                        "sensitive_calls": sensitive[:12],
                        "sensitive_call_total": len(sensitive),
                        "local_opcodes": [str(inst.get_name()) for _, inst in instructions[:24]],
                    })
    method_records.sort(key=lambda r: (-r["sensitive_call_total"], r["method"]))
    return {
        "dex_count": len(dex_names), "native_library_count": native_count,
        "method_count": total_methods, "sensitive_method_count": len(method_records),
        "role_call_counts": dict(sorted(role_counts.items())),
        "example_sensitive_methods": method_records[:20],
        "examples_truncated": len(method_records) > 20,
        "wall_seconds": round(time.monotonic() - started, 3),
        "limits": "Static invocations are not proof of execution, data use, consent, or malicious intent; top-20 methods are a review sample.",
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError("Output directory already exists; use a new version")
    OUT.mkdir(parents=True)
    before = {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                          "sha256": sha256(path)} for path in (REFERENCE, EMBEDDING, MAPPING)}
    write_json(OUT / "baseline_asset_manifest_before.json", before)

    with REFERENCE.open(encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    cohort = [r for r in all_rows if r["tag"] == "T3_C1" and r["year"] == "2020"
              and r["routed"] == "1" and r["stage1_prediction"] == "0"]
    fn = [r for r in cohort if r["label"] == "1" and r["category"] == "UNFIXED_FN"]
    tn = [r for r in cohort if r["label"] == "0" and r["final_prediction"] == "0"]
    if len(fn) != 24 or len(tn) < 24:
        raise RuntimeError(f"Frozen cohort mismatch: unfixed FN={len(fn)}, correct TN={len(tn)}")
    # Match on frozen Stage-I P31 probability only. Do not match on the expert outcome.
    costs = np.array([[abs(float(a["stage1_probability"]) - float(b["stage1_probability"]))
                       for b in tn] for a in fn])
    left, right = linear_sum_assignment(costs)
    pairs = [(fn[i], tn[j], float(costs[i, j])) for i, j in zip(left, right)]
    pairs.sort(key=lambda pair: (float(pair[0]["expert_probability"]), pair[0]["sha256"]))
    # Spread the bounded pilot across the unresolved-FN score distribution.
    selected = [pairs[int(round(k * (len(pairs) - 1) / (PILOT_PAIRS - 1)))] for k in range(PILOT_PAIRS)]

    asset_rows = []
    apk_paths = {}
    for row in fn + tn:
        sid = row["sha256"].lower()
        path = locate_apk(sid)
        digest = sha256(path)
        if digest != sid:
            raise RuntimeError(f"Filename/content SHA mismatch: {sid}")
        apk_paths[sid] = path
        asset_rows.append({"sha256": sid, "path": str(path), "bytes": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns, "content_sha256": digest})
    write_csv(OUT / "apk_asset_audit_restricted.csv", asset_rows)
    write_json(OUT / "apk_asset_coverage.json", {"unfixed_fn": len(fn), "correct_tn": len(tn),
                                                   "resolved_and_sha_verified": len(apk_paths),
                                                   "pilot_pairs": PILOT_PAIRS})

    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    rng = random.Random(20201015)
    key_rows = []
    blind_rows = []
    packet_dir = OUT / "blind_packets"
    packet_dir.mkdir()
    scanned = {}
    for case, (error, control, cost) in enumerate(selected, 1):
        names = [("FN", error), ("TN", control)]
        rng.shuffle(names)
        packet = {"case_id": f"P{case:02d}", "match": "2020, routed, Stage-I predicted benign; matched on Stage-I probability",
                  "sides": {}, "interpretation_boundary": "Neither side's class label or expert outcome is disclosed. DEX facts require human semantic review."}
        for side, (kind, row) in zip(("A", "B"), names):
            sid = row["sha256"].lower()
            if sid not in scanned:
                print(f"STATIC_SCAN {len(scanned)+1}/{2*PILOT_PAIRS}", flush=True)
                scanned[sid] = scan_static(apk_paths[sid], mapping)
            packet["sides"][side] = scanned[sid]
            key_rows.append({"case_id": packet["case_id"], "side": side, "sha256": sid,
                             "kind": kind, "apk_path": str(apk_paths[sid]), "stage1_probability": row["stage1_probability"],
                             "expert_probability": row["expert_probability"], "match_cost": cost})
        write_json(packet_dir / f"P{case:02d}.json", packet)
        blind_rows.append({"case_id": packet["case_id"], "verified_static_difference": "",
                           "dex_location_and_fact": "", "is_difference_in_existing_C43": "",
                           "can_fact_reliably_support_class_direction": "", "evidence_grade": "",
                           "reviewer_notes": ""})
    write_csv(OUT / "blind_review_form.csv", blind_rows)
    write_csv(OUT / "sealed_key_restricted.csv", key_rows)
    # The key is deliberately separate from the blind packets and review form.
    after = {str(path): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                         "sha256": sha256(path)} for path in (REFERENCE, EMBEDDING, MAPPING)}
    write_json(OUT / "baseline_asset_manifest_after.json", after)
    changed = [path for path in before if before[path] != after[path]]
    write_json(OUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed,
                                                     "scope": "Three reused baseline files; APK file hashes verified during asset audit"})
    if changed:
        raise RuntimeError("Baseline asset integrity changed")
    decision = {"status": "BLIND_STATIC_REVIEW_REQUIRED", "phase": "2020_UNFIXED_FN_VS_TN_PILOT",
                "cohort": {"routed_stage1_predicted_benign": len(cohort), "unfixed_fn": len(fn), "correct_tn": len(tn)},
                "matched_pairs": len(pairs), "pilot_pairs": PILOT_PAIRS, "pilot_apks_scanned": len(scanned),
                "all_candidate_apks_sha_verified": len(apk_paths), "dynamic_execution": False,
                "new_model_trained": False, "2020_labels_used_for_diagnostic_pair_selection": True,
                "2020_labels_used_for_model_or_threshold_selection": False,
                "claim_boundary": "No causal/error-mechanism conclusion before blind semantic review; no cross-year generalization from this pilot.",
                "next_action": "Complete blind_review_form.csv using blind_packets without opening sealed_key_restricted.csv; then freeze form and interpret against key."}
    write_json(OUT / "decision.json", decision)
    (OUT / "report.md").write_text(
        "# 2020 routed FN-versus-TN static DEX blind-review pilot\n\n"
        f"Status: **{decision['status']}**.\n\n"
        f"Frozen cohort: {len(cohort)} routed Stage-I-predicted-benign APKs, including {len(fn)} Direct-C43-unfixed FN and {len(tn)} Direct-C43-correct TN. "
        f"All {len(apk_paths)} candidate APKs were resolved and SHA256-verified. "
        f"One-to-one Stage-I probability matching produced {len(pairs)} pairs; {PILOT_PAIRS} score-spread pairs ({len(scanned)} APKs) received read-only DEX scans.\n\n"
        "The packets show Role31-mapped invocation examples and local method summaries, not execution, intent, consent, or class direction. "
        "The reviewer must inspect the packets while the key remains sealed. No classifier was fitted, no parameter selected, and no formal CARE asset modified.\n\n"
        "This diagnostic uses 2020 labels only to form FN/TN case-control pairs; it must not be used to choose a model for 2017→future evaluation. "
        "The top-20 sensitive methods per APK are a review sample, not full static coverage.\n",
        encoding="utf-8")
    print("DEX_REVIEW_PILOT_DONE " + json.dumps(decision, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
