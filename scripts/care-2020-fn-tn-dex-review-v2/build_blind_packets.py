#!/usr/bin/env python3
"""Build a structurally comparable, label-sealed static review pilot."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
OUT = ROOT / "outputs/care-2020-fn-tn-dex-review-v2"
V1 = ROOT / "outputs/care-2020-fn-tn-dex-review-v1"
MAPPING = ROOT / "scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml"
sys.path.insert(0, str(ROOT / "scripts/care-2020-fn-tn-dex-review-v1"))
from run_audit import scan_static


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if (OUT / "blind_packets").exists():
        raise RuntimeError("v2 blind packets already exist; refuse overwrite")
    with (OUT / "matched_pairs_restricted.csv").open(encoding="utf-8", newline="") as stream:
        selected = list(csv.DictReader(stream))[:8]
    if len(selected) != 8:
        raise RuntimeError("Need eight eligible matched pairs")
    with (V1 / "apk_asset_audit_restricted.csv").open(encoding="utf-8", newline="") as stream:
        asset = {r["sha256"]: r for r in csv.DictReader(stream)}
    with (V1 / "sealed_key_restricted.csv").open(encoding="utf-8", newline="") as stream:
        v1_key = list(csv.DictReader(stream))
    v1_scans = {}
    for case in range(1, 9):
        packet = json.loads((V1 / "blind_packets" / f"P{case:02d}.json").read_text(encoding="utf-8"))
        for row in v1_key:
            if row["case_id"] == f"P{case:02d}":
                v1_scans[row["sha256"]] = packet["sides"][row["side"]]
    selected_ids = {row[key] for row in selected for key in ("fn_sha256", "tn_sha256")}
    if len(selected_ids) != 16 or not selected_ids <= set(asset):
        raise RuntimeError("APK identity alignment failed")
    # This executable uses old labels solely to construct an externally diagnostic case-control audit.
    manifest_before = {}
    for sid in selected_ids:
        path = Path(asset[sid]["path"])
        digest = hash_file(path)
        if digest != sid:
            raise RuntimeError("APK content hash mismatch for " + sid)
        manifest_before[sid] = {"path": str(path), "bytes": path.stat().st_size,
                                "mtime_ns": path.stat().st_mtime_ns, "sha256": digest}
    dump(OUT / "pilot_asset_manifest_before_restricted.json", manifest_before)
    mapping = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
    packet_dir = OUT / "blind_packets"
    packet_dir.mkdir()
    rng = random.Random(20201016)
    sealed = []
    review = []
    scans = dict(v1_scans)
    reused = 0
    for n, pair in enumerate(selected, 1):
        sides = [("UNFIXED_FN", pair["fn_sha256"]), ("CORRECT_TN", pair["tn_sha256"])]
        rng.shuffle(sides)
        packet = {"case_id": f"M{n:02d}", "year": 2020,
                  "match_contract": "routed Stage-I-predicted-benign; probability delta<=0.15; Host ratio<=2; P31 cosine distance<=0.25",
                  "match_quality_without_labels": {"stage1_probability_difference": float(pair["stage1_probability_difference"]),
                                                   "host_count_ratio": float(pair["host_count_ratio"]),
                                                   "p31_standardized_cosine_distance": float(pair["p31_standardized_cosine_distance"])},
                  "sides": {},
                  "boundary": "Static DEX calls are not proof of execution, consent, outcome, or class direction."}
        for side, (kind, sid) in zip(("A", "B"), sides):
            if sid in v1_scans:
                reused += 1
            if sid not in scans:
                try:
                    from loguru import logger
                    logger.remove()
                except ImportError:
                    pass
                print(f"V2_STATIC_SCAN_NEW {len(scans)-len(v1_scans)+1}", flush=True)
                scans[sid] = scan_static(Path(asset[sid]["path"]), mapping)
            packet["sides"][side] = scans[sid]
            sealed.append({"case_id": packet["case_id"], "side": side, "sha256": sid,
                           "diagnostic_kind": kind, "apk_path": asset[sid]["path"]})
        dump(packet_dir / f"M{n:02d}.json", packet)
        review.append({"case_id": packet["case_id"], "reliably_interpretable_Dex_difference": "",
                       "fact_and_method_location": "", "possible_class_direction_before_unsealing": "",
                       "is_fact_in_frozen_C43": "", "uncertainty_or_counterexample": "", "review_notes": ""})
    with (OUT / "sealed_key_restricted.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sealed[0])); writer.writeheader(); writer.writerows(sealed)
    with (OUT / "blind_review_form.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review[0])); writer.writeheader(); writer.writerows(review)
    changed = []
    for sid, expected in manifest_before.items():
        path = Path(expected["path"])
        if hash_file(path) != sid or path.stat().st_size != expected["bytes"] or path.stat().st_mtime_ns != expected["mtime_ns"]:
            changed.append(sid)
    leak = []
    for path in packet_dir.glob("M*.json"):
        text = path.read_text(encoding="utf-8")
        if any(sid in text for sid in selected_ids) or '"diagnostic_kind"' in text or '"apk_path"' in text:
            leak.append(path.name)
    status = "PASS" if not changed and not leak else "FAIL"
    audit = {"status": status, "original_apks_before_after_hashed": len(selected_ids),
             "changed_original_apks": changed, "blind_identifier_leaks": leak,
             "packets": len(review), "scans_reused_from_v1": reused,
             "new_static_scans": len(selected_ids) - reused,
             "labels_used_for_diagnostic_case_control_selection": True,
             "labels_used_for_model_or_threshold_selection": False,
             "model_trained": False, "dynamic_execution": False}
    dump(OUT / "pilot_integrity_and_blinding.json", audit)
    if status != "PASS":
        raise RuntimeError("v2 integrity/blinding failed")
    dump(OUT / "final_decision.json", {"status": "STRUCTURALLY_MATCHED_BLIND_REVIEW_REQUIRED",
                                       "matched_eligible_pairs": 22, "pilot_pairs": len(review),
                                       "meaning": "Comparable static review material exists; no error cause established without blinded semantic assessment.",
                                       "next_action": "Complete blind_review_form.csv without sealed_key_restricted.csv; freeze form, then unseal for analysis."})
    (OUT / "report.md").write_text(
        "# 2020 Direct-C43 unfixed FN versus correct TN: matched static review pilot\n\n"
        f"Structural matching found 22/24 one-to-one eligible pairs. The eight lowest-cost pairs received static DEX review packets; "
        f"{reused} scans were reused and {len(selected_ids)-reused} were newly performed. "
        "All 16 original APKs were SHA-verified before and after and unchanged. "
        "Packets omit labels, APK hashes and source paths; the key is stored separately. "
        "Neither DEX invocation counts nor method/Role differences imply a reliable class direction. "
        "Manual blind review remains necessary before assigning a failure mechanism. No model was trained and no future result was used for model or threshold selection.\n",
        encoding="utf-8")
    print("V2_BLIND_REVIEW_DONE " + json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
