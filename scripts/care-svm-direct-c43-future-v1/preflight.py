#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from future_common import (DIRECT_METHOD, DIRECT_OUTPUT, DIRECT_SCRIPT, FAMILY_OUTPUT, FAMILY_SCRIPT,
    FORMAL_DIAG, OUTPUT, PROJECT, dump_json, file_record, read_csv)


def main():
    if not OUTPUT.is_relative_to((PROJECT / "outputs").resolve()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: unsafe output directory")
    if OUTPUT.name != "care-svm-direct-c43-future-v1":
        raise RuntimeError("BLOCKED_BY_CONTRACT: output name changed")
    resolved_path = FORMAL_DIAG / "resolved_inputs.json"
    required = {
        "source_direct_predictions": DIRECT_OUTPUT / "source/U2_DIRECT_STAGE2/source_predictions.csv",
        "source_direct_metrics": DIRECT_OUTPUT / "source_results.csv",
        "source_direct_variant": DIRECT_SCRIPT / "model_variants.py",
        "family_future_runner": FAMILY_SCRIPT / "run_frozen_arm.py",
        "family_model_adapter": FAMILY_SCRIPT / "model_adapter.py",
        "family_stage1": FAMILY_SCRIPT / "stage1_core.py",
        "family_common": FAMILY_SCRIPT / "common.py",
        "bounded_future_svm": FAMILY_OUTPUT / "frozen/SVM/predictions_label_free.csv",
        "future_member_reference": FORMAL_DIAG / "references/future_label_free.csv",
        "future_labels_summary_only": FORMAL_DIAG / "frozen_predictions_with_labels.csv",
        "formal_resolved_inputs": resolved_path,
        "v4_runner": PROJECT / "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing prerequisites:\n- " + "\n- ".join(missing))
    source_metrics = {row["arm"]: row for row in read_csv(required["source_direct_metrics"])}
    direct = source_metrics.get("U2_DIRECT_STAGE2")
    if direct is None:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source Direct-C43 result missing")
    exact = (int(direct["fixed"]), int(direct["broken"]), int(direct["net"])) == (49, 17, 32)
    if not exact or abs(float(direct["BA"]) - .9186152882205514) > 1e-12:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source Direct-C43 reference changed")
    source_predictions = read_csv(required["source_direct_predictions"])
    if len(source_predictions) != 799 or not source_predictions or DIRECT_METHOD + "__prob" not in source_predictions[0]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source Direct-C43 prediction schema changed")
    future_reference = read_csv(required["future_member_reference"])
    bounded_future = read_csv(required["bounded_future_svm"])
    # Do not parse the future-label table before predictions are frozen.  It is
    # present in the read-only manifest but opened semantically only by summarize.py.
    if len(future_reference) != 1989 or len(bounded_future) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort changed")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    dynamic = {
        "sealed_features": Path(resolved["sealed_features"]),
        "endpoint_cache": Path(resolved["endpoint_cache"]),
        "future_inter_cache": Path(resolved["future_inter_cache"]),
        "future_s7_cache": FORMAL_DIAG / "cache/future_s7.npz",
        "future_a0_cache": FORMAL_DIAG / "cache/a0_role_2017_2021.npz",
        "source_p31": PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",
    }
    dynamic_missing = [f"{name}: {path}" for name, path in dynamic.items() if not path.is_file() or not path.stat().st_size]
    if dynamic_missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: missing cached assets:\n- " + "\n- ".join(dynamic_missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = [file_record(path, name) for name, path in {**required, **dynamic}.items()]
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {"assets": assets})
    protocol = {
        "experiment": "SVM + Direct-C43 frozen 2017-to-2018..2021 diagnostic v1",
        "source_status": "post-hoc research mainline candidate; source broken exceeds bounded reference",
        "train": "all 799 source-2017 APKs; SVM context is strict cross-fit OOF",
        "test": "2018, 2019, 2020, 2021; predictions are written before labels are loaded",
        "stage1": "calibrated P31-SVM trained on 2017",
        "routing": "SVM-anchored four-member MeanOpposition; deterministic Top25% separately per year",
        "stage2": "A0 Role-C43 Relation-Query DirectStage2; Stage-I logit input zeroed",
        "decision": "routed sample uses Direct-C43 probability at threshold 0.5; unrouted sample keeps SVM",
        "bounded_correction": False,
        "threshold": .5,
        "model_seeds": [900000, 900001, 900002],
        "keep_weight": 1.0,
        "fix_weight": 1.0,
        "keep_consistency": 0.0,
        "future_used_for_selection": False,
        "no_future_tuning": True,
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    protocol_bytes = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (OUTPUT / "protocol_hash.txt").write_text(hashlib.sha256(protocol_bytes).hexdigest() + "\n", encoding="utf-8")
    dump_json(OUTPUT / "preflight.json", {"status": "PASS", "source_rows": 799, "future_rows": 1989,
        "source_direct": {key: direct[key] for key in ("BA", "AUC", "F1", "fixed", "broken", "net")},
        "future_labels_used_for_prediction": False, "output": str(OUTPUT)})
    print("SVM_DIRECT_C43_FUTURE_PREFLIGHT_PASS source=799 future=1989 source_direct_exact=true")


if __name__ == "__main__":
    main()
