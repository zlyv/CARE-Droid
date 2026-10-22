#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/care-litelocal-p31-c43-evidence-audit-v1")
P31 = Path("artifacts/care-litelocal-p31-c43-source-v1")
DIRECT = Path("artifacts/care-litelocal-p31-c43-direct-source-v1")
FAIL = Path("artifacts/care-litelocal-p31-c43-failure-audit-v1")
CACHE = Path("artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz")
BOUNDED = "P31Mean__DI25__C43RelationQuery__Bounded"
DIRECT_METHOD = "P31Mean__DI25__C43RelationQuery__Direct"
DIAGNOSTICS = (
    "critical_score", "mean_instance_score", "attention_entropy", "max_attention",
    "context_entropy", "mean_external_distance", "mean_internal_distance",
    "mean_shared_commonness", "max_local_residual", "graph_pool_entropy",
)


def read(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def norm(value):
    return str(value or "").strip().casefold()


def percentile(value, control):
    control = np.asarray(control, float)
    if len(control) == 0:
        return float("nan")
    return float((np.sum(control < value) + .5 * np.sum(control == value)) / len(control))


def expert_probability_column(rows, preferred, kind):
    fields = list(rows[0])
    exact = preferred + "__expert_prob"
    if exact in fields:
        return exact
    candidates = [field for field in fields
                  if field.endswith("__expert_prob") and kind.casefold() in field.casefold()]
    if len(candidates) != 1:
        raise RuntimeError("BLOCKED_BY_CONTRACT: cannot identify " + kind +
                           " expert probability column: " + repr(candidates))
    return candidates[0]


def load_compact():
    with np.load(CACHE, allow_pickle=False) as saved:
        ids = np.asarray([norm(value) for value in saved["sha256"]])
        offsets_key = next((key for key in saved.files
                            if "offset" in key.casefold() and saved[key].shape == (800,)), None)
        if offsets_key is None:
            raise RuntimeError("BLOCKED_BY_CONTRACT: no 800-row compact offsets")
        offsets = np.asarray(saved[offsets_key], int)
        candidates = []
        for key in saved.files:
            value = saved[key]
            if value.ndim == 2 and value.shape[0] == int(offsets[-1]) and value.shape[1] >= 74:
                candidates.append((key, value.shape[1]))
        if not candidates:
            raise RuntimeError("BLOCKED_BY_CONTRACT: no compact Host matrix with >=74 columns")
        candidates.sort(key=lambda item: (0 if "host" in item[0].casefold() else 1,
                                          abs(item[1] - 74), item[0]))
        values_key = candidates[0][0]
        matrix = np.asarray(saved[values_key], float)
        if matrix.shape[1] == 105:
            # Frozen cache layout: original Host74 followed by Relation31.
            relation_slice = (74, 105)
        elif matrix.shape[1] == 74:
            # Compact materialized layout: Compact43 followed by Relation31.
            relation_slice = (43, 74)
        else:
            raise RuntimeError("BLOCKED_BY_CONTRACT: unsupported Host/Relation layout " +
                               repr(matrix.shape))
        real_key = next((key for key in saved.files
                         if key.casefold() == "real_host_count" and saved[key].shape == (799,)), None)
        real = np.asarray(saved[real_key], int) if real_key else np.diff(offsets)
    result = {}
    for index, sha in enumerate(ids):
        begin = int(offsets[index]); count = int(real[index]); end = begin + count
        host = matrix[begin:end]
        relation = (host[:, relation_slice[0]:relation_slice[1]] if len(host)
                    else np.zeros((0, 31), float))
        nonzero = np.any(np.abs(relation) > 1e-12, axis=1) if len(relation) else np.zeros(0, bool)
        result[sha] = {
            "real_host_count": count,
            "relation_nonzero_host_count": int(nonzero.sum()),
            "relation_nonzero_host_ratio": float(nonzero.mean()) if len(nonzero) else 0.0,
            "relation_mean_l1": float(np.mean(np.sum(np.abs(relation), axis=1))) if len(relation) else 0.0,
            "relation_max_l1": float(np.max(np.sum(np.abs(relation), axis=1))) if len(relation) else 0.0,
        }
    return result, offsets_key, values_key, list(matrix.shape), list(relation_slice)


def group_summary(rows, features):
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"]].append(row)
    output = []
    for cohort, values in sorted(groups.items()):
        item = {"cohort": cohort, "n": len(values)}
        for feature in features:
            vector = np.asarray([float(row[feature]) for row in values], float)
            item[feature + "__mean"] = float(np.mean(vector))
            item[feature + "__median"] = float(np.median(vector))
        output.append(item)
    return output


def main():
    reference = sorted(read(P31 / "preflight/p31_reference/source_predictions.csv"),
                       key=lambda row: norm(row["sha256"]))
    bounded_rows = sorted(read(P31 / "arms/P31_C43_RQ/source_predictions.csv"),
                          key=lambda row: norm(row["sha256"]))
    direct_rows = sorted(read(DIRECT / "arms/P31_C43_DIRECT/source_predictions.csv"),
                         key=lambda row: norm(row["sha256"]))
    failure = {norm(row["sha256"]): row
               for row in read(FAIL / "routed_errors_mechanism.csv")}
    compact, offsets_key, values_key, matrix_shape, relation_slice = load_compact()
    bounded_expert = expert_probability_column(bounded_rows, BOUNDED, "Bounded")
    direct_expert = expert_probability_column(direct_rows, DIRECT_METHOD, "Direct")
    rows = []
    for base_row, bounded_row, direct_row in zip(reference, bounded_rows, direct_rows):
        if len({norm(base_row["sha256"]), norm(bounded_row["sha256"]),
                norm(direct_row["sha256"])}) != 1:
            raise RuntimeError("BLOCKED_BY_CONTRACT: prediction cohorts do not align")
        if int(float(base_row["P31_DI25__routed"])) != 1:
            continue
        sha = norm(base_row["sha256"])
        label = int(float(base_row["label"]))
        fold = int(float(base_row["fold"]))
        base = float(base_row["P31_MEAN_COMMITTEE__prob"])
        prediction = int(base >= .5)
        correct = int(prediction == label)
        if correct:
            cohort = "P31_CORRECT_PRED_BENIGN" if prediction == 0 else "P31_CORRECT_PRED_MALWARE"
            primary = "P31_CORRECT"
        else:
            failed = failure[sha]
            primary = failed.get("primary_failure_type") or failed.get("mechanism")
            if not primary:
                raise RuntimeError("BLOCKED_BY_CONTRACT: failure mechanism column missing")
            cohort = ("FN_" if label == 1 else "FP_") + primary
        item = {
            "sha256": sha, "fold": fold, "label": label,
            "P31_prediction": prediction, "P31_probability": base,
            "P31_correct": correct, "cohort": cohort,
            "mechanism": primary,
            "bounded_expert_probability": float(bounded_row[bounded_expert]),
            "direct_expert_probability": float(direct_row[direct_expert]),
            **compact[sha],
        }
        for name in DIAGNOSTICS:
            item[name] = float(bounded_row[BOUNDED + "__" + name])
        rows.append(item)
    if len(rows) != 200:
        raise RuntimeError("BLOCKED_BY_CONTRACT: routed accounting failed")
    # Fold- and predicted-class-matched correct controls. These percentiles ask
    # whether an error looks like the correct APKs that received the same P31 prediction.
    for row in rows:
        controls = [candidate for candidate in rows
                    if candidate["fold"] == row["fold"]
                    and candidate["P31_prediction"] == row["P31_prediction"]
                    and candidate["P31_correct"] == 1]
        for name in DIAGNOSTICS + ("real_host_count", "relation_nonzero_host_ratio",
                                   "relation_mean_l1", "relation_max_l1"):
            row[name + "__percentile_vs_same_prediction_correct"] = percentile(
                float(row[name]), [float(candidate[name]) for candidate in controls]
            )
    features = DIAGNOSTICS + ("real_host_count", "relation_nonzero_host_count",
                              "relation_nonzero_host_ratio", "relation_mean_l1",
                              "relation_max_l1", "direct_expert_probability")
    summary = group_summary(rows, features)
    percentile_features = tuple(name + "__percentile_vs_same_prediction_correct"
                                for name in DIAGNOSTICS +
                                ("real_host_count", "relation_nonzero_host_ratio",
                                 "relation_mean_l1", "relation_max_l1"))
    percentile_summary = group_summary(rows, percentile_features)
    errors = [row for row in rows if not row["P31_correct"]]
    wrong = [row for row in errors if row["mechanism"] == "WRONG_OR_ZERO_DIRECTION"]
    fixed = [row for row in errors if row["mechanism"] == "FIXED_BOUNDED"]
    weak = [row for row in errors if row["mechanism"] == "RIGHT_DIRECTION_TOO_WEAK"]
    fn_wrong = [row for row in wrong if row["label"] == 1]
    fp_wrong = [row for row in wrong if row["label"] == 0]
    # Direction-consistency is descriptive: FN direct p<.5 is benign-side;
    # FP direct p>=.5 is malware-side. Together they indicate confirmation,
    # not a global benign bias.
    fn_direct_benign = sum(row["direct_expert_probability"] < .5 for row in fn_wrong)
    fp_direct_malware = sum(row["direct_expert_probability"] >= .5 for row in fp_wrong)
    wrong_zero_relation = sum(row["relation_nonzero_host_count"] == 0 for row in wrong)
    fixed_zero_relation = sum(row["relation_nonzero_host_count"] == 0 for row in fixed)
    output = ROOT
    write(output / "routed_apk_evidence.csv", rows)
    write(output / "evidence_group_summary.csv", summary)
    write(output / "matched_percentile_group_summary.csv", percentile_summary)
    audit = {
        "source_rows": 799, "routed_rows": 200, "routed_errors": 46,
        "fixed": len(fixed), "right_direction_too_weak": len(weak),
        "wrong_or_zero_direction": len(wrong),
        "wrong_FN": len(fn_wrong), "wrong_FP": len(fp_wrong),
        "wrong_FN_direct_still_benign_side": fn_direct_benign,
        "wrong_FP_direct_still_malware_side": fp_direct_malware,
        "wrong_zero_relation_APKs": wrong_zero_relation,
        "fixed_zero_relation_APKs": fixed_zero_relation,
        "compact_offsets_key": offsets_key,
        "compact_values_key": values_key,
        "compact_values_shape": matrix_shape,
        "relation_slice": relation_slice,
        "bounded_expert_probability_column": bounded_expert,
        "direct_expert_probability_column": direct_expert,
        "training_invoked": False, "inference_invoked": False,
        "route_modified": False, "future_rows_loaded": 0, "ego_consumed": False,
    }
    (output / "hard_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "CARE-Droid P31-C43 extracted-evidence audit v1", "",
        "Read-only strict-2017 audit; no training, inference, route change, future row, or Ego tensor.",
        "Absolute background quantities are not class labels. Comparisons use fold- and P31-prediction-matched correct controls.", "",
        f"wrong_or_zero_direction={len(wrong)} fixed={len(fixed)} too_weak={len(weak)}",
        f"wrong_FN={len(fn_wrong)}; Direct remains benign-side for {fn_direct_benign}/{len(fn_wrong)}",
        f"wrong_FP={len(fp_wrong)}; Direct remains malware-side for {fp_direct_malware}/{len(fp_wrong)}",
        f"wrong_direction with zero Relation31 coverage={wrong_zero_relation}/{len(wrong)}",
        f"fixed with zero Relation31 coverage={fixed_zero_relation}/{len(fixed)}", "",
        "Interpretation: FN benign-side and FP malware-side together indicate confirmation of P31, not that all extracted evidence is benign.",
        "Inspect evidence_group_summary.csv and matched-control percentiles before attributing failure to Host, background, relation, or attention evidence.",
    ]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P31_C43_EVIDENCE_AUDIT_DONE")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
