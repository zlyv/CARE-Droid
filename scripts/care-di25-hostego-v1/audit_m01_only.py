#!/usr/bin/env python3
"""Post-hoc M0/M1 hard audits. Never trains or rewrites model predictions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
for path in (
    HERE,
    PROJECT / "scripts",
    PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
):
    sys.path.insert(0, str(path))

from experiment_contract import read_csv, write_csv
from host_ego_models import anomaly_classes
import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly

ROOT = Path("artifacts/care-di25-hostego-v1")
AUDIT = ROOT / "audits"
M0_PATH = ROOT / "source/M0_host_anomaly_bounded/source_predictions.csv"
M1_PATH = ROOT / "source/M1_hostego_anomaly_bounded/source_predictions.csv"
M0_DIR = M0_PATH.parent
M1_DIR = M1_PATH.parent
ASSET_ROOT = Path("artifacts/care-source-local-assets-v2")
EXPECTED_N = 799
EXPECTED_ROUTED = 200
EXPECTED = {
    "M0": {"BA": 0.9111058897243107, "AUC": 0.9410213032581454,
           "F1": 0.9086229086229086, "fixed": 87, "broken": 14, "net": 73},
    "M1": {"BA": 0.9148715538847118, "AUC": 0.9433521303258146,
           "F1": 0.9132653061224489, "fixed": 90, "broken": 14, "net": 76},
}
DI_CANDIDATES = [
    Path("artifacts/care-committee-decision-instability-router-source-v1/source_sample_scores.csv"),
    Path("artifacts/care-committee-uncertainty-router-source-v1/source_sample_scores.csv"),
    Path("artifacts/care-route-stage2-future-matrix-v1/preflight/source_sample_scores.csv"),
    Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/source_sample_scores.csv"),
]


def blocked(message: str) -> None:
    raise SystemExit("BLOCKED_BY_CONTRACT: " + message)


def normal(value: object) -> str:
    return str(value).strip().lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_binary(value: object, context: str) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        blocked(f"{context} is not binary: {value!r}")
    if result not in (0, 1):
        blocked(f"{context} is not binary: {value!r}")
    return result


def unique_index(rows: list[dict], name: str) -> dict[str, dict]:
    if len(rows) != EXPECTED_N:
        blocked(f"{name} rows={len(rows)}, expected 799")
    result: dict[str, dict] = {}
    for row in rows:
        sha = normal(row.get("sha256", row.get("sample_id", "")))
        if not sha or sha in result:
            blocked(f"{name} contains missing/duplicate SHA: {sha!r}")
        result[sha] = row
    return result


def method_probability_column(rows: list[dict], arm: str) -> str:
    columns = [key for key in rows[0] if key.startswith(f"DI25__") and key.endswith("__prob")]
    expected_token = "HostEgo" if arm == "M1" else "Host"
    columns = [key for key in columns if expected_token in key and (arm != "M0" or "HostEgo" not in key)]
    if len(columns) != 1:
        blocked(f"cannot identify {arm} bounded probability column: {columns}")
    return columns[0]


def route_column(rows: list[dict], arm: str) -> str:
    # A25__routed is an inherited loader compatibility name. Its values are
    # audited below against canonical frozen route_DI before being accepted.
    for column in ("route_DI25", "DI25__routed", "A25__routed"):
        if column in rows[0]:
            return column
    blocked(f"{arm} prediction has no routed column")


def find_canonical_di(cohort: set[str]) -> tuple[Path, dict[str, dict]]:
    diagnostics = []
    for path in DI_CANDIDATES:
        if not path.is_file():
            diagnostics.append(f"{path}: absent")
            continue
        try:
            rows = read_csv(path)
            index = unique_index(rows, str(path))
            if set(index) != cohort:
                diagnostics.append(f"{path}: cohort difference={len(set(index) ^ cohort)}")
                continue
            if "route_DI" not in rows[0]:
                diagnostics.append(f"{path}: route_DI absent (A25 is never accepted as substitute)")
                continue
            routes = [as_binary(row["route_DI"], f"{path}:route_DI") for row in rows]
            if sum(routes) != EXPECTED_ROUTED:
                diagnostics.append(f"{path}: routed={sum(routes)}, expected 200")
                continue
            return path, index
        except SystemExit:
            raise
        except Exception as exc:
            diagnostics.append(f"{path}: {type(exc).__name__}: {exc}")
    blocked("canonical frozen DI25 route not found:\n- " + "\n- ".join(diagnostics))


def compute_metrics(rows: list[dict], arm: str) -> tuple[dict, dict[str, dict]]:
    index = unique_index(rows, arm)
    prob_col = method_probability_column(rows, arm)
    y = np.asarray([int(float(row["label"])) for row in rows], dtype=int)
    prob = np.asarray([float(row[prob_col]) for row in rows], dtype=float)
    base = np.asarray([float(row["Global24__prob"]) for row in rows], dtype=float)
    pred = prob >= 0.5
    base_pred = base >= 0.5
    result = {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, prob)),
        "F1": float(f1_score(y, pred)),
        "fixed": int(np.sum((base_pred != y) & (pred == y))),
        "broken": int(np.sum((base_pred == y) & (pred != y))),
    }
    result["net"] = result["fixed"] - result["broken"]
    for key, expected in EXPECTED[arm].items():
        actual = result[key]
        if isinstance(expected, int):
            if actual != expected:
                blocked(f"{arm} result lock failed: {key}={actual}, expected {expected}")
        elif not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            blocked(f"{arm} result lock failed: {key}={actual:.17g}, expected {expected:.17g}")
    result["probability_column"] = prob_col
    return result, index


def exact_mcnemar(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def number_or_unavailable(value: object) -> object:
    if value is None or value == "":
        return "unavailable"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    return result if math.isfinite(result) else "unavailable"


def main() -> None:
    required = [
        M0_PATH, M1_PATH,
        M0_DIR / "fit_audit.csv", M1_DIR / "fit_audit.csv",
        M0_DIR / "architecture.json", M1_DIR / "architecture.json",
        AUDIT / "ego_host_diagnostic_M1.csv",
        AUDIT / "ego_apk_diagnostic_M1.csv",
        AUDIT / "ego_fallback_audit_M1.csv",
        ASSET_ROOT / "host_regions_2017.npz",
        ASSET_ROOT / "ego_1hop_regions_2017.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        blocked("audit-only preflight missing all of the following:\n- " + "\n- ".join(missing))
    AUDIT.mkdir(parents=True, exist_ok=True)

    protected_paths = [M0_PATH, M1_PATH]
    primary_summary = ROOT / "results/M0_M1_primary_comparison.csv"
    if primary_summary.is_file():
        protected_paths.append(primary_summary)
    hashes_before = {str(path): sha256_file(path) for path in protected_paths}

    # Result lock happens before any audit output is accepted.
    m0_rows, m1_rows = read_csv(M0_PATH), read_csv(M1_PATH)
    m0_metrics, m0 = compute_metrics(m0_rows, "M0")
    m1_metrics, m1 = compute_metrics(m1_rows, "M1")
    if set(m0) != set(m1):
        blocked(f"M0/M1 cohort difference={len(set(m0) ^ set(m1))}")
    cohort = set(m0)

    # 1. Exact frozen-DI25 route identity.
    di_path, di = find_canonical_di(cohort)
    m0_route_col, m1_route_col = route_column(m0_rows, "M0"), route_column(m1_rows, "M1")
    route_rows = []
    d0 = d1 = d01 = 0
    for row in m0_rows:
        sha = normal(row["sha256"])
        frozen = as_binary(di[sha]["route_DI"], "frozen route_DI")
        r0 = as_binary(m0[sha][m0_route_col], f"M0:{m0_route_col}")
        r1 = as_binary(m1[sha][m1_route_col], f"M1:{m1_route_col}")
        fold0, fold1 = int(float(m0[sha]["fold"])), int(float(m1[sha]["fold"]))
        if fold0 != fold1:
            blocked(f"fold differs for {sha}: M0={fold0}, M1={fold1}")
        route_rows.append({"sha256": sha, "fold": fold0, "route_frozen_DI25": frozen,
                           "route_M0": r0, "route_M1": r1})
        d0 += int(r0 != frozen); d1 += int(r1 != frozen); d01 += int(r0 != r1)
    route_summary = {
        "n": EXPECTED_N,
        "canonical_frozen_DI25_source": str(di_path.resolve()),
        "canonical_frozen_DI25_sha256": sha256_file(di_path),
        "canonical_route_column": "route_DI",
        "M0_saved_route_column": m0_route_col,
        "M1_saved_route_column": m1_route_col,
        "frozen_DI25_routed_n": sum(row["route_frozen_DI25"] for row in route_rows),
        "M0_routed_n": sum(row["route_M0"] for row in route_rows),
        "M1_routed_n": sum(row["route_M1"] for row in route_rows),
        "M0_vs_DI25_difference_count": d0,
        "M1_vs_DI25_difference_count": d1,
        "M0_vs_M1_difference_count": d01,
    }
    write_csv(AUDIT / "route_integrity_M01.csv", route_rows)
    json_write(AUDIT / "route_integrity_M01_summary.json", route_summary)
    if d0 or d1 or d01:
        blocked("route identity failed: " + json.dumps(route_summary, ensure_ascii=False))
    explanation = [
        "M0/M1 route identity explanation",
        "",
        f"Canonical route source: {di_path.resolve()}",
        "Canonical route field: route_DI (not an A25 field).",
        f"M0 saved route field: {m0_route_col}; M1 saved route field: {m1_route_col}.",
        "The saved field may retain the inherited name A25__routed because the frozen v4 loader expects that compatibility field.",
        "Its 799 row values are nevertheless proven identical by SHA to canonical frozen route_DI; all three difference counts are zero.",
        "",
        "Global24__A25__JointS7_ordinary and Global24__A25__CARE_DualProto_ES_FixKeep_reference",
        "are inherited historical/reference summary rows emitted by the legacy evaluator.",
        "They are not route generators, are not M0/M1 model names, and do not participate in M0/M1 routing.",
        "M0/M1 routing is determined only by the audited canonical frozen DI25 membership above.",
    ]
    (AUDIT / "reference_row_explanation_M01.txt").write_text("\n".join(explanation) + "\n", encoding="utf-8")

    # 2. Instantiate the exact formal classes without fitting and count params.
    cfg = SimpleNamespace(max_delta=2.25, graph_dim=32, embedding_dim=24,
                          hidden_dim=48, dropout=0.20, graph_heads=4, graph_layers=2)
    M0Class, M1Class = anomaly_classes(anomaly, v4)
    model0 = M0Class(24, "RGCondAttn", cfg)
    model1 = M1Class(24, "RGCondAttn", cfg)
    p0 = sum(parameter.numel() for parameter in model0.parameters())
    p1 = sum(parameter.numel() for parameter in model1.parameters())
    fit0 = sorted({int(float(row["parameter_n"])) for row in read_csv(M0_DIR / "fit_audit.csv")})
    fit1 = sorted({int(float(row["parameter_n"])) for row in read_csv(M1_DIR / "fit_audit.csv")})
    extra_names = sorted(set(dict(model1.named_parameters())) - set(dict(model0.named_parameters())))
    extra_rows = []
    named1 = dict(model1.named_parameters())
    for name in extra_names:
        tensor = named1[name]
        extra_rows.append({"name": name, "shape": "x".join(str(v) for v in tensor.shape),
                           "numel": tensor.numel()})
    write_csv(AUDIT / "parameter_tensors_M1_extra.csv", extra_rows)
    parameter_rows = [
        {"arm": "M0", "parameter_count": p0, "extra_vs_M0": 0, "relative_increase": 0.0},
        {"arm": "M1", "parameter_count": p1, "extra_vs_M0": p1-p0,
         "relative_increase": (p1-p0)/p0},
    ]
    write_csv(AUDIT / "parameter_count_M01.csv", parameter_rows)
    parameter_ok = p0 == 33388 and p1 == 34613 and p1-p0 == 1225 and fit0 == [p0] and fit1 == [p1]
    if not parameter_ok:
        json_write(AUDIT / "parameter_count_M01_mismatch.json", {
            "instantiated_M0": p0, "instantiated_M1": p1,
            "saved_fit_audit_M0": fit0, "saved_fit_audit_M1": fit1,
            "expected_M0": 33388, "expected_M1": 34613, "extra_tensors": extra_rows,
            "explanation": "Counts differ between the formal instantiated classes, saved fit audit, or predeclared expectation; no code was modified.",
        })
        blocked("parameter-count hard audit failed; see parameter_count_M01_mismatch.json")

    # 3. Ego usage from saved diagnostics, with duplicate/self checks from the
    # same frozen directional cache consumed by M1.
    host_diag = read_csv(AUDIT / "ego_host_diagnostic_M1.csv")
    apk_diag = unique_index(read_csv(AUDIT / "ego_apk_diagnostic_M1.csv"), "ego APK diagnostic M1")
    if set(apk_diag) != cohort:
        blocked(f"Ego APK diagnostic cohort difference={len(set(apk_diag) ^ cohort)}")
    total_hosts_from_diagnostic = sum(int(float(row["host_count"])) for row in apk_diag.values())
    hosts_with_ego_from_diagnostic = sum(int(float(row["nontrivial_ego_count"])) for row in apk_diag.values())
    nontrivial_host_diag = [row for row in host_diag if as_binary(row["ego_used"], "host diagnostic ego_used") == 1]
    if len(nontrivial_host_diag) != hosts_with_ego_from_diagnostic:
        blocked(
            "saved Host/APK Ego diagnostics disagree: "
            f"host rows with Ego={len(nontrivial_host_diag)}, APK aggregate={hosts_with_ego_from_diagnostic}"
        )
    with np.load(ASSET_ROOT / "host_regions_2017.npz", allow_pickle=False) as host_npz, \
         np.load(ASSET_ROOT / "ego_1hop_regions_2017.npz", allow_pickle=False) as ego_npz:
        asset_ids = [normal(value) for value in host_npz["sha256"].astype(str).tolist()]
        if set(asset_ids) != cohort or len(set(asset_ids)) != EXPECTED_N:
            blocked("Host/Ego asset cohort is not the exact M0/M1 cohort")
        real_hosts = np.asarray(host_npz["real_host_count"], dtype=int)
        ego_offsets = np.asarray(ego_npz["ego_offsets"], dtype=int)
        status = np.asarray(ego_npz["ego_status"], dtype=int)
        in_offsets = np.asarray(ego_npz["incoming_offsets"], dtype=int)
        out_offsets = np.asarray(ego_npz["outgoing_offsets"], dtype=int)
        in_pos = np.asarray(ego_npz["incoming_host_positions"], dtype=int)
        out_pos = np.asarray(ego_npz["outgoing_host_positions"], dtype=int)
        total_hosts_cache = int(real_hosts.sum())
        hosts_with_ego_cache = int(np.sum(status == 1))
        if total_hosts_cache != total_hosts_from_diagnostic or hosts_with_ego_cache != hosts_with_ego_from_diagnostic:
            blocked(
                "saved Ego diagnostics disagree with frozen M1 cache: "
                f"hosts {total_hosts_from_diagnostic}/{total_hosts_cache}, "
                f"nontrivial {hosts_with_ego_from_diagnostic}/{hosts_with_ego_cache}"
            )
        total_hosts = total_hosts_from_diagnostic
        hosts_with_ego = hosts_with_ego_from_diagnostic
        hosts_without_ego = total_hosts - hosts_with_ego
        duplicate_neighbors = 0
        self_neighbors = 0
        distinct_neighbor_counts = []
        in_counts, out_counts = [], []
        for apk_i in range(EXPECTED_N):
            begin, end = int(ego_offsets[apk_i]), int(ego_offsets[apk_i+1])
            if end - begin != int(real_hosts[apk_i]):
                blocked(f"Ego offset/real Host mismatch at asset row {apk_i}")
            for center in range(begin, end):
                incoming = in_pos[in_offsets[center]:in_offsets[center+1]].astype(int).tolist()
                outgoing = out_pos[out_offsets[center]:out_offsets[center+1]].astype(int).tolist()
                duplicate_neighbors += len(incoming)-len(set(incoming)) + len(outgoing)-len(set(outgoing))
                self_neighbors += incoming.count(center) + outgoing.count(center)
                if int(status[center]) == 1:
                    distinct_neighbor_counts.append(len(set(incoming) | set(outgoing)))
                    in_counts.append(len(incoming)); out_counts.append(len(outgoing))
    # Verify saved per-APK diagnostic route against the canonical route.
    for sha in cohort:
        if as_binary(apk_diag[sha]["routed"], "ego diagnostic routed") != as_binary(di[sha]["route_DI"], "DI route"):
            blocked(f"Ego diagnostic route differs from DI25 for {sha}")
    routed_sha = [sha for sha in cohort if as_binary(di[sha]["route_DI"], "DI route") == 1]
    routed_host_counts = np.asarray([int(float(apk_diag[sha]["host_count"])) for sha in routed_sha], dtype=int)
    routed_ego_counts = np.asarray([int(float(apk_diag[sha]["nontrivial_ego_count"])) for sha in routed_sha], dtype=int)
    ego_usage = {
        "total_apks": EXPECTED_N,
        "routed_apks": len(routed_sha),
        "total_hosts": total_hosts,
        "hosts_with_nontrivial_ego": hosts_with_ego,
        "hosts_without_nontrivial_ego": hosts_without_ego,
        "host_ego_usage_ratio": hosts_with_ego/total_hosts if total_hosts else 0.0,
        "routed_apks_with_any_nontrivial_ego": int(np.sum(routed_ego_counts > 0)),
        "routed_apks_without_nontrivial_ego": int(np.sum(routed_ego_counts == 0)),
        "routed_apk_ego_coverage": float(np.mean(routed_ego_counts > 0)),
        "mean_hosts_per_routed_apk": float(np.mean(routed_host_counts)),
        "median_hosts_per_routed_apk": float(np.median(routed_host_counts)),
        "mean_nontrivial_ego_hosts_per_routed_apk": float(np.mean(routed_ego_counts)),
        "mean_neighbor_count_among_nontrivial_ego": float(np.mean([
            float(row["neighbor_count"]) for row in nontrivial_host_diag
        ])),
        "mean_in_neighbor_count": float(np.mean([
            float(row["incoming_neighbor_count_if_available"]) for row in nontrivial_host_diag
        ])),
        "mean_out_neighbor_count": float(np.mean([
            float(row["outgoing_neighbor_count_if_available"]) for row in nontrivial_host_diag
        ])),
        "cache_crosscheck_mean_distinct_neighbor_count": float(np.mean(distinct_neighbor_counts)),
        "duplicate_neighbor_count": duplicate_neighbors,
        "center_self_neighbor_count": self_neighbors,
        "duplicate_definition": "duplicates within each directional list; a reciprocal neighbor appearing once incoming and once outgoing is not a duplicate",
        "source_host_diagnostic_rows": len(host_diag),
    }
    write_csv(AUDIT / "ego_usage_M1.csv", [ego_usage])
    json_write(AUDIT / "ego_usage_M1.json", ego_usage)
    if duplicate_neighbors or self_neighbors:
        print("AUDIT_WARNING: duplicate/self neighbor count is nonzero; reported without modification", flush=True)

    # 4. Exact no-Ego fallback summary from the already-saved M1 audit.
    fallback = read_csv(AUDIT / "ego_fallback_audit_M1.csv")
    no_ego = [row for row in fallback if as_binary(row["has_nontrivial_ego"], "fallback has_nontrivial_ego") == 0]
    diffs = np.asarray([float(row["max_abs_hostonly_vs_hostego_feature_diff"]) for row in no_ego], dtype=float)
    if not np.all(np.isfinite(diffs)):
        blocked("no-Ego fallback contains unavailable/non-finite differences")
    fallback_summary = {
        "fallback_rows_total": len(fallback),
        "no_ego_host_n": len(no_ego),
        "max_abs_diff": float(np.max(diffs)) if len(diffs) else 0.0,
        "mean_abs_diff": float(np.mean(diffs)) if len(diffs) else 0.0,
        "nonzero_diff_n": int(np.sum(diffs != 0.0)),
    }
    write_csv(AUDIT / "ego_fallback_summary_M1.csv", [fallback_summary])
    json_write(AUDIT / "ego_fallback_summary_M1.json", fallback_summary)
    if fallback_summary["max_abs_diff"] != 0.0 or fallback_summary["nonzero_diff_n"] != 0:
        blocked("fallback identity hard audit failed")

    # 5. Efficiency: only previously recorded values are reported.
    efficiency = []
    for arm, directory, params in (("M0", M0_DIR, p0), ("M1", M1_DIR, p1)):
        architecture = json.loads((directory / "architecture.json").read_text(encoding="utf-8"))
        inference = number_or_unavailable(architecture.get("model_inference_seconds"))
        per_apk = number_or_unavailable(architecture.get("model_inference_seconds_per_model_apk"))
        efficiency.append({
            "arm": arm,
            "routed_apks": EXPECTED_ROUTED,
            "total_hosts": total_hosts,
            "total_hosts_in_routed_apks": int(np.sum(routed_host_counts)),
            "hosts_with_ego": hosts_with_ego if arm == "M1" else 0,
            "hosts_with_ego_in_routed_apks": int(np.sum(routed_ego_counts)) if arm == "M1" else 0,
            "model_parameter_count": params,
            "stage2_training_seconds_if_available": "unavailable",
            "stage2_inference_seconds_if_available": inference,
            "seconds_per_routed_apk_if_available": per_apk,
            "original_wall_seconds_including_training_if_available": number_or_unavailable(architecture.get("wall_seconds_including_training")),
            "recorded_model_inference_apk_calls": architecture.get("model_inference_apk_calls", "unavailable"),
            "timing_scope_note": "training-only time was not recorded; inference values are copied from the original run and no timing rerun was performed",
        })
    write_csv(AUDIT / "efficiency_M01.csv", efficiency)

    # 6. Paired M0-vs-M1 bounded prediction changes over all 799 APKs.
    p0_col, p1_col = m0_metrics["probability_column"], m1_metrics["probability_column"]
    paired = []
    counts = Counter()
    for row in m0_rows:
        sha = normal(row["sha256"])
        label = int(float(m0[sha]["label"]))
        if label != int(float(m1[sha]["label"])):
            blocked(f"label differs for {sha}")
        base_correct = int((float(m0[sha]["Global24__prob"]) >= 0.5) == bool(label))
        prob0, prob1 = float(m0[sha][p0_col]), float(m1[sha][p1_col])
        correct0, correct1 = int((prob0 >= 0.5) == bool(label)), int((prob1 >= 0.5) == bool(label))
        if correct0 and correct1:
            status_name = "same_correct"
        elif not correct0 and not correct1:
            status_name = "same_wrong"
        elif not correct0 and correct1:
            status_name = "M0_wrong_M1_correct"
        else:
            status_name = "M0_correct_M1_wrong"
        counts[status_name] += 1
        paired.append({"sha256": sha, "label": label, "Global24_correct": base_correct,
                       "M0_correct": correct0, "M1_correct": correct1,
                       "M0_prob": prob0, "M1_prob": prob1,
                       "M0_to_M1_status": status_name})
    n01 = counts["M0_wrong_M1_correct"]
    n10 = counts["M0_correct_M1_wrong"]
    paired_summary = {
        "n": EXPECTED_N,
        "same_correct": counts["same_correct"],
        "same_wrong": counts["same_wrong"],
        "M0_wrong_M1_correct": n01,
        "M0_correct_M1_wrong": n10,
        "n01_definition": "M0 wrong, M1 correct",
        "n10_definition": "M0 correct, M1 wrong",
        "n01": n01, "n10": n10,
        "exact_McNemar_two_sided_p_value": exact_mcnemar(n01, n10),
    }
    write_csv(AUDIT / "M0_M1_paired_changes.csv", paired)
    write_csv(AUDIT / "M0_M1_paired_summary.csv", [paired_summary])
    json_write(AUDIT / "M0_M1_paired_summary.json", paired_summary)

    # 7. Prove protected inputs were not changed by this process.
    hashes_after = {str(path): sha256_file(path) for path in protected_paths}
    if hashes_before != hashes_after:
        blocked("an existing M0/M1 result artifact changed during audit")
    result_lock = {
        "passed": True,
        "M0_Bounded": {key: value for key, value in m0_metrics.items() if key != "probability_column"},
        "M1_Bounded": {key: value for key, value in m1_metrics.items() if key != "probability_column"},
        "protected_sha256_before": hashes_before,
        "protected_sha256_after": hashes_after,
        "existing_results_modified": False,
        "training_or_inference_rerun": False,
        "M2_M3_started": False,
    }
    json_write(AUDIT / "result_lock_M01.json", result_lock)

    report = [
        "CARE-Droid M0/M1 post-hoc hard audit", "",
        "AUDIT_M01_PASS", "No model fitting, prediction inference, routing change, or M2/M3 execution occurred.", "",
        "Route identity:",
        *(f"  {key}: {value}" for key, value in route_summary.items()), "",
        "Parameter count:",
        f"  M0={p0}; M1={p1}; extra={p1-p0}; relative={(p1-p0)/p0:.8f}", "",
        "Ego usage:",
        *(f"  {key}: {value}" for key, value in ego_usage.items() if key != "duplicate_definition"), "",
        "Fallback:",
        *(f"  {key}: {value}" for key, value in fallback_summary.items()), "",
        "Paired M0 vs M1:",
        *(f"  {key}: {value}" for key, value in paired_summary.items()), "",
        "Locked metrics:",
        f"  M0: {m0_metrics}", f"  M1: {m1_metrics}",
    ]
    (AUDIT / "AUDIT_M01_REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
