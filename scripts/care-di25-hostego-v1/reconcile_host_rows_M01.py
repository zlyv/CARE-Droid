#!/usr/bin/env python3
"""Reconcile the two deterministic no-Host placeholders in saved M1 audits."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/care-di25-hostego-v1")
AUDIT = ROOT / "audits"
ASSETS = Path("artifacts/care-source-local-assets-v2")
M0_PRED = ROOT / "source/M0_host_anomaly_bounded/source_predictions.csv"
M1_PRED = ROOT / "source/M1_hostego_anomaly_bounded/source_predictions.csv"
HOST_DIAG = AUDIT / "ego_host_diagnostic_M1.csv"
FALLBACK = AUDIT / "ego_fallback_audit_M1.csv"
EGO_USAGE = AUDIT / "ego_usage_M1.json"
OUTPUT_CSV = AUDIT / "host_row_count_reconciliation.csv"
OUTPUT_TXT = AUDIT / "host_row_count_reconciliation.txt"
ANNOTATED_FALLBACK = AUDIT / "ego_fallback_audit_M1_reconciled.csv"


def blocked(message: str) -> None:
    raise SystemExit("BLOCKED_BY_CONTRACT: " + message)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normal(value: object) -> str:
    return str(value).strip().lower()


def binary(value: object, context: str) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        blocked(f"{context} is not binary: {value!r}")
    if result not in (0, 1):
        blocked(f"{context} is not binary: {value!r}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[normal(row["sha256"])].append(row)
    return result


def main() -> None:
    required = [
        M0_PRED, M1_PRED, HOST_DIAG, FALLBACK, EGO_USAGE,
        ASSETS / "host_regions_2017.npz", ASSETS / "ego_1hop_regions_2017.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        blocked("reconciliation preflight missing:\n- " + "\n- ".join(missing))

    protected = {str(path): sha256_file(path) for path in (M0_PRED, M1_PRED)}
    host_rows = read_csv(HOST_DIAG)
    fallback_rows = read_csv(FALLBACK)
    host_by_sha, fallback_by_sha = group(host_rows), group(fallback_rows)

    with np.load(ASSETS / "host_regions_2017.npz", allow_pickle=False) as host, \
         np.load(ASSETS / "ego_1hop_regions_2017.npz", allow_pickle=False) as ego:
        ids = [normal(value) for value in host["sha256"].astype(str).tolist()]
        ego_ids = [normal(value) for value in ego["sha256"].astype(str).tolist()]
        real_count = np.asarray(host["real_host_count"], dtype=int)
        host_offsets = np.asarray(host["host_offsets"], dtype=int)
        ego_offsets = np.asarray(ego["ego_offsets"], dtype=int)
        if len(ids) != 799 or len(set(ids)) != 799 or ids != ego_ids:
            blocked("Host/Ego cache lineage is not the same exact 799-row cohort")
        zero_indices = np.flatnonzero(real_count == 0).astype(int).tolist()
        reconciliation = []
        errors = []
        for index, sha in enumerate(ids):
            real_n = int(real_count[index])
            value_n = int(host_offsets[index+1] - host_offsets[index])
            ego_n = int(ego_offsets[index+1] - ego_offsets[index])
            diag = host_by_sha.get(sha, [])
            fallback = fallback_by_sha.get(sha, [])
            expected_rows = real_n if real_n else 1
            if len(diag) != expected_rows:
                errors.append(f"{sha}: diagnostic_rows={len(diag)}, expected={expected_rows}")
            if len(fallback) != expected_rows:
                errors.append(f"{sha}: fallback_rows={len(fallback)}, expected={expected_rows}")
            if real_n:
                if value_n != real_n or ego_n != real_n:
                    errors.append(f"{sha}: real={real_n}, host_values={value_n}, ego_rows={ego_n}")
                continue

            # The no-Host contract deliberately creates exactly one model-input
            # row while retaining zero real Host and zero Ego records.
            host_id = str(diag[0].get("host_index_or_id", "")) if len(diag) == 1 else ""
            fallback_id = str(fallback[0].get("host_id", "")) if len(fallback) == 1 else ""
            fold = str(diag[0].get("fold", "")) if len(diag) == 1 else ""
            ego_used = binary(diag[0].get("ego_used"), f"{sha}:ego_used") if len(diag) == 1 else -1
            neighbor_n = int(float(diag[0].get("neighbor_count", -1))) if len(diag) == 1 else -1
            fallback_nontrivial = binary(fallback[0].get("has_nontrivial_ego"), f"{sha}:fallback") if len(fallback) == 1 else -1
            fallback_diff = float(fallback[0].get("max_abs_hostonly_vs_hostego_feature_diff", "nan")) if len(fallback) == 1 else float("nan")
            placeholder = int(
                real_n == 0 and value_n == 1 and ego_n == 0 and len(diag) == 1 and len(fallback) == 1
                and host_id == "__missing_host__" and fallback_id == "__missing_host__"
                and ego_used == 0 and neighbor_n == 0 and fallback_nontrivial == 0
                and np.isfinite(fallback_diff) and fallback_diff == 0.0
            )
            reconciliation.append({
                "sha256": sha,
                "fold": fold,
                "host_id": host_id,
                "fallback_host_id": fallback_id,
                "real_host_count": real_n,
                "host_value_row_count": value_n,
                "ego_cache_row_count": ego_n,
                "diagnostic_row_count": len(diag),
                "fallback_row_count": len(fallback),
                "ego_used": ego_used,
                "neighbor_count": neighbor_n,
                "fallback_has_nontrivial_ego": fallback_nontrivial,
                "fallback_max_abs_diff": fallback_diff,
                "placeholder": placeholder,
                "classification": "deterministic_no_real_host_placeholder" if placeholder else "unresolved_non_real_host_row",
            })
            if not placeholder:
                errors.append(f"{sha}: zero-real-Host row does not satisfy deterministic placeholder contract")

    real_host_rows = int(real_count.sum())
    placeholder_rows = len(reconciliation)
    total_diagnostic_rows = len(host_rows)
    total_fallback_rows = len(fallback_rows)
    no_ego_total = sum(binary(row["has_nontrivial_ego"], "fallback has_nontrivial_ego") == 0 for row in fallback_rows)
    no_ego_real = no_ego_total - placeholder_rows

    if len(zero_indices) != 2:
        errors.append(f"zero-real-Host APKs={len(zero_indices)}, expected 2")
    if placeholder_rows != 2:
        errors.append(f"placeholder_rows={placeholder_rows}, expected 2")
    if real_host_rows != 95314:
        errors.append(f"real_host_rows={real_host_rows}, expected 95314")
    if total_diagnostic_rows != real_host_rows + placeholder_rows:
        errors.append(
            f"diagnostic rows do not reconcile: {total_diagnostic_rows} != {real_host_rows}+{placeholder_rows}"
        )
    if total_fallback_rows != real_host_rows + placeholder_rows:
        errors.append(
            f"fallback rows do not reconcile: {total_fallback_rows} != {real_host_rows}+{placeholder_rows}"
        )
    if no_ego_real != 70568 or no_ego_total != 70570:
        errors.append(f"no-Ego counts do not reconcile: real={no_ego_real}, with placeholders={no_ego_total}")

    usage = json.loads(EGO_USAGE.read_text(encoding="utf-8"))
    expected_ratio = (real_host_rows - no_ego_real) / real_host_rows
    if int(usage["total_hosts"]) != real_host_rows:
        errors.append("ego_usage total_hosts includes non-real rows")
    if int(usage["hosts_without_nontrivial_ego"]) != no_ego_real:
        errors.append("ego_usage hosts_without_nontrivial_ego includes placeholders")
    if not np.isclose(float(usage["host_ego_usage_ratio"]), expected_ratio, rtol=0.0, atol=1e-15):
        errors.append("ego_usage ratio denominator/numerator is not real-Host-only")

    write_csv(OUTPUT_CSV, reconciliation)
    # Preserve the original fallback audit byte-for-byte and emit an annotated
    # derived copy so placeholder membership is explicit.
    placeholder_keys = {(row["sha256"], row["host_id"]) for row in reconciliation if row["placeholder"] == 1}
    annotated = []
    for row in fallback_rows:
        item = dict(row)
        item["placeholder"] = int((normal(row["sha256"]), str(row["host_id"])) in placeholder_keys)
        annotated.append(item)
    write_csv(ANNOTATED_FALLBACK, annotated)

    lines = [
        "CARE-Droid M0/M1 Host-row count reconciliation", "",
        f"real_host_rows: {real_host_rows}",
        f"placeholder_rows: {placeholder_rows}",
        f"total_diagnostic_rows: {total_diagnostic_rows}",
        f"total_fallback_rows: {total_fallback_rows}",
        f"real_hosts_without_nontrivial_ego: {no_ego_real}",
        f"fallback_no_ego_rows_including_placeholders: {no_ego_total}",
        "why_the_difference_is_2: two APKs have real_host_count=0; each receives exactly one __missing_host__ model-input/fallback row, but zero Ego-cache rows.",
        "counting_policy: placeholders are excluded from total_hosts, host_ego_usage_ratio, hosts_without_nontrivial_ego, and neighbor means.",
        "fallback_policy: the original fallback audit is unchanged; ego_fallback_audit_M1_reconciled.csv is a derived copy with placeholder=1.",
        "", "Placeholder rows:",
    ]
    for row in reconciliation:
        lines.append(
            f"  sha256={row['sha256']} fold={row['fold']} host_id={row['host_id']} "
            f"real={row['real_host_count']} host_value_rows={row['host_value_row_count']} "
            f"ego_rows={row['ego_cache_row_count']} placeholder={row['placeholder']}"
        )
    if errors:
        lines.extend(["", "RECONCILIATION_FAILED", *["  " + value for value in errors]])
    else:
        lines.extend(["", "HOST_ROW_COUNT_RECONCILIATION_PASS"])
    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    protected_after = {str(path): sha256_file(path) for path in (M0_PRED, M1_PRED)}
    if protected_after != protected:
        blocked("locked M0/M1 prediction changed during bookkeeping audit")
    if errors:
        blocked("Host-row reconciliation is not a pure two-placeholder explanation; see " + str(OUTPUT_TXT))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
