#!/usr/bin/env python3
"""Resolve and audit all inputs for the frozen CARE future 2x2x2 matrix."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path


EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
MEMBERS = ("p_lr", "p_rf", "p_xgb", "p_svm")


def norm(value) -> str:
    return str(value or "").strip().lower()


def row_id(row: dict) -> str:
    return norm(row.get("sha256") or row.get("sample_id") or row.get("apk_id"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_top(score: list[float], groups: list[int], fraction: float = 0.25) -> list[int]:
    result = [0] * len(score)
    for group in sorted(set(groups)):
        index = [i for i, value in enumerate(groups) if value == group]
        count = max(1, min(len(index), int(round(fraction * len(index)))))
        ordered = sorted(index, key=lambda i: (score[i], i))
        for i in ordered[-count:]:
            result[i] = 1
    return result


def di_score(row: dict) -> float:
    missing = [name for name in MEMBERS if name not in row]
    if missing:
        raise RuntimeError("cannot reconstruct U_DI; missing " + repr(missing))
    mean = sum(float(row[name]) for name in MEMBERS) / len(MEMBERS)
    return 1.0 - mean if float(row["p_stage1"]) >= 0.5 else mean


def feature_sha_set(path: Path) -> set[str]:
    values = set()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sha256", "status", "metadata_year"}
        if not required.issubset(reader.fieldnames or []):
            return set()
        for row in reader:
            if str(row.get("status", "")).strip() == "OK":
                value = norm(row.get("sha256"))
                if value:
                    values.add(value)
    return values


def build_label_safe_features(
    originals: list[Path],
    source_ids: list[str],
    future_ids: list[str],
    output: Path,
) -> tuple[Path, list[str]]:
    """Materialize the frozen cohort while sealing future label columns.

    Source labels must remain available because every expert is fitted on 2017.
    All columns whose names contain ``label`` are blanked for 2018--2021 before
    a model subprocess can open the compact-feature file.
    """
    wanted = set(source_ids) | set(future_ids)
    rows: dict[str, dict] = {}
    fields: list[str] = []
    seen_paths: set[str] = set()
    for original in originals:
        key = str(original.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        opener = gzip.open if original.name.endswith(".gz") else open
        with opener(original, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or [])
            if "sha256" not in current_fields:
                raise RuntimeError("compact feature file has no sha256 column: " + str(original))
            if fields and current_fields != fields:
                raise RuntimeError(
                    "source/future compact feature column schemas differ; refusing implicit alignment")
            fields = current_fields
            for row in reader:
                value = norm(row.get("sha256"))
                if value not in wanted or str(row.get("status", "OK")).strip() != "OK":
                    continue
                if value in rows:
                    raise RuntimeError("duplicate frozen SHA across compact feature files: " + value)
                rows[value] = row
    missing = wanted - set(rows)
    if missing:
        raise RuntimeError(f"label-safe feature materialization is missing {len(missing)} frozen SHA values")
    label_fields = [name for name in fields if "label" in name.casefold()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for value in source_ids:
            writer.writerow(rows[value])
        for value in future_ids:
            row = dict(rows[value])
            for name in label_fields:
                row[name] = ""
            writer.writerow(row)
    if sum(1 for _ in output.open("r", encoding="utf-8")) - 1 != 2788:
        raise RuntimeError("label-safe feature row-count contract failed")
    return output.resolve(), label_fields


def resolve_feature_part(
    wanted: set[str], explicit: str, env_name: str, default_candidates: list[Path], label: str,
) -> Path:
    candidates: list[Path] = []
    for value in (explicit, os.environ.get(env_name, ""), os.environ.get("CARE_FULL_FEATURES", "")):
        if value:
            candidates.append(Path(value))
    candidates.extend(default_candidates)
    seen = set()
    audits = []
    for path in candidates:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        try:
            ids = feature_sha_set(path)
        except Exception as exc:
            audits.append(f"{path}: unreadable ({exc})")
            continue
        missing = wanted - ids
        audits.append(f"{path}: ids={len(ids)} missing_frozen={len(missing)}")
        if not missing:
            return path.resolve()
    raise RuntimeError(
        f"No compact feature CSV covers the frozen {label} cohort.\n"
        + "\n".join(audits)
        + f"\nSet {env_name} to the matching features.csv."
    )


def resolve_feature_parts(
    source_ids: set[str], future_ids: set[str], source_explicit: str, future_explicit: str,
) -> tuple[Path, Path]:
    base = Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/datasets/android/AndroZoo/features/v12_selective_role_v3_full_2017_2021")
    full_candidates = [base / "features.csv", base / "compact_features.csv", base / "v12_full_features.csv"]
    source = resolve_feature_part(
        source_ids, source_explicit, "CARE_SOURCE_FEATURES",
        [Path("artifacts/p1-features/features.csv")] + full_candidates, "2017 source",
    )
    future = resolve_feature_part(
        future_ids, future_explicit, "CARE_FUTURE_FEATURES",
        [Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-route-stage2-future-matrix-v1/compact_future_2018_2021_fixed3/features.csv")]
        + full_candidates,
        "2018--2021 future",
    )
    return source, future


def resolve_schema(features: Path, explicit: str, env_name: str) -> Path:
    candidates = []
    for value in (explicit, os.environ.get(env_name, "")):
        if value:
            candidates.append(Path(value))
    candidates.append(features.with_name("feature_schema.json"))
    if os.environ.get("CARE_FULL_SCHEMA", ""):
        candidates.append(Path(os.environ["CARE_FULL_SCHEMA"]))
    candidates.append(Path("artifacts/p1-features/feature_schema.json"))
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return path.resolve()
    raise RuntimeError("No compatible feature_schema.json was found")


def audit_schema_pair(source: Path, future: Path) -> None:
    left = json.loads(source.read_text(encoding="utf-8"))
    right = json.loads(future.read_text(encoding="utf-8"))
    for key in ("feature_fields", "mapping_sha256", "priority_sha256"):
        if left.get(key) != right.get(key):
            raise RuntimeError(f"source/future feature schema mismatch: {key}")


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Preflight found all missing files:\n- " + "\n- ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    parser.add_argument("--future-lineage", type=Path, default=Path(
        "artifacts/care-future-local-assets-v2/future_lineage_2018_2021.csv"))
    parser.add_argument("--features", default="")
    parser.add_argument("--future-features", default="")
    parser.add_argument("--schema", default="")
    parser.add_argument("--future-schema", default="")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/preflight"))
    args = parser.parse_args()

    package = Path(__file__).resolve().parent
    required_scripts = [package / name for name in (
        "run_care_rgcondattn_dedup_bounded_source_v1.py",
        "run_care_malben_role_source_v2.py", "run_malben_host_v2.py",
        "run_stage2only_malben_v1.py", "run_stage2only_anomaly_v1.py",
        "stage2only_malben_base.py", "shared_role_prototypes.py",
        "source_label_contract.py", "committee_uncertainty_core.py",
        "vendor/run_care_rolegraph_conditioned_dualstream_v4.py",
    )]
    required_artifacts = [
        args.source_manifest, args.future_lineage,
        Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"),
        Path("artifacts/care-source-local-assets-v2/source_asset_gate.txt"),
        Path("artifacts/care-source-local-assets-v2/regions_2017.npz"),
        Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
        Path("artifacts/care-future-local-assets-v2/future_asset_gate.txt"),
        Path("artifacts/care-future-local-assets-v2/regions_2018_2021.npz"),
        Path("artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"),
    ]
    require_files(required_scripts + required_artifacts)
    for gate in (required_artifacts[3], required_artifacts[6]):
        text = gate.read_text(encoding="utf-8").lower()
        if "passed: true" not in text:
            raise RuntimeError("hard gate is not passed: " + str(gate))

    source = read_csv(args.source_manifest)
    future = read_csv(args.future_lineage)
    if len(source) != 799 or len(future) != 1989:
        raise RuntimeError(f"cohort changed: source={len(source)} future={len(future)}")
    source_ids = [row_id(row) for row in source]
    future_ids = [row_id(row) for row in future]
    if len(set(source_ids)) != 799 or len(set(future_ids)) != 1989 or set(source_ids) & set(future_ids):
        raise RuntimeError("source/future SHA uniqueness or disjointness failed")
    years = [int(row.get("year") or row.get("metadata_year")) for row in future]
    counts = {year: years.count(year) for year in sorted(set(years))}
    if counts != EXPECTED_YEARS:
        raise RuntimeError("future year counts changed: " + repr(counts))

    args.output.mkdir(parents=True, exist_ok=True)
    cohort_label_free = args.output / "future_cohort_label_free.csv"
    write_csv(cohort_label_free, [
        {"sha256": future_ids[i], "year": years[i]} for i in range(len(future_ids))
    ])
    source_features, future_features = resolve_feature_parts(
        set(source_ids), set(future_ids), args.features, args.future_features)
    schema = resolve_schema(source_features, args.schema, "CARE_SOURCE_SCHEMA")
    future_schema = resolve_schema(future_features, args.future_schema, "CARE_FUTURE_SCHEMA")
    audit_schema_pair(schema, future_schema)
    features, sealed_fields = build_label_safe_features(
        [source_features, future_features], source_ids, future_ids,
        args.output / "compact_features_2017_train_2018_2021_label_sealed.csv")
    resolved = {
        "experiment": "CARE independent within-year route x Stage-II x head matrix v1",
        "source_n": 799, "future_n": 1989,
        "year_counts": EXPECTED_YEARS,
        "features": str(features), "schema": str(schema),
        "source_features": str(source_features),
        "source_features_sha256": sha256(source_features),
        "future_features": str(future_features),
        "future_features_sha256": sha256(future_features),
        "future_schema": str(future_schema),
        "future_label_fields_sealed_in_model_input": sealed_fields,
        "label_safe_features_sha256": sha256(features),
        "source_manifest": str(args.source_manifest.resolve()),
        "within_year_label_file": str(args.future_lineage.resolve()),
        "future_cohort_label_free": str(cohort_label_free.resolve()),
        "future_cohort_label_free_sha256": sha256(cohort_label_free),
        "source_scores": str((args.output / "source_sample_scores.csv").resolve()),
        "future_router_inputs_label_free": str((args.output / "future_router_inputs_label_free.csv").resolve()),
        "feature_dictionary_frozen_from_2017": True,
        "within_year_labels_used_for_training_and_routing": True,
        "future_labels_used_for_training_routing_or_model_selection": True,
        "future_labels_unsealed_only_by_final_summarizer": False,
        "temporal_generalization_result": False,
    }
    (args.output / "resolved_inputs.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("WITHIN_YEAR_MATRIX_PREFLIGHT_PASS")
    print(json.dumps(resolved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
