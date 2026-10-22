#!/usr/bin/env python3
"""Train one 2017-only Stage-II arm and emit label-free future probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SEARCH = [
    HERE, HERE / "vendor",
    SCRIPTS / "care-malben-host-v2", SCRIPTS / "care-malben-contrast-v2",
    SCRIPTS / "care-rgcondattn-dedup-bounded-source-v1",
    SCRIPTS / "care-stage2only-malben-v1", SCRIPTS / "care-stage2only-anomaly-v1",
    SCRIPTS / "care-rolegraph-conditioned-v4-script", SCRIPTS,
]
for path in reversed(SEARCH):
    sys.path.insert(0, str(path))

import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly_bounded
import run_care_malben_role_source_v2 as malben_bounded
import run_malben_host_v2 as malben_host
import run_stage2only_malben_v1 as direct_malben
import run_stage2only_anomaly_v1 as direct_anomaly
from shared_role_prototypes import evidence_columns, fit_role_prototypes
from source_label_contract import install_source_label_contract


ORIGINAL_ANOMALY_FIT = v4.v3.fit_scaled_dualproto_bags
ORIGINAL_V4_PREDICT = v4.predict_model
ORIGINAL_AUGMENTED_DIM = v4.AUGMENTED_REGION_DIM


def norm(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aligned_values(rows: list[dict], path: Path, column: str, labels=None) -> np.ndarray:
    records = read_csv(path)
    by_sha = {norm(row["sha256"]): row for row in records}
    if len(by_sha) != len(records):
        raise RuntimeError("duplicate SHA in " + str(path))
    values = []
    for i, row in enumerate(rows):
        sha = norm(row["sha256"])
        if sha not in by_sha:
            raise RuntimeError("score file is missing SHA " + sha)
        record = by_sha[sha]
        if labels is not None and int(record["label"]) != int(labels[i]):
            raise RuntimeError("source label mismatch for " + sha)
        values.append(float(record[column]))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite " + column)
    return result


def load_bags(path: Path, rows: list[dict]) -> tuple[list[np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        sha = [norm(value) for value in saved["sha256"].astype(str)]
        values = np.asarray(saved["host_values"], dtype=np.float32)
        offsets = np.asarray(saved["host_offsets"], dtype=np.int64)
        real = np.asarray(saved["real_host_count"], dtype=np.int64)
    lookup = {value: i for i, value in enumerate(sha)}
    if len(lookup) != len(sha):
        raise RuntimeError("duplicate SHA in region cache " + str(path))
    bags, counts = [], []
    for row in rows:
        value = norm(row["sha256"])
        if value not in lookup:
            raise RuntimeError("region cache missing SHA " + value)
        i = lookup[value]
        bag = values[offsets[i]:offsets[i + 1]]
        if len(bag) < 1 or bag.shape[1] != 74:
            raise RuntimeError("invalid deterministic Host bag for " + value)
        bags.append(bag); counts.append(int(real[i]))
    return bags, np.asarray(counts, dtype=np.int64)


def merge_endpoint_caches(source: Path, future: Path):
    left = v4.audit.exp.read_cache(source)
    right = v4.audit.exp.read_cache(future)
    overlap = set(left) & set(right)
    if overlap:
        raise RuntimeError("source/future endpoint cache overlap")
    merged = dict(left); merged.update(right)
    return merged


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", choices=("anomaly", "malben"), required=True)
    p.add_argument("--head", choices=("direct", "bounded"), required=True)
    p.add_argument("--resolved", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/preflight/resolved_inputs.json"))
    p.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    p.add_argument("--source-multiscale-features", type=Path, default=Path(
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    p.add_argument("--source-cache", type=Path, default=Path(
        "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))
    p.add_argument("--future-cache", type=Path, default=Path(
        "artifacts/care-future-local-assets-v2/endpoint_centered_tensor_cache_2018_2021.jsonl.gz"))
    p.add_argument("--source-regions", type=Path, default=Path(
        "artifacts/care-source-local-assets-v2/regions_2017.npz"))
    p.add_argument("--future-regions", type=Path, default=Path(
        "artifacts/care-future-local-assets-v2/regions_2018_2021.npz"))
    p.add_argument("--future-inter-cache", type=Path, default=Path(
        "artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"))
    p.add_argument("--rebuild-future-inter-cache", action="store_true")
    p.add_argument("--seeds", default="900000,900001,900002")
    p.add_argument("--inner-folds", type=int, default=4)
    p.add_argument("--split-seed", type=int, default=20260901)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--early-stop-fraction", type=float, default=0.18)
    p.add_argument("--early-stop-patience", type=int, default=5)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--embedding-dim", type=int, default=24)
    p.add_argument("--graph-dim", type=int, default=32)
    p.add_argument("--graph-heads", type=int, default=4)
    p.add_argument("--graph-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-delta", type=float, default=2.25)
    p.add_argument("--keep-weight", type=float, default=2.0)
    p.add_argument("--fix-weight", type=float, default=1.0)
    p.add_argument("--keep-consistency", type=float, default=0.5)
    p.add_argument("--shared-prototypes", type=int, default=32)
    p.add_argument("--benign-prototypes", type=int, default=24)
    p.add_argument("--prototype-regions-per-apk", type=int, default=16)
    p.add_argument("--internal-background-coverage", type=float, default=0.70)
    p.add_argument("--null-samples", type=int, default=64)
    p.add_argument("--swaps-per-edge", type=int, default=10)
    p.add_argument("--null-seed", type=int, default=20260827)
    p.add_argument("--z-clip", type=float, default=5.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    return p


def load_common(args):
    # The recovered compact feature file stores textual benign/malware labels.
    # Install the already-audited normalization contract before load_inputs.
    install_source_label_contract(v4)
    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    args.features = Path(resolved["features"])
    args.schema = Path(resolved["schema"])
    args.cache = args.source_cache
    args.region_cache = args.source_regions
    args.rebuild_region_cache = False
    manifest, source_rows, _, compact_by_sha, schema, names = v4.audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    if len(source_y) != 799:
        raise RuntimeError("expected exact 799 source rows")
    future_input = read_csv(Path(resolved["future_router_inputs_label_free"]))
    target_rows = []
    for row in future_input:
        sha = norm(row["sha256"])
        if sha not in compact_by_sha:
            raise RuntimeError("complete compact features missing future SHA " + sha)
        target = compact_by_sha[sha]
        leaked = {key: value for key, value in target.items()
                  if "label" in key.casefold() and str(value).strip()}
        if leaked:
            raise RuntimeError("future label seal failed for " + sha + ": " + repr(leaked))
        target_rows.append(target)
    if len(target_rows) != 1989:
        raise RuntimeError("expected exact 1989 future rows")
    base_source = aligned_values(source_rows, Path(resolved["source_scores"]), "p_stage1", source_y)
    base_target = np.asarray([float(row["p_stage1"]) for row in future_input], dtype=np.float64)
    endpoint = merge_endpoint_caches(args.source_cache, args.future_cache)
    source_s3, _ = v4.audit.future.reconstruct_structure(source_rows, endpoint, 31)
    target_s3, _ = v4.audit.future.reconstruct_structure(target_rows, endpoint, 31)
    source_s7, source_s7_audit = v4.v3.future_s7.source_s7_from_reference(
        args.source_multiscale_features, source_rows, source_s3)
    target_inter, _, future_inter_audit = v4.v3.future_s7.load_or_build_future_inter(
        target_rows, target_s3, endpoint, args)
    target_s7 = np.concatenate([target_s3, target_inter], axis=1)
    if source_s7.shape != (799, v4.S7_DIM) or target_s7.shape != (1989, v4.S7_DIM):
        raise RuntimeError("S7 shape contract failed")
    source_bags, source_count = load_bags(args.source_regions, source_rows)
    target_bags, target_count = load_bags(args.future_regions, target_rows)
    return SimpleNamespace(
        resolved=resolved, manifest=manifest, source_rows=source_rows,
        target_rows=target_rows, future_input=future_input, source_y=source_y,
        base_source=base_source, base_target=base_target,
        source_s7=source_s7, target_s7=target_s7,
        source_bags=source_bags, target_bags=target_bags,
        source_count=source_count, target_count=target_count,
        compact_by_sha=compact_by_sha, schema=schema, names=names,
        source_s7_audit=source_s7_audit, future_inter_audit=future_inter_audit,
    )


def train_direct(data, args, seeds, device):
    if args.representation == "anomaly":
        direct_malben.StageIIOnlyModel = direct_anomaly.StageIIOnlyAnomalyModel
        direct_malben.fit_scaled_shared_role_bags = direct_anomaly.fit_scaled_anomaly_bags
    else:
        malben_bounded.fit_role_prototypes = fit_role_prototypes
        malben_bounded.evidence_columns = evidence_columns
    source_feature, target_feature, audits = direct_malben.fit_fold_features(
        data.source_s7, data.target_s7, data.source_bags, data.target_bags,
        data.source_y, seeds, args, device)
    if source_feature.shape != (799, 81) or target_feature.shape != (1989, 81):
        raise RuntimeError("direct Stage-II feature contract failed")
    scaler = StandardScaler().fit(source_feature)
    classifier = LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
        random_state=int(args.split_seed))
    classifier.fit(scaler.transform(source_feature), data.source_y)
    probability = classifier.predict_proba(scaler.transform(target_feature))[:, 1]
    return probability, audits, {
        "decision_head": "absolute Stage-II StandardScaler + L2 LogisticRegression(C=1)",
        "uses_stage1_in_expert": False, "bounded_correction": False,
        "feature_dim": 81,
    }


def train_bounded(data, args, seeds, device):
    x_source, x_target, feature_names = v4.audit.compact_matrix_variant(
        v4.VARIANT, data.source_rows, data.target_rows,
        data.compact_by_sha, data.schema, data.names)
    global_index, _ = v4.evidence.split_view_indices(feature_names)
    global_source = np.asarray(x_source[:, global_index], dtype=np.float32)
    global_target = np.asarray(x_target[:, global_index], dtype=np.float32)
    if global_source.shape != (799, 24) or global_target.shape != (1989, 24):
        raise RuntimeError("Global24 feature contract failed")
    if args.representation == "anomaly":
        v4.AUGMENTED_REGION_DIM = ORIGINAL_AUGMENTED_DIM
        v4.v3.fit_scaled_dualproto_bags = ORIGINAL_ANOMALY_FIT
        v4.RoleGraphConditionedEvidence = anomaly_bounded.DedupBoundedRGCondAttn
        v4.predict_model = ORIGINAL_V4_PREDICT
        fit = v4.fit_predict_variant
        variant = "RGCondAttn"
    else:
        malben_bounded.fit_role_prototypes = fit_role_prototypes
        malben_bounded.evidence_columns = evidence_columns
        v4.v3.fit_scaled_dualproto_bags = malben_host.fit_scaled_shared_role_bags
        v4.AUGMENTED_REGION_DIM = malben_bounded.FULL_REGION_DIM
        v4.RoleGraphConditionedEvidence = malben_bounded.RoleConditionedEvidenceModel
        # The source audit captured every region for mechanism analysis.  The
        # frozen future matrix needs probabilities only; retaining all 1989 x
        # three-seed captures would waste substantial RAM without changing the
        # model forward path.
        v4.predict_model = ORIGINAL_V4_PREDICT
        fit = malben_bounded.grouped_fit_predict
        variant = "MalBenContrast"
    probability, diagnostics, audits = fit(
        global_source, global_target, data.source_s7, data.target_s7,
        data.source_bags, data.target_bags, data.base_source, data.base_target,
        data.source_y, variant, seeds, args, device)

    # Diagnostic capture only.
    # No model/forward/training arithmetic is changed.
    diagnostics = np.asarray(
        diagnostics,
        dtype=np.float64,
    )

    if args.representation == "malben":

        names = tuple(
            malben_bounded.DIAGNOSTIC_NAMES
        )

        expected_shape = (
            1989,
            len(names),
        )

        if diagnostics.shape != expected_shape:
            raise RuntimeError(
                "MalBen diagnostic shape mismatch: "
                + repr(diagnostics.shape)
                + " expected="
                + repr(expected_shape)
            )

        diagnostic_rows = []

        for i in range(1989):

            row = {
                "sha256": norm(
                    data.target_rows[i]["sha256"]
                ),
                "year": int(
                    data.future_input[i]["year"]
                ),
                "diagnostic_only": True,
            }

            for j, name in enumerate(names):
                row[name] = float(
                    diagnostics[i, j]
                )

            diagnostic_rows.append(row)

        write_csv(
            args.output / "malben_diagnostics.csv",
            diagnostic_rows,
        )

    return probability, audits, {
        "decision_head": "FixKeep logit-space bounded correction",
        "uses_stage1_in_expert": True, "bounded_correction": True,
        "max_delta_logit": float(args.max_delta),
    }


def main() -> None:
    args = parser().parse_args()
    if float(args.max_delta) != 2.25:
        raise SystemExit("frozen experiment requires --max-delta 2.25")
    seeds = v4.parse_ints(args.seeds)
    if seeds != [900000, 900001, 900002]:
        raise SystemExit("frozen experiment requires seeds 900000,900001,900002")
    device = v4.resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_common(args)
    print("Training future arm", args.representation, args.head, "device", device, flush=True)
    if args.head == "direct":
        probability, audits, contract = train_direct(data, args, seeds, device)
    else:
        probability, audits, contract = train_bounded(data, args, seeds, device)
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (1989,) or not np.isfinite(probability).all():
        raise RuntimeError("incomplete future expert probability")
    rows = [{
        "sha256": norm(data.target_rows[i]["sha256"]),
        "year": int(data.future_input[i]["year"]),
        "representation": args.representation,
        "head": args.head,
        "p_expert": float(probability[i]),
        "real_host_count": int(data.target_count[i]),
        "diagnostic_only": True,
    } for i in range(1989)]
    write_csv(args.output / "future_expert_predictions.csv", rows)
    if audits:
        fields = []
        flattened = []
        for item in audits:
            row = {}
            for key, value in item.items():
                row[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                if key not in fields: fields.append(key)
            flattened.append(row)
        with (args.output / "fit_audit.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(flattened)
    protocol = {
        "representation": args.representation, "head": args.head,
        "source_train_n": 799, "future_n": 1989,
        "source_only_training": True, "future_labels_loaded_by_model_arm": False,
        "future_label_columns_hard_asserted_blank": True,
        "seeds": seeds, "ensemble": "equal mean across three frozen neural seeds",
        "local_input": "Host-only 74-D assets; cached Ego is not loaded",
        "source_asset_gate": "passed", "future_asset_gate": "passed",
        **contract,
        "source_s7_audit": data.source_s7_audit,
        "future_inter_audit": data.future_inter_audit,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print("ARM_DONE", args.representation, args.head, args.output, flush=True)


if __name__ == "__main__":
    main()
