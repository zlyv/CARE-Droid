#!/usr/bin/env python3
"""Strict 2017 OOF Stage-II-only MalBen classification.

The final classifier receives only [g, z_mal-z_ben, h_critical,
e_mal-e_ben].  No Global24 feature, probability, router, or bounded
correction enters either representation learning or final inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
V4_DIR = SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script"
for path in (SCRIPT_DIR, SCRIPTS_ROOT, V4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_care_malben_role_source_v2 as malben  # noqa: E402

v4 = malben.v4


class StageIIRepresentationModel(malben.RoleConditionedEvidenceModel):
    """Unbounded auxiliary head; reported decisions come from L2 LR only."""

    def __init__(self, global_dim: int, variant: str, args):
        super().__init__(global_dim, variant, args)
        self.direct_stage2_training = True


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_features(model, graph_values, bags, device) -> np.ndarray:
    graph = v4.graph_tensors(graph_values, device)
    bag_tensors = [
        torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags
    ]
    zero_global = torch.zeros(1, dtype=torch.float32, device=device)
    zero_logit = torch.zeros((), dtype=torch.float32, device=device)
    result = []
    model.eval()
    with torch.no_grad():
        for index in range(len(bags)):
            one_graph = {key: value[index] for key, value in graph.items()}
            model(zero_global, zero_logit, one_graph, bag_tensors[index])
            result.append(np.asarray(
                model.last_apk_capture["stage2_feature"], dtype=np.float32
            ))
    return np.stack(result)


def fit_fold_features(s7_train, s7_val, raw_train_bags, raw_val_bags,
                      labels, seeds, args, device):
    train_features = []
    val_features = []
    audits = []
    neutral_train = np.full(len(labels), 0.5, dtype=np.float64)
    neutral_val = np.full(len(raw_val_bags), 0.5, dtype=np.float64)
    zero_train = np.zeros((len(labels), 1), dtype=np.float32)
    zero_val = np.zeros((len(raw_val_bags), 1), dtype=np.float32)
    for seed in seeds:
        selected_epochs, selection_loss = v4.select_epochs(
            zero_train, s7_train, raw_train_bags, neutral_train, labels,
            "MalBenContrast", args, device, int(seed),
        )
        train_bags, val_bags, prototype_audit = malben.fit_scaled_role_bags(
            raw_train_bags, raw_val_bags, labels, args, int(seed) + 1707
        )
        graph_scaler = v4.fit_role_graph_scaler(s7_train)
        train_graph = v4.transform_role_graphs(s7_train, graph_scaler)
        val_graph = v4.transform_role_graphs(s7_val, graph_scaler)
        train_graph_t = v4.graph_tensors(train_graph, device)
        train_bag_t = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in train_bags
        ]
        global_t = torch.as_tensor(zero_train, dtype=torch.float32, device=device)
        weights, correct = v4.fixkeep_weights(labels, neutral_train, args)
        v4.seed_everything(seed)
        model = StageIIRepresentationModel(1, "MalBenContrast", args).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        final_loss = v4.train_epochs(
            model, optimizer, global_t, neutral_train, train_graph_t,
            train_bag_t, labels, weights, correct, selected_epochs, args,
            int(seed) + 3301,
        )
        train_features.append(extract_features(model, train_graph, train_bags, device))
        val_features.append(extract_features(model, val_graph, val_bags, device))
        audits.append({
            "seed": int(seed),
            "selected_epochs": int(selected_epochs),
            "selection_loss": float(selection_loss),
            "final_training_loss": float(final_loss),
            **prototype_audit,
        })
        del model, train_graph_t, train_bag_t, global_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        np.mean(np.stack(train_features), axis=0),
        np.mean(np.stack(val_features), axis=0),
        audits,
    )


def metrics(y, probability) -> dict:
    prediction = np.asarray(probability) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "BA": float(balanced_accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, probability)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def valid_region_stats(audit_rows: list[dict]) -> tuple[int, int]:
    valid = 0
    missing = 0
    for row in audit_rows:
        raw = str(row.get("host_was_empty", "")).strip().lower()
        if raw in {"1", "true", "yes"}:
            missing += 1
        elif raw in {"0", "false", "no"}:
            valid += 1
        else:
            # A cached bag is always non-empty after deterministic handling,
            # but does not reveal whether the raw host set was empty.
            count = str(row.get("real_host_region_n", "")).strip()
            if count.isdigit() and int(count) == 0:
                missing += 1
            else:
                valid += 1
    return valid, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--cache", type=Path, default=Path(
        "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    parser.add_argument("--source-multiscale-features", type=Path, default=Path(
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    parser.add_argument("--previous-result", type=Path, default=Path(
        "artifacts/global24-a25-route-conditioned-joints7-v1-full"))
    parser.add_argument("--care-reference", type=Path, default=Path(
        "artifacts/care-bounded-fixkeep-v3-full/source_predictions.csv"))
    parser.add_argument("--region-cache", type=Path, default=Path(
        "artifacts/care-local-region-cache-v1/regions_2017_2021.npz"))
    parser.add_argument("--region-audit-reference", type=Path, default=Path(
        "artifacts/care-di-budget-bound-selection-v1/retrained_source/max_delta_2p25/region_bag_audit.csv"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260901)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop-fraction", type=float, default=0.18)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--graph-dim", type=int, default=32)
    parser.add_argument("--graph-heads", type=int, default=4)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-delta", type=float, default=2.25)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--fix-weight", type=float, default=1.0)
    parser.add_argument("--keep-consistency", type=float, default=0.0)
    parser.add_argument("--shared-prototypes", type=int, default=32)
    parser.add_argument("--benign-prototypes", type=int, default=24)
    parser.add_argument("--prototype-regions-per-apk", type=int, default=16)
    parser.add_argument("--internal-background-coverage", type=float, default=0.70)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-malben-contrast-v2/source/stage2only"))
    args = parser.parse_args()

    malben.FIT_CALL_IN_GROUP = 0
    malben.CURRENT_GROUP = -1
    malben.CURRENT_VARIANT = "StageIIOnly-MalBen"
    v4.v3.fit_scaled_dualproto_bags = malben.fit_scaled_role_bags
    v4.AUGMENTED_REGION_DIM = malben.FULL_REGION_DIM
    v4.RoleGraphConditionedEvidence = StageIIRepresentationModel
    device = v4.resolve_device(args.device)
    seeds = v4.parse_ints(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    source_s7, _ = v4.v3.load_source_s7(source_rows, endpoint_cache, args)
    host_bags, bag_audit, cache_loaded = v4.get_source_host_bags(source_rows, endpoint_cache, args)
    probability = np.full(len(y), np.nan, dtype=np.float64)
    fit_rows = []
    feature_dim = None
    print("Strict 2017 OOF StageIIOnly-MalBen; no Stage-I input or router.", flush=True)
    for fold in sorted(np.unique(folds)):
        train_index = np.flatnonzero(folds != fold)
        val_index = np.flatnonzero(folds == fold)
        fold_seeds = [100000 * int(fold) + seed for seed in seeds]
        train_feature, val_feature, audits = fit_fold_features(
            source_s7[train_index], source_s7[val_index],
            [host_bags[i] for i in train_index],
            [host_bags[i] for i in val_index],
            y[train_index], fold_seeds, args, device,
        )
        feature_dim = int(train_feature.shape[1])
        scaler = StandardScaler().fit(train_feature)
        classifier = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
            random_state=int(args.split_seed) + int(fold),
        )
        classifier.fit(scaler.transform(train_feature), y[train_index])
        probability[val_index] = classifier.predict_proba(
            scaler.transform(val_feature)
        )[:, 1]
        for row in audits:
            fit_rows.append({"fold": int(fold), **row})
        print("fold", int(fold), "done", flush=True)
    if np.any(~np.isfinite(probability)):
        raise RuntimeError("incomplete Stage-II-only OOF probabilities")

    coverage_audit = bag_audit
    if args.region_audit_reference.is_file():
        with args.region_audit_reference.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reference_audit = list(csv.DictReader(handle))
        by_sha = {str(row.get("sha256", "")).strip().lower(): row for row in reference_audit}
        expected_sha = [str(row["sha256"]).strip().lower() for row in manifest]
        if len(by_sha) == len(manifest) and set(by_sha) == set(expected_sha):
            coverage_audit = [by_sha[sha] for sha in expected_sha]
    valid, missing = valid_region_stats(coverage_audit)
    overall = metrics(y, probability)
    overall.update({
        "method": "StageIIOnly-MalBen",
        "stage2_invocation_rate": 1.0,
        "valid_region_coverage": float(valid / len(y)),
        "valid_region_count": int(valid),
        "missing_region_count": int(missing),
    })
    per_fold = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        row = metrics(y[selected], probability[selected])
        row.update({"fold": int(fold), "method": "StageIIOnly-MalBen"})
        per_fold.append(row)
    prediction_rows = [{
        "sha256": str(manifest[i]["sha256"]).strip().lower(),
        "label": int(y[i]), "fold": int(folds[i]),
        "StageIIOnly_MalBen__prob": float(probability[i]),
    } for i in range(len(y))]
    write_csv(args.output / "source_summary.csv", [overall])
    write_csv(args.output / "source_per_fold.csv", per_fold)
    write_csv(args.output / "source_predictions.csv", prediction_rows)
    write_csv(args.output / "fit_audit.csv", fit_rows)
    write_csv(args.output / "region_bag_audit.csv", bag_audit)
    architecture = {
        "experiment": "StageIIOnly-MalBen strict source OOF v1",
        "future_data_loaded": False,
        "outer_folds": sorted(int(x) for x in np.unique(folds)),
        "representation_training": "Stage-II-only direct BCE auxiliary head with zero global input; no base logit or correction bound",
        "final_classifier": "StandardScaler + L2 LogisticRegression(C=1.0)",
        "final_input": "[g_i, z_mal_i-z_ben_i, h_critical_i, e_mal_i-e_ben_i]",
        "feature_dim": feature_dim,
        "forbidden_inputs": ["Global24 features", "p_stage1", "A25", "DI25", "bounded correction"],
        "region_cache_loaded": bool(cache_loaded),
        "stage2_invocation_rate": 1.0,
        "valid_region_coverage": overall["valid_region_coverage"],
        "missing_region_count": missing,
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "CARE-Droid Stage-II-only MalBen source comparison v1", "",
        "Strict 2017 outer-fold OOF only; no future row is loaded.",
        "No Global24 input, router, p_stage1, or bounded correction is used.",
        "Stage-II invocation rate: 100%", "",
        f"BA={overall['BA']:.6f} AUC={overall['AUC']:.6f} F1={overall['F1']:.6f}",
        f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}",
        f"valid_region_coverage={overall['valid_region_coverage']:.6f}",
        f"missing_region_count={missing}",
    ]
    (args.output / "REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
