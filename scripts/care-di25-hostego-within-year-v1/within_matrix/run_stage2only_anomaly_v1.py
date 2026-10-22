#!/usr/bin/env python3
"""Independent strict-2017 Stage-II-only anomaly representation control."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (
    HERE, ROOT, ROOT / "care-stage2only-malben-v1",
    ROOT / "care-malben-host-v2", ROOT / "care-malben-contrast-v2",
    ROOT / "care-rolegraph-conditioned-v4-script",
):
    sys.path.insert(0, str(path))

import stage2only_malben_base as base
from source_label_contract import install_source_label_contract


v4 = base.v4
BASE_REGION_DIM = base.BASE_REGION_DIM
OLD_AUGMENTED_DIM = base.OLD_AUGMENTED_DIM
ORIGINAL_ANOMALY_BAG_FIT = v4.v3.fit_scaled_dualproto_bags


def fit_scaled_anomaly_bags(train_bags, test_bags, labels, args, seed):
    """Established fold-local external/internal/shared anomaly features."""
    return ORIGINAL_ANOMALY_BAG_FIT(
        train_bags, test_bags, labels, args, seed
    )


class StageIIOnlyAnomalyModel(nn.Module):
    """Equal-width no-Stage-I counterpart of StageIIOnlyModel."""

    def __init__(self, args):
        super().__init__()
        graph_dim = int(args.graph_dim)
        local_dim = int(args.embedding_dim)
        hidden_dim = int(args.hidden_dim)
        self.graph_encoder = v4.RoleGraphEncoder(args)
        self.region_encoder = nn.Sequential(
            nn.Linear(OLD_AUGMENTED_DIM, hidden_dim), nn.ReLU(),
            nn.Dropout(float(args.dropout)), nn.Linear(hidden_dim, local_dim), nn.ReLU(),
        )
        self.local_risk = nn.Linear(local_dim, 1)
        self.graph_query = nn.Linear(graph_dim, local_dim)
        self.local_key = nn.Linear(local_dim, local_dim)
        self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))
        self.auxiliary_head = nn.Sequential(
            nn.Linear(graph_dim + 2 * local_dim + 1, hidden_dim), nn.ReLU(),
            nn.Dropout(float(args.dropout)), nn.Linear(hidden_dim, 1),
        )
        self.last_feature = None

    def forward(self, graph, regions):
        graph_token, _ = self.graph_encoder(
            graph["node"], graph["edge"], graph["global"]
        )
        local = self.region_encoder(regions[:, :OLD_AUGMENTED_DIM])
        residual = regions[:, BASE_REGION_DIM + 3].clamp(min=0.0)
        score = self.local_risk(local).reshape(-1)
        score = score + nn.functional.softplus(self.residual_beta_raw) * torch.log1p(residual)
        score = score + torch.mv(
            self.local_key(local), self.graph_query(graph_token)
        ) / math.sqrt(local.shape[1])
        attention = torch.softmax(score, dim=0)
        critical = local[torch.argmax(score)]
        z_anomaly = torch.sum(attention[:, None] * local, dim=0)
        e_anomaly = torch.sum(attention * residual)
        feature = torch.cat([
            graph_token, z_anomaly, critical, e_anomaly.reshape(1)
        ])
        self.last_feature = feature
        return self.auxiliary_head(feature).reshape(())


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    value.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    value.add_argument("--cache", type=Path, default=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))
    value.add_argument("--p72-oof", type=Path, default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    value.add_argument("--source-multiscale-features", type=Path, default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    value.add_argument("--region-cache", type=Path, default=Path("artifacts/care-source-local-assets-v2/regions_2017.npz"))
    value.add_argument("--rebuild-region-cache", action="store_true")
    value.add_argument("--seeds", default="0,1,2")
    value.add_argument("--inner-folds", type=int, default=4)
    value.add_argument("--split-seed", type=int, default=20260901)
    value.add_argument("--max-epochs", type=int, default=30)
    value.add_argument("--early-stop-fraction", type=float, default=0.18)
    value.add_argument("--early-stop-patience", type=int, default=5)
    value.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--hidden-dim", type=int, default=48)
    value.add_argument("--embedding-dim", type=int, default=24)
    value.add_argument("--graph-dim", type=int, default=32)
    value.add_argument("--graph-heads", type=int, default=4)
    value.add_argument("--graph-layers", type=int, default=2)
    value.add_argument("--dropout", type=float, default=0.20)
    value.add_argument("--learning-rate", type=float, default=8e-4)
    value.add_argument("--weight-decay", type=float, default=5e-4)
    value.add_argument("--grad-clip", type=float, default=5.0)
    value.add_argument("--shared-prototypes", type=int, default=32)
    value.add_argument("--benign-prototypes", type=int, default=24)
    value.add_argument("--prototype-regions-per-apk", type=int, default=16)
    value.add_argument("--internal-background-coverage", type=float, default=0.70)
    value.add_argument("--device", default="auto")
    value.add_argument("--progress-every", type=int, default=100)
    value.add_argument("--output", type=Path, default=Path("artifacts/care-stage2only-anomaly-v1/source"))
    return value


def main() -> None:
    args = parser().parse_args()
    install_source_label_contract(v4)
    # Reuse the exact training/early-stopping/feature-ensemble code from the
    # completed MalBen control; only these two experiment-specific objects vary.
    base.StageIIOnlyModel = StageIIOnlyAnomalyModel
    base.fit_scaled_shared_role_bags = fit_scaled_anomaly_bags
    device = v4.resolve_device(args.device)
    seeds = v4.parse_ints(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    if len(labels) != 799:
        raise RuntimeError("Stage-II-only anomaly must retain all 799 source APKs")
    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    source_s7, _ = v4.v3.load_source_s7(source_rows, endpoint_cache, args)
    host_bags, bag_audit, cache_loaded = v4.get_source_host_bags(
        source_rows, endpoint_cache, args
    )

    probability = np.full(len(labels), np.nan, dtype=np.float64)
    fit_rows = []
    feature_dim = None
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold)
        val = np.flatnonzero(folds == fold)
        fold_seeds = [100000 * int(fold) + seed for seed in seeds]
        train_feature, val_feature, audits = base.fit_fold_features(
            source_s7[train], source_s7[val],
            [host_bags[i] for i in train], [host_bags[i] for i in val],
            labels[train], fold_seeds, args, device,
        )
        feature_dim = int(train_feature.shape[1])
        expected_dim = int(args.graph_dim) + 2 * int(args.embedding_dim) + 1
        if feature_dim != expected_dim or expected_dim != 81:
            raise RuntimeError("Stage-II-only anomaly feature contract failed")
        scaler = StandardScaler().fit(train_feature)
        classifier = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
            random_state=int(args.split_seed) + int(fold),
        )
        classifier.fit(scaler.transform(train_feature), labels[train])
        probability[val] = classifier.predict_proba(
            scaler.transform(val_feature)
        )[:, 1]
        fit_rows.extend({"fold": int(fold), **row} for row in audits)
        print("StageIIOnly-Anomaly fold", int(fold), "done", flush=True)
    if np.any(~np.isfinite(probability)):
        raise RuntimeError("incomplete Stage-II-only anomaly OOF probabilities")

    with np.load(args.region_cache, allow_pickle=False) as saved:
        real_host_count = np.asarray(saved["real_host_count"], dtype=np.int64)
    if len(real_host_count) != 799:
        raise RuntimeError("region coverage does not cover all 799 APKs")
    overall = base.metrics(labels, probability)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        fold_rows.append({
            "fold": int(fold), **base.metrics(labels[selected], probability[selected]),
            "host_coverage": float(np.mean(real_host_count[selected] > 0)),
            "stage2_invocation_rate": 1.0,
        })
    predictions = [{
        "sha256": str(manifest[i]["sha256"]).strip().lower(),
        "label": int(labels[i]), "fold": int(folds[i]),
        "StageIIOnly_Anomaly__prob": float(probability[i]),
        "has_host": int(real_host_count[i] > 0),
        "real_host_count": int(real_host_count[i]),
    } for i in range(799)]
    base.write_csv(args.output / "fold_metrics.csv", fold_rows)
    base.write_csv(args.output / "oof_predictions.csv", predictions)
    base.write_csv(args.output / "fit_audit.csv", fit_rows)
    coverage = {
        "total_apks": 799,
        "apks_with_host": int(np.sum(real_host_count > 0)),
        "apks_without_host": int(np.sum(real_host_count == 0)),
        "host_coverage": float(np.mean(real_host_count > 0)),
        "stage2_invocation_rate": 1.0,
    }
    base.write_csv(args.output / "coverage.csv", [coverage])
    architecture = {
        "experiment": "independent Stage-II-only Anomaly v1",
        "future_data_loaded": False, "outer_train_only": True,
        "stage2_invocation_rate": 1.0,
        "model_class": "StageIIOnlyAnomalyModel",
        "forward_signature": "forward(graph, regions)",
        "absent_paths": [
            "Global24", "p_stage1", "A25", "DI25",
            "bounded correction", "FixKeep", "MalBen role-class prototypes",
        ],
        "final_feature": "[g_i, z_anomaly, h_critical, e_anomaly]",
        "final_feature_dim": feature_dim,
        "z_anomaly": "attention-weighted local embedding",
        "e_anomaly": "attention-weighted established local residual",
        "anomaly_residual": "(1-shared_commonness)*sqrt(external_benign_distance*internal_background_distance)",
        "final_classifier": "outer-train StandardScaler + L2 LogisticRegression(C=1)",
        "local_encoder_input": "fold-standardized 74-D Host plus four established anomaly/background columns",
        "missing_host": "deterministic one-row zero Host bag; APK remains in all metrics",
        "region_cache_loaded": bool(cache_loaded),
        "fairness_contract": "same folds, seeds, graph/local widths, 81-D final width and L2 classifier as StageIIOnly-MalBen",
    }
    (args.output / "architecture.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "CARE-Droid independent Stage-II-only Anomaly v1", "",
        "Strict 2017 outer-fold OOF; 2018-2021 are not loaded.",
        "Global24 and p_stage1 do not exist in the model forward path.",
        "Stage-II invocation rate: 100%", "",
        f"BA={overall['BA']:.6f} AUC={overall['AUC']:.6f} F1={overall['F1']:.6f}",
        f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}",
        f"host_coverage={coverage['host_coverage']:.6f}",
        f"apks_without_host={coverage['apks_without_host']}",
    ]
    (args.output / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
