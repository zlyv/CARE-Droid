#!/usr/bin/env python3
"""Independent strict-2017 Stage-II-only MalBen model and OOF evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (
    HERE, ROOT, ROOT / "care-malben-host-v2", ROOT / "care-malben-contrast-v2",
    ROOT / "care-rolegraph-conditioned-v4-script",
):
    sys.path.insert(0, str(path))

import run_care_malben_role_source_v2 as malben
from shared_role_prototypes import evidence_columns, fit_role_prototypes
from source_label_contract import install_source_label_contract

v4 = malben.v4


def fit_scaled_shared_role_bags(*args, **kwargs):
    train, test, audit = malben.fit_scaled_role_bags(*args, **kwargs)
    audit = dict(audit)
    audit["prototype_similarity"] = "exp(-RMS-Euclidean/shared-role-or-global-scale)"
    audit["prototype_scale_scope"] = "outer-train only; one common tau for S_ben and S_mal"
    return train, test, audit


class StageIIOnlyModel(nn.Module):
    """No Global24/base-logit slots exist in this model or forward signature."""

    def __init__(self, args):
        super().__init__()
        graph_dim = int(args.graph_dim)
        local_dim = int(args.embedding_dim)
        hidden_dim = int(args.hidden_dim)
        self.graph_encoder = v4.RoleGraphEncoder(args)
        self.region_encoder = nn.Sequential(
            nn.Linear(malben.OLD_AUGMENTED_DIM, hidden_dim), nn.ReLU(),
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
        graph_token, _ = self.graph_encoder(graph["node"], graph["edge"], graph["global"])
        local = self.region_encoder(regions[:, :malben.OLD_AUGMENTED_DIM])
        residual = regions[:, malben.BASE_REGION_DIM + 3]
        score = self.local_risk(local).reshape(-1)
        score = score + nn.functional.softplus(self.residual_beta_raw) * torch.log1p(
            torch.clamp(residual, min=0.0)
        )
        score = score + torch.mv(self.local_key(local), self.graph_query(graph_token)) / math.sqrt(local.shape[1])
        attention = torch.softmax(score, dim=0)
        critical = local[torch.argmax(score)]
        role_columns = regions[:, malben.OLD_AUGMENTED_DIM:]
        margin = role_columns[:, 2].clamp(-1.0, 1.0)
        e_mal_region = torch.relu(margin)
        e_ben_region = torch.relu(-margin)
        z_mal = torch.sum((attention * e_mal_region)[:, None] * local, dim=0)
        z_ben = torch.sum((attention * e_ben_region)[:, None] * local, dim=0)
        e_contrast = torch.sum(attention * (e_mal_region - e_ben_region))
        feature = torch.cat([graph_token, z_mal - z_ben, critical, e_contrast.reshape(1)])
        self.last_feature = feature
        return self.auxiliary_head(feature).reshape(())


def write_csv(path: Path, rows: list[dict]) -> None:
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


def class_weights(labels):
    labels = np.asarray(labels, dtype=np.int64)
    result = np.ones(len(labels), dtype=np.float32)
    for label in (0, 1):
        selected = labels == label
        result[selected] = len(labels) / max(2 * int(np.sum(selected)), 1)
    return result


def forward_batch(model, graph, bags, indices):
    logits = []
    for value in indices:
        index = int(value)
        logits.append(model({key: item[index] for key, item in graph.items()}, bags[index]))
    return torch.stack(logits)


def train_epochs(model, optimizer, graph, bags, labels, weights, epochs, args, seed):
    device = next(model.parameters()).device
    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)
    rng = np.random.RandomState(int(seed) + 911)
    final = float("nan")
    for _ in range(int(epochs)):
        model.train(); order = rng.permutation(len(labels)); total = 0.0
        for start in range(0, len(order), int(args.batch_size)):
            batch = order[start:start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, graph, bags, batch)
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            element = nn.functional.binary_cross_entropy_with_logits(logits, labels_t[idx], reduction="none")
            loss = torch.sum(element * weights_t[idx]) / torch.sum(weights_t[idx])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)); optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
        final = total / len(labels)
    return final


def evaluate_loss(model, graph_values, bags, labels, args, device):
    graph = v4.graph_tensors(graph_values, device)
    bags_t = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
    weights = torch.as_tensor(class_weights(labels), dtype=torch.float32, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = forward_batch(model, graph, bags_t, np.arange(len(labels)))
        element = nn.functional.binary_cross_entropy_with_logits(logits, labels_t, reduction="none")
        return float((torch.sum(element * weights) / torch.sum(weights)).cpu())


def select_epochs(s7, raw_bags, labels, args, device, seed):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=float(args.early_stop_fraction),
        random_state=int(seed) + int(args.split_seed),
    )
    fit_index, val_index = next(splitter.split(np.zeros(len(labels)), labels))
    fit_raw = [raw_bags[i] for i in fit_index]; val_raw = [raw_bags[i] for i in val_index]
    fit_bags, val_bags, _ = fit_scaled_shared_role_bags(
        fit_raw, val_raw, np.asarray(labels)[fit_index], args, int(seed) + 707
    )
    scaler = v4.fit_role_graph_scaler(np.asarray(s7)[fit_index])
    fit_graph_values = v4.transform_role_graphs(np.asarray(s7)[fit_index], scaler)
    val_graph_values = v4.transform_role_graphs(np.asarray(s7)[val_index], scaler)
    fit_graph = v4.graph_tensors(fit_graph_values, device)
    fit_bags_t = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in fit_bags]
    fit_labels = np.asarray(labels)[fit_index]
    v4.seed_everything(seed)
    model = StageIIOnlyModel(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    best_loss = float("inf"); best_epoch = 1; patience = 0
    for epoch in range(1, int(args.max_epochs) + 1):
        train_epochs(model, optimizer, fit_graph, fit_bags_t, fit_labels,
                     class_weights(fit_labels), 1, args, int(seed) + epoch * 17)
        value = evaluate_loss(model, val_graph_values, val_bags, np.asarray(labels)[val_index], args, device)
        if value < best_loss - float(args.early_stop_min_delta):
            best_loss = value; best_epoch = epoch; patience = 0
        else:
            patience += 1
        if patience >= int(args.early_stop_patience):
            break
    del model, fit_graph, fit_bags_t
    if device.type == "cuda": torch.cuda.empty_cache()
    return best_epoch, best_loss


def extract_features(model, graph_values, bags, device):
    graph = v4.graph_tensors(graph_values, device)
    bags_t = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
    output = []
    model.eval()
    with torch.no_grad():
        for index in range(len(bags)):
            model({key: value[index] for key, value in graph.items()}, bags_t[index])
            output.append(model.last_feature.detach().cpu().numpy())
    return np.stack(output)


def fit_fold_features(s7_train, s7_val, raw_train, raw_val, labels, seeds, args, device):
    train_features = []; val_features = []; audits = []
    for seed in seeds:
        epochs, selection_loss = select_epochs(s7_train, raw_train, labels, args, device, seed)
        train_bags, val_bags, proto_audit = fit_scaled_shared_role_bags(
            raw_train, raw_val, labels, args, int(seed) + 1707
        )
        graph_scaler = v4.fit_role_graph_scaler(s7_train)
        train_graph_values = v4.transform_role_graphs(s7_train, graph_scaler)
        val_graph_values = v4.transform_role_graphs(s7_val, graph_scaler)
        train_graph = v4.graph_tensors(train_graph_values, device)
        train_bags_t = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in train_bags]
        v4.seed_everything(seed)
        model = StageIIOnlyModel(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
        final_loss = train_epochs(model, optimizer, train_graph, train_bags_t, labels,
                                  class_weights(labels), epochs, args, int(seed) + 3301)
        train_features.append(extract_features(model, train_graph_values, train_bags, device))
        val_features.append(extract_features(model, val_graph_values, val_bags, device))
        audits.append({
            "seed": int(seed), "selected_epochs": int(epochs),
            "selection_loss": float(selection_loss), "final_training_loss": float(final_loss),
            "model_parameter_n": int(sum(value.numel() for value in model.parameters())),
            **proto_audit,
        })
        del model, train_graph, train_bags_t
        if device.type == "cuda": torch.cuda.empty_cache()
    return np.mean(np.stack(train_features), axis=0), np.mean(np.stack(val_features), axis=0), audits


def metrics(labels, probability):
    prediction = np.asarray(probability) >= 0.5
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "BA": float(balanced_accuracy_score(labels, prediction)),
        "AUC": float(roc_auc_score(labels, probability)),
        "F1": float(f1_score(labels, prediction, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"))
    parser.add_argument("--p72-oof", type=Path, default=Path("artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"))
    parser.add_argument("--source-multiscale-features", type=Path, default=Path("artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"))
    parser.add_argument("--region-cache", type=Path, default=Path("artifacts/care-source-local-assets-v2/regions_2017.npz"))
    parser.add_argument("--rebuild-region-cache", action="store_true")
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
    parser.add_argument("--shared-prototypes", type=int, default=32)
    parser.add_argument("--benign-prototypes", type=int, default=24)
    parser.add_argument("--prototype-regions-per-apk", type=int, default=16)
    parser.add_argument("--internal-background-coverage", type=float, default=0.70)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("artifacts/care-stage2only-malben-v1/source"))
    args = parser.parse_args()

    install_source_label_contract(v4)
    malben.fit_role_prototypes = fit_role_prototypes
    malben.evidence_columns = evidence_columns
    device = v4.resolve_device(args.device)
    seeds = v4.parse_ints(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, source_rows, _, compact_by_sha, compact_schema, original_names = v4.audit.load_inputs(args)
    labels = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    folds = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    if len(labels) != 799:
        raise RuntimeError("Stage-II-only must retain all 799 source APKs")
    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    source_s7, _ = v4.v3.load_source_s7(source_rows, endpoint_cache, args)
    host_bags, bag_audit, cache_loaded = v4.get_source_host_bags(source_rows, endpoint_cache, args)
    probability = np.full(len(labels), np.nan, dtype=np.float64)
    fit_rows = []
    feature_dim = None
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold); val = np.flatnonzero(folds == fold)
        fold_seeds = [100000 * int(fold) + seed for seed in seeds]
        train_feature, val_feature, audits = fit_fold_features(
            source_s7[train], source_s7[val], [host_bags[i] for i in train],
            [host_bags[i] for i in val], labels[train], fold_seeds, args, device,
        )
        feature_dim = int(train_feature.shape[1])
        if feature_dim != int(args.graph_dim) + 2 * int(args.embedding_dim) + 1:
            raise RuntimeError("Stage-II feature contract failed")
        scaler = StandardScaler().fit(train_feature)
        classifier = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=2000,
            random_state=int(args.split_seed) + int(fold),
        )
        classifier.fit(scaler.transform(train_feature), labels[train])
        probability[val] = classifier.predict_proba(scaler.transform(val_feature))[:, 1]
        fit_rows.extend({"fold": int(fold), **row} for row in audits)
        print("StageIIOnly fold", int(fold), "done", flush=True)
    if np.any(~np.isfinite(probability)):
        raise RuntimeError("incomplete OOF probabilities")

    with np.load(args.region_cache, allow_pickle=False) as region_saved:
        real_host_count = np.asarray(region_saved["real_host_count"], dtype=np.int64)
    if len(real_host_count) != 799:
        raise RuntimeError("region coverage does not cover all 799 APKs")
    overall = metrics(labels, probability)
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        selected = folds == fold
        fold_rows.append({"fold": int(fold), **metrics(labels[selected], probability[selected]),
                          "host_coverage": float(np.mean(real_host_count[selected] > 0)),
                          "stage2_invocation_rate": 1.0})
    predictions = [{
        "sha256": str(manifest[i]["sha256"]).strip().lower(), "label": int(labels[i]),
        "fold": int(folds[i]), "StageIIOnly_MalBen__prob": float(probability[i]),
        "has_host": int(real_host_count[i] > 0), "real_host_count": int(real_host_count[i]),
    } for i in range(799)]
    write_csv(args.output / "fold_metrics.csv", fold_rows)
    write_csv(args.output / "oof_predictions.csv", predictions)
    write_csv(args.output / "fit_audit.csv", fit_rows)
    coverage = [{
        "total_apks": 799, "apks_with_host": int(np.sum(real_host_count > 0)),
        "apks_without_host": int(np.sum(real_host_count == 0)),
        "host_coverage": float(np.mean(real_host_count > 0)),
        "local_region_min": int(np.min(real_host_count)),
        "local_region_median": float(np.median(real_host_count)),
        "local_region_p90": float(np.percentile(real_host_count, 90)),
        "local_region_max": int(np.max(real_host_count)),
        "stage2_invocation_rate": 1.0,
    }]
    write_csv(args.output / "coverage.csv", coverage)
    mean_values = {name: float(np.mean([float(row[name]) for row in fold_rows]))
                   for name in ("BA", "AUC", "F1")}
    summary = [
        {"method": "StageIIOnly-MalBen", "aggregation": "OOF_pooled", **overall,
         "stage2_invocation_rate": 1.0, "host_coverage": coverage[0]["host_coverage"]},
        {"method": "StageIIOnly-MalBen", "aggregation": "fold_mean",
         "BA": mean_values["BA"], "AUC": mean_values["AUC"], "F1": mean_values["F1"],
         "TN": "", "FP": "", "FN": "", "TP": "",
         "stage2_invocation_rate": 1.0, "host_coverage": coverage[0]["host_coverage"]},
    ]
    write_csv(args.output / "source_summary.csv", summary)
    architecture = {
        "experiment": "independent Stage-II-only MalBen v1", "future_data_loaded": False,
        "outer_train_only": True, "stage2_invocation_rate": 1.0,
        "model_class": "StageIIOnlyModel", "forward_signature": "forward(graph, regions)",
        "absent_paths": ["Global24", "p_stage1", "A25", "DI25", "bounded correction", "FixKeep"],
        "final_feature": "[g_i, z_mal-z_ben, h_critical, e_mal-e_ben]",
        "final_feature_dim": feature_dim,
        "final_classifier": "outer-train StandardScaler + L2 LogisticRegression(C=1)",
        "local_encoder_input": "standardized 74-D Host + four established anomaly/background columns; no Stage-I fields",
        "prototype_space": "outer-train standardized raw 74-D Host space",
        "similarity_scale": "shared tau_role for both class similarities; shared tau_global on either-side fallback",
        "missing_host": "deterministic one-row zero Host bag; APK remains in all metrics",
        "region_cache_loaded": bool(cache_loaded),
    }
    (args.output / "architecture.json").write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "CARE-Droid independent Stage-II-only MalBen v1", "",
        "Strict 2017 outer-fold OOF; 2018-2021 are not loaded.",
        "Global24 and p_stage1 do not exist in the model forward path.",
        "Stage-II invocation rate: 100%", "",
        f"BA={overall['BA']:.6f} AUC={overall['AUC']:.6f} F1={overall['F1']:.6f}",
        f"TN={overall['TN']} FP={overall['FP']} FN={overall['FN']} TP={overall['TP']}",
        f"host_coverage={coverage[0]['host_coverage']:.6f}",
        f"apks_without_host={coverage[0]['apks_without_host']}",
    ]
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
