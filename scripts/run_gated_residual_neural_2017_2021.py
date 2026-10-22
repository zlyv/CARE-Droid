#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: " + str(exc))


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_graph_pool_diagnostic import read_csv_rows, read_split_rows
from run_dual_view_error_attribution import compact_matrix_named
from run_sensitive_upstream_structural_eval import label_value, standardize_named
from run_pilot_compact_eval import metrics


YEARS = (2018, 2019, 2020, 2021)
EXPECTED_TEST_COUNTS = {2018: 789, 2019: 399, 2020: 397, 2021: 400}
THRESHOLD = 0.5


def read_gz_csv(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_rows(rows):
    result = {}
    for row in rows:
        sha = row["sha256"].strip().casefold()
        if sha in result:
            raise SystemExit("duplicate SHA: " + sha)
        result[sha] = row
    return result


def support150_names(header):
    names = sorted(
        name for name in header if name.startswith("semrel_support__")
    )
    selected = [
        name for name in names
        if name.split("__")[1] != name.split("__")[2]
    ]
    if len(names) != 155 or len(selected) != 150:
        raise SystemExit("unexpected support150 definition")
    return selected


def coverage_values(row, names):
    pair_count = 0
    anchors = set()
    for name in names:
        if float(row[name]) > 0.0:
            pair_count += 1
            anchors.add(name.split("__")[1])
    return np.asarray(
        [float(pair_count > 0), float(pair_count), float(len(anchors))],
        dtype=np.float64,
    )


def standardize_coverage(train_rows, test_rows, semantic_by_sha, names):
    train = np.asarray(
        [coverage_values(semantic_by_sha[row["sha256"]], names) for row in train_rows],
        dtype=np.float64,
    )
    test = np.asarray(
        [coverage_values(semantic_by_sha[row["sha256"]], names) for row in test_rows],
        dtype=np.float64,
    )
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    usable = scale > 1e-12
    if not np.any(usable):
        return np.zeros((len(train_rows), 0)), np.zeros((len(test_rows), 0)), []
    return (
        (train[:, usable] - mean[usable]) / scale[usable],
        (test[:, usable] - mean[usable]) / scale[usable],
        [name for name, keep in zip(("any", "pair_count", "anchor_count"), usable) if keep],
    )


def presence_values(rows, semantic_by_sha, names):
    return np.asarray(
        [
            [float(any(float(semantic_by_sha[row["sha256"]][name]) > 0.0 for name in names))]
            for row in rows
        ],
        dtype=np.float32,
    )


def labels_for(rows, compact_by_sha):
    return np.asarray(
        [
            label_value(compact_by_sha[row["sha256"]].get("metadata_label", ""))
            for row in rows
        ],
        dtype=np.float32,
    )


class GatedResidualNet(nn.Module):
    def __init__(self, compact_dim, semantic_dim, coverage_dim, hidden, dropout):
        super().__init__()
        self.compact_encoder = nn.Sequential(
            nn.Linear(compact_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(max(semantic_dim, 1), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(compact_dim, 1)
        gate_hidden = max(4, hidden // 2)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden + coverage_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden * 3 + coverage_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, compact, semantic, coverage, presence):
        if semantic.shape[1] == 0:
            semantic = torch.zeros(
                (semantic.shape[0], 1),
                dtype=compact.dtype,
                device=compact.device,
            )
        compact_hidden = self.compact_encoder(compact)
        semantic_hidden = self.semantic_encoder(semantic)
        interaction = compact_hidden * semantic_hidden
        gate_input = torch.cat([semantic_hidden, coverage], dim=1)
        gate = torch.sigmoid(self.gate_head(gate_input)) * presence
        delta_input = torch.cat(
            [compact_hidden, semantic_hidden, interaction, coverage],
            dim=1,
        )
        delta = self.delta_head(delta_input)
        base_logit = self.base_head(compact)
        final_logit = base_logit + gate * delta
        return final_logit, base_logit, gate, delta


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def fit_one(
    x_train,
    s_train,
    c_train,
    p_train,
    y_train,
    x_val,
    s_val,
    c_val,
    p_val,
    y_val,
    config,
    seed,
    device,
):
    set_seed(seed)
    model = GatedResidualNet(
        x_train.shape[1],
        s_train.shape[1],
        c_train.shape[1],
        config["hidden"],
        config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.BCEWithLogitsLoss()
    tx = tensor(x_train, device)
    ts = tensor(s_train, device)
    tc = tensor(c_train, device)
    tp = tensor(p_train, device)
    ty = tensor(y_train.reshape(-1, 1), device)
    vx = tensor(x_val, device)
    vs = tensor(s_val, device)
    vc = tensor(c_val, device)
    vp = tensor(p_val, device)
    vy = tensor(y_val.reshape(-1, 1), device)

    best_score = -float("inf")
    best_epoch = 1
    best_state = None
    patience_left = config["patience"]
    history = []

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, base_logits, _, _ = model(tx, ts, tc, tp)
        main_loss = loss_fn(logits, ty)
        base_loss = loss_fn(base_logits, ty)
        loss = main_loss + config["base_aux_weight"] * base_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _, _, _ = model(vx, vs, vc, vp)
            val_probability = torch.sigmoid(val_logits).detach().cpu().numpy().ravel()
        val_metrics = metrics(y_val.astype(int), val_probability, threshold=THRESHOLD)
        score = float(val_metrics["balanced_accuracy"])
        history.append({"epoch": epoch, **val_metrics})

        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = config["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_epoch, history


def evaluate_model(model, compact, semantic, coverage, presence, labels, device):
    model.eval()
    with torch.no_grad():
        logits, base_logits, gate, delta = model(
            tensor(compact, device),
            tensor(semantic, device),
            tensor(coverage, device),
            tensor(presence, device),
        )
        probability = torch.sigmoid(logits).cpu().numpy().ravel()
        base_probability = torch.sigmoid(base_logits).cpu().numpy().ravel()
        gate_value = gate.cpu().numpy().ravel()
        delta_value = delta.cpu().numpy().ravel()
    result = metrics(labels.astype(int), probability, threshold=THRESHOLD)
    return result, probability, base_probability, gate_value, delta_value


def build_fold_data(train_rows, test_rows, compact_by_sha, compact_schema, semantic_by_sha, names150):
    compact_train, compact_test, compact_names = compact_matrix_named(
        train_rows, test_rows, compact_by_sha, compact_schema
    )
    semantic_train, semantic_test, semantic_names = standardize_named(
        train_rows, test_rows, semantic_by_sha, names150
    )
    coverage_train, coverage_test, coverage_names = standardize_coverage(
        train_rows, test_rows, semantic_by_sha, names150
    )
    presence_train = presence_values(train_rows, semantic_by_sha, names150)
    presence_test = presence_values(test_rows, semantic_by_sha, names150)
    return {
        "compact_train": compact_train,
        "compact_test": compact_test,
        "semantic_train": semantic_train,
        "semantic_test": semantic_test,
        "coverage_train": coverage_train,
        "coverage_test": coverage_test,
        "presence_train": presence_train,
        "presence_test": presence_test,
        "compact_names": compact_names,
        "semantic_names": semantic_names,
        "coverage_names": coverage_names,
    }


CONFIGS = [
    {
        "name": "gated16_d10",
        "hidden": 16,
        "dropout": 0.10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "base_aux_weight": 0.20,
        "max_epochs": 800,
        "patience": 100,
    },
    {
        "name": "gated32_d10",
        "hidden": 32,
        "dropout": 0.10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "base_aux_weight": 0.20,
        "max_epochs": 800,
        "patience": 100,
    },
    {
        "name": "gated32_d25",
        "hidden": 32,
        "dropout": 0.25,
        "learning_rate": 1e-3,
        "weight_decay": 3e-4,
        "base_aux_weight": 0.20,
        "max_epochs": 800,
        "patience": 100,
    },
    {
        "name": "gated32_strong_aux",
        "hidden": 32,
        "dropout": 0.15,
        "learning_rate": 7e-4,
        "weight_decay": 3e-4,
        "base_aux_weight": 0.50,
        "max_epochs": 800,
        "patience": 100,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--compact-schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--semantic-2017", type=Path, default=Path("artifacts/clean-local-context-2017/semantic_relation_features.csv.gz"))
    parser.add_argument("--semantic-future", type=Path, default=Path("artifacts/clean-local-context-future/semantic_relation_features.csv.gz"))
    parser.add_argument("--rolling-splits", type=Path, default=Path("artifacts/p9-2017-rolling-splits/split_manifest.csv"))
    parser.add_argument("--future-splits", type=Path, default=Path("artifacts/p1-splits-fixed-2017-test-2018-2021/split_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gated-residual-neural-2017-2021"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--configs", nargs="+", default=[config["name"] for config in CONFIGS])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto" else "cpu"
    )
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    selected_configs = [config for config in CONFIGS if config["name"] in set(args.configs)]
    if not selected_configs:
        raise SystemExit("no matching configs")

    compact_by_sha = normalized_rows(read_csv_rows(args.compact_features))
    compact_schema = json.loads(args.compact_schema.read_text(encoding="utf-8"))
    semantic_2017 = read_gz_csv(args.semantic_2017)
    semantic_future = read_gz_csv(args.semantic_future)
    names150 = support150_names(semantic_2017[0].keys())
    if support150_names(semantic_future[0].keys()) != names150:
        raise SystemExit("2017/future support150 columns differ")
    semantic_by_sha = normalized_rows(semantic_2017 + semantic_future)

    rolling_parts = read_split_rows(args.rolling_splits)
    rolling_splits = {
        name: parts
        for name, parts in rolling_parts.items()
        if "rolling" in name and parts["train"] and parts["test"]
    }
    if not rolling_splits:
        raise SystemExit("no rolling splits found")

    future_parts = read_split_rows(args.future_splits)
    future_splits = {}
    split_to_year = {}
    for name, parts in future_parts.items():
        for year in YEARS:
            if name == f"train_fixed_2017__test_{year}":
                if len(parts["train"]) != 799 or len(parts["test"]) != EXPECTED_TEST_COUNTS[year]:
                    raise SystemExit(f"unexpected fixed split size: {name}")
                future_splits[name] = parts
                split_to_year[name] = year
    if set(split_to_year.values()) != set(YEARS):
        raise SystemExit("future fixed splits are incomplete")

    print("======================================")
    print("Gated residual neural network")
    print("======================================")
    print("device =", device)
    print("rolling splits =", sorted(rolling_splits))
    print("configs =", [config["name"] for config in selected_configs])
    print("threshold =", THRESHOLD)
    print("seeds =", args.seeds)

    cv_rows = []
    config_summaries = []

    for config in selected_configs:
        print()
        print("config =", config["name"], flush=True)
        for split_name in sorted(rolling_splits):
            train_rows = rolling_splits[split_name]["train"]
            val_rows = rolling_splits[split_name]["test"]
            data = build_fold_data(
                train_rows,
                val_rows,
                compact_by_sha,
                compact_schema,
                semantic_by_sha,
                names150,
            )
            y_train = labels_for(train_rows, compact_by_sha)
            y_val = labels_for(val_rows, compact_by_sha)
            for seed in args.seeds:
                model, best_epoch, history = fit_one(
                    data["compact_train"],
                    data["semantic_train"],
                    data["coverage_train"],
                    data["presence_train"],
                    y_train,
                    data["compact_test"],
                    data["semantic_test"],
                    data["coverage_test"],
                    data["presence_test"],
                    y_val,
                    config,
                    seed,
                    device,
                )
                best = max(history, key=lambda row: row["balanced_accuracy"])
                row = {
                    "config": config["name"],
                    "split_name": split_name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "BA": best["balanced_accuracy"],
                    "AUC": best["auc"],
                    "ACC": best["accuracy"],
                    "F1": best["f1"],
                    "FP": best["fp"],
                    "FN": best["fn"],
                }
                cv_rows.append(row)
                print(
                    f"{split_name} seed={seed} "
                    f"epoch={best_epoch} BA={best['balanced_accuracy']:.6f} "
                    f"AUC={best['auc']:.6f}",
                    flush=True,
                )

        current = [row for row in cv_rows if row["config"] == config["name"]]
        by_split = defaultdict(list)
        for row in current:
            by_split[row["split_name"]].append(row)
        split_ba = [float(np.mean([row["BA"] for row in rows])) for rows in by_split.values()]
        config_summaries.append({
            "config": config["name"],
            "mean_BA": float(np.mean(split_ba)),
            "min_split_BA": float(np.min(split_ba)),
            "mean_AUC": float(np.mean([row["AUC"] for row in current])),
            "mean_ACC": float(np.mean([row["ACC"] for row in current])),
            "median_best_epoch": int(round(np.median([row["best_epoch"] for row in current]))),
        })

    config_summaries.sort(
        key=lambda row: (row["mean_BA"], row["min_split_BA"], row["mean_AUC"]),
        reverse=True,
    )
    best_config_name = config_summaries[0]["config"]
    best_config = next(config for config in selected_configs if config["name"] == best_config_name)
    final_epochs = config_summaries[0]["median_best_epoch"]
    final_epochs = max(50, final_epochs)

    print()
    print("===== source-only config ranking =====")
    for row in config_summaries:
        print(
            row["config"],
            f"BA={row['mean_BA']:.6f}",
            f"minBA={row['min_split_BA']:.6f}",
            f"AUC={row['mean_AUC']:.6f}",
            f"epoch={row['median_best_epoch']}",
        )
    print("selected config =", best_config_name)
    print("selected final epochs =", final_epochs)

    # Train the source-selected configuration on all 799 2017 samples and
    # evaluate every fixed future year with threshold 0.5.
    fixed_train_rows = next(iter(future_splits.values()))["train"]
    future_rows = []
    future_predictions = []
    for split_name in sorted(future_splits, key=lambda name: split_to_year[name]):
        test_rows = future_splits[split_name]["test"]
        data = build_fold_data(
            fixed_train_rows,
            test_rows,
            compact_by_sha,
            compact_schema,
            semantic_by_sha,
            names150,
        )
        y_train = labels_for(fixed_train_rows, compact_by_sha)
        y_test = labels_for(test_rows, compact_by_sha)
        seed_results = []
        for seed in args.seeds:
            model, _, _ = fit_one(
                data["compact_train"],
                data["semantic_train"],
                data["coverage_train"],
                data["presence_train"],
                y_train,
                data["compact_train"],
                data["semantic_train"],
                data["coverage_train"],
                data["presence_train"],
                y_train,
                {**best_config, "max_epochs": final_epochs, "patience": final_epochs + 1},
                seed,
                device,
            )
            result, probability, base_probability, gate, delta = evaluate_model(
                model,
                data["compact_test"],
                data["semantic_test"],
                data["coverage_test"],
                data["presence_test"],
                y_test,
                device,
            )
            seed_results.append(result)
            for row, label, prob, base_p, gate_value, delta_value in zip(
                test_rows,
                y_test,
                probability,
                base_probability,
                gate,
                delta,
            ):
                future_predictions.append({
                    "year": split_to_year[split_name],
                    "split_name": split_name,
                    "sha256": row["sha256"],
                    "seed": seed,
                    "label": int(label),
                    "probability": float(prob),
                    "base_probability": float(base_p),
                    "gate": float(gate_value),
                    "semantic_delta": float(delta_value),
                })
        future_rows.append({
            "year": split_to_year[split_name],
            "split_name": split_name,
            "test_n": len(test_rows),
            "BA": float(np.mean([row["balanced_accuracy"] for row in seed_results])),
            "AUC": float(np.mean([row["auc"] for row in seed_results])),
            "ACC": float(np.mean([row["accuracy"] for row in seed_results])),
            "F1": float(np.mean([row["f1"] for row in seed_results])),
            "FP": float(np.mean([row["fp"] for row in seed_results])),
            "FN": float(np.mean([row["fn"] for row in seed_results])),
        })
        print(
            f"future {split_to_year[split_name]} "
            f"BA={future_rows[-1]['BA']:.6f} "
            f"AUC={future_rows[-1]['AUC']:.6f} "
            f"ACC={future_rows[-1]['ACC']:.6f}",
            flush=True,
        )

    future_rows.append({
        "year": "four_year_mean",
        "BA": float(np.mean([row["BA"] for row in future_rows])),
        "AUC": float(np.mean([row["AUC"] for row in future_rows])),
        "ACC": float(np.mean([row["ACC"] for row in future_rows])),
        "F1": float(np.mean([row["F1"] for row in future_rows])),
        "FP": float(np.mean([row["FP"] for row in future_rows])),
        "FN": float(np.mean([row["FN"] for row in future_rows])),
    })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "source_cv_per_seed.csv", cv_rows)
    write_csv(args.output / "source_cv_summary.csv", config_summaries)
    write_csv(args.output / "future_per_year.csv", future_rows)
    write_csv(args.output / "future_predictions.csv", future_predictions)
    (args.output / "selected_config.json").write_text(
        json.dumps(
            {
                "config": best_config,
                "selected_config_name": best_config_name,
                "final_epochs": final_epochs,
                "threshold": THRESHOLD,
                "device": str(device),
                "seeds": args.seeds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
