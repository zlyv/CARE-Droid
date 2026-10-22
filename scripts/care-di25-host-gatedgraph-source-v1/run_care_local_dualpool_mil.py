#!/usr/bin/env python3
"""CARE-Droid v1: Global24 + A25 + local sparse/diffuse dual-pool MIL.

This is the first classifier experiment after the label-free local-region asset
audit.  It keeps the existing first layer and route completely frozen:

* layer 1: Global24 (24 dimensions; no role-ratio dimensions);
* route: the frozen A25 decisions from the previous fair experiment;
* reference expert: frozen ordinary Joint-S7 probabilities.

Only the new second-layer representation changes.  Two small MIL experts are
compared:

* CARE_HostMIL: one local instance per observed caller method;
* CARE_HostEgoMIL: caller-method instances plus unique undirected 1-hop egos.

Every region uses the audited 72-dimensional label-free vector plus a two-way
scale indicator.  A shared encoder produces region embeddings and region risk
logits.  Top-fraction pooling models sparse payload evidence, while mean pooling
models diffuse evidence.  The two pooled embeddings and Global24 are classified
jointly.  There is no confidence-overwrite rule and no community module in v1.

2017 uses the fixed five outer folds.  2018--2021 train only on all 2017 APKs
and remain diagnostic.  Previous future labels are never used for training,
routing, scaling, epoch selection, or hyperparameter selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - environment check
    raise SystemExit("PyTorch is required: " + str(exc))

import run_care_local_region_asset_audit as care_region  # noqa: E402
import run_global24_r_joints7_2017_2021 as g24  # noqa: E402


audit = g24.audit
evidence = g24.evidence
acn = g24.acn
host_asset = g24.host_asset
VARIANT = g24.VARIANT
THRESHOLD = g24.THRESHOLD
REGION_DIM = int(care_region.REGION_FEATURE_DIM) + 2


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_ints(text):
    values = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            value = int(token)
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("integer list is empty")
    return values


def norm_sha(value):
    return str(value or "").strip().casefold()


def resolve_device(name):
    name = str(name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def empty_region():
    return np.zeros((1, REGION_DIM), dtype=np.float32)


def build_region_bags(rows, cache, progress_every=100):
    host_bags = []
    ego_bags = []
    audits = []
    for index, row in enumerate(rows, 1):
        sha = norm_sha(row["sha256"])
        if sha not in cache:
            raise RuntimeError("endpoint cache is missing: " + sha)
        asset = host_asset.reconstruct_host_roles(cache[sha]["payload"])
        host_roles = asset["host_roles"]
        edges = asset["host_host_edges"]
        hosts = sorted(host_roles)
        adjacency = care_region.undirected_adjacency(hosts, edges)
        host_regions = [frozenset([host]) for host in hosts]
        ego_regions = care_region.unique_ego_regions(adjacency)
        host_vectors = [
            np.concatenate([
                care_region.region_vector(region, host_roles, edges),
                np.asarray([1.0, 0.0]),
            ])
            for region in host_regions
        ]
        ego_vectors = [
            np.concatenate([
                care_region.region_vector(region, host_roles, edges),
                np.asarray([0.0, 1.0]),
            ])
            for region in ego_regions
        ]
        host_matrix = (
            np.asarray(host_vectors, dtype=np.float32)
            if host_vectors else empty_region()
        )
        ego_matrix = (
            np.asarray(ego_vectors, dtype=np.float32)
            if ego_vectors else np.zeros((0, REGION_DIM), dtype=np.float32)
        )
        if host_matrix.shape[1] != REGION_DIM or ego_matrix.shape[1] != REGION_DIM:
            raise RuntimeError("bad local-region matrix width")
        if not np.isfinite(host_matrix).all() or not np.isfinite(ego_matrix).all():
            raise RuntimeError("non-finite local-region matrix")
        host_bags.append(host_matrix)
        ego_bags.append(ego_matrix)
        audits.append({
            "sha256": sha,
            "metadata_year": str(row.get("metadata_year", "")).strip(),
            "host_region_n": int(len(host_matrix)),
            "real_host_region_n": int(len(host_vectors)),
            "ego_region_n": int(len(ego_matrix)),
            "host_was_empty": int(not host_vectors),
            "hostego_total_region_n": int(len(host_matrix) + len(ego_matrix)),
        })
        if index % int(progress_every) == 0 or index == len(rows):
            print("  local bags:", index, "/", len(rows), flush=True)
    return host_bags, ego_bags, audits


def save_region_cache(path, rows, host_bags, ego_bags):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    host_offsets = [0]
    ego_offsets = [0]
    for bag in host_bags:
        host_offsets.append(host_offsets[-1] + len(bag))
    for bag in ego_bags:
        ego_offsets.append(ego_offsets[-1] + len(bag))
    host_values = np.concatenate(host_bags, axis=0)
    ego_values = (
        np.concatenate([bag for bag in ego_bags if len(bag)], axis=0)
        if any(len(bag) for bag in ego_bags)
        else np.zeros((0, REGION_DIM), dtype=np.float32)
    )
    np.savez_compressed(
        path,
        sha256=np.asarray([norm_sha(row["sha256"]) for row in rows]),
        host_values=host_values.astype(np.float32),
        host_offsets=np.asarray(host_offsets, dtype=np.int64),
        ego_values=ego_values.astype(np.float32),
        ego_offsets=np.asarray(ego_offsets, dtype=np.int64),
        region_dim=np.asarray(REGION_DIM, dtype=np.int64),
    )


def load_region_cache(path, rows):
    path = Path(path)
    if not path.exists():
        return None
    saved = np.load(path, allow_pickle=False)
    expected = np.asarray([norm_sha(row["sha256"]) for row in rows])
    if (
        int(saved["region_dim"]) != REGION_DIM
        or not np.array_equal(saved["sha256"].astype(str), expected)
    ):
        return None
    host_values = np.asarray(saved["host_values"], dtype=np.float32)
    ego_values = np.asarray(saved["ego_values"], dtype=np.float32)
    host_offsets = np.asarray(saved["host_offsets"], dtype=np.int64)
    ego_offsets = np.asarray(saved["ego_offsets"], dtype=np.int64)
    host_bags = [
        host_values[host_offsets[i]:host_offsets[i + 1]]
        for i in range(len(rows))
    ]
    ego_bags = [
        ego_values[ego_offsets[i]:ego_offsets[i + 1]]
        for i in range(len(rows))
    ]
    return host_bags, ego_bags


def get_region_bags(rows, cache, args):
    loaded = None if args.rebuild_region_cache else load_region_cache(
        args.region_cache, rows
    )
    if loaded is not None:
        host_bags, ego_bags = loaded
        print("Loaded local-region cache:", args.region_cache, flush=True)
        audits = [{
            "sha256": norm_sha(row["sha256"]),
            "metadata_year": str(row.get("metadata_year", "")).strip(),
            "host_region_n": int(len(host_bags[i])),
            "real_host_region_n": "cached",
            "ego_region_n": int(len(ego_bags[i])),
            "host_was_empty": "cached",
            "hostego_total_region_n": int(len(host_bags[i]) + len(ego_bags[i])),
        } for i, row in enumerate(rows)]
        return host_bags, ego_bags, audits, True
    print("Building local-region cache...", flush=True)
    host_bags, ego_bags, audits = build_region_bags(
        rows, cache, args.progress_every
    )
    save_region_cache(args.region_cache, rows, host_bags, ego_bags)
    return host_bags, ego_bags, audits, False


def variant_bags(host_bags, ego_bags, variant):
    if variant == "HostMIL":
        return host_bags
    if variant == "HostEgoMIL":
        return [
            np.concatenate([host, ego], axis=0) if len(ego) else host
            for host, ego in zip(host_bags, ego_bags)
        ]
    raise ValueError("unknown MIL variant: " + str(variant))


def fit_equal_apk_scaler(bags, indices):
    bag_means = []
    bag_seconds = []
    for index in indices:
        bag = np.asarray(bags[int(index)], dtype=np.float64)
        bag_means.append(np.mean(bag, axis=0))
        bag_seconds.append(np.mean(np.square(bag), axis=0))
    mean = np.mean(np.asarray(bag_means), axis=0)
    second = np.mean(np.asarray(bag_seconds), axis=0)
    variance = np.maximum(second - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    # Scale indicators are already in [0,1] and must keep their meaning.
    mean[-2:] = 0.0
    std[-2:] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def tensorize_bags(bags, indices, mean, std, device):
    return [
        torch.as_tensor(
            (np.asarray(bags[int(index)], dtype=np.float32) - mean) / std,
            dtype=torch.float32,
            device=device,
        )
        for index in indices
    ]


class DualPoolMIL(nn.Module):
    def __init__(
        self,
        global_dim,
        region_dim,
        hidden_dim,
        embedding_dim,
        sparse_fraction,
        dropout,
    ):
        super().__init__()
        self.sparse_fraction = float(sparse_fraction)
        self.encoder = nn.Sequential(
            nn.Linear(region_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.instance_head = nn.Linear(embedding_dim, 1)
        fused_dim = global_dim + 2 * embedding_dim + 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features, regions):
        embedding = self.encoder(regions)
        instance_logit = self.instance_head(embedding).reshape(-1)
        k = max(1, int(np.ceil(len(instance_logit) * self.sparse_fraction)))
        top_values, top_index = torch.topk(instance_logit, k=k, largest=True)
        sparse_embedding = torch.mean(embedding[top_index], dim=0)
        diffuse_embedding = torch.mean(embedding, dim=0)
        sparse_logit = torch.mean(top_values)
        diffuse_logit = torch.mean(instance_logit)
        max_logit = torch.max(instance_logit)
        fused = torch.cat([
            global_features.reshape(-1),
            sparse_embedding,
            diffuse_embedding,
            torch.stack([sparse_logit, diffuse_logit, max_logit]),
        ])
        apk_logit = self.classifier(fused).reshape(())
        diagnostic = torch.stack([sparse_logit, diffuse_logit, max_logit])
        return apk_logit, diagnostic


def train_one_model(
    global_train,
    region_train,
    labels,
    args,
    device,
    seed,
):
    seed_everything(seed)
    model = DualPoolMIL(
        global_dim=global_train.shape[1],
        region_dim=REGION_DIM,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        sparse_fraction=args.sparse_fraction,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    global_tensor = torch.as_tensor(
        global_train, dtype=torch.float32, device=device
    )
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
    rng = np.random.RandomState(int(seed) + 1701)
    n = len(labels)
    model.train()
    final_loss = float("nan")
    for epoch in range(int(args.max_epochs)):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, int(args.batch_size)):
            batch = order[start:start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = []
            for local_index in batch:
                logit, _ = model(
                    global_tensor[int(local_index)],
                    region_train[int(local_index)],
                )
                logits.append(logit)
            stacked = torch.stack(logits)
            batch_index = torch.as_tensor(
                batch, dtype=torch.long, device=device
            )
            loss = criterion(stacked, label_tensor[batch_index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch)
        final_loss = epoch_loss / n
        if args.verbose_epochs and (
            epoch == 0 or (epoch + 1) % args.verbose_epochs == 0
            or epoch + 1 == args.max_epochs
        ):
            print(
                "    seed", seed, "epoch", epoch + 1, "/", args.max_epochs,
                "loss", f"{final_loss:.6f}", flush=True,
            )
    return model, final_loss


def predict_model(model, global_values, region_values, device):
    global_tensor = torch.as_tensor(
        global_values, dtype=torch.float32, device=device
    )
    probabilities = []
    diagnostics = []
    model.eval()
    with torch.no_grad():
        for index in range(len(region_values)):
            logit, diagnostic = model(global_tensor[index], region_values[index])
            probabilities.append(float(torch.sigmoid(logit).cpu()))
            diagnostics.append(diagnostic.detach().cpu().numpy())
    return (
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(diagnostics, dtype=np.float64),
    )


def fit_predict_ensemble(
    global_train,
    global_test,
    train_bags,
    test_bags,
    labels,
    seeds,
    args,
    device,
):
    train_indices = np.arange(len(train_bags), dtype=np.int64)
    mean, std = fit_equal_apk_scaler(train_bags, train_indices)
    train_tensor_bags = tensorize_bags(
        train_bags, train_indices, mean, std, device
    )
    test_tensor_bags = tensorize_bags(
        test_bags, np.arange(len(test_bags)), mean, std, device
    )
    probabilities = []
    diagnostics = []
    losses = []
    for seed in seeds:
        model, loss = train_one_model(
            global_train,
            train_tensor_bags,
            labels,
            args,
            device,
            int(seed),
        )
        probability, diagnostic = predict_model(
            model, global_test, test_tensor_bags, device
        )
        probabilities.append(probability)
        diagnostics.append(diagnostic)
        losses.append(loss)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        np.mean(np.asarray(probabilities), axis=0),
        np.mean(np.asarray(diagnostics), axis=0),
        {
            "region_scaler_zero_std_n": int(np.sum(std == 1.0)),
            "final_training_loss_mean": float(np.mean(losses)),
            "final_training_loss_by_seed": [float(value) for value in losses],
        },
    )


def reference_arrays(path, rows, source):
    records = read_csv(path)
    by_sha = {norm_sha(row.get("sha256")): row for row in records}
    required = {
        "Global24__prob",
        "A25__score",
        "A25__routed",
        "Global24__A25__JointS7_ordinary__prob",
    }
    if not records or not required.issubset(records[0]):
        raise RuntimeError("frozen reference lacks required A25 columns: " + str(path))
    coarse = []
    ordinary = []
    score = []
    route = []
    for row in rows:
        sha = norm_sha(row["sha256"])
        if sha not in by_sha:
            raise RuntimeError("frozen reference is missing SHA: " + sha)
        record = by_sha[sha]
        coarse.append(float(record["Global24__prob"]))
        ordinary.append(float(record["Global24__A25__JointS7_ordinary__prob"]))
        score.append(float(record["A25__score"]))
        route.append(bool(int(record["A25__routed"])))
        if source:
            if int(record["label"]) != int(row["label"]):
                raise RuntimeError("source reference label mismatch: " + sha)
            if int(record["fold"]) != int(row["fold"]):
                raise RuntimeError("source reference fold mismatch: " + sha)
    return {
        "coarse": np.asarray(coarse, dtype=np.float64),
        "ordinary": np.asarray(ordinary, dtype=np.float64),
        "score": np.asarray(score, dtype=np.float64),
        "route": np.asarray(route, dtype=bool),
    }


def metric_rows(labels, probabilities, coarse, route, year=None):
    rows = []
    for method, values in probabilities.items():
        method_route = (
            np.zeros(len(labels), dtype=bool)
            if method == "Global24" else route
        )
        rows.append(g24.metric_row(
            method, labels, values, coarse, method_route, year
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("artifacts/p1-features/features.csv"))
    parser.add_argument("--schema", type=Path, default=Path("artifacts/p1-features/feature_schema.json"))
    parser.add_argument("--cache", type=Path, default=Path(
        "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz"
    ))
    parser.add_argument("--p72-oof", type=Path, default=Path(
        "artifacts/p72-verified-dependency-source-ablation/p72_oof_predictions.csv"
    ))
    parser.add_argument("--previous-result", type=Path, default=Path(
        "artifacts/global24-a25-route-conditioned-joints7-v1-full"
    ))
    parser.add_argument("--region-cache", type=Path, default=Path(
        "artifacts/care-local-region-cache-v1/regions_2017_2021.npz"
    ))
    parser.add_argument("--rebuild-region-cache", action="store_true")
    parser.add_argument("--years", default="2018,2019,2020,2021")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--sparse-fraction", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose-epochs", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-future", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-local-dualpool-mil-v1"
    ))
    args = parser.parse_args()

    if args.max_epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch size must be positive")
    if not 0.0 < args.sparse_fraction <= 1.0:
        raise SystemExit("--sparse-fraction must be in (0,1]")
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0,1)")
    seeds = parse_ints(args.seeds)
    years = sorted({int(value.strip()) for value in args.years.split(",") if value.strip()})
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Device:", device, "seeds:", seeds, flush=True)

    (
        manifest,
        source_rows,
        clean_rows,
        compact_by_sha,
        compact_schema,
        original_names,
    ) = audit.load_inputs(args)
    source_y = np.asarray([row["label"] for row in manifest], dtype=np.int64)
    source_fold = np.asarray([row["fold"] for row in manifest], dtype=np.int64)
    source_reference = reference_arrays(
        args.previous_result / "source_predictions.csv", manifest, True
    )

    target_rows = []
    target_y = np.zeros(0, dtype=np.int64)
    target_year = np.zeros(0, dtype=np.int64)
    target_reference = None
    if not args.skip_future:
        requested = {str(year) for year in years}
        target_rows = [
            row for row in clean_rows
            if str(row.get("metadata_year", "")).strip() in requested
        ]
        observed = sorted({int(row["metadata_year"]) for row in target_rows})
        if observed != years:
            raise RuntimeError("requested/observed future years differ")
        target_y = np.asarray(
            [audit.future.label_of(row) for row in target_rows], dtype=np.int64
        )
        target_year = np.asarray(
            [int(row["metadata_year"]) for row in target_rows], dtype=np.int64
        )
        target_reference = reference_arrays(
            args.previous_result / "future_predictions_diagnostic.csv",
            target_rows,
            False,
        )

    all_rows = list(source_rows) + list(target_rows)
    endpoint_cache = audit.exp.read_cache(args.cache)
    host_bags, ego_bags, bag_audit, region_cache_loaded = get_region_bags(
        all_rows, endpoint_cache, args
    )
    source_n = len(source_rows)
    source_host = host_bags[:source_n]
    source_ego = ego_bags[:source_n]
    target_host = host_bags[source_n:]
    target_ego = ego_bags[source_n:]
    variants = ("HostMIL", "HostEgoMIL")
    method_for_variant = {
        "HostMIL": "Global24__A25__CARE_HostMIL",
        "HostEgoMIL": "Global24__A25__CARE_HostEgoMIL",
    }
    source_expert = {
        variant: np.full(source_n, np.nan) for variant in variants
    }
    source_diagnostic = {
        variant: np.full((source_n, 3), np.nan) for variant in variants
    }
    training_audit = []

    print("Running strict 2017 fixed five-fold CARE comparison...", flush=True)
    for fold in sorted(np.unique(source_fold)):
        train_index = np.flatnonzero(source_fold != fold)
        val_index = np.flatnonzero(source_fold == fold)
        train_rows = [source_rows[i] for i in train_index]
        val_rows = [source_rows[i] for i in val_index]
        x_train, x_val, names = audit.compact_matrix_variant(
            VARIANT,
            train_rows,
            val_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        global_train = np.asarray(x_train[:, global_index], dtype=np.float32)
        global_val = np.asarray(x_val[:, global_index], dtype=np.float32)
        for variant in variants:
            all_variant_bags = variant_bags(source_host, source_ego, variant)
            train_bags = [all_variant_bags[i] for i in train_index]
            val_bags = [all_variant_bags[i] for i in val_index]
            print("  fold", int(fold), variant, flush=True)
            probability, diagnostic, train_info = fit_predict_ensemble(
                global_train,
                global_val,
                train_bags,
                val_bags,
                source_y[train_index],
                [100000 * int(fold) + seed for seed in seeds],
                args,
                device,
            )
            source_expert[variant][val_index] = probability
            source_diagnostic[variant][val_index] = diagnostic
            training_audit.append({
                "stage": "source_outer_fold",
                "fold": int(fold),
                "variant": variant,
                "train_n": int(len(train_index)),
                "validation_n": int(len(val_index)),
                "train_region_n": int(sum(len(value) for value in train_bags)),
                "validation_region_n": int(sum(len(value) for value in val_bags)),
                **train_info,
            })
        print("fold", int(fold), "done", flush=True)

    if any(np.any(~np.isfinite(value)) for value in source_expert.values()):
        raise RuntimeError("incomplete source CARE probabilities")
    source_probabilities = {
        "Global24": source_reference["coarse"],
        "Global24__A25__JointS7_ordinary": source_reference["ordinary"],
    }
    for variant in variants:
        source_probabilities[method_for_variant[variant]] = acn.routed_probability(
            source_reference["coarse"],
            source_expert[variant],
            source_reference["route"],
        )
    source_summary = metric_rows(
        source_y,
        source_probabilities,
        source_reference["coarse"],
        source_reference["route"],
    )
    source_per_fold = []
    for fold in sorted(np.unique(source_fold)):
        selected = source_fold == fold
        source_per_fold.extend([
            {
                **row,
                "fold": int(fold),
            }
            for row in metric_rows(
                source_y[selected],
                {key: value[selected] for key, value in source_probabilities.items()},
                source_reference["coarse"][selected],
                source_reference["route"][selected],
            )
        ])
    source_route_diagnostics = []
    for variant in variants:
        source_route_diagnostics.append(g24.route_diagnostic(
            method_for_variant[variant],
            source_y,
            source_reference["coarse"],
            source_expert[variant],
            source_reference["route"],
            source_reference["score"],
        ))

    source_prediction_rows = []
    for index, row in enumerate(manifest):
        output = {
            "sha256": row["sha256"],
            "label": int(source_y[index]),
            "fold": int(source_fold[index]),
            "Global24__prob": float(source_reference["coarse"][index]),
            "A25__score": float(source_reference["score"][index]),
            "A25__routed": int(source_reference["route"][index]),
            "Global24__A25__JointS7_ordinary__prob": float(
                source_reference["ordinary"][index]
            ),
        }
        for variant in variants:
            method = method_for_variant[variant]
            output[method + "__expert_prob"] = float(source_expert[variant][index])
            output[method + "__prob"] = float(source_probabilities[method][index])
            output[method + "__sparse_logit"] = float(source_diagnostic[variant][index, 0])
            output[method + "__diffuse_logit"] = float(source_diagnostic[variant][index, 1])
            output[method + "__max_logit"] = float(source_diagnostic[variant][index, 2])
        source_prediction_rows.append(output)

    g24.write_csv(args.output / "source_summary.csv", source_summary)
    g24.write_csv(args.output / "source_per_fold.csv", source_per_fold)
    g24.write_csv(args.output / "source_route_diagnostics.csv", source_route_diagnostics)
    g24.write_csv(args.output / "source_predictions.csv", source_prediction_rows)

    annual_rows = []
    future_mean = []
    future_prediction_rows = []
    future_expert = {}
    future_diagnostic = {}
    if not args.skip_future:
        x_source, x_target, names = audit.compact_matrix_variant(
            VARIANT,
            source_rows,
            target_rows,
            compact_by_sha,
            compact_schema,
            original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        global_source = np.asarray(x_source[:, global_index], dtype=np.float32)
        global_target = np.asarray(x_target[:, global_index], dtype=np.float32)
        for variant in variants:
            source_variant = variant_bags(source_host, source_ego, variant)
            target_variant = variant_bags(target_host, target_ego, variant)
            print("Training all-2017", variant, "for future diagnostics...", flush=True)
            probability, diagnostic, train_info = fit_predict_ensemble(
                global_source,
                global_target,
                source_variant,
                target_variant,
                source_y,
                [900000 + seed for seed in seeds],
                args,
                device,
            )
            future_expert[variant] = probability
            future_diagnostic[variant] = diagnostic
            training_audit.append({
                "stage": "all_2017_to_future",
                "fold": "all_2017",
                "variant": variant,
                "train_n": int(len(source_y)),
                "validation_n": int(len(target_y)),
                "train_region_n": int(sum(len(value) for value in source_variant)),
                "validation_region_n": int(sum(len(value) for value in target_variant)),
                **train_info,
            })
        future_probabilities = {
            "Global24": target_reference["coarse"],
            "Global24__A25__JointS7_ordinary": target_reference["ordinary"],
        }
        for variant in variants:
            future_probabilities[method_for_variant[variant]] = acn.routed_probability(
                target_reference["coarse"],
                future_expert[variant],
                target_reference["route"],
            )
        for year in years:
            selected = target_year == year
            annual_rows.extend(metric_rows(
                target_y[selected],
                {key: value[selected] for key, value in future_probabilities.items()},
                target_reference["coarse"][selected],
                target_reference["route"][selected],
                year,
            ))
        future_mean = g24.mean_future_rows(annual_rows)
        for index, row in enumerate(target_rows):
            output = {
                "sha256": row["sha256"],
                "year": int(target_year[index]),
                "label": int(target_y[index]),
                "Global24__prob": float(target_reference["coarse"][index]),
                "A25__score": float(target_reference["score"][index]),
                "A25__routed": int(target_reference["route"][index]),
                "Global24__A25__JointS7_ordinary__prob": float(
                    target_reference["ordinary"][index]
                ),
                "diagnostic_only": True,
            }
            for variant in variants:
                method = method_for_variant[variant]
                output[method + "__expert_prob"] = float(future_expert[variant][index])
                output[method + "__prob"] = float(future_probabilities[method][index])
                output[method + "__sparse_logit"] = float(future_diagnostic[variant][index, 0])
                output[method + "__diffuse_logit"] = float(future_diagnostic[variant][index, 1])
                output[method + "__max_logit"] = float(future_diagnostic[variant][index, 2])
            future_prediction_rows.append(output)
        g24.write_csv(args.output / "future_annual_summary_diagnostic.csv", annual_rows)
        g24.write_csv(args.output / "future_mean_summary_diagnostic.csv", future_mean)
        g24.write_csv(
            args.output / "future_predictions_diagnostic.csv", future_prediction_rows
        )

    g24.write_csv(args.output / "training_audit.csv", training_audit)
    g24.write_csv(args.output / "region_bag_audit.csv", bag_audit)
    metadata = {
        "experiment": "care-local-dualpool-mil-v1",
        "first_layer": "frozen Global24",
        "router": "frozen A25 from global24-a25-route-conditioned-joints7-v1-full",
        "reference_expert": "frozen ordinary Joint-S7",
        "new_experts": list(variants),
        "region_dimension": int(REGION_DIM),
        "region_base_dimension": int(care_region.REGION_FEATURE_DIM),
        "scale_indicator_dimension": 2,
        "pooling": {
            "sparse": "mean embedding/logit of top ceil(rho*m) regions",
            "diffuse": "mean embedding/logit of all regions",
            "rho": float(args.sparse_fraction),
        },
        "classifier_input": "Global24 + sparse embedding + diffuse embedding + three pooled logits",
        "community_module_used": False,
        "reason_community_deferred": "audit found highly fragmented host topology; first isolate host and ego value",
        "seeds": seeds,
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "dropout": float(args.dropout),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "device": str(device),
        "region_cache": str(args.region_cache),
        "region_cache_loaded": bool(region_cache_loaded),
        "source_protocol": "fixed 2017 five-fold OOF",
        "future_protocol": "train only on all 2017; evaluate 2018-2021 afterward",
        "future_diagnostic_only": True,
        "future_labels_used_for_training_routing_scaling_or_selection": False,
        "selection_note": "do not discard solely on 2017; report future diagnostics without tuning on them",
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "CARE-Droid v1: local sparse/diffuse dual-pool MIL",
        "",
        "Global24 and A25 are frozen from the previous fair experiment.",
        "HostMIL uses observed caller-method regions.",
        "HostEgoMIL adds unique undirected one-hop regions.",
        "No community module, prototype module, correction loss, or counterfactual loss is used yet.",
        "This experiment isolates whether local multi-instance evidence itself is useful.",
        f"Device: {device}; seeds: {seeds}; epochs: {args.max_epochs}",
    ]
    g24.append_report_table(
        lines,
        "Strict 2017 source summary",
        source_summary,
        ["BA", "AUC", "F1", "fixed", "broken", "net", "routed_n"],
    )
    g24.append_report_table(
        lines,
        "Strict 2017 route diagnostics",
        source_route_diagnostics,
        ["coarse_errors", "routed_coarse_errors", "error_capture_rate", "expert_fixable_total", "expert_breakable_total"],
    )
    if not args.skip_future:
        lines.extend(["", "2018-2021 results are diagnostic only."])
        for year in years:
            g24.append_report_table(
                lines,
                "Future diagnostic " + str(year),
                [row for row in annual_rows if row["year"] == year],
                ["BA", "AUC", "F1", "fixed", "broken", "net", "routed_n"],
            )
        g24.append_report_table(
            lines,
            "2018-2021 macro mean (diagnostic only)",
            future_mean,
            ["mean_BA", "min_BA", "mean_AUC", "mean_F1", "mean_fixed", "mean_broken", "mean_net"],
        )
    lines.extend([
        "",
        "How to judge",
        "------------",
        "1. Compare HostMIL and HostEgoMIL with frozen ordinary Joint-S7.",
        "2. Inspect fixed, broken, and net; BA alone cannot show correction safety.",
        "3. HostEgo > Host indicates that one-hop organization adds evidence beyond individual methods.",
        "4. Even if 2017 is lower, inspect every future year before deciding, but never tune hyperparameters on those years.",
        "5. Prototype/correction/counterfactual modules are added only after choosing the viable local scale.",
    ])
    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
