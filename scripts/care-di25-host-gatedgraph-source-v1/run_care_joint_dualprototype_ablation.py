#!/usr/bin/env python3
"""CARE-Droid v2: Joint-S7 enhanced by local background/prototype evidence.

The deployed structure stays fixed:

* layer 1 is the frozen role-free Global24 detector;
* the frozen A25 route sends ambiguous APKs to layer 2;
* layer 2 is one joint classifier over Global24, S7, and local evidence.

This is not a confidence overwrite experiment and the local branch does not
replace Joint-S7.  Three second-layer variants isolate the contribution of the
new evidence:

* JointRawHost: Global24 + S7 + raw host-region dual-pool MIL;
* JointSharedExternal: additionally uses label-balanced shared-background
  commonness and distance to external benign prototypes;
* JointDualPrototype: additionally uses an APK-internal dominant-background
  distance, producing the complete external/internal dual-normal residual.

All scalers, S7 transforms, clusters, cluster commonness values, and neural
parameters are fit inside the current 2017 training fold.  The 2018--2021
labels are read only after predictions have been fixed and are diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
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
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: " + str(exc))

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as exc:  # pragma: no cover
    raise SystemExit("scikit-learn with MiniBatchKMeans is required: " + str(exc))

import run_care_local_dualpool_mil as care_v1  # noqa: E402


g24 = care_v1.g24
audit = care_v1.audit
evidence = care_v1.evidence
future_s7 = g24.future_s7
acn = care_v1.acn
VARIANT = care_v1.VARIANT
BASE_REGION_DIM = int(care_v1.REGION_DIM)
PROTOTYPE_DIM = 4
AUGMENTED_REGION_DIM = BASE_REGION_DIM + PROTOTYPE_DIM

VARIANTS = (
    "JointRawHost",
    "JointSharedExternal",
    "JointDualPrototype",
)
METHOD_FOR_VARIANT = {
    "JointRawHost": "Global24__A25__CARE_JointRawHost",
    "JointSharedExternal": "Global24__A25__CARE_JointSharedExternal",
    "JointDualPrototype": "Global24__A25__CARE_JointDualPrototype",
}
DIAGNOSTIC_NAMES = (
    "sparse_logit",
    "diffuse_logit",
    "max_logit",
    "mean_external_distance",
    "mean_internal_distance",
    "mean_shared_commonness",
    "max_local_residual",
)


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name):
    name = str(name).strip().lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    return device


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


def sample_bag_rows(bag, limit, rng):
    bag = np.asarray(bag, dtype=np.float32)
    if len(bag) <= int(limit):
        return bag
    index = np.sort(rng.choice(len(bag), size=int(limit), replace=False))
    return bag[index]


def balanced_region_sample(bags, labels, per_apk, seed):
    rng = np.random.RandomState(int(seed))
    labels = np.asarray(labels, dtype=np.int64)
    by_class = {}
    class_sizes = []
    for label in (0, 1):
        parts = [
            sample_bag_rows(bags[index], per_apk, rng)
            for index in np.flatnonzero(labels == label)
        ]
        values = np.concatenate(parts, axis=0)
        by_class[label] = values
        class_sizes.append(len(values))
    target = min(class_sizes)
    balanced = []
    for label in (0, 1):
        values = by_class[label]
        if len(values) > target:
            chosen = np.sort(rng.choice(len(values), size=target, replace=False))
            values = values[chosen]
        balanced.append(values)
    return np.concatenate(balanced, axis=0), by_class[0]


def pairwise_rms(values, centers):
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    squared = np.mean(
        np.square(values[:, None, :] - centers[None, :, :]), axis=2
    )
    return np.sqrt(np.maximum(squared, 0.0))


def safe_scale(values, fallback=1.0):
    values = np.asarray(values, dtype=np.float64)
    selected = values[np.isfinite(values) & (values > 1e-8)]
    if not len(selected):
        return float(fallback)
    return max(float(np.median(selected)), 1e-6)


def fit_prototype_state(bags, labels, args, seed):
    """Fit all background statistics from training APKs only."""
    labels = np.asarray(labels, dtype=np.int64)
    balanced, benign = balanced_region_sample(
        bags, labels, args.prototype_regions_per_apk, seed
    )
    shared_k = min(int(args.shared_prototypes), len(balanced))
    benign_k = min(int(args.benign_prototypes), len(benign))
    if shared_k < 2 or benign_k < 1:
        raise RuntimeError("too few sampled regions for prototype fitting")

    shared_model = MiniBatchKMeans(
        n_clusters=shared_k,
        random_state=int(seed) + 101,
        batch_size=min(2048, max(256, len(balanced))),
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.01,
    ).fit(balanced)
    benign_model = MiniBatchKMeans(
        n_clusters=benign_k,
        random_state=int(seed) + 202,
        batch_size=min(2048, max(256, len(benign))),
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.01,
    ).fit(benign)

    presence = np.zeros((2, shared_k), dtype=np.float64)
    apk_count = np.asarray([
        np.sum(labels == 0), np.sum(labels == 1)
    ], dtype=np.float64)
    for bag, label in zip(bags, labels):
        assignment = shared_model.predict(np.asarray(bag, dtype=np.float32))
        presence[int(label), np.unique(assignment)] += 1.0
    prevalence = presence / np.maximum(apk_count[:, None], 1.0)
    commonness = np.clip(2.0 * np.minimum(prevalence[0], prevalence[1]), 0.0, 1.0)

    balanced_assignment = shared_model.predict(balanced)
    balanced_distance = pairwise_rms(
        balanced, shared_model.cluster_centers_
    )[np.arange(len(balanced)), balanced_assignment]
    cluster_scale = np.zeros(shared_k, dtype=np.float32)
    global_shared_scale = safe_scale(balanced_distance)
    for cluster in range(shared_k):
        selected = balanced_distance[balanced_assignment == cluster]
        cluster_scale[cluster] = safe_scale(selected, global_shared_scale)

    benign_distance = np.min(
        pairwise_rms(benign, benign_model.cluster_centers_), axis=1
    )
    external_scale = safe_scale(benign_distance)
    return {
        "shared_centers": np.asarray(shared_model.cluster_centers_, dtype=np.float32),
        "benign_centers": np.asarray(benign_model.cluster_centers_, dtype=np.float32),
        "commonness": commonness.astype(np.float32),
        "shared_scale": cluster_scale.astype(np.float32),
        "external_scale": float(external_scale),
        "shared_k": int(shared_k),
        "benign_k": int(benign_k),
        "sampled_balanced_n": int(len(balanced)),
        "sampled_benign_n": int(len(benign)),
    }


def internal_dominant_distance(bag, assignment, coverage):
    counts = np.bincount(assignment)
    ordered = np.argsort(-counts)
    target = max(1, int(math.ceil(float(coverage) * len(bag))))
    selected = []
    covered = 0
    for cluster in ordered:
        if counts[cluster] <= 0:
            continue
        selected.append(int(cluster))
        covered += int(counts[cluster])
        if covered >= target:
            break
    centers = np.asarray([
        np.mean(bag[assignment == cluster], axis=0)
        for cluster in selected
    ], dtype=np.float32)
    distance = np.min(pairwise_rms(bag, centers), axis=1)
    scale = safe_scale(distance)
    return np.log1p(distance / scale).astype(np.float32)


def prototype_columns(bag, state, args):
    bag = np.asarray(bag, dtype=np.float32)
    shared_distance_all = pairwise_rms(bag, state["shared_centers"])
    assignment = np.argmin(shared_distance_all, axis=1)
    shared_distance = shared_distance_all[np.arange(len(bag)), assignment]
    similarity = np.exp(
        -shared_distance / np.maximum(state["shared_scale"][assignment], 1e-6)
    )
    common = state["commonness"][assignment] * similarity

    external_distance = np.min(
        pairwise_rms(bag, state["benign_centers"]), axis=1
    )
    external = np.log1p(
        external_distance / max(float(state["external_scale"]), 1e-6)
    ).astype(np.float32)
    internal = internal_dominant_distance(
        bag, assignment, args.internal_background_coverage
    )
    residual = (
        (1.0 - np.clip(common, 0.0, 1.0))
        * np.sqrt(np.maximum(external, 0.0) * np.maximum(internal, 0.0))
    )
    return np.stack([external, internal, common, residual], axis=1).astype(np.float32)


def augment_bags(bags, state, variant, args):
    result = []
    summaries = []
    for bag in bags:
        bag = np.asarray(bag, dtype=np.float32)
        columns = prototype_columns(bag, state, args)
        if variant == "JointRawHost":
            columns[:] = 0.0
        elif variant == "JointSharedExternal":
            columns[:, 1] = 0.0
            columns[:, 3] = columns[:, 0] * (1.0 - columns[:, 2])
        elif variant != "JointDualPrototype":
            raise ValueError("unknown variant: " + str(variant))
        result.append(np.concatenate([bag, columns], axis=1))
        summaries.append([
            float(np.mean(columns[:, 0])),
            float(np.mean(columns[:, 1])),
            float(np.mean(columns[:, 2])),
            float(np.max(columns[:, 3])),
        ])
    return result, np.asarray(summaries, dtype=np.float64)


class JointPrototypeMIL(nn.Module):
    def __init__(self, global_dim, s7_dim, args):
        super().__init__()
        self.sparse_fraction = float(args.sparse_fraction)
        self.region_encoder = nn.Sequential(
            nn.Linear(AUGMENTED_REGION_DIM, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(args.hidden_dim, args.embedding_dim),
            nn.ReLU(),
        )
        self.instance_head = nn.Linear(args.embedding_dim, 1)
        self.s7_encoder = nn.Sequential(
            nn.Linear(s7_dim, args.s7_hidden_dim),
            nn.LayerNorm(args.s7_hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(args.s7_hidden_dim, args.s7_embedding_dim),
            nn.ReLU(),
        )
        fused_dim = (
            global_dim + args.s7_embedding_dim
            + 2 * args.embedding_dim + 3 + PROTOTYPE_DIM
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(args.hidden_dim, 1),
        )

    def forward(self, global_features, s7_features, regions):
        embedding = self.region_encoder(regions)
        instance_logit = self.instance_head(embedding).reshape(-1)
        k = max(1, int(math.ceil(len(instance_logit) * self.sparse_fraction)))
        top_values, top_index = torch.topk(instance_logit, k=k, largest=True)
        sparse_embedding = torch.mean(embedding[top_index], dim=0)
        diffuse_embedding = torch.mean(embedding, dim=0)
        pooled_logits = torch.stack([
            torch.mean(top_values),
            torch.mean(instance_logit),
            torch.max(instance_logit),
        ])
        prototype_summary = torch.stack([
            torch.mean(regions[:, -4]),
            torch.mean(regions[:, -3]),
            torch.mean(regions[:, -2]),
            torch.max(regions[:, -1]),
        ])
        s7_embedding = self.s7_encoder(s7_features.reshape(-1))
        fused = torch.cat([
            global_features.reshape(-1),
            s7_embedding,
            sparse_embedding,
            diffuse_embedding,
            pooled_logits,
            prototype_summary,
        ])
        apk_logit = self.classifier(fused).reshape(())
        diagnostic = torch.cat([pooled_logits, prototype_summary])
        return apk_logit, diagnostic


def train_one_model(global_train, s7_train, region_train, labels, args, device, seed):
    seed_everything(seed)
    model = JointPrototypeMIL(global_train.shape[1], s7_train.shape[1], args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    global_tensor = torch.as_tensor(global_train, dtype=torch.float32, device=device)
    s7_tensor = torch.as_tensor(s7_train, dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device)
    rng = np.random.RandomState(int(seed) + 2702)
    final_loss = float("nan")
    model.train()
    for epoch in range(int(args.max_epochs)):
        order = rng.permutation(len(labels))
        epoch_loss = 0.0
        for start in range(0, len(labels), int(args.batch_size)):
            batch = order[start:start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = []
            for index in batch:
                logit, _ = model(
                    global_tensor[int(index)],
                    s7_tensor[int(index)],
                    region_train[int(index)],
                )
                logits.append(logit)
            batch_index = torch.as_tensor(batch, dtype=torch.long, device=device)
            loss = criterion(torch.stack(logits), label_tensor[batch_index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch)
        final_loss = epoch_loss / len(labels)
        if args.verbose_epochs and (
            epoch == 0 or (epoch + 1) % args.verbose_epochs == 0
            or epoch + 1 == args.max_epochs
        ):
            print(
                "    seed", seed, "epoch", epoch + 1, "/", args.max_epochs,
                "loss", f"{final_loss:.6f}", flush=True,
            )
    return model, final_loss


def predict_model(model, global_values, s7_values, region_values, device):
    global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    s7_tensor = torch.as_tensor(s7_values, dtype=torch.float32, device=device)
    probabilities = []
    diagnostics = []
    model.eval()
    with torch.no_grad():
        for index in range(len(region_values)):
            logit, diagnostic = model(
                global_tensor[index], s7_tensor[index], region_values[index]
            )
            probabilities.append(float(torch.sigmoid(logit).cpu()))
            diagnostics.append(diagnostic.detach().cpu().numpy())
    return np.asarray(probabilities), np.asarray(diagnostics)


def fit_predict_variant(
    global_train,
    global_test,
    s7_train,
    s7_test,
    raw_train_bags,
    raw_test_bags,
    labels,
    variant,
    seeds,
    args,
    device,
):
    mean, std = care_v1.fit_equal_apk_scaler(
        raw_train_bags, np.arange(len(raw_train_bags))
    )
    scaled_train = [
        (np.asarray(bag, dtype=np.float32) - mean) / std
        for bag in raw_train_bags
    ]
    scaled_test = [
        (np.asarray(bag, dtype=np.float32) - mean) / std
        for bag in raw_test_bags
    ]
    z_s7_train, z_s7_test, _ = audit.p72.standardize_added(
        np.asarray(s7_train, dtype=np.float64),
        np.asarray(s7_test, dtype=np.float64),
        [f"s7__{index:04d}" for index in range(s7_train.shape[1])],
    )
    probabilities = []
    diagnostics = []
    losses = []
    prototype_rows = []
    for seed in seeds:
        state = fit_prototype_state(scaled_train, labels, args, int(seed))
        train_augmented, train_proto_summary = augment_bags(
            scaled_train, state, variant, args
        )
        test_augmented, test_proto_summary = augment_bags(
            scaled_test, state, variant, args
        )
        train_tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in train_augmented
        ]
        test_tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in test_augmented
        ]
        model, loss = train_one_model(
            global_train,
            z_s7_train.astype(np.float32),
            train_tensors,
            labels,
            args,
            device,
            int(seed),
        )
        probability, diagnostic = predict_model(
            model,
            global_test,
            z_s7_test.astype(np.float32),
            test_tensors,
            device,
        )
        probabilities.append(probability)
        diagnostics.append(diagnostic)
        losses.append(loss)
        prototype_rows.append({
            "seed": int(seed),
            "shared_prototype_n": state["shared_k"],
            "benign_prototype_n": state["benign_k"],
            "sampled_balanced_region_n": state["sampled_balanced_n"],
            "sampled_benign_region_n": state["sampled_benign_n"],
            "shared_commonness_mean": float(np.mean(state["commonness"])),
            "shared_commonness_max": float(np.max(state["commonness"])),
            "external_distance_scale": float(state["external_scale"]),
            "train_external_mean": float(np.mean(train_proto_summary[:, 0])),
            "test_external_mean": float(np.mean(test_proto_summary[:, 0])),
            "train_internal_mean": float(np.mean(train_proto_summary[:, 1])),
            "test_internal_mean": float(np.mean(test_proto_summary[:, 1])),
            "train_commonness_mean": float(np.mean(train_proto_summary[:, 2])),
            "test_commonness_mean": float(np.mean(test_proto_summary[:, 2])),
        })
        del model, train_tensors, test_tensors
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
        prototype_rows,
    )


def load_s7(source_rows, target_rows, endpoint_cache, args):
    print("Reconstructing source Stats120+PairRaw465...", flush=True)
    source_s3, _ = audit.future.reconstruct_structure(
        source_rows, endpoint_cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if source_s3.shape != (len(source_rows), 585):
        raise RuntimeError("bad source S3 shape: " + repr(source_s3.shape))
    source_s7, source_reference = future_s7.source_s7_from_reference(
        args.source_multiscale_features, source_rows, source_s3
    )
    target_s7 = np.zeros((0, source_s7.shape[1]), dtype=np.float64)
    future_cache_audit = {"skipped": True}
    if target_rows:
        print("Reconstructing future Stats120+PairRaw465...", flush=True)
        target_s3, _ = audit.future.reconstruct_structure(
            target_rows, endpoint_cache, int(audit.exp.ROLE_VOCABULARY_SIZE)
        )
        target_inter, _, future_cache_audit = future_s7.load_or_build_future_inter(
            target_rows, target_s3, endpoint_cache, args
        )
        target_s7 = np.concatenate([target_s3, target_inter], axis=1)
    if source_s7.shape[1] != 1546 or target_s7.shape[1] != 1546:
        raise RuntimeError("bad S7 dimensions")
    return source_s7, target_s7, source_reference, future_cache_audit


def metric_rows(labels, probabilities, coarse, route, year=None):
    return care_v1.metric_rows(labels, probabilities, coarse, route, year)


def add_prediction_columns(output, variants, expert, final, diagnostic, index):
    for variant in variants:
        method = METHOD_FOR_VARIANT[variant]
        output[method + "__expert_prob"] = float(expert[variant][index])
        output[method + "__prob"] = float(final[method][index])
        for column_index, name in enumerate(DIAGNOSTIC_NAMES):
            output[method + "__" + name] = float(
                diagnostic[variant][index, column_index]
            )


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
    parser.add_argument("--source-multiscale-features", type=Path, default=Path(
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/multiscale_features.npz"
    ))
    parser.add_argument("--future-inter-cache", type=Path, default=Path(
        "artifacts/s7-future-interz-cache-v1/future_interz.npz"
    ))
    parser.add_argument("--rebuild-future-inter-cache", action="store_true")
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
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--s7-hidden-dim", type=int, default=64)
    parser.add_argument("--s7-embedding-dim", type=int, default=24)
    parser.add_argument("--sparse-fraction", type=float, default=0.10)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--shared-prototypes", type=int, default=32)
    parser.add_argument("--benign-prototypes", type=int, default=24)
    parser.add_argument("--prototype-regions-per-apk", type=int, default=16)
    parser.add_argument("--internal-background-coverage", type=float, default=0.70)
    parser.add_argument("--null-samples", type=int, default=64)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--null-seed", type=int, default=20260827)
    parser.add_argument("--z-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose-epochs", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-future", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-joint-dualprototype-v2"
    ))
    args = parser.parse_args()

    if args.max_epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch size must be positive")
    if not 0.0 < args.sparse_fraction <= 1.0:
        raise SystemExit("--sparse-fraction must be in (0,1]")
    if not 0.0 < args.internal_background_coverage <= 1.0:
        raise SystemExit("--internal-background-coverage must be in (0,1]")
    if min(args.shared_prototypes, args.benign_prototypes, args.prototype_regions_per_apk) < 1:
        raise SystemExit("prototype counts must be positive")
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
    source_reference = care_v1.reference_arrays(
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
        target_reference = care_v1.reference_arrays(
            args.previous_result / "future_predictions_diagnostic.csv",
            target_rows,
            False,
        )

    endpoint_cache = audit.exp.read_cache(args.cache)
    source_s7, target_s7, source_s7_reference, future_cache_audit = load_s7(
        source_rows, target_rows, endpoint_cache, args
    )
    all_rows = list(source_rows) + list(target_rows)
    host_bags, _, bag_audit, region_cache_loaded = care_v1.get_region_bags(
        all_rows, endpoint_cache, args
    )
    source_n = len(source_rows)
    source_host = host_bags[:source_n]
    target_host = host_bags[source_n:]

    source_expert = {
        variant: np.full(source_n, np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    source_diagnostic = {
        variant: np.full((source_n, len(DIAGNOSTIC_NAMES)), np.nan)
        for variant in VARIANTS
    }
    training_audit = []
    prototype_audit = []

    print("Running strict 2017 joint prototype ablation...", flush=True)
    for fold in sorted(np.unique(source_fold)):
        train_index = np.flatnonzero(source_fold != fold)
        val_index = np.flatnonzero(source_fold == fold)
        train_rows = [source_rows[index] for index in train_index]
        val_rows = [source_rows[index] for index in val_index]
        x_train, x_val, names = audit.compact_matrix_variant(
            VARIANT, train_rows, val_rows,
            compact_by_sha, compact_schema, original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        global_train = np.asarray(x_train[:, global_index], dtype=np.float32)
        global_val = np.asarray(x_val[:, global_index], dtype=np.float32)
        train_bags = [source_host[index] for index in train_index]
        val_bags = [source_host[index] for index in val_index]
        for variant in VARIANTS:
            print("  fold", int(fold), variant, flush=True)
            probability, diagnostic, train_info, proto_rows = fit_predict_variant(
                global_train,
                global_val,
                source_s7[train_index],
                source_s7[val_index],
                train_bags,
                val_bags,
                source_y[train_index],
                variant,
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
            for row in proto_rows:
                prototype_audit.append({
                    "stage": "source_outer_fold",
                    "fold": int(fold),
                    "variant": variant,
                    **row,
                })
        print("fold", int(fold), "done", flush=True)

    if any(np.any(~np.isfinite(values)) for values in source_expert.values()):
        raise RuntimeError("incomplete source probabilities")
    source_probability = {
        "Global24": source_reference["coarse"],
        "Global24__A25__JointS7_ordinary": source_reference["ordinary"],
    }
    for variant in VARIANTS:
        method = METHOD_FOR_VARIANT[variant]
        source_probability[method] = acn.routed_probability(
            source_reference["coarse"],
            source_expert[variant],
            source_reference["route"],
        )
    source_summary = metric_rows(
        source_y, source_probability, source_reference["coarse"], source_reference["route"]
    )
    source_per_fold = []
    for fold in sorted(np.unique(source_fold)):
        selected = source_fold == fold
        source_per_fold.extend([
            {**row, "fold": int(fold)}
            for row in metric_rows(
                source_y[selected],
                {key: value[selected] for key, value in source_probability.items()},
                source_reference["coarse"][selected],
                source_reference["route"][selected],
            )
        ])
    source_route = [
        g24.route_diagnostic(
            METHOD_FOR_VARIANT[variant],
            source_y,
            source_reference["coarse"],
            source_expert[variant],
            source_reference["route"],
            source_reference["score"],
        )
        for variant in VARIANTS
    ]
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
        add_prediction_columns(
            output, VARIANTS, source_expert, source_probability, source_diagnostic, index
        )
        source_prediction_rows.append(output)

    g24.write_csv(args.output / "source_summary.csv", source_summary)
    g24.write_csv(args.output / "source_per_fold.csv", source_per_fold)
    g24.write_csv(args.output / "source_route_diagnostics.csv", source_route)
    g24.write_csv(args.output / "source_predictions.csv", source_prediction_rows)

    annual_rows = []
    future_mean = []
    future_prediction_rows = []
    if not args.skip_future:
        x_source, x_target, names = audit.compact_matrix_variant(
            VARIANT, source_rows, target_rows,
            compact_by_sha, compact_schema, original_names,
        )
        global_index, _ = evidence.split_view_indices(names)
        global_source = np.asarray(x_source[:, global_index], dtype=np.float32)
        global_target = np.asarray(x_target[:, global_index], dtype=np.float32)
        future_expert = {}
        future_diagnostic = {}
        for variant in VARIANTS:
            print("Training all-2017", variant, "for future diagnostics...", flush=True)
            probability, diagnostic, train_info, proto_rows = fit_predict_variant(
                global_source,
                global_target,
                source_s7,
                target_s7,
                source_host,
                target_host,
                source_y,
                variant,
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
                "train_region_n": int(sum(len(value) for value in source_host)),
                "validation_region_n": int(sum(len(value) for value in target_host)),
                **train_info,
            })
            for row in proto_rows:
                prototype_audit.append({
                    "stage": "all_2017_to_future",
                    "fold": "all_2017",
                    "variant": variant,
                    **row,
                })
        future_probability = {
            "Global24": target_reference["coarse"],
            "Global24__A25__JointS7_ordinary": target_reference["ordinary"],
        }
        for variant in VARIANTS:
            method = METHOD_FOR_VARIANT[variant]
            future_probability[method] = acn.routed_probability(
                target_reference["coarse"],
                future_expert[variant],
                target_reference["route"],
            )
        for year in years:
            selected = target_year == year
            annual_rows.extend(metric_rows(
                target_y[selected],
                {key: value[selected] for key, value in future_probability.items()},
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
            add_prediction_columns(
                output, VARIANTS, future_expert, future_probability,
                future_diagnostic, index,
            )
            future_prediction_rows.append(output)
        g24.write_csv(args.output / "future_annual_summary_diagnostic.csv", annual_rows)
        g24.write_csv(args.output / "future_mean_summary_diagnostic.csv", future_mean)
        g24.write_csv(
            args.output / "future_predictions_diagnostic.csv", future_prediction_rows
        )

    g24.write_csv(args.output / "training_audit.csv", training_audit)
    g24.write_csv(args.output / "prototype_audit.csv", prototype_audit)
    g24.write_csv(args.output / "region_bag_audit.csv", bag_audit)
    metadata = {
        "experiment": "care-joint-dualprototype-v2",
        "first_layer": "frozen Global24 without 52 APK-level role-ratio dimensions",
        "router": "frozen A25",
        "reference_expert": "frozen ordinary Global24+S7 joint classifier",
        "new_experts": list(VARIANTS),
        "joint_classifier_input": (
            "Global24 + fold-standardized S7 embedding + local sparse/diffuse pools "
            "+ prototype summaries"
        ),
        "raw_region_dimension": BASE_REGION_DIM,
        "augmented_region_dimension": AUGMENTED_REGION_DIM,
        "prototype_columns": [
            "external_benign_distance",
            "internal_dominant_background_distance",
            "label_balanced_shared_commonness",
            "local_residual",
        ],
        "shared_commonness": "2*min(P(cluster present|benign), P(cluster present|malware)), clipped to 1",
        "internal_background": "dominant within-APK shared clusters covering the configured region mass",
        "future_labels_used_for_training_scaling_clustering_or_selection": False,
        "source_protocol": "fixed 2017 five-fold OOF",
        "future_protocol": "train only on all 2017; 2018-2021 diagnostic only",
        "source_s7_reference": source_s7_reference,
        "future_inter_cache_audit": future_cache_audit,
        "region_cache": str(args.region_cache),
        "region_cache_loaded": bool(region_cache_loaded),
        "seeds": seeds,
        "device": str(device),
        "max_epochs": int(args.max_epochs),
        "shared_prototypes": int(args.shared_prototypes),
        "benign_prototypes": int(args.benign_prototypes),
        "prototype_regions_per_apk": int(args.prototype_regions_per_apk),
        "internal_background_coverage": float(args.internal_background_coverage),
        "fix_keep_loss_used": False,
        "counterfactual_loss_used": False,
        "selection_rule": (
            "First require a 2017 gain from JointRawHost to establish local complement; "
            "then require monotonic Fix/Break improvement from external and dual prototypes."
        ),
    }
    (args.output / "audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "CARE-Droid v2: Joint-S7 + local shared-background dual prototypes",
        "",
        "Global24 and A25 remain frozen.",
        "Every new expert jointly classifies Global24, S7, and local Host evidence.",
        "RawHost isolates joint local complement; SharedExternal adds balanced background",
        "and benign prototypes; DualPrototype additionally adds APK-internal background.",
        "No future label is used before prediction. Fix/Keep is deliberately deferred.",
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
        source_route,
        [
            "coarse_errors", "routed_coarse_errors", "error_capture_rate",
            "expert_fixable_total", "expert_breakable_total",
        ],
    )
    if not args.skip_future:
        for year in years:
            g24.append_report_table(
                lines,
                f"Future diagnostic {year}",
                [row for row in annual_rows if int(row["year"]) == year],
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
        "1. JointRawHost must be compared with frozen ordinary Joint-S7; this isolates joint local evidence.",
        "2. SharedExternal should reduce Break without destroying Fix; otherwise external prototypes are unsafe.",
        "3. DualPrototype should improve over SharedExternal; otherwise the internal-background mechanism is unsupported.",
        "4. Do not tune prototype counts or thresholds using the already inspected future years.",
        "5. Add Fix/Keep or counterfactual loss only after a prototype variant passes the strict 2017 test.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output, flush=True)


if __name__ == "__main__":
    main()
