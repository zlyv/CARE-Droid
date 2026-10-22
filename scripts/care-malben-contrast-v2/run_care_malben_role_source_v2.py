#!/usr/bin/env python3
"""Strict 2017 OOF role-conditioned Mal-vs-Ben evidence ablation."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
V4_DIR = SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script"
# The outputs/* fallbacks make the package locally auditable in this workspace.
# On the Linux experiment host, the established dependency scripts already
# live under scripts/ and the first three entries are sufficient.
LOCAL_FALLBACKS = (
    SCRIPTS_ROOT / "outputs" / "care-bounded-fixkeep-v3-script",
    SCRIPTS_ROOT / "outputs" / "care-joint-dualprototype-v2-script",
    SCRIPTS_ROOT / "outputs" / "care-local-dualpool-mil-v1-script",
    SCRIPTS_ROOT / "outputs" / "care-local-region-asset-audit-v1-script",
    SCRIPTS_ROOT / "outputs" / "global24-r-joints7-2017-2021-v1-script" / "scripts",
    SCRIPTS_ROOT / "outputs",
    SCRIPTS_ROOT / "work" / "compact76_trace" / "scripts",
    SCRIPTS_ROOT / "work" / "role_masking_code_bundle_uploaded"
    / "role_masking_code_bundle" / "repository" / "scripts",
)
for path in (SCRIPT_DIR, SCRIPTS_ROOT, V4_DIR, *LOCAL_FALLBACKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_care_rolegraph_conditioned_dualstream_v4 as v4  # noqa: E402
from role_conditioned_prototypes import (  # noqa: E402
    ROLE_COUNT,
    evidence_columns,
    fit_role_prototypes,
)


MIN_SUPPORT = 20
VARIANTS = ("BenignOnly", "MaliciousOnly", "MalBenContrast")
METHODS = {
    "BenignOnly": "Global24__A25__CARE_RoleBenignOnly_FixKeep",
    "MaliciousOnly": "Global24__A25__CARE_RoleMaliciousOnly_FixKeep",
    "MalBenContrast": "Global24__A25__CARE_RoleMalBenContrast_FixKeep",
}
BASE_REGION_DIM = int(v4.v3.v2.BASE_REGION_DIM)
OLD_AUGMENTED_DIM = BASE_REGION_DIM + 4
FULL_REGION_DIM = OLD_AUGMENTED_DIM + 9
DIAGNOSTIC_NAMES = (
    "bounded_delta",
    "critical_score",
    "mean_score",
    "attention_entropy",
    "max_attention",
    "e_mal",
    "e_ben",
    "mean_evidence_strength",
    "max_abs_margin",
    "evidence_difference",
    "graph_entropy",
)
REGION_CAPTURE_NAMES = (
    "attention_alpha",
    "S_ben",
    "S_mal",
    "margin",
    "E_ben",
    "E_mal",
    "evidence_strength",
    "center_role",
    "prototype_support_ben",
    "prototype_support_mal",
    "fallback_ben",
    "fallback_mal",
)


CURRENT_GROUP = -1
CURRENT_VARIANT = ""
FIT_CALL_IN_GROUP = 0
NEXT_GROUP = 0
FIT_RECORDS: list[dict] = []
PREDICT_CAPTURES: list[dict] = []


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fit_scaled_role_bags(train_bags, test_bags, labels, args, seed):
    """Keep old four features, then append role-conditioned audit evidence."""
    global FIT_CALL_IN_GROUP
    mean, std = v4.care_v1.fit_equal_apk_scaler(
        train_bags, np.arange(len(train_bags), dtype=np.int64)
    )
    scaled_train = [
        (np.asarray(bag, dtype=np.float32) - mean) / std for bag in train_bags
    ]
    scaled_test = [
        (np.asarray(bag, dtype=np.float32) - mean) / std for bag in test_bags
    ]
    old_state = v4.v3.v2.fit_prototype_state(
        scaled_train, np.asarray(labels), args, int(seed)
    )
    role_state = fit_role_prototypes(
        scaled_train,
        [np.asarray(bag, dtype=np.float32) for bag in train_bags],
        np.asarray(labels),
        int(args.prototype_regions_per_apk),
        MIN_SUPPORT,
        # K=1 centroids do not need model-seed randomness.  A fixed sampling
        # seed keeps role support/fallback identical across neural seeds and
        # across M1/M2/M3 for the same training partition.
        20260917,
    )

    def augment(raw_values, scaled_values):
        result = []
        for raw, scaled in zip(raw_values, scaled_values):
            old = v4.v3.v2.prototype_columns(scaled, old_state, args)
            role = evidence_columns(scaled, raw, role_state)
            result.append(np.concatenate([scaled, old, role], axis=1).astype(np.float32))
        return result

    train_aug = augment(train_bags, scaled_train)
    test_aug = augment(test_bags, scaled_test)
    call_index = int(FIT_CALL_IN_GROUP)
    FIT_CALL_IN_GROUP += 1
    FIT_RECORDS.append({
        "group": int(CURRENT_GROUP),
        "variant": str(CURRENT_VARIANT),
        "call_index": call_index,
        "purpose": "final_outer_fit" if call_index % 2 == 1 else "epoch_selection_fit",
        "seed_argument": int(seed),
        "quality": [dict(row) for row in role_state.quality_rows],
    })
    audit = {
        "region_scaler_zero_std_n": int(np.sum(std == 1.0)),
        "shared_prototype_n": int(old_state["shared_k"]),
        "benign_prototype_n": int(old_state["benign_k"]),
        "role_conditioned_K": 1,
        "role_min_support": MIN_SUPPORT,
        "role_count": ROLE_COUNT,
        "role_ben_fallback_n": int(np.sum(~role_state.use_role[0])),
        "role_mal_fallback_n": int(np.sum(~role_state.use_role[1])),
        "prototype_similarity": "exp(-RMS-Euclidean/within-class-median-scale)",
        "prototype_l2_normalized": False,
    }
    return train_aug, test_aug, audit


class RoleConditionedEvidenceModel(nn.Module):
    """Same trainable width as the deduplicated anomaly baseline."""

    def __init__(self, global_dim: int, variant: str, args):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("unknown evidence variant: " + str(variant))
        self.variant = str(variant)
        self.max_delta = float(args.max_delta)
        graph_dim = int(args.graph_dim)
        local_dim = int(args.embedding_dim)
        hidden_dim = int(args.hidden_dim)
        self.graph_encoder = v4.RoleGraphEncoder(args)
        # The nine new evidence/audit columns are deliberately excluded here.
        # Therefore Local Encoder width and parameter count stay unchanged.
        self.region_encoder = nn.Sequential(
            nn.Linear(OLD_AUGMENTED_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(hidden_dim, local_dim),
            nn.ReLU(),
        )
        self.local_risk = nn.Linear(local_dim, 1)
        self.graph_query = nn.Linear(graph_dim, local_dim)
        self.local_key = nn.Linear(local_dim, local_dim)
        self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))

        # Identical fused width to the deduplicated baseline:
        # global + base + graph + critical + one evidence vector + 8 scalars.
        fused_dim = int(global_dim) + 1 + graph_dim + 2 * local_dim + 8
        self.delta_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.last_region_capture = None
        self.last_apk_capture = None
        self.direct_stage2_training = False

    def forward(self, global_features, base_logit, graph, regions):
        graph_token, graph_entropy = self.graph_encoder(
            graph["node"], graph["edge"], graph["global"]
        )
        local = self.region_encoder(regions[:, :OLD_AUGMENTED_DIM])
        residual = regions[:, BASE_REGION_DIM + 3]
        score = self.local_risk(local).reshape(-1)
        score = score + nn.functional.softplus(self.residual_beta_raw) * torch.log1p(
            torch.clamp(residual, min=0.0)
        )
        query = self.graph_query(graph_token)
        key = self.local_key(local)
        score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
        attention = torch.softmax(score, dim=0)
        critical_index = torch.argmax(score)
        critical = local[critical_index]

        role_columns = regions[:, OLD_AUGMENTED_DIM:]
        s_ben = role_columns[:, 0].clamp(0.0, 1.0)
        s_mal = role_columns[:, 1].clamp(0.0, 1.0)
        margin = role_columns[:, 2].clamp(-1.0, 1.0)
        strength = role_columns[:, 3].clamp(0.0, 1.0)
        if self.variant == "BenignOnly":
            e_ben_region = s_ben
            e_mal_region = torch.zeros_like(s_mal)
        elif self.variant == "MaliciousOnly":
            e_ben_region = torch.zeros_like(s_ben)
            e_mal_region = s_mal
        else:
            e_ben_region = torch.relu(-margin)
            e_mal_region = torch.relu(margin)
        signed_evidence = e_mal_region - e_ben_region
        z_mal = torch.sum((attention * e_mal_region)[:, None] * local, dim=0)
        z_ben = torch.sum((attention * e_ben_region)[:, None] * local, dim=0)
        z_difference = z_mal - z_ben
        e_mal = torch.sum(attention * e_mal_region)
        e_ben = torch.sum(attention * e_ben_region)
        evidence_difference = e_mal - e_ben
        attention_entropy = -torch.sum(
            attention * torch.log(attention.clamp_min(1e-8))
        ) / math.log(max(int(len(attention)), 2))
        scalar_summary = torch.stack([
            score[critical_index],
            torch.mean(score),
            attention_entropy,
            torch.max(attention),
            e_mal,
            e_ben,
            torch.mean(strength),
            torch.max(torch.abs(margin)),
        ])
        fused = torch.cat([
            global_features.reshape(-1),
            base_logit.reshape(1),
            graph_token,
            critical,
            z_difference,
            scalar_summary,
        ])
        raw_delta = self.delta_head(fused).reshape(())
        if self.direct_stage2_training:
            # Auxiliary direct-classification head used only to learn the
            # Stage-II representation.  It neither adds a Stage-I logit nor
            # applies bounded correction; the reported classifier is still
            # the outer-train L2 logistic regression on stage2_feature.
            bounded_delta = raw_delta
            final_logit = raw_delta
        else:
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            final_logit = base_logit.reshape(()) + bounded_delta
        diagnostic = torch.stack([
            bounded_delta,
            score[critical_index],
            torch.mean(score),
            attention_entropy,
            torch.max(attention),
            e_mal,
            e_ben,
            torch.mean(strength),
            torch.max(torch.abs(margin)),
            evidence_difference,
            graph_entropy,
        ])
        self.last_region_capture = torch.stack([
            attention,
            s_ben,
            s_mal,
            margin,
            e_ben_region,
            e_mal_region,
            strength,
            role_columns[:, 4],
            role_columns[:, 5],
            role_columns[:, 6],
            role_columns[:, 7],
            role_columns[:, 8],
        ], dim=1).detach()
        self.last_apk_capture = {
            "e_mal": float(e_mal.detach().cpu()),
            "e_ben": float(e_ben.detach().cpu()),
            "z_mal": z_mal.detach().cpu().numpy(),
            "z_ben": z_ben.detach().cpu().numpy(),
            # Stage-II-only representation.  It intentionally excludes the
            # Global24 feature vector, base_logit and bounded correction.
            "stage2_feature": torch.cat([
                graph_token,
                z_difference,
                critical,
                evidence_difference.reshape(1),
            ]).detach().cpu().numpy(),
        }
        return final_logit, diagnostic


def capture_predict_model(model, global_values, base_probability, graph_values, bags, device):
    global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
    base_logit = torch.as_tensor(
        v4.probability_to_logit(base_probability), dtype=torch.float32, device=device
    )
    graph = v4.graph_tensors(graph_values, device)
    bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
    probability = []
    diagnostic = []
    regions = []
    apk = []
    model.eval()
    with torch.no_grad():
        for index in range(len(bags)):
            one_graph = {key: value[index] for key, value in graph.items()}
            logit, diag = model(
                global_tensor[index], base_logit[index], one_graph, bag_tensors[index]
            )
            probability.append(float(torch.sigmoid(logit).cpu()))
            diagnostic.append(diag.detach().cpu().numpy())
            regions.append(model.last_region_capture.cpu().numpy())
            apk.append(dict(model.last_apk_capture))
    PREDICT_CAPTURES.append({
        "group": int(CURRENT_GROUP),
        "variant": str(model.variant),
        "regions": regions,
        "apk": apk,
    })
    return np.asarray(probability), np.asarray(diagnostic)


ORIGINAL_FIT_PREDICT = v4.fit_predict_variant


def grouped_fit_predict(
    global_train, global_test, s7_train, s7_test,
    raw_train_bags, raw_test_bags, base_train, base_test,
    labels, variant, seeds, args, device,
):
    global CURRENT_GROUP, CURRENT_VARIANT, FIT_CALL_IN_GROUP, NEXT_GROUP
    CURRENT_GROUP = int(NEXT_GROUP)
    NEXT_GROUP += 1
    CURRENT_VARIANT = str(variant)
    FIT_CALL_IN_GROUP = 0
    return ORIGINAL_FIT_PREDICT(
        global_train, global_test, s7_train, s7_test,
        raw_train_bags, raw_test_bags, base_train, base_test,
        labels, variant, seeds, args, device,
    )


def parse_cli_value(name: str, default: str) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def average_arrays(values: list[np.ndarray]) -> np.ndarray:
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise RuntimeError("ensemble region captures changed shape")
    return np.mean(np.stack(values, axis=0), axis=0)


def emit_mechanism_artifacts(output: Path) -> None:
    prediction_path = output / "source_predictions.csv"
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    folds = sorted({int(row["fold"]) for row in prediction_rows})
    seeds = [int(value) for value in parse_cli_value("--seeds", "0").split(",")]
    expected_groups = len(folds) * len(VARIANTS)
    if NEXT_GROUP != expected_groups:
        raise RuntimeError(f"expected {expected_groups} fit groups, got {NEXT_GROUP}")

    region_rows: list[dict] = []
    apk_rows: list[dict] = []
    group = 0
    for fold in folds:
        source_index = [i for i, row in enumerate(prediction_rows) if int(row["fold"]) == fold]
        for variant in VARIANTS:
            captures = [row for row in PREDICT_CAPTURES if row["group"] == group]
            if len(captures) != len(seeds):
                raise RuntimeError("missing ensemble capture for group " + str(group))
            for local_index, global_index in enumerate(source_index):
                region = average_arrays([
                    row["regions"][local_index] for row in captures
                ])
                apk_values = [row["apk"][local_index] for row in captures]
                base = prediction_rows[global_index]
                for region_index, value in enumerate(region):
                    item = {
                        "sample_id": base["sha256"],
                        "fold": fold,
                        "true_label": int(base["label"]),
                        "stage1_probability": float(base["Global24__prob"]),
                        "stage1_pred": int(float(base["Global24__prob"]) >= 0.5),
                        "stage1_correct": int(
                            (float(base["Global24__prob"]) >= 0.5) == bool(int(base["label"]))
                        ),
                        "variant": variant,
                        "region_index": region_index,
                    }
                    for column, name in enumerate(REGION_CAPTURE_NAMES):
                        number = float(value[column])
                        if name in {
                            "center_role", "prototype_support_ben", "prototype_support_mal",
                            "fallback_ben", "fallback_mal",
                        }:
                            number = int(round(number))
                        item[name] = number
                    item["fallback_used"] = int(
                        item["fallback_ben"] or item["fallback_mal"]
                    )
                    region_rows.append(item)
                z_mal = np.mean(np.stack([row["z_mal"] for row in apk_values]), axis=0)
                z_ben = np.mean(np.stack([row["z_ben"] for row in apk_values]), axis=0)
                apk_rows.append({
                    "sample_id": base["sha256"],
                    "fold": fold,
                    "true_label": int(base["label"]),
                    "variant": variant,
                    "e_mal": float(np.mean([row["e_mal"] for row in apk_values])),
                    "e_ben": float(np.mean([row["e_ben"] for row in apk_values])),
                    "z_mal": json.dumps(z_mal.tolist(), separators=(",", ":")),
                    "z_ben": json.dumps(z_ben.tolist(), separators=(",", ":")),
                })
            group += 1
    write_csv(output.parent / "source_region_evidence.csv", region_rows)
    write_csv(output.parent / "source_apk_evidence.csv", apk_rows)

    # Only final outer-training fits (the odd call of each seed pair) are used.
    quality_output: list[dict] = []
    group = 0
    for fold in folds:
        for variant in VARIANTS:
            if variant != "MalBenContrast":
                group += 1
                continue
            records = [
                row for row in FIT_RECORDS
                if row["group"] == group and row["purpose"] == "final_outer_fit"
            ]
            if len(records) != len(seeds):
                raise RuntimeError("missing final prototype quality for fold " + str(fold))
            for role in range(ROLE_COUNT):
                values = [record["quality"][role] for record in records]
                row = {"fold": fold, "role": role, "K": 1, "min_support": MIN_SUPPORT}
                for name in (
                    "num_ben_regions", "num_mal_regions", "prototype_distance",
                    "prototype_cosine", "within_ben_dispersion",
                    "within_mal_dispersion", "fallback_ben", "fallback_mal", "fallback",
                ):
                    raw = np.asarray([item[name] for item in values], dtype=np.float64)
                    mean = float(np.nanmean(raw)) if np.any(np.isfinite(raw)) else float("nan")
                    row[name] = int(round(mean)) if name.startswith("num_") or name.startswith("fallback") else mean
                quality_output.append(row)
            group += 1
    write_csv(output.parent / "source_prototype_quality.csv", quality_output)


def rewrite_metadata(output: Path) -> None:
    path = output / "architecture.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({
        "experiment": "CARE role-conditioned malicious-vs-benign evidence v2",
        "evaluation": "strict fixed 2017 outer-fold OOF only",
        "primary_router_for_selection": "A25",
        "secondary_router_sensitivity": "DI25 (post-hoc application of same OOF experts)",
        "changed_component": "prototype-based local evidence only",
        "role_conditioning": {
            "role_source": "argmax of the 31 role-fraction columns in each cached host region",
            "tie_rule": "smallest role id",
            "K": 1,
            "min_support": MIN_SUPPORT,
            "fallback": "role/class centroid -> class-global centroid -> unavailable",
        },
        "similarity": "exp(-RMS-Euclidean distance / training within-class median scale)",
        "l2_normalization_before_similarity": False,
        "saved_region_evidence": ["S_ben", "S_mal", "margin", "evidence_strength"],
        "aggregation": (
            "fixed alpha separately aggregates positive malicious and benign evidence; "
            "only z_mal-z_ben and scalar evidence summaries enter correction"
        ),
        "parameter_control": (
            "new evidence columns are excluded from Local Encoder; correction input width "
            "equals the deduplicated anomaly baseline"
        ),
        "input_implementation_note": (
            "the current v4 loader consumes cached host_values only; cached ego_values are "
            "not added in this controlled experiment"
        ),
        "future_data_loaded": False,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    v4.v3.fit_scaled_dualproto_bags = fit_scaled_role_bags
    v4.AUGMENTED_REGION_DIM = FULL_REGION_DIM
    v4.MODEL_VARIANTS = VARIANTS
    v4.METHOD_FOR_VARIANT = METHODS
    v4.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    v4.RoleGraphConditionedEvidence = RoleConditionedEvidenceModel
    v4.predict_model = capture_predict_model
    v4.fit_predict_variant = grouped_fit_predict
    v4.main()

    output = Path("artifacts/care-malben-contrast-v2/source/model")
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    rewrite_metadata(output)
    emit_mechanism_artifacts(output)
    print("Role-conditioned mechanism artifacts written to:", output.parent, flush=True)


if __name__ == "__main__":
    main()
