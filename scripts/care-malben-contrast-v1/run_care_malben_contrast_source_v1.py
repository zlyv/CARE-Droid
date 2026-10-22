#!/usr/bin/env python3
"""Strict 2017 OOF malicious-evidence vs benign-explanation ablation.

Only the Stage-II local evidence representation changes.  The Stage-I model,
router inputs, one-hop Host/Ego regions, graph-conditioned attention equation,
FixKeep objective and bounded residual correction remain unchanged.
"""

from __future__ import annotations

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
for path in (SCRIPT_DIR, SCRIPTS_ROOT, V4_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_care_rolegraph_conditioned_dualstream_v4 as v4  # noqa: E402
from contrast_prototype_core import contrast_columns, fit_contrast_state  # noqa: E402


METHOD = "Global24__A25__CARE_MalBenContrast_FixKeep"
BASE_REGION_DIM = int(v4.v3.v2.BASE_REGION_DIM)
CONTRAST_REGION_DIM = BASE_REGION_DIM + 8
DIAGNOSTIC_NAMES = (
    "bounded_delta",
    "critical_score",
    "mean_score",
    "attention_entropy",
    "max_attention",
    "mean_external_benign_distance",
    "mean_internal_distance",
    "mean_shared_commonness",
    "max_local_residual",
    "graph_entropy",
    "mean_benign_similarity",
    "mean_malware_similarity",
    "max_malware_margin",
    "max_benign_margin",
    "attended_evidence_contrast",
)


def fit_scaled_contrast_bags(train_bags, test_bags, labels, args, seed):
    """Scale and fit every reference using the current training partition only."""
    mean, std = v4.care_v1.fit_equal_apk_scaler(
        train_bags, np.arange(len(train_bags), dtype=np.int64)
    )
    scaled_train = [
        (np.asarray(bag, dtype=np.float32) - mean) / std for bag in train_bags
    ]
    scaled_test = [
        (np.asarray(bag, dtype=np.float32) - mean) / std for bag in test_bags
    ]

    # Preserve the established external/intra/shared/residual features.
    old_state = v4.v3.v2.fit_prototype_state(
        scaled_train, np.asarray(labels), args, int(seed)
    )
    contrast_state = fit_contrast_state(
        scaled_train,
        np.asarray(labels),
        int(args.prototype_regions_per_apk),
        int(args.benign_prototypes),
        int(args.benign_prototypes),
        int(seed) + 5000,
    )

    def augment(values):
        result = []
        summary = []
        for bag in values:
            old = v4.v3.v2.prototype_columns(bag, old_state, args)
            new = contrast_columns(bag, contrast_state)
            result.append(np.concatenate([bag, old, new], axis=1).astype(np.float32))
            summary.append(np.concatenate([
                np.mean(old, axis=0),
                np.mean(new, axis=0),
            ]))
        return result, np.asarray(summary, dtype=np.float64)

    augmented_train, train_summary = augment(scaled_train)
    augmented_test, test_summary = augment(scaled_test)
    audit = {
        "region_scaler_zero_std_n": int(np.sum(std == 1.0)),
        "shared_prototype_n": int(old_state["shared_k"]),
        "benign_prototype_n": int(len(contrast_state.benign_centers)),
        "malware_prototype_n": int(len(contrast_state.malware_centers)),
        "sampled_benign_region_n": int(contrast_state.sampled_benign_n),
        "sampled_malware_region_n": int(contrast_state.sampled_malware_n),
        "benign_distance_scale": float(contrast_state.benign_scale),
        "malware_distance_scale": float(contrast_state.malware_scale),
        "train_benign_similarity_mean": float(np.mean(train_summary[:, 4])),
        "test_benign_similarity_mean": float(np.mean(test_summary[:, 4])),
        "train_malware_similarity_mean": float(np.mean(train_summary[:, 5])),
        "test_malware_similarity_mean": float(np.mean(test_summary[:, 5])),
        "train_evidence_margin_mean": float(np.mean(train_summary[:, 6])),
        "test_evidence_margin_mean": float(np.mean(test_summary[:, 6])),
    }
    return augmented_train, augmented_test, audit


class MalBenContrastRGCondAttn(nn.Module):
    """Original attention plus two separately aggregated evidence streams."""

    def __init__(self, global_dim: int, variant: str, args):
        super().__init__()
        if str(variant) != "RGCondAttn":
            raise ValueError("MalBen contrast supports RGCondAttn only")
        self.max_delta = float(args.max_delta)
        graph_dim = int(args.graph_dim)
        local_dim = int(args.embedding_dim)
        hidden_dim = int(args.hidden_dim)
        self.graph_encoder = v4.RoleGraphEncoder(args)
        self.region_encoder = nn.Sequential(
            nn.Linear(CONTRAST_REGION_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(hidden_dim, local_dim),
            nn.ReLU(),
        )
        self.local_risk = nn.Linear(local_dim, 1)
        self.graph_query = nn.Linear(graph_dim, local_dim)
        self.local_key = nn.Linear(local_dim, local_dim)
        self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))

        # Four local vectors: critical, ordinary attention, malignant evidence,
        # and benign explanation.  Attention itself is unchanged.
        summary_dim = 12
        fused_dim = int(global_dim) + 1 + graph_dim + 4 * local_dim + summary_dim
        self.delta_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features, base_logit, graph, regions):
        graph_token, graph_entropy = self.graph_encoder(
            graph["node"], graph["edge"], graph["global"]
        )
        local = self.region_encoder(regions)

        # Layout: raw | external, internal, common, residual |
        # benign-similarity, malware-similarity, margin, strength.
        external = regions[:, -8]
        internal = regions[:, -7]
        common = regions[:, -6]
        residual = regions[:, -5]
        benign_similarity = regions[:, -4].clamp(0.0, 1.0)
        malware_similarity = regions[:, -3].clamp(0.0, 1.0)
        margin = regions[:, -2].clamp(-1.0, 1.0)

        # This is exactly the established attention form.  Contrast evidence
        # is not allowed to change which regions attention selects in v1.
        score = self.local_risk(local).reshape(-1)
        score = score + nn.functional.softplus(self.residual_beta_raw) * torch.log1p(
            torch.clamp(residual, min=0.0)
        )
        query = self.graph_query(graph_token)
        key = self.local_key(local)
        score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
        attention = torch.softmax(score, dim=0)

        attended = torch.sum(attention[:, None] * local, dim=0)
        critical_index = torch.argmax(score)
        critical = local[critical_index]
        malignant_evidence = torch.sum(
            (attention * malware_similarity)[:, None] * local, dim=0
        )
        benign_explanation = torch.sum(
            (attention * benign_similarity)[:, None] * local, dim=0
        )
        evidence_contrast = torch.sum(attention * margin)
        attention_entropy = -torch.sum(
            attention * torch.log(attention.clamp_min(1e-8))
        ) / math.log(max(int(len(attention)), 2))

        scalar_summary = torch.stack([
            score[critical_index],
            torch.mean(score),
            attention_entropy,
            torch.max(attention),
            torch.mean(external),
            torch.mean(internal),
            torch.mean(common),
            torch.max(residual),
            torch.mean(benign_similarity),
            torch.mean(malware_similarity),
            evidence_contrast,
            torch.max(torch.abs(margin)),
        ])
        fused = torch.cat([
            global_features.reshape(-1),
            base_logit.reshape(1),
            graph_token,
            critical,
            attended,
            malignant_evidence,
            benign_explanation,
            scalar_summary,
        ])
        raw_delta = self.delta_head(fused).reshape(())
        bounded_delta = self.max_delta * torch.tanh(raw_delta)
        final_logit = base_logit.reshape(()) + bounded_delta
        diagnostic = torch.stack([
            bounded_delta,
            score[critical_index],
            torch.mean(score),
            attention_entropy,
            torch.max(attention),
            torch.mean(external),
            torch.mean(internal),
            torch.mean(common),
            torch.max(residual),
            graph_entropy,
            torch.mean(benign_similarity),
            torch.mean(malware_similarity),
            torch.max(margin),
            torch.max(-margin),
            evidence_contrast,
        ])
        return final_logit, diagnostic


def rewrite_metadata(output: Path) -> None:
    path = output / "architecture.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({
        "experiment": "CARE malicious-evidence vs benign-explanation source ablation v1",
        "evaluation": "strict fixed 2017 outer-fold OOF only",
        "changed_component": "local evidence representation only",
        "unchanged_components": [
            "Global24 Stage I", "A25 route stored by the source protocol",
            "one-hop Host/Ego extraction", "graph-conditioned attention equation",
            "FixKeep objective", "bounded residual correction",
        ],
        "prototype_leakage_control": (
            "benign and malware prototypes are refit inside every training partition, "
            "including the epoch-selection split"
        ),
        "weak_label_limitation": (
            "malware-APK regions may include benign or library code because only APK labels exist"
        ),
        "augmented_region_dimension": CONTRAST_REGION_DIM,
        "max_delta_logit": float(data["parameters"]["max_delta"]),
        "future_data_loaded": False,
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = output / "REPORT.txt"
    original = report.read_text(encoding="utf-8")
    header = (
        "CARE-Droid malicious evidence vs benign explanation v1\n\n"
        "This is a source-only component ablation. No 2018-2021 sample is loaded.\n"
        "Both class-reference libraries are fit from the current training partition only.\n"
        "The malware reference is weakly supervised at APK level; it is not claimed to be\n"
        "a ground-truth library of malicious functions.\n\n"
    )
    report.write_text(header + original, encoding="utf-8")


def main() -> None:
    # Patch only the local bag augmentation and the Stage-II model used by the
    # established strict OOF runner.
    v4.v3.fit_scaled_dualproto_bags = fit_scaled_contrast_bags
    v4.AUGMENTED_REGION_DIM = CONTRAST_REGION_DIM
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": METHOD}
    v4.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    v4.RoleGraphConditionedEvidence = MalBenContrastRGCondAttn
    v4.main()

    output = Path("artifacts/care-malben-contrast-v1/source")
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    rewrite_metadata(output)


if __name__ == "__main__":
    main()

