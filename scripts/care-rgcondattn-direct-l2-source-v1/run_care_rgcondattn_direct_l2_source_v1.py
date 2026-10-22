#!/usr/bin/env python3
"""Strict 2017 source-only ablation for an independent RGCondAttn classifier.

The first-layer Global24 model and its frozen A25 route are retained only for
deployment routing and for the final comparison.  The second-layer classifier
does not receive Global24 features or the first-layer logit.  It predicts an
absolute malware logit from the role graph and local-region evidence:

    [graph_token; critical_region; attended_regions; scalar_summary] -> logit

The model is trained with ordinary binary cross entropy on every source
training row.  There is no residual addition, tanh bound, FixKeep weighting,
or keep-consistency penalty.  Evaluation uses the established fixed 2017
outer folds; future years are not loaded.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
LOCAL_V4_BUNDLE = SCRIPTS_ROOT / "care-rolegraph-conditioned-v4-script"
for path in (SCRIPT_DIR, SCRIPTS_ROOT, LOCAL_V4_BUNDLE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required: " + str(exc))

try:
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing CARE v4 and its dependencies in scripts/: " + str(exc))


METHOD = "Global24__A25__CARE_RGCondAttn_DirectL2"
DIAGNOSTIC_NAMES = (
    "direct_logit",
    "critical_score",
    "mean_instance_score",
    "attention_entropy",
    "max_attention",
    "context_entropy",
    "mean_external_distance",
    "mean_internal_distance",
    "mean_shared_commonness",
    "max_local_residual",
    "graph_pool_entropy",
)


class DirectRGCondAttn(nn.Module):
    """Independent second-layer classifier with graph-conditioned attention."""

    def __init__(self, global_dim: int, variant: str, args):
        super().__init__()
        if str(variant) != "RGCondAttn":
            raise ValueError("Direct-L2 ablation supports RGCondAttn only")
        del global_dim
        graph_dim = int(args.graph_dim)
        local_dim = int(args.embedding_dim)
        hidden_dim = int(args.hidden_dim)
        self.graph_encoder = v4.RoleGraphEncoder(args)
        self.region_encoder = nn.Sequential(
            nn.Linear(v4.AUGMENTED_REGION_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(hidden_dim, local_dim),
            nn.ReLU(),
        )
        self.local_risk = nn.Linear(local_dim, 1)
        self.graph_query = nn.Linear(graph_dim, local_dim)
        self.local_key = nn.Linear(local_dim, local_dim)
        self.residual_beta_raw = nn.Parameter(torch.tensor(0.0))

        # graph token + critical local evidence + attended local evidence + 8 scalars
        fused_dim = graph_dim + 2 * local_dim + 8
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(args.dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_features, base_logit, graph, regions):
        # These arguments remain in the signature only for compatibility with
        # the established trainer.  They are deliberately excluded from the
        # independent second-layer representation.
        del global_features, base_logit
        graph_token, graph_entropy = self.graph_encoder(
            graph["node"], graph["edge"], graph["global"]
        )
        local = self.region_encoder(regions)
        local_score = self.local_risk(local).reshape(-1)
        residual = torch.log1p(torch.clamp(regions[:, -1], min=0.0))
        score = (
            local_score
            + nn.functional.softplus(self.residual_beta_raw) * residual
        )
        query = self.graph_query(graph_token)
        key = self.local_key(local)
        score = score + torch.mv(key, query) / math.sqrt(local.shape[1])
        attention = torch.softmax(score, dim=0)
        attended = torch.sum(attention[:, None] * local, dim=0)
        critical_index = torch.argmax(score)
        critical = local[critical_index]
        attention_entropy = -torch.sum(
            attention * torch.log(attention.clamp_min(1e-8))
        ) / math.log(max(int(len(attention)), 2))
        prototype_summary = torch.stack([
            torch.mean(regions[:, -4]),
            torch.mean(regions[:, -3]),
            torch.mean(regions[:, -2]),
            torch.max(regions[:, -1]),
        ])
        scalar_summary = torch.cat([
            score[critical_index].reshape(1),
            torch.mean(score).reshape(1),
            attention_entropy.reshape(1),
            torch.max(attention).reshape(1),
            prototype_summary,
        ])
        fused = torch.cat([
            graph_token,
            critical,
            attended,
            scalar_summary,
        ])
        direct_logit = self.classifier(fused).reshape(())
        diagnostic = torch.cat([
            direct_logit.reshape(1),
            score[critical_index].reshape(1),
            torch.mean(score).reshape(1),
            attention_entropy.reshape(1),
            torch.max(attention).reshape(1),
            attention_entropy.reshape(1),
            prototype_summary,
            graph_entropy.reshape(1),
        ])
        return direct_logit, diagnostic


def ordinary_weights(labels, base_probability, args):
    """Uniform training weights; no FixKeep supervision."""
    del base_probability, args
    labels = np.asarray(labels, dtype=np.int64)
    return np.ones(len(labels), dtype=np.float32), np.zeros(len(labels), dtype=bool)


def ordinary_bce(logits, labels, weights, base_probability, base_correct, args):
    """Ordinary BCE for an absolute classifier logit."""
    del weights, base_probability, base_correct, args
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def unused_inner_oof(rows, labels, compact_by_sha, compact_schema,
                     original_names, folds, split_seed, model_seed):
    """Avoid fitting an unused inner Global24 model in the direct-L2 arm."""
    del rows, compact_by_sha, compact_schema, original_names
    del folds, split_seed, model_seed
    return np.full(len(labels), 0.5, dtype=np.float64)


def rewrite_metadata(output: Path) -> None:
    architecture_path = output / "architecture.json"
    if architecture_path.exists():
        data = json.loads(architecture_path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update({
        "experiment": "CARE RGCondAttn independent second-layer source ablation v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "second_layer_input": [
            "31-role graph token",
            "critical Host/Ego region",
            "graph-conditioned attended Host/Ego evidence",
            "eight scalar evidence summaries",
        ],
        "second_layer_excludes": [
            "Global24 feature vector",
            "first-layer probability/logit",
        ],
        "training_loss": "ordinary binary cross entropy",
        "candidate_base": "none; the expert predicts an absolute logit",
        "residual_correction": False,
        "bounded_correction": False,
        "fixkeep": False,
        "variants": {
            "RGCondAttn-DirectL2": (
                "31-role graph conditions Host/Ego attention; the resulting "
                "evidence is classified independently"
            )
        },
        "deployment": "frozen Global24 outside A25; Direct-L2 probability inside A25",
        "future_data_loaded": False,
    })
    architecture_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_path = output / "source_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "CARE-Droid RGCondAttn Direct-L2 source ablation v1",
        "",
        "Strict source-only evaluation: fixed 2017 outer folds.",
        "Future years are not loaded.",
        "The frozen Global24/A25 pair is used only for routing and comparison.",
        "The second layer excludes Global24 features and the first-layer logit.",
        "It directly predicts an absolute malware logit from [g; h_critical; e; m].",
        "Training uses ordinary BCE: no residual addition, tanh bound, FixKeep weighting,",
        "or keep-consistency penalty. The direct prediction replaces Global24 only on A25.",
        "",
        "Source summary:",
    ]
    for row in rows:
        lines.append(
            f"{row['method']}: BA={float(row['BA']):.6f} "
            f"AUC={float(row['AUC']):.6f} F1={float(row['F1']):.6f} "
            f"fixed={int(float(row['fixed']))} broken={int(float(row['broken']))} "
            f"net={int(float(row['net']))}"
        )
    lines.extend([
        "",
        "Decision rule:",
        "Compare Direct-L2 against the frozen Global24 baseline and the established",
        "bounded RGCondAttn result using BA, AUC, F1, fixed, broken, net, and per-fold BA.",
        "Only after the source-side result is accepted should the frozen future diagnostic run.",
    ])
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Reuse the audited data loading, fixed outer folds, role-graph construction,
    # prototype fitting and exports from v4, while replacing only the model and
    # loss required by this ablation.
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": METHOD}
    v4.DIAGNOSTIC_NAMES = DIAGNOSTIC_NAMES
    v4.RoleGraphConditionedEvidence = DirectRGCondAttn
    v4.fixkeep_weights = ordinary_weights
    v4.batch_loss = ordinary_bce
    v4.inner_oof_global24 = unused_inner_oof
    v4.main()

    # argparse in v4 has already parsed the output argument. Recover it here
    # without maintaining a second conflicting argument parser.
    output = Path("artifacts/care-rgcondattn-direct-l2-source-v1")
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    rewrite_metadata(output)


if __name__ == "__main__":
    main()
