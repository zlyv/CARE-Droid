#!/usr/bin/env python3
"""Strict 2017 ablation for a deduplicated bounded RGCondAttn expert.

This keeps the established Global24 base, frozen A25 route, FixKeep objective,
and bounded logit correction.  It removes two implementation artifacts from
the shared RGCondAttn/RGCondDual class:

1. `context` was identical to `attended` for RGCondAttn and was concatenated
   twice into the correction head;
2. critical-query/context-key layers existed only for RGCondDual and were
   never used by RGCondAttn.

The cleaned correction representation is:

    [Global24; base_logit; graph_token; critical; attended; 8 summaries]
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


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
    import run_care_rolegraph_attncontext_v4 as v4
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing CARE v4 and its dependencies in scripts/: " + str(exc))


METHOD = "Global24__A25__CARE_RGCondAttn_Dedup_FixKeep"


class DedupBoundedRGCondAttn(nn.Module):
    """RGCondAttn with one attended vector and no unused dual-stream layers."""

    def __init__(self, global_dim: int, variant: str, args):
        super().__init__()
        if str(variant) != "RGCondAttn":
            raise ValueError("deduplicated model supports RGCondAttn only")
        self.max_delta = float(args.max_delta)
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

        # One critical vector and one attended vector.  No duplicate context.
        fused_dim = int(global_dim) + 1 + graph_dim + 2 * local_dim + 8
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
            global_features.reshape(-1),
            base_logit.reshape(1),
            graph_token,
            critical,
            attended,
            scalar_summary,
        ])
        raw_delta = self.delta_head(fused).reshape(())
        bounded_delta = self.max_delta * torch.tanh(raw_delta)
        final_logit = base_logit.reshape(()) + bounded_delta
        diagnostic = torch.cat([
            bounded_delta.reshape(1),
            score[critical_index].reshape(1),
            torch.mean(score).reshape(1),
            attention_entropy.reshape(1),
            torch.max(attention).reshape(1),
            # Retain the established diagnostic schema. For single-stream
            # RGCondAttn, context entropy is attention entropy by definition.
            attention_entropy.reshape(1),
            prototype_summary,
            graph_entropy.reshape(1),
        ])
        return final_logit, diagnostic


def rewrite_metadata(output: Path) -> None:
    architecture_path = output / "architecture.json"
    data = json.loads(architecture_path.read_text(encoding="utf-8"))
    data.update({
        "experiment": "CARE bounded RGCondAttn deduplication source ablation v1",
        "evaluation": "strict fixed 2017 outer folds only",
        "correction_input": [
            "Global24 vector (24)",
            "base logit (1)",
            "31-role graph token (32)",
            "critical Host/Ego vector (24)",
            "attended Host/Ego vector (24)",
            "scalar summaries (8)",
        ],
        "correction_input_width": 113,
        "removed_duplicate": "context was identical to attended in RGCondAttn",
        "removed_unused_modules": ["critical_query", "context_key"],
        "residual_correction": True,
        "bounded_correction": 1.5,
        "fixkeep": True,
        "future_data_loaded": False,
        "variants": {
            "RGCondAttn-Dedup": (
                "role token conditions local attention; one critical and one "
                "attended vector enter the bounded correction head"
            )
        },
    })
    architecture_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (output / "source_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "CARE-Droid bounded RGCondAttn deduplication source ablation v1",
        "",
        "Strict source-only evaluation: fixed 2017 outer folds.",
        "Future years are not loaded.",
        "Global24, frozen A25, FixKeep, and max-delta tanh correction are unchanged.",
        "Only the duplicated attended/context slot and unused RGCondDual projections",
        "are removed. The correction input is [xG; z1; g; h_critical; e; m].",
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
        "Prefer the deduplicated model if it is non-inferior in aggregate and per-fold",
        "BA while preserving or improving fixed/broken/net. Future evaluation follows",
        "only after this source-side decision.",
    ])
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": METHOD}
    v4.RoleGraphConditionedEvidence = DedupBoundedRGCondAttn
    v4.main()
    output = Path("artifacts/care-rgcondattn-dedup-bounded-source-v1")
    if "--output" in sys.argv:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    rewrite_metadata(output)


if __name__ == "__main__":
    main()
