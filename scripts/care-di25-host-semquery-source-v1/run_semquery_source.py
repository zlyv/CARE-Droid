#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

for path in (
    HERE,
    PROJECT / "scripts",
    PROJECT / "scripts/care-di25-host-input-replay-v1",
    PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    PROJECT / "scripts/care-rgcondattn-dedup-bounded-source-v1",
    PROJECT / "scripts/care-malben-host-v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from masked_host_model import make_masked_host_class
from experiment_contract import MAX_DELTA, build_di_reference
from source_label_contract import install_source_label_contract

import run_care_rolegraph_conditioned_dualstream_v4 as v4
import run_care_rgcondattn_dedup_bounded_source_v1 as anomaly


METHOD = "DI25__Host__Anomaly__SemQuery__Bounded"

# FULL keeps all original 113 correction dimensions.
BaseFull = make_masked_host_class(anomaly, "FULL")


def encode_graph_parts(encoder, node, edge, graph_global):
    """
    Compute the original role-graph encoder in ONE pass.

    semantic_token:
        RoleNode + Pair + Inter only.
        This is the pooled detailed role-semantic representation.

    graph_token:
        Fuse(semantic_token, GraphGlobal27 context).
        This remains the representation used by the final correction head.
    """

    role_index = torch.arange(
        v4.ROLE_COUNT,
        device=node.device,
    )

    hidden = (
        encoder.node_input(node)
        + encoder.role_embedding(role_index)
    )

    # Pair / Inter information enters through edge-biased
    # graph-attention blocks, exactly as in original M0.
    for block in encoder.blocks:
        hidden = block(hidden, edge)

    pool_weight = torch.softmax(
        encoder.pool_score(hidden).reshape(-1),
        dim=0,
    )

    # Fine-grained role-semantic structure only.
    semantic_token = torch.sum(
        pool_weight[:, None] * hidden,
        dim=0,
    )

    # GraphGlobal27 remains the global structural context.
    global_hidden = encoder.global_encoder(
        graph_global.reshape(-1)
    )

    # Original fused graph representation.
    graph_token = encoder.fuse(
        torch.cat([
            semantic_token,
            global_hidden,
        ])
    )

    graph_entropy = -torch.sum(
        pool_weight
        * torch.log(
            pool_weight.clamp_min(1e-8)
        )
    ) / math.log(v4.ROLE_COUNT)

    return (
        graph_token,
        semantic_token,
        graph_entropy,
    )


class SemanticQueryHostAnomaly(BaseFull):
    """
    CARE-Droid SemQuery.

    Only architectural change relative to FULL M0:

        original:
            query = graph_query(graph_token)

        SemQuery:
            query = graph_query(semantic_token)

    GraphGlobal27 remains in graph_token and therefore remains
    available to the final bounded correction head.
    """

    semantic_query_only = True

    def forward(
        self,
        global_features,
        base_logit,
        graph,
        regions,
    ):
        (
            graph_token,
            semantic_token,
            graph_entropy,
        ) = encode_graph_parts(
            self.graph_encoder,
            graph["node"],
            graph["edge"],
            graph["global"],
        )

        # --------------------------------------------------
        # Host local evidence: unchanged.
        # --------------------------------------------------

        local = self.region_encoder(regions)

        local_score = self.local_risk(
            local
        ).reshape(-1)

        residual = torch.log1p(
            torch.clamp(
                regions[:, -1],
                min=0.0,
            )
        )

        score = (
            local_score
            + nn.functional.softplus(
                self.residual_beta_raw
            )
            * residual
        )

        # ==================================================
        # ONLY CHANGE:
        #
        # OLD:
        # query = self.graph_query(graph_token)
        #
        # NEW:
        # Role-semantic structure alone selects Host.
        # ==================================================

        query = self.graph_query(
            semantic_token
        )

        key = self.local_key(local)

        alignment_score = (
            torch.mv(
                key,
                query,
            )
            / math.sqrt(
                local.shape[1]
            )
        )

        score = (
            score
            + alignment_score
        )

        attention = torch.softmax(
            score,
            dim=0,
        )

        attended = torch.sum(
            attention[:, None] * local,
            dim=0,
        )

        critical_index = torch.argmax(
            score
        )

        critical = local[
            critical_index
        ]

        attention_entropy = -torch.sum(
            attention
            * torch.log(
                attention.clamp_min(1e-8)
            )
        ) / math.log(
            max(
                int(len(attention)),
                2,
            )
        )

        # --------------------------------------------------
        # Original summary features: unchanged.
        # --------------------------------------------------

        prototype_summary = torch.stack([
            torch.mean(
                regions[:, -4]
            ),
            torch.mean(
                regions[:, -3]
            ),
            torch.mean(
                regions[:, -2]
            ),
            torch.max(
                regions[:, -1]
            ),
        ])

        scalar_summary = torch.cat([
            score[
                critical_index
            ].reshape(1),

            torch.mean(
                score
            ).reshape(1),

            attention_entropy.reshape(1),

            torch.max(
                attention
            ).reshape(1),

            prototype_summary,
        ])

        # --------------------------------------------------
        # IMPORTANT:
        # Final correction still receives graph_token,
        # which still contains GraphGlobal27 context.
        # --------------------------------------------------

        fused = torch.cat([
            global_features.reshape(-1),
            base_logit.reshape(1),

            # Original fused role graph + GraphGlobal context.
            graph_token,

            critical,
            attended,
            scalar_summary,
        ])

        if fused.numel() != 113:
            raise RuntimeError(
                "SemQuery correction input width "
                f"{fused.numel()} != 113"
            )

        masked = (
            fused
            * self.correction_mask.to(
                dtype=fused.dtype
            )
        )

        raw_delta = self.delta_head(
            masked
        ).reshape(())

        bounded_delta = (
            self.max_delta
            * torch.tanh(
                raw_delta
            )
        )

        final_logit = (
            base_logit.reshape(())
            + bounded_delta
        )

        # Keep diagnostic schema identical to M0.
        diagnostic = torch.cat([
            bounded_delta.reshape(1),

            score[
                critical_index
            ].reshape(1),

            torch.mean(
                score
            ).reshape(1),

            attention_entropy.reshape(1),

            torch.max(
                attention
            ).reshape(1),

            attention_entropy.reshape(1),

            prototype_summary,

            graph_entropy.reshape(1),
        ])

        return (
            final_logit,
            diagnostic,
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "care-di25-host-semquery-source-v1"
        ),
    )

    parser.add_argument(
        "--seeds",
        default="0,1,2",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    if float(MAX_DELTA) != 2.25:
        raise RuntimeError(
            "max_delta must remain 2.25"
        )

    root = args.output
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same frozen DI25 router.
    build_di_reference(
        PROJECT,
        root,
    )

    install_source_label_contract(
        v4
    )

    # Only one model variant.
    v4.MODEL_VARIANTS = (
        "RGCondAttn",
    )

    v4.METHOD_FOR_VARIANT = {
        "RGCondAttn":
            METHOD
    }

    # Replace only Stage-II model class.
    v4.RoleGraphConditionedEvidence = (
        SemanticQueryHostAnomaly
    )

    # EXACT same formal source protocol as M0.
    sys.argv = [
        sys.argv[0],

        "--previous-result",
        str(
            root
            / "preflight"
            / "di25_reference"
        ),

        "--cache",
        "artifacts/"
        "care-source-local-assets-v2/"
        "endpoint_centered_tensor_cache_2017.jsonl.gz",

        "--region-cache",
        "artifacts/"
        "care-source-local-assets-v2/"
        "regions_2017.npz",

        "--variants",
        "RGCondAttn",

        "--seeds",
        args.seeds,

        "--max-delta",
        "2.25",

        "--device",
        args.device,

        "--output",
        str(root),
    ]

    v4.main()

    architecture_path = (
        root
        / "architecture.json"
    )

    architecture = json.loads(
        architecture_path.read_text(
            encoding="utf-8"
        )
    )

    architecture.update({
        "experiment":
            "CARE-Droid SemQuery source-only v1",

        "method":
            METHOD,

        "changed_component":
            "Host attention query only",

        "original_query":
            "fused role-graph token "
            "(semantic structure + GraphGlobal27)",

        "new_query":
            "semantic role-graph token "
            "(RoleNode + Pair + Inter)",

        "graphglobal_removed":
            False,

        "graphglobal_role":
            "global structural context retained "
            "in final fused correction representation",

        "correction_input_width":
            113,

        "max_delta_logit":
            2.25,

        "router":
            "frozen DI25",

        "source_only":
            True,

        "future_data_loaded":
            False,

        "parameter_addition":
            0,
    })

    architecture_path.write_text(
        json.dumps(
            architecture,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "SEMQUERY_SOURCE_DONE"
    )

    print(
        "output:",
        root,
    )


if __name__ == "__main__":
    main()
