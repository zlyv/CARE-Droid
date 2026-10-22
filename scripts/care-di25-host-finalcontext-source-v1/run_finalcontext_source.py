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


BaseFull = make_masked_host_class(anomaly, "FULL")


def make_final_context_model(beta: float):

    beta = float(beta)

    class FinalContextHostAnomaly(BaseFull):

        final_context_beta = beta

        def forward(
            self,
            global_features,
            base_logit,
            graph,
            regions,
        ):
            # ==================================================
            # 1. Exact role-semantic encoder
            #    RoleNode + Pair + Inter
            # ==================================================
            role_index = torch.arange(
                v4.ROLE_COUNT,
                device=graph["node"].device,
            )

            hidden = (
                self.graph_encoder.node_input(
                    graph["node"]
                )
                +
                self.graph_encoder.role_embedding(
                    role_index
                )
            )

            for block in self.graph_encoder.blocks:
                hidden = block(
                    hidden,
                    graph["edge"],
                )

            pool_weight = torch.softmax(
                self.graph_encoder
                .pool_score(hidden)
                .reshape(-1),
                dim=0,
            )

            semantic_token = torch.sum(
                pool_weight[:, None] * hidden,
                dim=0,
            )

            # ==================================================
            # 2. Exact GraphGlobal27 context encoder
            # ==================================================
            global_hidden = (
                self.graph_encoder.global_encoder(
                    graph["global"].reshape(-1)
                )
            )

            # ==================================================
            # 3. FULL token for Host Query
            #
            # IMPORTANT:
            # Host alignment is ALWAYS original M0.
            # beta does NOT affect this path.
            # ==================================================
            full_graph_token = (
                self.graph_encoder.fuse(
                    torch.cat([
                        semantic_token,
                        global_hidden,
                    ])
                )
            )

            # ==================================================
            # 4. Controlled DIRECT token for final correction
            # ==================================================
            if abs(beta - 1.0) < 1e-12:
                # Exact original M0 path.
                correction_graph_token = (
                    full_graph_token
                )
            else:
                correction_graph_token = (
                    self.graph_encoder.fuse(
                        torch.cat([
                            semantic_token,
                            beta * global_hidden,
                        ])
                    )
                )

            graph_entropy = -torch.sum(
                pool_weight
                * torch.log(
                    pool_weight.clamp_min(1e-8)
                )
            ) / math.log(v4.ROLE_COUNT)

            # ==================================================
            # 5. Host evidence -- exact M0
            # ==================================================
            local = self.region_encoder(
                regions
            )

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
                +
                nn.functional.softplus(
                    self.residual_beta_raw
                )
                * residual
            )

            # FULL GraphGlobal context retained here.
            query = self.graph_query(
                full_graph_token
            )

            key = self.local_key(
                local
            )

            score = (
                score
                +
                torch.mv(
                    key,
                    query,
                )
                / math.sqrt(
                    local.shape[1]
                )
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

            prototype_summary = torch.stack([
                torch.mean(regions[:, -4]),
                torch.mean(regions[:, -3]),
                torch.mean(regions[:, -2]),
                torch.max(regions[:, -1]),
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

            # ==================================================
            # 6. Final correction
            #
            # ONLY experimental change:
            # graph token here is beta-controlled.
            # ==================================================
            fused = torch.cat([
                global_features.reshape(-1),
                base_logit.reshape(1),

                correction_graph_token,

                critical,
                attended,
                scalar_summary,
            ])

            if fused.numel() != 113:
                raise RuntimeError(
                    f"Correction width "
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

            return final_logit, diagnostic

    return FinalContextHostAnomaly


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--beta",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/"
            "care-di25-host-finalcontext-source-v1"
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

    beta = float(args.beta)

    if not 0.0 <= beta <= 1.0:
        raise RuntimeError(
            "beta must be in [0,1]"
        )

    if float(MAX_DELTA) != 2.25:
        raise RuntimeError(
            "max_delta must remain 2.25"
        )

    tag = f"{int(round(beta * 100)):03d}"

    root = (
        args.output_root
        / f"beta_{tag}"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    method = (
        "DI25__Host__Anomaly__"
        f"FinalCtxB{tag}__Bounded"
    )

    build_di_reference(
        PROJECT,
        root,
    )

    install_source_label_contract(
        v4
    )

    v4.MODEL_VARIANTS = (
        "RGCondAttn",
    )

    v4.METHOD_FOR_VARIANT = {
        "RGCondAttn": method
    }

    v4.RoleGraphConditionedEvidence = (
        make_final_context_model(
            beta
        )
    )

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
        root / "architecture.json"
    )

    architecture = json.loads(
        architecture_path.read_text(
            encoding="utf-8"
        )
    )

    architecture.update({
        "experiment":
            "CARE-Droid final-context beta sweep",

        "beta":
            beta,

        "host_query":
            "FULL Fuse(RoleSemantic, GraphGlobal)",

        "final_correction_graph":
            "Fuse(RoleSemantic, beta * GraphGlobal)",

        "graphglobal_host_query":
            "full",

        "graphglobal_direct_correction_strength":
            beta,

        "correction_input_width":
            113,

        "parameter_addition":
            0,

        "max_delta":
            2.25,

        "source_only":
            True,

        "future_data_loaded":
            False,
    })

    architecture_path.write_text(
        json.dumps(
            architecture,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"FINAL_CONTEXT_BETA_{tag}_DONE"
    )


if __name__ == "__main__":
    main()
