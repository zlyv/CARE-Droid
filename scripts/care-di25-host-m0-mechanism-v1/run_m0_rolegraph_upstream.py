#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path("/home/boyuanjie/projects/tasa-droid")

BASE_PATH = (
    PROJECT
    / "scripts"
    / "care-di25-host-m0-mechanism-v1"
    / "run_m0_block_replacement.py"
)

spec = importlib.util.spec_from_file_location(
    "m0_block_base",
    BASE_PATH,
)

base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

v4 = base.v4


ARMS = (
    "NODE",
    "GRAPH_GLOBAL",
    "PAIR",
    "INTER",
    "FULL_GRAPH",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("empty rows")

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def clone_graph(one_graph):
    return {
        key: value.clone()
        for key, value
        in one_graph.items()
    }


def apply_replacement(
    one_graph,
    arm,
    reference,
):

    cf = clone_graph(
        one_graph
    )

    if arm == "NODE":
        cf["node"] = (
            reference["node"]
            .clone()
        )

    elif arm == "GRAPH_GLOBAL":
        cf["global"] = (
            reference["global"]
            .clone()
        )

    elif arm == "PAIR":
        edge = (
            cf["edge"]
            .clone()
        )

        edge[..., 0] = (
            reference["edge"]
            [..., 0]
        )

        cf["edge"] = edge

    elif arm == "INTER":
        edge = (
            cf["edge"]
            .clone()
        )

        # channel 1 = inter
        # channel 2 = transpose(inter)
        edge[..., 1:3] = (
            reference["edge"]
            [..., 1:3]
        )

        cf["edge"] = edge

    elif arm == "FULL_GRAPH":
        cf = {
            key:
            reference[key].clone()
            for key
            in (
                "node",
                "edge",
                "global",
            )
        }

    else:
        raise RuntimeError(
            "unknown arm: " + arm
        )

    return cf


def upstream_predict(
    model,
    global_values,
    base_probability,
    graph_values,
    bags,
    device,
):

    if base.PREDICT_CALL_INDEX >= len(
        base.SEEDS
    ):
        raise RuntimeError(
            "predict called too many times"
        )

    seed = base.SEEDS[
        base.PREDICT_CALL_INDEX
    ]

    base.PREDICT_CALL_INDEX += 1

    # -------------------------------------------------
    # Exact source cohort:
    # routed Stage-I false negatives.
    # -------------------------------------------------

    source_fn = (
        (base.SOURCE_Y == 1)
        &
        (base.SOURCE_BASE < 0.5)
        &
        base.SOURCE_ROUTE
    )

    if int(
        np.sum(source_fn)
    ) != 85:
        raise RuntimeError(
            "source FN cohort changed"
        )

    # SOURCE_GRAPH has already been transformed by the
    # exact source-fitted role-graph scaler.
    source_reference_np = {
        key: np.mean(
            np.asarray(
                base.SOURCE_GRAPH[key]
            )[source_fn],
            axis=0,
        ).astype(np.float32)
        for key in (
            "node",
            "edge",
            "global",
        )
    }

    source_reference = {
        key: torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
        for key, value
        in source_reference_np.items()
    }

    # Audit reference magnitudes.
    base.CENTROID_ROWS.extend([
        {
            "seed": seed,
            "block": "NODE",
            "source_routed_fn_n": 85,
            "centroid_l2": float(
                np.linalg.norm(
                    source_reference_np[
                        "node"
                    ]
                )
            ),
        },
        {
            "seed": seed,
            "block": "GRAPH_GLOBAL",
            "source_routed_fn_n": 85,
            "centroid_l2": float(
                np.linalg.norm(
                    source_reference_np[
                        "global"
                    ]
                )
            ),
        },
        {
            "seed": seed,
            "block": "PAIR",
            "source_routed_fn_n": 85,
            "centroid_l2": float(
                np.linalg.norm(
                    source_reference_np[
                        "edge"
                    ][..., 0]
                )
            ),
        },
        {
            "seed": seed,
            "block": "INTER",
            "source_routed_fn_n": 85,
            "centroid_l2": float(
                np.linalg.norm(
                    source_reference_np[
                        "edge"
                    ][..., 1:3]
                )
            ),
        },
        {
            "seed": seed,
            "block": "FULL_GRAPH",
            "source_routed_fn_n": 85,
            "centroid_l2": float(
                np.sqrt(
                    np.sum(
                        source_reference_np[
                            "node"
                        ] ** 2
                    )
                    +
                    np.sum(
                        source_reference_np[
                            "edge"
                        ] ** 2
                    )
                    +
                    np.sum(
                        source_reference_np[
                            "global"
                        ] ** 2
                    )
                )
            ),
        },
    ])

    # -------------------------------------------------
    # Exact official prediction preparation.
    # -------------------------------------------------

    global_tensor = torch.as_tensor(
        global_values,
        dtype=torch.float32,
        device=device,
    )

    base_logit = torch.as_tensor(
        v4.probability_to_logit(
            base_probability
        ),
        dtype=torch.float32,
        device=device,
    )

    graph = v4.graph_tensors(
        graph_values,
        device,
    )

    bag_tensors = [
        torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
        for value in bags
    ]

    probability = []
    diagnostic = []

    model.eval()

    with torch.no_grad():

        for index in range(
            len(bags)
        ):

            one_graph = {
                key: value[index]
                for key, value
                in graph.items()
            }

            # -----------------------------------------
            # Original frozen forward.
            # -----------------------------------------

            original_logit, original_diag = (
                model(
                    global_tensor[index],
                    base_logit[index],
                    one_graph,
                    bag_tensors[index],
                )
            )

            original_probability = float(
                torch.sigmoid(
                    original_logit
                ).cpu()
            )

            probability.append(
                original_probability
            )

            diagnostic.append(
                original_diag
                .detach()
                .cpu()
                .numpy()
            )

            row = {
                "sha256":
                    base.TARGET_SHA[index],

                "year":
                    int(
                        base.TARGET_YEAR[
                            index
                        ]
                    ),

                "seed":
                    int(seed),

                "p_stage1":
                    float(
                        base_probability[
                            index
                        ]
                    ),

                "original_probability":
                    original_probability,

                "original_bounded_delta":
                    float(
                        original_diag[
                            0
                        ].cpu()
                    ),

                "original_critical_score":
                    float(
                        original_diag[
                            1
                        ].cpu()
                    ),

                "original_mean_score":
                    float(
                        original_diag[
                            2
                        ].cpu()
                    ),

                "original_attention_entropy":
                    float(
                        original_diag[
                            3
                        ].cpu()
                    ),

                "original_max_attention":
                    float(
                        original_diag[
                            4
                        ].cpu()
                    ),
            }

            # -----------------------------------------
            # Upstream role-graph replacements.
            # -----------------------------------------

            for arm in ARMS:

                cf_graph = (
                    apply_replacement(
                        one_graph,
                        arm,
                        source_reference,
                    )
                )

                cf_logit, cf_diag = model(
                    global_tensor[index],
                    base_logit[index],
                    cf_graph,
                    bag_tensors[index],
                )

                token = arm.lower()

                row[
                    f"cf_{token}_probability"
                ] = float(
                    torch.sigmoid(
                        cf_logit
                    ).cpu()
                )

                row[
                    f"cf_{token}_bounded_delta"
                ] = float(
                    cf_diag[
                        0
                    ].cpu()
                )

                row[
                    f"cf_{token}_critical_score"
                ] = float(
                    cf_diag[
                        1
                    ].cpu()
                )

                row[
                    f"cf_{token}_mean_score"
                ] = float(
                    cf_diag[
                        2
                    ].cpu()
                )

                row[
                    f"cf_{token}_attention_entropy"
                ] = float(
                    cf_diag[
                        3
                    ].cpu()
                )

                row[
                    f"cf_{token}_max_attention"
                ] = float(
                    cf_diag[
                        4
                    ].cpu()
                )

            base.ATTR_ROWS.append(
                row
            )

    return (
        np.asarray(
            probability,
            dtype=np.float64,
        ),
        np.asarray(
            diagnostic,
            dtype=np.float64,
        ),
    )


def main():

    # Dedicated output; never overwrite the already
    # validated token-level attribution.
    args = (
        base.future
        .parser()
        .parse_args()
    )

    output = args.output

    base.ATTR_ROWS.clear()
    base.CENTROID_ROWS.clear()

    # Monkeypatch only prediction-time diagnostic logic.
    # Training code remains untouched.
    base.custom_predict = (
        upstream_predict
    )

    base.main()

    # Rewrite a correct contract for THIS experiment.
    contract = {
        "passed": True,
        "diagnostic_only": True,
        "future_labels_loaded_by_runner":
            False,

        "model":
            "Frozen M0 Host-Anomaly-Bounded",

        "source_reference":
            "2017 DI25-routed Stage-I FN",

        "source_reference_n":
            85,

        "replacement_location":
            "transformed role-graph input before graph encoder",

        "arms": {
            "NODE":
                "replace transformed node tensor",
            "GRAPH_GLOBAL":
                "replace transformed graph-global tensor",
            "PAIR":
                "replace edge channel 0 (pair_z)",
            "INTER":
                "replace edge channels 1-2 (inter and transpose)",
            "FULL_GRAPH":
                "replace node + edge + graph-global",
        },

        "important":
            "All replacements pass through the frozen graph encoder, graph-conditioned Host attention, fused correction head, and bounded delta.",

        "seeds":
            base.SEEDS,

        "max_delta":
            2.25,
    }

    (
        output
        / "rolegraph_upstream_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "M0_ROLEGRAPH_UPSTREAM_PASS"
    )


if __name__ == "__main__":
    main()
