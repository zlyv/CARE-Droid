#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
MATRIX = (
    PROJECT
    / "scripts"
    / "care-route-stage2-future-matrix-v1"
)

sys.path.insert(
    0,
    str(MATRIX),
)

import run_future_arm as future


v4 = future.v4


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1"
)

SOURCE_ROUTER = Path(
    "artifacts/care-source-di25-reconstructed-v1/"
    "source_router_inputs.csv"
)

LOCKED = (
    ROOT
    / "anomaly_bounded"
    / "future_expert_predictions.csv"
)


SEEDS = [
    900000,
    900001,
    900002,
]


BLOCKS = {
    "GLOBAL24": (0, 24),
    "STAGE1_LOGIT": (24, 25),
    "ROLE_GRAPH": (25, 57),
    "CRITICAL_HOST": (57, 81),
    "ATTENDED_HOST": (81, 105),
    "SUMMARY": (105, 113),
}


SOURCE_GLOBAL = None
SOURCE_BASE = None
SOURCE_GRAPH = None
SOURCE_BAGS = None
SOURCE_Y = None
SOURCE_ROUTE = None

# Exact per-seed source bags returned by the original
# fit_scaled_dualproto_bags() call.
LATEST_TRAIN_BAGS = None

TARGET_SHA = None
TARGET_YEAR = None

PREDICT_CALL_INDEX = 0

ATTR_ROWS = []
CENTROID_ROWS = []


def norm(value):
    return str(value).strip().lower()


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_csv(path, rows):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "refusing to write empty CSV"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

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


def capture_scaled_dualproto_bags(*args, **kwargs):
    global LATEST_TRAIN_BAGS

    result = future.ORIGINAL_ANOMALY_FIT(
        *args,
        **kwargs,
    )

    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "unexpected fit_scaled_dualproto_bags "
            "return contract"
        )

    train_bags, test_bags, audit = result

    # fit_predict_variant() calls this function twice
    # within each seed:
    #
    # 1) select_epochs(): inner split, e.g. 655/144
    # 2) final fit(): exact 799 source / 1989 future
    #
    # Only the second call belongs to the final trained
    # model whose delta-head input we are attributing.
    if (
        len(train_bags) == 799
        and
        len(test_bags) == 1989
    ):
        LATEST_TRAIN_BAGS = train_bags

    # All calls, including the inner epoch-selection call,
    # are returned completely unchanged.
    return train_bags, test_bags, audit


def probability_to_logit(p):

    p = np.clip(
        np.asarray(p, dtype=np.float64),
        1e-12,
        1.0 - 1e-12,
    )

    return np.log(
        p / (1.0 - p)
    )


def model_inputs(
    global_values,
    base_probability,
    graph_values,
    bags,
    device,
):

    global_tensor = torch.as_tensor(
        global_values,
        dtype=torch.float32,
        device=device,
    )

    # Use the exact frozen CARE implementation.
    # Do not duplicate the probability-to-logit arithmetic
    # in the diagnostic harness.
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

    return (
        global_tensor,
        base_logit,
        graph,
        bag_tensors,
    )


def capture_fused_for_dataset(
    model,
    global_values,
    base_probability,
    graph_values,
    bags,
    device,
):

    (
        global_tensor,
        base_logit,
        graph,
        bag_tensors,
    ) = model_inputs(
        global_values,
        base_probability,
        graph_values,
        bags,
        device,
    )

    captured = {
        "fused": None,
    }

    def pre_hook(
        module,
        inputs,
    ):
        del module

        x = inputs[0]

        captured["fused"] = (
            x.detach()
            .cpu()
            .numpy()
            .copy()
        )

    handle = (
        model.delta_head
        .register_forward_pre_hook(
            pre_hook
        )
    )

    fused_rows = []

    model.eval()

    try:

        with torch.no_grad():

            for i in range(
                len(bag_tensors)
            ):

                one_graph = {
                    key: value[i]
                    for key, value
                    in graph.items()
                }

                model(
                    global_tensor[i],
                    base_logit[i],
                    one_graph,
                    bag_tensors[i],
                )

                fused = captured[
                    "fused"
                ]

                if fused is None:
                    raise RuntimeError(
                        "delta-head pre-hook "
                        "did not capture fused input"
                    )

                fused = np.asarray(
                    fused,
                    dtype=np.float64,
                ).reshape(-1)

                if fused.shape != (113,):
                    raise RuntimeError(
                        "fused width changed: "
                        + repr(
                            fused.shape
                        )
                    )

                fused_rows.append(
                    fused.copy()
                )

    finally:

        handle.remove()

    result = np.asarray(
        fused_rows,
        dtype=np.float64,
    )

    if result.shape != (
        len(bag_tensors),
        113,
    ):
        raise RuntimeError(
            "fused matrix shape mismatch: "
            + repr(result.shape)
        )

    return result


def custom_predict(
    model,
    global_values,
    base_probability,
    graph_values,
    bags,
    device,
):

    global PREDICT_CALL_INDEX

    if PREDICT_CALL_INDEX >= len(SEEDS):
        raise RuntimeError(
            "predict_model called more "
            "than three times"
        )

    seed = SEEDS[
        PREDICT_CALL_INDEX
    ]

    PREDICT_CALL_INDEX += 1

    # -------------------------------------------------
    # 1. Source reference in EXACT SAME trained model.
    # -------------------------------------------------

    if LATEST_TRAIN_BAGS is None:
        raise RuntimeError(
            "per-seed transformed source bags "
            "were not captured"
        )

    if len(LATEST_TRAIN_BAGS) != 799:
        raise RuntimeError(
            "per-seed transformed source bag "
            "cohort changed"
        )

    source_fused = (
        capture_fused_for_dataset(
            model,
            SOURCE_GLOBAL,
            SOURCE_BASE,
            SOURCE_GRAPH,
            LATEST_TRAIN_BAGS,
            device,
        )
    )

    source_fn = (
        (SOURCE_Y == 1)
        &
        (SOURCE_BASE < 0.5)
        &
        SOURCE_ROUTE
    )

    source_fn_n = int(
        np.sum(source_fn)
    )

    if source_fn_n != 85:
        raise RuntimeError(
            "expected 85 routed source FN, "
            f"got {source_fn_n}"
        )

    source_centroid = np.mean(
        source_fused[
            source_fn
        ],
        axis=0,
    )

    if source_centroid.shape != (
        113,
    ):
        raise RuntimeError(
            "source centroid width mismatch"
        )

    for block, (
        left,
        right,
    ) in BLOCKS.items():

        values = source_centroid[
            left:right
        ]

        CENTROID_ROWS.append({
            "seed": seed,
            "block": block,
            "left": left,
            "right": right,
            "width": right - left,
            "source_routed_fn_n":
                source_fn_n,
            "centroid_l2":
                float(
                    np.linalg.norm(
                        values
                    )
                ),
            "centroid_mean":
                float(
                    np.mean(
                        values
                    )
                ),
            "centroid_std":
                float(
                    np.std(
                        values
                    )
                ),
        })

    # -------------------------------------------------
    # 2. Future original forward +
    #    head-input block replacement.
    # -------------------------------------------------

    (
        global_tensor,
        base_logit,
        graph,
        bag_tensors,
    ) = model_inputs(
        global_values,
        base_probability,
        graph_values,
        bags,
        device,
    )

    captured = {
        "fused": None,
    }

    def pre_hook(
        module,
        inputs,
    ):
        del module

        captured["fused"] = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    handle = (
        model.delta_head
        .register_forward_pre_hook(
            pre_hook
        )
    )

    probability = []
    diagnostic = []

    model.eval()

    try:

        with torch.no_grad():

            for i in range(
                len(bag_tensors)
            ):

                one_graph = {
                    key: value[i]
                    for key, value
                    in graph.items()
                }

                final_logit, diag = model(
                    global_tensor[i],
                    base_logit[i],
                    one_graph,
                    bag_tensors[i],
                )

                p_original = float(
                    torch.sigmoid(
                        final_logit
                    ).cpu()
                )

                probability.append(
                    p_original
                )

                diagnostic.append(
                    diag.detach()
                    .cpu()
                    .numpy()
                )

                fused = np.asarray(
                    captured["fused"],
                    dtype=np.float64,
                ).reshape(-1)

                if fused.shape != (113,):
                    raise RuntimeError(
                        "future fused width "
                        "mismatch"
                    )

                original_raw = (
                    float(
                        model.delta_head(
                            torch.as_tensor(
                                fused,
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        .reshape(())
                        .cpu()
                    )
                )

                max_delta = float(
                    model.max_delta
                )

                original_bounded = (
                    max_delta
                    * np.tanh(
                        original_raw
                    )
                )

                base_z = float(
                    base_logit[i]
                    .detach()
                    .cpu()
                )

                row = {
                    "sha256":
                        TARGET_SHA[i],
                    "year":
                        int(
                            TARGET_YEAR[i]
                        ),
                    "seed":
                        seed,
                    "p_stage1":
                        float(
                            base_probability[i]
                        ),
                    "stage1_logit":
                        base_z,
                    "original_raw_delta":
                        original_raw,
                    "original_bounded_delta":
                        original_bounded,
                    "original_probability":
                        p_original,
                }

                for block, (
                    left,
                    right,
                ) in BLOCKS.items():

                    cf = fused.copy()

                    cf[
                        left:right
                    ] = source_centroid[
                        left:right
                    ]

                    cf_tensor = (
                        torch.as_tensor(
                            cf,
                            dtype=torch.float32,
                            device=device,
                        )
                    )

                    cf_raw = float(
                        model.delta_head(
                            cf_tensor
                        )
                        .reshape(())
                        .cpu()
                    )

                    cf_bounded = (
                        max_delta
                        * np.tanh(
                            cf_raw
                        )
                    )

                    cf_final_z = (
                        base_z
                        + cf_bounded
                    )

                    cf_probability = (
                        1.0
                        /
                        (
                            1.0
                            +
                            np.exp(
                                -cf_final_z
                            )
                        )
                    )

                    token = block.lower()

                    row[
                        "cf_"
                        + token
                        + "_raw_delta"
                    ] = cf_raw

                    row[
                        "cf_"
                        + token
                        + "_bounded_delta"
                    ] = cf_bounded

                    row[
                        "cf_"
                        + token
                        + "_probability"
                    ] = float(
                        cf_probability
                    )

                ATTR_ROWS.append(
                    row
                )

    finally:

        handle.remove()

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

    global SOURCE_GLOBAL
    global SOURCE_BASE
    global SOURCE_GRAPH
    global SOURCE_BAGS
    global SOURCE_Y
    global SOURCE_ROUTE
    global TARGET_SHA
    global TARGET_YEAR
    global PREDICT_CALL_INDEX
    global LATEST_TRAIN_BAGS

    args = (
        future.parser()
        .parse_args()
    )

    if args.representation != "anomaly":
        raise SystemExit(
            "requires --representation anomaly"
        )

    if args.head != "bounded":
        raise SystemExit(
            "requires --head bounded"
        )

    seeds = v4.parse_ints(
        args.seeds
    )

    if seeds != SEEDS:
        raise SystemExit(
            "requires seeds "
            "900000,900001,900002"
        )

    if float(
        args.max_delta
    ) != 2.25:
        raise SystemExit(
            "requires --max-delta 2.25"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = v4.resolve_device(
        args.device
    )

    data = future.load_common(
        args
    )

    # -------------------------------------------------
    # Reconstruct same Global24 matrices used by M0.
    # -------------------------------------------------

    (
        x_source,
        x_target,
        feature_names,
    ) = (
        v4.audit
        .compact_matrix_variant(
            v4.VARIANT,
            data.source_rows,
            data.target_rows,
            data.compact_by_sha,
            data.schema,
            data.names,
        )
    )

    global_index, _ = (
        v4.evidence
        .split_view_indices(
            feature_names
        )
    )

    SOURCE_GLOBAL = np.asarray(
        x_source[
            :,
            global_index,
        ],
        dtype=np.float32,
    )

    target_global = np.asarray(
        x_target[
            :,
            global_index,
        ],
        dtype=np.float32,
    )

    if SOURCE_GLOBAL.shape != (
        799,
        24,
    ):
        raise RuntimeError(
            "source Global24 mismatch"
        )

    if target_global.shape != (
        1989,
        24,
    ):
        raise RuntimeError(
            "future Global24 mismatch"
        )

    SOURCE_BASE = np.asarray(
        data.base_source,
        dtype=np.float64,
    )

    # Exact role-graph preprocessing used by the
    # frozen fit_predict_variant().
    source_graph_scaler = (
        v4.fit_role_graph_scaler(
            data.source_s7
        )
    )

    SOURCE_GRAPH = (
        v4.transform_role_graphs(
            data.source_s7,
            source_graph_scaler,
        )
    )

    if set(SOURCE_GRAPH) != {
        "node",
        "edge",
        "global",
    }:
        raise RuntimeError(
            "source transformed graph contract "
            "changed: "
            + repr(
                sorted(
                    SOURCE_GRAPH.keys()
                )
            )
        )

    # Raw bags kept only as source assets.
    # Actual source forward uses LATEST_TRAIN_BAGS.
    SOURCE_BAGS = data.source_bags

    SOURCE_Y = np.asarray(
        data.source_y,
        dtype=np.int64,
    )

    # -------------------------------------------------
    # Canonical source DI25 route.
    # -------------------------------------------------

    router = read_csv(
        SOURCE_ROUTER
    )

    router_map = {
        norm(row["sha256"]): row
        for row in router
    }

    if len(router_map) != 799:
        raise RuntimeError(
            "source router cohort mismatch"
        )

    source_sha = [
        norm(
            row["sha256"]
        )
        for row
        in data.source_rows
    ]

    SOURCE_ROUTE = np.asarray(
        [
            int(
                float(
                    router_map[sha][
                        "route_DI"
                    ]
                )
            )
            == 1
            for sha
            in source_sha
        ],
        dtype=bool,
    )

    if int(
        np.sum(
            SOURCE_ROUTE
        )
    ) != 200:
        raise RuntimeError(
            "source DI25 route changed"
        )

    # -------------------------------------------------
    # Source score alignment audit.
    #
    # IMPORTANT:
    # - SOURCE_BASE is the exact Stage-I probability
    #   actually consumed by the frozen M0 runner.
    # - router_map supplies the canonical DI25 route.
    #
    # The two historical source-score files are not
    # required to be numerically identical.  What is
    # required here is that they induce the same
    # Stage-I decisions and the same routed FN/FP
    # membership.
    # -------------------------------------------------

    router_probability = np.asarray(
        [
            float(
                router_map[sha][
                    "p_stage1"
                ]
            )
            for sha in source_sha
        ],
        dtype=np.float64,
    )

    # Label alignment remains exact.
    for i, sha in enumerate(
        source_sha
    ):

        rr = router_map[sha]

        if int(
            float(
                rr["label"]
            )
        ) != int(
            SOURCE_Y[i]
        ):
            raise RuntimeError(
                "source label alignment failed"
            )

    m0_prediction = (
        SOURCE_BASE >= 0.5
    )

    router_prediction = (
        router_probability >= 0.5
    )

    probability_difference = np.abs(
        SOURCE_BASE
        - router_probability
    )

    disagreement = (
        m0_prediction
        != router_prediction
    )

    routed_disagreement = (
        SOURCE_ROUTE
        &
        disagreement
    )

    m0_fn = (
        SOURCE_ROUTE
        &
        (SOURCE_Y == 1)
        &
        (~m0_prediction)
    )

    router_fn = (
        SOURCE_ROUTE
        &
        (SOURCE_Y == 1)
        &
        (~router_prediction)
    )

    m0_fp = (
        SOURCE_ROUTE
        &
        (SOURCE_Y == 0)
        &
        m0_prediction
    )

    router_fp = (
        SOURCE_ROUTE
        &
        (SOURCE_Y == 0)
        &
        router_prediction
    )

    print()
    print(
        "SOURCE_SCORE_ALIGNMENT"
    )

    print(
        "max |M0-router| =",
        float(
            np.max(
                probability_difference
            )
        ),
    )

    print(
        "mean |M0-router| =",
        float(
            np.mean(
                probability_difference
            )
        ),
    )

    print(
        "decision disagreements =",
        int(
            np.sum(
                disagreement
            )
        ),
    )

    print(
        "routed decision disagreements =",
        int(
            np.sum(
                routed_disagreement
            )
        ),
    )

    print(
        "M0 routed FN =",
        int(
            np.sum(
                m0_fn
            )
        ),
    )

    print(
        "router routed FN =",
        int(
            np.sum(
                router_fn
            )
        ),
    )

    print(
        "M0 routed FP =",
        int(
            np.sum(
                m0_fp
            )
        ),
    )

    print(
        "router routed FP =",
        int(
            np.sum(
                router_fp
            )
        ),
    )

    # Hard safety gates.
    if np.any(
        disagreement
    ):
        raise RuntimeError(
            "source Stage-I decision "
            "alignment failed"
        )

    if np.any(
        routed_disagreement
    ):
        raise RuntimeError(
            "routed Stage-I decision "
            "alignment failed"
        )

    if not np.array_equal(
        m0_fn,
        router_fn,
    ):
        raise RuntimeError(
            "source routed FN membership "
            "changed between score files"
        )

    if not np.array_equal(
        m0_fp,
        router_fp,
    ):
        raise RuntimeError(
            "source routed FP membership "
            "changed between score files"
        )

    if int(
        np.sum(
            m0_fn
        )
    ) != 85:
        raise RuntimeError(
            "expected exact 85 M0 "
            "routed source FN"
        )

    if int(
        np.sum(
            m0_fp
        )
    ) != 39:
        raise RuntimeError(
            "expected exact 39 M0 "
            "routed source FP"
        )

    TARGET_SHA = [
        norm(
            row["sha256"]
        )
        for row
        in data.target_rows
    ]

    TARGET_YEAR = [
        int(
            row["year"]
        )
        for row
        in data.future_input
    ]

    # -------------------------------------------------
    # Install exact current M0 Anomaly model.
    # -------------------------------------------------

    v4.AUGMENTED_REGION_DIM = (
        future.ORIGINAL_AUGMENTED_DIM
    )

    v4.v3.fit_scaled_dualproto_bags = (
        capture_scaled_dualproto_bags
    )

    v4.RoleGraphConditionedEvidence = (
        future.anomaly_bounded
        .DedupBoundedRGCondAttn
    )

    v4.predict_model = (
        custom_predict
    )

    PREDICT_CALL_INDEX = 0
    LATEST_TRAIN_BAGS = None

    (
        probability,
        diagnostic,
        audits,
    ) = v4.fit_predict_variant(
        SOURCE_GLOBAL,
        target_global,
        data.source_s7,
        data.target_s7,
        data.source_bags,
        data.target_bags,
        data.base_source,
        data.base_target,
        data.source_y,
        "RGCondAttn",
        SEEDS,
        args,
        device,
    )

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    if probability.shape != (
        1989,
    ):
        raise RuntimeError(
            "future probability shape mismatch"
        )

    if PREDICT_CALL_INDEX != 3:
        raise RuntimeError(
            "expected exactly three "
            "seed prediction calls"
        )

    if len(ATTR_ROWS) != (
        1989 * 3
    ):
        raise RuntimeError(
            "attribution row count mismatch: "
            + str(
                len(
                    ATTR_ROWS
                )
            )
        )

    # -------------------------------------------------
    # Locked M0 reproduction gate.
    # -------------------------------------------------

    locked = read_csv(
        LOCKED
    )

    locked_map = {
        norm(
            row["sha256"]
        ):
        float(
            row["p_expert"]
        )
        for row in locked
    }

    if len(locked_map) != 1989:
        raise RuntimeError(
            "locked M0 cohort mismatch"
        )

    expected = np.asarray(
        [
            locked_map[sha]
            for sha
            in TARGET_SHA
        ],
        dtype=np.float64,
    )

    max_diff = float(
        np.max(
            np.abs(
                probability
                - expected
            )
        )
    )

    absolute_difference = np.abs(
        probability - expected
    )

    worst_index = int(
        np.argmax(
            absolute_difference
        )
    )

    decision_disagreement = (
        (probability >= 0.5)
        !=
        (expected >= 0.5)
    )

    print(
        "max |attribution original "
        "- locked M0| =",
        max_diff,
    )

    print(
        "mean probability diff =",
        float(
            np.mean(
                absolute_difference
            )
        ),
    )

    print(
        "median probability diff =",
        float(
            np.median(
                absolute_difference
            )
        ),
    )

    print(
        "n diff > 1e-7 =",
        int(
            np.sum(
                absolute_difference
                > 1e-7
            )
        ),
    )

    print(
        "n diff > 1e-6 =",
        int(
            np.sum(
                absolute_difference
                > 1e-6
            )
        ),
    )

    print(
        "prediction disagreements =",
        int(
            np.sum(
                decision_disagreement
            )
        ),
    )

    print(
        "worst SHA =",
        TARGET_SHA[
            worst_index
        ],
    )

    print(
        "worst year =",
        TARGET_YEAR[
            worst_index
        ],
    )

    print(
        "worst observed =",
        float(
            probability[
                worst_index
            ]
        ),
    )

    print(
        "worst locked =",
        float(
            expected[
                worst_index
            ]
        ),
    )

    if max_diff > 1e-10:
        raise RuntimeError(
            "BLOCKED_BY_REPRODUCTION: "
            "M0 probability changed"
        )

    write_csv(
        args.output
        / "per_seed_block_replacement.csv",
        ATTR_ROWS,
    )

    write_csv(
        args.output
        / "source_fn_centroid_audit.csv",
        CENTROID_ROWS,
    )

    audit = {
        "passed": True,
        "diagnostic_only": True,
        "future_labels_loaded": False,
        "method":
            "M0 Host Anomaly Bounded",
        "seeds": SEEDS,
        "max_delta": 2.25,
        "source_reference":
            "2017 DI25-routed Stage-I FN",
        "source_reference_n": 85,
        "replacement_scope":
            "delta-head fused input only",
        "fused_width": 113,
        "blocks": {
            key: [
                int(left),
                int(right),
            ]
            for key, (
                left,
                right,
            )
            in BLOCKS.items()
        },
        "max_probability_diff_vs_locked":
            max_diff,
        "per_seed_rows":
            len(ATTR_ROWS),
    }

    (
        args.output
        / "attribution_contract.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "M0_BLOCK_REPLACEMENT_PASS"
    )

    print(
        "saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
