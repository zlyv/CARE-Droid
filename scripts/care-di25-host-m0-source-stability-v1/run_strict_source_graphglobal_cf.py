#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT = Path("/home/boyuanjie/projects/tasa-droid")

for path in (
    PROJECT / "scripts/care-di25-host-input-replay-v1",
    PROJECT / "scripts",
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


# ============================================================
# Fixed experiment contract
# ============================================================

if float(MAX_DELTA) != 2.25:
    raise RuntimeError("max_delta contract changed")


METHOD = "DI25__Host__Anomaly__FULL__Bounded"

OUTPUT_ROOT = (
    PROJECT
    / "artifacts"
    / "care-di25-host-m0-source-stability-v1"
    / "outer_counterfactual"
)

FORMAL_OUT = (
    OUTPUT_ROOT
    / "formal_m0"
)

STRICT_REF_ROOT = (
    PROJECT
    / "artifacts"
    / "care-di25-host-m0-source-stability-v1"
    / "strict_reference"
)

LOCKED_PATH = (
    PROJECT
    / "artifacts"
    / "care-di25-host-input-replay-v1"
    / "arms"
    / "FULL"
    / "source_predictions.csv"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

FORMAL_OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# GraphGlobal27 natural groups.
GROUP_SLICES = {
    "ORDER": (0, 5),
    "SCALE": (5, 13),
    "SPECTRAL": (13, 21),
    "TOPOLOGY": (21, 27),
}

ARMS = {
    "ORDER": (
        "ORDER",
    ),

    "SCALE": (
        "SCALE",
    ),

    "SPECTRAL": (
        "SPECTRAL",
    ),

    "TOPOLOGY": (
        "TOPOLOGY",
    ),

    "ORDER_TOPOLOGY": (
        "ORDER",
        "TOPOLOGY",
    ),

    "ORDER_TOPOLOGY_SCALE": (
        "ORDER",
        "TOPOLOGY",
        "SCALE",
    ),

    "ORDER_TOPOLOGY_SPECTRAL": (
        "ORDER",
        "TOPOLOGY",
        "SPECTRAL",
    ),

    "GLOBAL_FULL27": (
        "ORDER",
        "SCALE",
        "SPECTRAL",
        "TOPOLOGY",
    ),
}


EXPECTED_REFERENCE_N = {
    1: 73,
    2: 67,
    3: 58,
    4: 67,
    5: 68,
}


# ============================================================
# Locked source ordering / formal M0
# ============================================================

if not LOCKED_PATH.is_file():
    raise RuntimeError(
        "missing locked M0: "
        + str(LOCKED_PATH)
    )

locked = pd.read_csv(
    LOCKED_PATH
)

if len(locked) != 799:
    raise RuntimeError(
        "locked source n != 799"
    )

locked["sha256"] = (
    locked["sha256"]
    .astype(str)
    .str.strip()
    .str.lower()
)

if locked["sha256"].duplicated().any():
    raise RuntimeError(
        "duplicate locked source SHA"
    )

SHA = locked["sha256"].to_numpy()
FOLD = locked["fold"].to_numpy(
    dtype=np.int64
)
LABEL = locked["label"].to_numpy(
    dtype=np.int64
)

BASE = locked[
    "Global24__prob"
].to_numpy(
    dtype=np.float64
)

ROUTE = locked[
    "A25__routed"
].to_numpy(
    dtype=np.int64
).astype(bool)

LOCKED_PROB = locked[
    METHOD + "__prob"
].to_numpy(
    dtype=np.float64
)

LOCKED_DELTA = locked[
    METHOD + "__bounded_delta"
].to_numpy(
    dtype=np.float64
)


if int(np.sum(ROUTE)) != 200:
    raise RuntimeError(
        "locked DI25 route != 200"
    )

fold_sizes = {
    int(fold):
        int(np.sum(FOLD == fold))
    for fold in sorted(
        np.unique(FOLD)
    )
}

if fold_sizes != {
    1: 160,
    2: 160,
    3: 160,
    4: 160,
    5: 159,
}:
    raise RuntimeError(
        "source folds changed: "
        + repr(fold_sizes)
    )


# ============================================================
# Load strict nested FN references
# ============================================================

REFERENCE_SETS = {}

for fold in range(1, 6):

    path = (
        STRICT_REF_ROOT
        / (
            f"outer_{fold}"
            "_routed_fn_reference.csv"
        )
    )

    if not path.is_file():
        raise RuntimeError(
            "missing strict reference: "
            + str(path)
        )

    d = pd.read_csv(path)

    d["sha256"] = (
        d["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    ids = set(
        d["sha256"].tolist()
    )

    if len(ids) != EXPECTED_REFERENCE_N[fold]:
        raise RuntimeError(
            f"outer={fold} "
            f"reference n={len(ids)} "
            f"expected={EXPECTED_REFERENCE_N[fold]}"
        )

    # Strict contract:
    # no held-out outer fold is allowed.
    if "sample_original_fold" in d.columns:
        if np.any(
            d[
                "sample_original_fold"
            ].astype(int)
            == fold
        ):
            raise RuntimeError(
                f"outer fold {fold} leaked "
                "into strict reference"
            )

    REFERENCE_SETS[fold] = ids


# ============================================================
# Counterfactual capture
# ============================================================

CF_ROWS = []


def infer_outer_fold(seeds):
    if not seeds:
        raise RuntimeError(
            "empty seed list"
        )

    folds = {
        int(seed) // 100000
        for seed in seeds
    }

    if len(folds) != 1:
        raise RuntimeError(
            "cannot infer unique outer fold "
            "from formal seeds: "
            + repr(seeds)
        )

    fold = int(
        next(iter(folds))
    )

    if fold not in {
        1, 2, 3, 4, 5
    }:
        raise RuntimeError(
            "invalid inferred fold: "
            + str(fold)
        )

    return fold


def make_cf_graph(
    original_graph,
    reference_global,
    groups,
):
    result = {
        key: np.array(
            value,
            copy=True,
        )
        for key, value
        in original_graph.items()
    }

    if result[
        "global"
    ].shape[1] != 27:
        raise RuntimeError(
            "GraphGlobal width changed"
        )

    for group in groups:

        left, right = (
            GROUP_SLICES[
                group
            ]
        )

        result[
            "global"
        ][:, left:right] = (
            reference_global[
                left:right
            ][None, :]
        )

    return result


def fit_predict_variant_cf(
    global_train,
    global_test,
    s7_train,
    s7_test,
    raw_train_bags,
    raw_test_bags,
    base_train,
    base_test,
    labels,
    variant,
    seeds,
    args,
    device,
):
    outer_fold = infer_outer_fold(
        seeds
    )

    train_index = np.flatnonzero(
        FOLD != outer_fold
    )

    val_index = np.flatnonzero(
        FOLD == outer_fold
    )

    if len(train_index) != len(
        s7_train
    ):
        raise RuntimeError(
            "outer train alignment failed"
        )

    if len(val_index) != len(
        s7_test
    ):
        raise RuntimeError(
            "outer validation alignment failed"
        )

    if not np.array_equal(
        LABEL[train_index],
        np.asarray(
            labels,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "outer train label alignment failed"
        )

    # base_test must be the exact frozen outer-OOF
    # Global24 probability.
    max_base_diff = float(
        np.max(
            np.abs(
                np.asarray(
                    base_test,
                    dtype=np.float64,
                )
                -
                BASE[val_index]
            )
        )
    )

    if max_base_diff > 1e-12:
        raise RuntimeError(
            "formal outer base alignment failed: "
            + str(max_base_diff)
        )

    train_sha = SHA[
        train_index
    ]

    val_sha = SHA[
        val_index
    ]

    reference_set = (
        REFERENCE_SETS[
            outer_fold
        ]
    )

    reference_mask = np.asarray(
        [
            sha in reference_set
            for sha in train_sha
        ],
        dtype=bool,
    )

    reference_n = int(
        np.sum(reference_mask)
    )

    if reference_n != (
        EXPECTED_REFERENCE_N[
            outer_fold
        ]
    ):
        raise RuntimeError(
            f"outer={outer_fold} "
            f"mapped reference n="
            f"{reference_n} "
            f"expected="
            f"{EXPECTED_REFERENCE_N[outer_fold]}"
        )

    probabilities = []
    diagnostics = []
    audits = []

    print()
    print(
        "=" * 90,
        flush=True,
    )

    print(
        f"SOURCE CF OUTER FOLD "
        f"{outer_fold}: "
        f"train={len(train_index)} "
        f"validation={len(val_index)} "
        f"strict_ref={reference_n}",
        flush=True,
    )

    print(
        "=" * 90,
        flush=True,
    )

    for seed in seeds:

        # --------------------------------------------------
        # EXACT original formal M0 training sequence.
        # --------------------------------------------------

        selected_epochs, selection_loss = (
            v4.select_epochs(
                global_train,
                s7_train,
                raw_train_bags,
                base_train,
                labels,
                variant,
                args,
                device,
                int(seed),
            )
        )

        (
            train_bags,
            test_bags,
            prototype_audit,
        ) = (
            v4.v3.fit_scaled_dualproto_bags(
                raw_train_bags,
                raw_test_bags,
                labels,
                args,
                int(seed) + 1707,
            )
        )

        scaler = (
            v4.fit_role_graph_scaler(
                s7_train
            )
        )

        train_graph = (
            v4.transform_role_graphs(
                s7_train,
                scaler,
            )
        )

        test_graph = (
            v4.transform_role_graphs(
                s7_test,
                scaler,
            )
        )

        # Strict reference is constructed AFTER applying
        # this outer-train-only graph scaler.
        reference_global = np.mean(
            train_graph[
                "global"
            ][reference_mask],
            axis=0,
        ).astype(
            np.float32
        )

        if reference_global.shape != (
            27,
        ):
            raise RuntimeError(
                "reference global width != 27"
            )

        if not np.isfinite(
            reference_global
        ).all():
            raise RuntimeError(
                "non-finite source reference"
            )

        global_tensor = (
            torch.as_tensor(
                global_train,
                dtype=torch.float32,
                device=device,
            )
        )

        train_graph_t = (
            v4.graph_tensors(
                train_graph,
                device,
            )
        )

        train_bag_t = [
            torch.as_tensor(
                value,
                dtype=torch.float32,
                device=device,
            )
            for value
            in train_bags
        ]

        weights, correct = (
            v4.fixkeep_weights(
                labels,
                base_train,
                args,
            )
        )

        v4.seed_everything(
            seed
        )

        model = (
            v4.RoleGraphConditionedEvidence(
                global_train.shape[1],
                variant,
                args,
            ).to(device)
        )

        optimizer = (
            torch.optim.Adam(
                model.parameters(),
                lr=float(
                    args.learning_rate
                ),
                weight_decay=float(
                    args.weight_decay
                ),
            )
        )

        final_loss = (
            v4.train_epochs(
                model,
                optimizer,
                global_tensor,
                base_train,
                train_graph_t,
                train_bag_t,
                labels,
                weights,
                correct,
                selected_epochs,
                args,
                int(seed) + 3301,
            )
        )

        # --------------------------------------------------
        # Original prediction.
        # --------------------------------------------------

        (
            original_probability,
            original_diagnostic,
        ) = (
            v4.predict_model(
                model,
                global_test,
                base_test,
                test_graph,
                test_bags,
                device,
            )
        )

        if (
            original_diagnostic.ndim
            != 2
            or
            original_diagnostic.shape[
                1
            ] < 1
        ):
            raise RuntimeError(
                "diagnostic schema changed"
            )

        arm_result = {}

        # --------------------------------------------------
        # Frozen-model counterfactual prediction.
        # No re-training.
        # --------------------------------------------------

        for arm, groups in (
            ARMS.items()
        ):

            cf_graph = make_cf_graph(
                test_graph,
                reference_global,
                groups,
            )

            (
                cf_probability,
                cf_diagnostic,
            ) = (
                v4.predict_model(
                    model,
                    global_test,
                    base_test,
                    cf_graph,
                    test_bags,
                    device,
                )
            )

            arm_result[arm] = {
                "prob":
                    cf_probability,

                "delta":
                    cf_diagnostic[
                        :,
                        0,
                    ],
            }

        # --------------------------------------------------
        # Save per-seed / per-APK result.
        # --------------------------------------------------

        for local_index, (
            original_index
        ) in enumerate(
            val_index
        ):

            row = {
                "sha256":
                    val_sha[
                        local_index
                    ],

                "outer_fold":
                    int(
                        outer_fold
                    ),

                "seed":
                    int(seed),

                "label":
                    int(
                        LABEL[
                            original_index
                        ]
                    ),

                "route_DI":
                    int(
                        ROUTE[
                            original_index
                        ]
                    ),

                "base_probability":
                    float(
                        base_test[
                            local_index
                        ]
                    ),

                "original_probability":
                    float(
                        original_probability[
                            local_index
                        ]
                    ),

                "original_bounded_delta":
                    float(
                        original_diagnostic[
                            local_index,
                            0,
                        ]
                    ),

                "strict_reference_n":
                    int(
                        reference_n
                    ),
            }

            for arm in ARMS:

                row[
                    f"cf_{arm}_probability"
                ] = float(
                    arm_result[
                        arm
                    ][
                        "prob"
                    ][
                        local_index
                    ]
                )

                row[
                    f"cf_{arm}_bounded_delta"
                ] = float(
                    arm_result[
                        arm
                    ][
                        "delta"
                    ][
                        local_index
                    ]
                )

            CF_ROWS.append(
                row
            )

        probabilities.append(
            original_probability
        )

        diagnostics.append(
            original_diagnostic
        )

        parameter_n = int(
            sum(
                value.numel()
                for value
                in model.parameters()
            )
        )

        audits.append({
            "seed":
                int(seed),

            "variant":
                variant,

            "selected_epochs":
                int(
                    selected_epochs
                ),

            "selection_loss":
                float(
                    selection_loss
                ),

            "final_training_loss":
                float(
                    final_loss
                ),

            "parameter_n":
                parameter_n,

            **prototype_audit,
        })

        print(
            f"outer={outer_fold} "
            f"seed={seed} "
            f"epochs={selected_epochs} "
            f"ref={reference_n} "
            f"done",
            flush=True,
        )

        del (
            model,
            train_graph_t,
            train_bag_t,
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (
        np.mean(
            np.asarray(
                probabilities
            ),
            axis=0,
        ),

        np.mean(
            np.asarray(
                diagnostics
            ),
            axis=0,
        ),

        audits,
    )


# ============================================================
# Run exact formal five-fold M0 with counterfactual capture.
# ============================================================

def main():

    build_di_reference(
        PROJECT,
        OUTPUT_ROOT,
    )

    install_source_label_contract(
        v4
    )

    Model = (
        make_masked_host_class(
            anomaly,
            "FULL",
        )
    )

    v4.MODEL_VARIANTS = (
        "RGCondAttn",
    )

    v4.METHOD_FOR_VARIANT = {
        "RGCondAttn":
            METHOD
    }

    v4.RoleGraphConditionedEvidence = (
        Model
    )

    # Replace ONLY the formal fitting wrapper.
    # Training sequence remains identical to v4.
    v4.fit_predict_variant = (
        fit_predict_variant_cf
    )

    sys.argv = [
        sys.argv[0],

        "--previous-result",
        str(
            OUTPUT_ROOT
            / "preflight"
            / "di25_reference"
        ),

        "--cache",
        "artifacts/care-source-local-assets-v2/"
        "endpoint_centered_tensor_cache_2017.jsonl.gz",

        "--region-cache",
        "artifacts/care-source-local-assets-v2/"
        "regions_2017.npz",

        "--variants",
        "RGCondAttn",

        "--seeds",
        "0,1,2",

        "--max-delta",
        "2.25",

        "--device",
        "auto",

        "--output",
        str(
            FORMAL_OUT
        ),
    ]

    v4.main()

    # --------------------------------------------------------
    # Save counterfactual table.
    # --------------------------------------------------------

    cf = pd.DataFrame(
        CF_ROWS
    )

    expected_rows = (
        799 * 3
    )

    if len(cf) != expected_rows:
        raise RuntimeError(
            f"counterfactual rows="
            f"{len(cf)} "
            f"expected={expected_rows}"
        )

    if cf[
        [
            "sha256",
            "seed",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "duplicate SHA/seed "
            "counterfactual row"
        )

    cf_path = (
        OUTPUT_ROOT
        / "per_seed_counterfactual.csv"
    )

    cf.to_csv(
        cf_path,
        index=False,
    )

    # --------------------------------------------------------
    # Formal M0 reproduction gate.
    # --------------------------------------------------------

    generated_path = (
        FORMAL_OUT
        / "source_predictions.csv"
    )

    generated = pd.read_csv(
        generated_path
    )

    generated["sha256"] = (
        generated["sha256"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    g = generated.set_index(
        "sha256"
    )

    l = locked.set_index(
        "sha256"
    )

    g = g.loc[
        l.index
    ]

    max_probability_difference = (
        float(
            np.max(
                np.abs(
                    g[
                        METHOD
                        + "__prob"
                    ].to_numpy(float)
                    -
                    l[
                        METHOD
                        + "__prob"
                    ].to_numpy(float)
                )
            )
        )
    )

    max_delta_difference = (
        float(
            np.max(
                np.abs(
                    g[
                        METHOD
                        + "__bounded_delta"
                    ].to_numpy(float)
                    -
                    l[
                        METHOD
                        + "__bounded_delta"
                    ].to_numpy(float)
                )
            )
        )
    )

    prediction_disagreements = int(
        np.sum(
            (
                g[
                    METHOD
                    + "__prob"
                ].to_numpy(float)
                >= 0.5
            )
            !=
            (
                l[
                    METHOD
                    + "__prob"
                ].to_numpy(float)
                >= 0.5
            )
        )
    )

    route_disagreements = int(
        np.sum(
            g[
                "A25__routed"
            ].to_numpy(int)
            !=
            l[
                "A25__routed"
            ].to_numpy(int)
        )
    )

    reproduction = {
        "max_probability_difference":
            max_probability_difference,

        "max_delta_difference":
            max_delta_difference,

        "prediction_disagreements":
            prediction_disagreements,

        "route_disagreements":
            route_disagreements,

        "counterfactual_rows":
            int(len(cf)),

        "future_rows_loaded":
            False,

        "strict_reference_counts":
            EXPECTED_REFERENCE_N,
    }

    (
        OUTPUT_ROOT
        / "reproduction_gate.json"
    ).write_text(
        json.dumps(
            reproduction,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 90
    )
    print(
        "FORMAL M0 REPRODUCTION GATE"
    )
    print(
        "=" * 90
    )

    print(
        "max probability difference =",
        max_probability_difference,
    )

    print(
        "max delta difference =",
        max_delta_difference,
    )

    print(
        "prediction disagreements =",
        prediction_disagreements,
    )

    print(
        "route disagreements =",
        route_disagreements,
    )

    print(
        "counterfactual rows =",
        len(cf),
    )

    # Exact prediction identity is mandatory.
    # Allow only tiny floating-point noise.
    if (
        prediction_disagreements != 0
        or
        route_disagreements != 0
        or
        max_probability_difference
        > 5e-6
        or
        max_delta_difference
        > 5e-6
    ):
        raise RuntimeError(
            "SOURCE_CF_REPRODUCTION_FAILED"
        )

    contract = {
        "experiment":
            "2017 strict OOF GraphGlobal "
            "counterfactual sensitivity audit",

        "source_only":
            True,

        "future_rows_loaded":
            False,

        "formal_model":
            "DI25 Host Anomaly FULL Bounded",

        "outer_folds":
            [1, 2, 3, 4, 5],

        "seeds":
            [0, 1, 2],

        "strict_reference_counts":
            EXPECTED_REFERENCE_N,

        "reference_semantics":
            "strict nested outer-train "
            "DI25-routed Stage-I FN centroid",

        "reference_space":
            "outer-train-standardized "
            "GraphGlobal27",

        "counterfactual_only":
            True,

        "model_retrained_per_arm":
            False,

        "groups":
            {
                key: list(value)
                for key, value
                in ARMS.items()
            },

        "interpretation":
            "counterfactual sensitivity, "
            "not causal identification",
    }

    (
        OUTPUT_ROOT
        / "counterfactual_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "SOURCE_CF_REPRODUCTION_PASS"
    )

    print(
        "saved:",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
