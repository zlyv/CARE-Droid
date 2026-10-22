#!/usr/bin/env python3
"""
Read-only instrumentation of the frozen Host-only M0 future replay.

IMPORTANT
---------
This script does NOT modify the M0 forward equation.
It only installs forward hooks on delta_head in order to capture
the exact 113-D correction input and raw delta.

No future labels are loaded here.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

FUTURE_DIR = (
    PROJECT
    / "scripts"
    / "care-route-stage2-future-matrix-v1"
)

if str(FUTURE_DIR) not in sys.path:
    sys.path.insert(0, str(FUTURE_DIR))

import run_future_arm as future


SEEDS = [900000, 900001, 900002]
MAX_DELTA = 2.25
FUSED_DIM = 113

LOCKED = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "anomaly_bounded/future_expert_predictions.csv"
)

ROUTER = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "preflight/future_router_inputs_label_free.csv"
)

DIAGNOSTIC_NAMES = (
    "bounded_delta",
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

REAL_PREDICT = future.ORIGINAL_V4_PREDICT

# One entry per seed.
SEED_CAPTURES = []


def norm(value):
    return str(value).strip().lower()


def read_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError(
            "refusing to write empty CSV: "
            + str(path)
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def capture_predict(
    model,
    global_values,
    base_probability,
    graph_values,
    bags,
    device,
):
    """
    Wrap the ORIGINAL predict_model.

    The hooks are observational only:
      - pre-hook records the exact input to delta_head
      - forward-hook records delta_head output

    Neither hook returns a replacement value.
    """

    fused_rows = []
    raw_rows = []

    def capture_input(module, inputs):
        value = (
            inputs[0]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32, copy=True)
        )

        if value.shape != (FUSED_DIM,):
            raise RuntimeError(
                "correction-input width changed: "
                f"{value.shape}"
            )

        fused_rows.append(value)

    def capture_output(module, inputs, output):
        value = float(
            output
            .detach()
            .cpu()
            .reshape(-1)[0]
        )

        raw_rows.append(value)

    input_hook = (
        model.delta_head
        .register_forward_pre_hook(
            capture_input
        )
    )

    output_hook = (
        model.delta_head
        .register_forward_hook(
            capture_output
        )
    )

    try:
        probability, diagnostic = REAL_PREDICT(
            model,
            global_values,
            base_probability,
            graph_values,
            bags,
            device,
        )
    finally:
        input_hook.remove()
        output_hook.remove()

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    diagnostic = np.asarray(
        diagnostic,
        dtype=np.float64,
    )

    fused = np.asarray(
        fused_rows,
        dtype=np.float32,
    )

    raw = np.asarray(
        raw_rows,
        dtype=np.float64,
    )

    n = len(bags)

    if probability.shape != (n,):
        raise RuntimeError(
            "probability shape changed: "
            + str(probability.shape)
        )

    if fused.shape != (n, FUSED_DIM):
        raise RuntimeError(
            "fused capture shape changed: "
            + str(fused.shape)
        )

    if raw.shape != (n,):
        raise RuntimeError(
            "raw delta capture shape changed: "
            + str(raw.shape)
        )

    if diagnostic.shape != (
        n,
        len(DIAGNOSTIC_NAMES),
    ):
        raise RuntimeError(
            "diagnostic shape changed: "
            + str(diagnostic.shape)
        )

    parameter_n = int(
        sum(
            value.numel()
            for value in model.parameters()
        )
    )

    SEED_CAPTURES.append({
        "probability": probability.copy(),
        "fused": fused.copy(),
        "raw_delta": raw.copy(),
        "diagnostic": diagnostic.copy(),
        "parameter_n": parameter_n,
    })

    return probability, diagnostic


def main():
    args = future.parser().parse_args()

    if args.representation != "anomaly":
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            "representation must be anomaly"
        )

    if args.head != "bounded":
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            "head must be bounded"
        )

    seeds = future.v4.parse_ints(
        args.seeds
    )

    if seeds != SEEDS:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            f"seeds changed: {seeds}"
        )

    if float(args.max_delta) != MAX_DELTA:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: "
            f"max_delta changed: {args.max_delta}"
        )

    if not LOCKED.is_file():
        raise RuntimeError(
            "missing locked M0 replay: "
            + str(LOCKED)
        )

    if not ROUTER.is_file():
        raise RuntimeError(
            "missing frozen DI25 router: "
            + str(ROUTER)
        )

    # Crucial patch:
    #
    # train_bounded() restores:
    #   v4.predict_model = ORIGINAL_V4_PREDICT
    #
    # Therefore replace that variable with our
    # observational wrapper. The real predictor
    # itself remains REAL_PREDICT.
    future.ORIGINAL_V4_PREDICT = (
        capture_predict
    )

    # Run the original producer unchanged.
    future.main()

    if len(SEED_CAPTURES) != 3:
        raise RuntimeError(
            "BLOCKED_BY_REPRODUCTION: "
            "expected exactly 3 prediction calls, "
            f"got {len(SEED_CAPTURES)}"
        )

    parameter_counts = [
        int(value["parameter_n"])
        for value in SEED_CAPTURES
    ]

    if len(set(parameter_counts)) != 1:
        raise RuntimeError(
            "parameter count differs by seed: "
            + repr(parameter_counts)
        )

    generated_path = (
        args.output
        / "future_expert_predictions.csv"
    )

    generated = read_csv(
        generated_path
    )

    locked = {
        norm(row["sha256"]): row
        for row in read_csv(LOCKED)
    }

    router = {
        norm(row["sha256"]): row
        for row in read_csv(ROUTER)
    }

    if (
        len(generated) != 1989
        or len(locked) != 1989
        or len(router) != 1989
    ):
        raise RuntimeError(
            "future cohort size changed"
        )

    generated_sha = [
        norm(row["sha256"])
        for row in generated
    ]

    if (
        len(set(generated_sha)) != 1989
        or set(generated_sha) != set(locked)
        or set(generated_sha) != set(router)
    ):
        raise RuntimeError(
            "future SHA alignment changed"
        )

    generated_probability = np.asarray(
        [
            float(row["p_expert"])
            for row in generated
        ],
        dtype=np.float64,
    )

    locked_probability = np.asarray(
        [
            float(
                locked[sha]["p_expert"]
            )
            for sha in generated_sha
        ],
        dtype=np.float64,
    )

    seed_probability = np.stack(
        [
            value["probability"]
            for value in SEED_CAPTURES
        ],
        axis=0,
    )

    seed_fused = np.stack(
        [
            value["fused"]
            for value in SEED_CAPTURES
        ],
        axis=0,
    )

    seed_raw = np.stack(
        [
            value["raw_delta"]
            for value in SEED_CAPTURES
        ],
        axis=0,
    )

    seed_diagnostic = np.stack(
        [
            value["diagnostic"]
            for value in SEED_CAPTURES
        ],
        axis=0,
    )

    if seed_probability.shape != (
        3,
        1989,
    ):
        raise RuntimeError(
            "seed probability shape changed"
        )

    if seed_fused.shape != (
        3,
        1989,
        113,
    ):
        raise RuntimeError(
            "seed fused shape changed"
        )

    if seed_diagnostic.shape != (
        3,
        1989,
        11,
    ):
        raise RuntimeError(
            "seed diagnostic shape changed"
        )

    ensemble_from_seed = np.mean(
        seed_probability,
        axis=0,
    )

    locked_diff = float(
        np.max(
            np.abs(
                generated_probability
                - locked_probability
            )
        )
    )

    ensemble_diff = float(
        np.max(
            np.abs(
                generated_probability
                - ensemble_from_seed
            )
        )
    )

    bounded_from_diag = (
        seed_diagnostic[:, :, 0]
    )

    tanh_diff = float(
        np.max(
            np.abs(
                bounded_from_diag
                - MAX_DELTA
                * np.tanh(seed_raw)
            )
        )
    )

    print()
    print("=" * 90)
    print("INSTRUMENTED M0 EQUIVALENCE")
    print("=" * 90)

    print(
        "max |instrumented - locked| =",
        locked_diff,
    )

    print(
        "max |ensemble(seed probs) "
        "- instrumented| =",
        ensemble_diff,
    )

    print(
        "max |bounded - "
        "2.25*tanh(raw)| =",
        tanh_diff,
    )

    print(
        "parameter counts =",
        parameter_counts,
    )

    # Hard lineage guard.
    if locked_diff > 1e-10:
        raise RuntimeError(
            "BLOCKED_BY_REPRODUCTION: "
            "instrumented replay does not "
            "reproduce locked M0 probabilities"
        )

    if ensemble_diff > 1e-10:
        raise RuntimeError(
            "BLOCKED_BY_INTERNAL_CONSISTENCY: "
            "three-seed ensemble mismatch"
        )

    if tanh_diff > 1e-5:
        raise RuntimeError(
            "BLOCKED_BY_INTERNAL_CONSISTENCY: "
            "raw/bounded delta mismatch"
        )

    route = np.asarray(
        [
            int(
                float(
                    router[sha]["route_DI"]
                )
            )
            == 1
            for sha in generated_sha
        ],
        dtype=bool,
    )

    year = np.asarray(
        [
            int(row["year"])
            for row in generated
        ],
        dtype=np.int64,
    )

    routed_index = np.flatnonzero(
        route
    )

    if len(routed_index) != 497:
        raise RuntimeError(
            "DI25 routed cohort changed: "
            + str(len(routed_index))
        )

    np.savez_compressed(
        args.output
        / "routed_internal_capture.npz",

        sha256=np.asarray(
            [
                generated_sha[i]
                for i in routed_index
            ],
            dtype="U64",
        ),

        year=year[routed_index],

        fused=seed_fused[
            :,
            routed_index,
            :,
        ],

        raw_delta=seed_raw[
            :,
            routed_index,
        ],

        diagnostic=seed_diagnostic[
            :,
            routed_index,
            :,
        ],

        seed_probability=seed_probability[
            :,
            routed_index,
        ],

        diagnostic_names=np.asarray(
            DIAGNOSTIC_NAMES,
            dtype="U40",
        ),
    )

    rows = []

    for i, sha in enumerate(
        generated_sha
    ):
        rows.append({
            "sha256": sha,
            "year": int(year[i]),
            "route_DI": int(route[i]),
            "locked_p_expert":
                float(locked_probability[i]),
            "instrumented_p_expert":
                float(
                    generated_probability[i]
                ),
            "seed_probability_mean":
                float(
                    np.mean(
                        seed_probability[:, i]
                    )
                ),
            "seed_probability_sd":
                float(
                    np.std(
                        seed_probability[:, i]
                    )
                ),
        })

    write_csv(
        args.output
        / "instrumented_predictions.csv",
        rows,
    )

    equivalence = {
        "passed": True,
        "arithmetic_changed": False,
        "model_class_changed": False,
        "forward_rewritten": False,
        "instrumentation":
            "delta_head forward hooks only",
        "representation": "anomaly",
        "head": "bounded",
        "local_input": "Host-only",
        "seeds": SEEDS,
        "max_delta": MAX_DELTA,
        "future_n": 1989,
        "routed_n": 497,
        "fused_dim": 113,
        "parameter_counts":
            parameter_counts,
        "max_abs_probability_diff_vs_locked":
            locked_diff,
        "max_abs_probability_diff_vs_seed_mean":
            ensemble_diff,
        "max_abs_bounded_tanh_identity_diff":
            tanh_diff,
        "future_labels_loaded":
            False,
    }

    (
        args.output
        / "instrumentation_equivalence.json"
    ).write_text(
        json.dumps(
            equivalence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "routed internal rows =",
        len(routed_index),
    )

    print(
        "capture saved:",
        args.output
        / "routed_internal_capture.npz",
    )

    print()
    print(
        "INSTRUMENTED_M0_REPLAY_PASS"
    )


if __name__ == "__main__":
    main()
