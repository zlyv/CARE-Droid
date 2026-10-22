#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT = Path("/home/boyuanjie/projects/tasa-droid")

UPSTREAM_PATH = (
    PROJECT
    / "scripts"
    / "care-di25-host-m0-mechanism-v1"
    / "run_m0_rolegraph_upstream.py"
)

spec = importlib.util.spec_from_file_location(
    "m0_rolegraph_upstream",
    UPSTREAM_PATH,
)

up = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = up
spec.loader.exec_module(up)


# ---------------------------------------------------------
# Natural groups of the 27-D graph_global vector.
#
# 0:5   -> hyperedge order histogram
# 5:13  -> scale/diversity/provenance
# 13:21 -> spectral structure
# 21:27 -> graph topology
# ---------------------------------------------------------

up.ARMS = (
    "ORDER5",
    "SCALE_DIVERSITY8",
    "SPECTRAL8",
    "TOPOLOGY6",
    "GLOBAL_FULL27",
)


GROUPS = {
    "ORDER5": (0, 5),
    "SCALE_DIVERSITY8": (5, 13),
    "SPECTRAL8": (13, 21),
    "TOPOLOGY6": (21, 27),
    "GLOBAL_FULL27": (0, 27),
}


def grouped_apply_replacement(
    one_graph,
    arm,
    reference,
):
    if arm not in GROUPS:
        raise RuntimeError(
            "unknown graph-global arm: "
            + str(arm)
        )

    cf = {
        key: value.clone()
        for key, value
        in one_graph.items()
    }

    if int(cf["global"].numel()) != 27:
        raise RuntimeError(
            "expected graph_global dim=27, got "
            + str(
                tuple(
                    cf["global"].shape
                )
            )
        )

    if int(reference["global"].numel()) != 27:
        raise RuntimeError(
            "bad source graph_global reference"
        )

    left, right = GROUPS[arm]

    new_global = (
        cf["global"].clone()
    )

    new_global[left:right] = (
        reference["global"][left:right]
    )

    cf["global"] = new_global

    return cf


# Replace ONLY the counterfactual editing rule.
# Training, frozen weights, preprocessing, source reference,
# Host bags, graph encoder and correction head stay unchanged.
up.apply_replacement = (
    grouped_apply_replacement
)


def output_path_from_argv():
    if "--output" not in sys.argv:
        raise RuntimeError(
            "--output is required"
        )

    index = sys.argv.index(
        "--output"
    )

    return Path(
        sys.argv[index + 1]
    )


def main():
    out = output_path_from_argv()

    up.main()

    # up.main() writes its generic upstream contract.
    # Write an additional exact contract for this experiment.
    contract = {
        "diagnostic_only": True,
        "future_labels_used_for_replacement": False,

        "source_reference":
            "2017 DI25-routed Stage-I false-negative centroid",

        "replacement_location":
            "standardized 27-D graph_global input before global_encoder",

        "groups": {
            "ORDER5":
                [0, 1, 2, 3, 4],

            "SCALE_DIVERSITY8":
                [5, 6, 7, 8, 9, 10, 11, 12],

            "SPECTRAL8":
                [13, 14, 15, 16, 17, 18, 19, 20],

            "TOPOLOGY6":
                [21, 22, 23, 24, 25, 26],

            "GLOBAL_FULL27":
                list(range(27)),
        },

        "group_meanings": {
            "ORDER5":
                "2/3/4/5/>=6 role hyperedge-order counts",

            "SCALE_DIVERSITY8":
                "hyperedge scale, pattern diversity, mean/max order, pattern entropy, provenance proportions",

            "SPECTRAL8":
                "top eight absolute eigenvalue features of normalized role co-occurrence graph",

            "TOPOLOGY6":
                "density, connected components, spectral entropy, mean/std/max degree",

            "GLOBAL_FULL27":
                "all graph-global features",
        },

        "positive_control":
            "GLOBAL_FULL27 must reproduce previous GRAPH_GLOBAL counterfactual",

        "model":
            "Frozen M0 Anomaly + bounded correction",

        "max_delta":
            2.25,
    }

    (
        out
        / "graphglobal_group_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "M0_GRAPHGLOBAL_GROUP_PASS"
    )


if __name__ == "__main__":
    main()
