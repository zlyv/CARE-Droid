#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT = Path("/home/boyuanjie/projects/tasa-droid")

BASE_PATH = (
    PROJECT
    / "scripts"
    / "care-di25-host-m0-mechanism-v1"
    / "run_m0_graphglobal_groups.py"
)

spec = importlib.util.spec_from_file_location(
    "graphglobal_groups",
    BASE_PATH,
)

gg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gg
spec.loader.exec_module(gg)


SLICES = {
    "ORDER": (0, 5),
    "SCALE": (5, 13),
    "SPECTRAL": (13, 21),
    "TOPOLOGY": (21, 27),
}


PAIR_GROUPS = {
    "ORDER_SCALE":
        ("ORDER", "SCALE"),

    "ORDER_SPECTRAL":
        ("ORDER", "SPECTRAL"),

    "ORDER_TOPOLOGY":
        ("ORDER", "TOPOLOGY"),

    "SCALE_SPECTRAL":
        ("SCALE", "SPECTRAL"),

    "SCALE_TOPOLOGY":
        ("SCALE", "TOPOLOGY"),

    "SPECTRAL_TOPOLOGY":
        ("SPECTRAL", "TOPOLOGY"),

    "GLOBAL_FULL27":
        ("ORDER", "SCALE", "SPECTRAL", "TOPOLOGY"),
}


gg.up.ARMS = tuple(
    PAIR_GROUPS.keys()
)


def pair_apply_replacement(
    one_graph,
    arm,
    reference,
):

    if arm not in PAIR_GROUPS:
        raise RuntimeError(
            "unknown arm: " + str(arm)
        )

    cf = {
        key: value.clone()
        for key, value
        in one_graph.items()
    }

    new_global = (
        cf["global"].clone()
    )

    if int(new_global.numel()) != 27:
        raise RuntimeError(
            "expected graph_global=27"
        )

    for group in PAIR_GROUPS[arm]:

        left, right = SLICES[group]

        new_global[left:right] = (
            reference["global"][left:right]
        )

    cf["global"] = new_global

    return cf


gg.up.apply_replacement = (
    pair_apply_replacement
)


def get_output():
    index = sys.argv.index(
        "--output"
    )
    return Path(
        sys.argv[index + 1]
    )


def main():

    out = get_output()

    gg.up.main()

    contract = {
        "diagnostic_only": True,
        "experiment":
            "GraphGlobal pairwise counterfactual replacement",

        "groups":
            PAIR_GROUPS,

        "source_reference":
            "2017 DI25-routed Stage-I FN centroid",

        "replacement_location":
            "standardized GraphGlobal27 before global_encoder",

        "positive_control":
            "GLOBAL_FULL27",

        "note":
            "Pair effects are counterfactual sensitivity, not causal effects.",
    }

    (
        out
        / "graphglobal_pair_contract.json"
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
        "M0_GRAPHGLOBAL_PAIR_PASS"
    )


if __name__ == "__main__":
    main()
