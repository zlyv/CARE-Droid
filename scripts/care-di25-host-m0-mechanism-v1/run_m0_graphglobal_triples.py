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


TRIPLE_GROUPS = {
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


gg.up.ARMS = tuple(TRIPLE_GROUPS.keys())


def triple_apply_replacement(
    one_graph,
    arm,
    reference,
):
    if arm not in TRIPLE_GROUPS:
        raise RuntimeError(
            "unknown arm: " + str(arm)
        )

    cf = {
        key: value.clone()
        for key, value in one_graph.items()
    }

    new_global = cf["global"].clone()

    if int(new_global.numel()) != 27:
        raise RuntimeError(
            "expected graph_global dim=27"
        )

    if int(reference["global"].numel()) != 27:
        raise RuntimeError(
            "expected reference graph_global dim=27"
        )

    for group in TRIPLE_GROUPS[arm]:
        left, right = SLICES[group]

        new_global[left:right] = (
            reference["global"][left:right]
        )

    cf["global"] = new_global

    return cf


gg.up.apply_replacement = triple_apply_replacement


def output_path():
    if "--output" not in sys.argv:
        raise RuntimeError("--output is required")

    idx = sys.argv.index("--output")
    return Path(sys.argv[idx + 1])


def main():
    out = output_path()

    gg.up.main()

    contract = {
        "diagnostic_only": True,
        "experiment":
            "GraphGlobal targeted triple-group counterfactual",

        "groups": {
            key: list(value)
            for key, value in TRIPLE_GROUPS.items()
        },

        "source_reference":
            "2017 DI25-routed Stage-I FN centroid",

        "replacement_location":
            "standardized GraphGlobal27 before global_encoder",

        "motivation":
            "Pairwise diagnosis identified ORDER+TOPOLOGY as the strongest interaction; add SCALE or SPECTRAL separately.",

        "positive_control":
            "GLOBAL_FULL27",

        "note":
            "Future labels are used only for post-hoc diagnostic grouping, not for training or model selection.",
    }

    (
        out
        / "graphglobal_triple_contract.json"
    ).write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("M0_GRAPHGLOBAL_TRIPLE_PASS")


if __name__ == "__main__":
    main()
