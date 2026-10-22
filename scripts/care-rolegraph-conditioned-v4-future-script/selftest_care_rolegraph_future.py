#!/usr/bin/env python3
"""Dependency-free structural self-test for the CARE v4 future runner."""

from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("run_care_rolegraph_conditioned_v4_future.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_functions = {
        "strict_reference_probability",
        "direct_comparison_row",
        "validate_args",
        "main",
    }
    missing = required_functions - functions
    if missing:
        raise AssertionError("missing functions: " + repr(sorted(missing)))
    required_text = [
        'FROZEN_VARIANTS = ("RGConcat", "RGCondAttn")',
        "FROZEN_SEEDS = (0, 1, 2)",
        "load_or_build_future_inter",
        "source_s7_from_reference",
        "future labels and reference CARE probabilities are used only post hoc",
        "future_seed_summary_diagnostic.csv",
        "future_pairwise_diagnostic.csv",
    ]
    for text in required_text:
        if text.casefold() not in source.casefold():
            raise AssertionError("required frozen-protocol marker missing: " + text)
    print("SELFTEST_OK")
    print("runner:", path.name)
    print("frozen_variants: RGConcat,RGCondAttn")
    print("per_seed_export: 0,1,2")
    print("future_labels: post-hoc only")


if __name__ == "__main__":
    main()
