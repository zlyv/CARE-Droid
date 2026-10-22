#!/usr/bin/env python3
"""Fast contract tests; no project artifact is opened or modified."""
from __future__ import annotations

import ast
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    required = {
        "preflight_future.py": (
            "EXPECTED_ROUTE", "lock_source", "DI25_HOSTEGO_FUTURE_PREFLIGHT_PASS",
        ),
        "run_m1_future.py": (
            "source_only_training", "M0_retrained", "34613", "load_packed",
        ),
        "summarize_future.py": (
            "future_label_file_posthoc_only", "M0_wrong_M1_correct", "exact_p",
        ),
        "host_ego_encoder.py": ("DirectedHostEgoFusion", "make_fit_adapter"),
        "host_ego_models.py": ("HostEgoAnomaly", "direct_probability_from_delta"),
    }
    for name, tokens in required.items():
        path = HERE / name
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        missing = [token for token in tokens if token not in source]
        if missing:
            raise RuntimeError(f"{name}: missing contract tokens {missing}")

    # Exact two-sided binomial McNemar values used by the final summarizer.
    namespace: dict[str, object] = {"math": math}
    tree = ast.parse((HERE / "summarize_future.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "exact_p")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<exact_p>", "exec"), namespace)
    exact_p = namespace["exact_p"]
    assert exact_p(0, 0) == 1.0
    assert exact_p(2, 5) == 0.453125
    assert exact_p(5, 2) == 0.453125

    run_source = (HERE / "run_m1_future.py").read_text(encoding="utf-8")
    forbidden = (
        "future_label_file_posthoc_only]).open",
        "future_label_file_posthoc_only\").open",
        "parse_label(",
    )
    if any(token in run_source for token in forbidden):
        raise RuntimeError("model runner contains a future-label access path")
    print("DI25_HOSTEGO_FUTURE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
