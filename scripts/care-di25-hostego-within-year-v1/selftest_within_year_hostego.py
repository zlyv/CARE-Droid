#!/usr/bin/env python3
"""Fast static contract tests; no experiment artifact is opened."""
from __future__ import annotations

import ast
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    required = {
        "preflight_within_year.py": (
            "strict independent within-year five-fold OOF", "M0_reused_not_retrained", "EXPECTED_ROUTES",
        ),
        "run_m1_within_year.py": (
            "outer_validation_labels_used_for_training_or_epoch_selection", "34613", "inner_oof_base", "load_packed",
        ),
        "summarize_within_year.py": (
            "Do not describe these same-year OOF results", "M0_wrong_M1_correct", "exact_p",
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

    namespace: dict[str, object] = {"math": math}
    tree = ast.parse((HERE / "summarize_within_year.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "exact_p")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<exact_p>", "exec"), namespace)
    exact_p = namespace["exact_p"]
    assert exact_p(0, 0) == 1.0
    assert exact_p(2, 5) == 0.453125
    assert exact_p(5, 2) == 0.453125

    run_source = (HERE / "run_m1_within_year.py").read_text(encoding="utf-8")
    for required_text in (
        "data.years == year", "data.folds != fold", "data.folds == fold",
        "within.inner_oof_base", "make_fit_adapter", "2.25",
    ):
        if required_text not in run_source:
            raise RuntimeError("missing strict OOF token: " + required_text)
    print("DI25_HOSTEGO_WITHIN_YEAR_SELFTEST_PASS")


if __name__ == "__main__":
    main()
