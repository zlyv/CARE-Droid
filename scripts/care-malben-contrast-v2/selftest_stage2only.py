#!/usr/bin/env python3
"""Static contract checks for the Stage-II-only experiment."""

from pathlib import Path


def main() -> None:
    text = (Path(__file__).parent / "run_stage2only_malben_source_v1.py").read_text(encoding="utf-8")
    required = (
        "StageIIOnly-MalBen", "StandardScaler", "LogisticRegression",
        "stage2_invocation_rate", "missing_region_count",
        "np.full(len(labels), 0.5", "np.zeros((len(labels), 1)",
    )
    for token in required:
        if token not in text:
            raise AssertionError("missing Stage-II-only contract: " + token)
    model = (Path(__file__).parent / "run_care_malben_role_source_v2.py").read_text(encoding="utf-8")
    feature = model.split('"stage2_feature"', 1)[1].split("}", 1)[0]
    for forbidden in ("global_features", "base_logit", "bounded_delta"):
        if forbidden in feature:
            raise AssertionError("forbidden Stage-II-only input: " + forbidden)
    print("Stage-II-only contract self-test passed")


if __name__ == "__main__":
    main()
