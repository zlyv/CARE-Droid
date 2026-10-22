#!/usr/bin/env python3
"""Post-hoc source error analysis for A25 and class-asymmetric A25.

This script does not train or select a model.  It reconstructs the frozen
Global24, A25+deduplicated-RGCondAttn and AsymA25+deduplicated-RGCondAttn
decisions, identifies error mechanisms, and describes their static-feature
profiles.  Results are descriptive rather than causal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_FEATURES = (
    "manifest_permission_count",
    "manifest_activity_count",
    "manifest_service_count",
    "manifest_receiver_count",
    "manifest_provider_count",
    "manifest_component_count",
    "min_sdk",
    "target_sdk",
    "method_count",
    "graph_node_count",
    "graph_edge_count",
    "mapped_api_node_count",
    "sensitive_node_count",
    "sensitive_edge_count",
    "manifest_context_match_count",
    "internal_node_count",
    "internal_edge_count",
    "external_node_count",
    "external_edge_count",
    "unmapped_external_node_count",
    "mapped_external_ratio",
    "context_role__boot_persistence",
)


def find_sha_column(frame: pd.DataFrame) -> str:
    candidates = ("sha256", "sample_id", "sha", "hash")
    by_lower = {str(column).strip().casefold(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in by_lower:
            return str(by_lower[candidate])
    raise RuntimeError("could not find a SHA/sample identifier column")


def normalize_sha(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold()


def binary(probability: pd.Series) -> np.ndarray:
    return (probability.to_numpy(dtype=np.float64) >= 0.5).astype(np.int64)


def transition(base_error: np.ndarray, final_error: np.ndarray) -> np.ndarray:
    return np.select(
        [base_error & ~final_error, ~base_error & final_error, base_error & final_error],
        ["fixed", "broken", "unchanged_wrong"],
        default="unchanged_correct",
    )


def outcome(labels: pd.Series, prediction: np.ndarray) -> np.ndarray:
    labels_array = labels.to_numpy(dtype=np.int64)
    prediction_array = np.asarray(prediction, dtype=np.int64)
    return np.select(
        [
            (labels_array == 1) & (prediction_array == 1),
            (labels_array == 0) & (prediction_array == 0),
            (labels_array == 0) & (prediction_array == 1),
        ],
        ["TP", "TN", "FP"],
        default="FN",
    )


def grouped_error_rows(frame: pd.DataFrame, group_columns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update({
            "n": int(len(group)),
            "base_error_n": int(group["base_error"].sum()),
            "base_error_rate": float(group["base_error"].mean()),
            "a25_error_n": int(group["a25_error"].sum()),
            "a25_error_rate": float(group["a25_error"].mean()),
            "asym_error_n": int(group["asym_error"].sum()),
            "asym_error_rate": float(group["asym_error"].mean()),
            "a25_route_ratio": float(group["A25__routed"].mean()),
            "asym_route_ratio": float(group["AsymA25__routed"].mean()),
            "mean_base_probability": float(group["Global24__prob"].mean()),
            "mean_base_confidence": float(group["base_confidence"].mean()),
        })
        rows.append(row)
    return rows


def numeric_feature_frame(features: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    result = {}
    for column in features.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        if float(values.notna().mean()) < 0.95 or values.nunique(dropna=True) < 2:
            continue
        result[str(column)] = values.astype(float)
    return pd.DataFrame(result, index=features.index)


def effect_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    error_column: str,
    method: str,
) -> list[dict]:
    rows: list[dict] = []
    scopes = [("all", frame)]
    scopes.extend((f"label_{label}", frame[frame["label"] == label]) for label in (0, 1))
    for scope, subset in scopes:
        wrong = subset[subset[error_column]]
        correct = subset[~subset[error_column]]
        if len(wrong) < 5 or len(correct) < 5:
            continue
        for feature in feature_columns:
            wrong_value = wrong[feature].dropna().to_numpy(dtype=float)
            correct_value = correct[feature].dropna().to_numpy(dtype=float)
            if len(wrong_value) < 5 or len(correct_value) < 5:
                continue
            combined = np.concatenate([wrong_value, correct_value])
            scale = float(np.std(combined, ddof=0))
            if not np.isfinite(scale) or scale <= 1e-12:
                continue
            mean_wrong = float(np.mean(wrong_value))
            mean_correct = float(np.mean(correct_value))
            standardized_difference = (mean_wrong - mean_correct) / scale
            rows.append({
                "method": method,
                "scope": scope,
                "feature": feature,
                "wrong_n": int(len(wrong_value)),
                "correct_n": int(len(correct_value)),
                "wrong_mean": mean_wrong,
                "correct_mean": mean_correct,
                "wrong_median": float(np.median(wrong_value)),
                "correct_median": float(np.median(correct_value)),
                "standardized_mean_difference": float(standardized_difference),
                "absolute_standardized_difference": float(abs(standardized_difference)),
            })
    return rows


def quartile_rows(frame: pd.DataFrame, features: list[str]) -> list[dict]:
    rows: list[dict] = []
    for feature in features:
        values = frame[feature]
        try:
            bins = pd.qcut(values, q=4, duplicates="drop")
        except (ValueError, TypeError):
            continue
        if bins.nunique(dropna=True) < 2:
            continue
        temporary = frame.assign(feature_bin=bins)
        for row in grouped_error_rows(temporary, ["feature_bin"]):
            row["feature"] = feature
            row["bin"] = str(row.pop("feature_bin"))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path(
        "artifacts/care-asymmetric-a25-router-source-v1/source_predictions.csv"
    ))
    parser.add_argument("--features", type=Path, default=Path(
        "artifacts/p1-features/features.csv"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/care-source-error-audit-v1"
    ))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)
    required = {
        "sha256", "label", "fold", "Global24__prob", "A25__routed",
        "AsymA25__routed", "FrozenDedupExpert__prob",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise RuntimeError("prediction columns missing: " + repr(missing))
    predictions["_sha"] = normalize_sha(predictions["sha256"])
    if predictions["_sha"].duplicated().any():
        raise RuntimeError("duplicate prediction SHA")

    features = pd.read_csv(args.features, low_memory=False)
    feature_sha = find_sha_column(features)
    features["_sha"] = normalize_sha(features[feature_sha])
    if features["_sha"].duplicated().any():
        raise RuntimeError("duplicate feature SHA")
    feature_by_sha = features.set_index("_sha")
    missing_feature = sorted(set(predictions["_sha"]) - set(feature_by_sha.index))
    if missing_feature:
        raise RuntimeError("features missing prediction SHA: " + repr(missing_feature[:5]))
    aligned_features = feature_by_sha.loc[predictions["_sha"]].reset_index(drop=True)

    excluded = {
        feature_sha, "sha256", "sample_id", "sha", "hash", "label", "y",
        "fold", "split", "year", "metadata_year", "apk_path", "path",
    }
    numeric = numeric_feature_frame(aligned_features, excluded)
    frame = pd.concat([predictions.reset_index(drop=True), numeric], axis=1)
    frame["label"] = frame["label"].astype(int)
    frame["A25__routed"] = frame["A25__routed"].astype(int)
    frame["AsymA25__routed"] = frame["AsymA25__routed"].astype(int)
    frame["base_pred"] = binary(frame["Global24__prob"])
    frame["expert_pred"] = binary(frame["FrozenDedupExpert__prob"])
    frame["A25__final_prob"] = np.where(
        frame["A25__routed"].astype(bool),
        frame["FrozenDedupExpert__prob"],
        frame["Global24__prob"],
    )
    frame["AsymA25__final_prob_reconstructed"] = np.where(
        frame["AsymA25__routed"].astype(bool),
        frame["FrozenDedupExpert__prob"],
        frame["Global24__prob"],
    )
    frame["a25_pred"] = binary(frame["A25__final_prob"])
    frame["asym_pred"] = binary(frame["AsymA25__final_prob_reconstructed"])
    frame["base_outcome"] = outcome(frame["label"], frame["base_pred"])
    frame["a25_outcome"] = outcome(frame["label"], frame["a25_pred"])
    frame["asym_outcome"] = outcome(frame["label"], frame["asym_pred"])
    frame["base_error"] = frame["base_pred"] != frame["label"]
    frame["a25_error"] = frame["a25_pred"] != frame["label"]
    frame["asym_error"] = frame["asym_pred"] != frame["label"]
    frame["a25_transition"] = transition(
        frame["base_error"].to_numpy(), frame["a25_error"].to_numpy()
    )
    frame["asym_transition"] = transition(
        frame["base_error"].to_numpy(), frame["asym_error"].to_numpy()
    )
    frame["route_relationship"] = np.select(
        [
            (frame["A25__routed"] == 1) & (frame["AsymA25__routed"] == 1),
            (frame["A25__routed"] == 1) & (frame["AsymA25__routed"] == 0),
            (frame["A25__routed"] == 0) & (frame["AsymA25__routed"] == 1),
        ],
        ["both", "A25_only", "Asym_only"],
        default="neither",
    )
    frame["base_confidence"] = 2.0 * np.abs(frame["Global24__prob"] - 0.5)
    frame["base_uncertainty"] = 1.0 - frame["base_confidence"]
    frame["confidence_bin"] = pd.cut(
        frame["base_confidence"],
        bins=[-1e-12, 0.2, 0.4, 0.6, 0.8, 1.0000001],
        labels=["[0,.2]", "(.2,.4]", "(.4,.6]", "(.6,.8]", "(.8,1]"],
        include_lowest=True,
    )

    group_rows = []
    for columns in (
        ["label"], ["fold"], ["confidence_bin"], ["label", "confidence_bin"],
        ["route_relationship"], ["a25_transition"], ["asym_transition"],
        ["base_outcome"], ["a25_outcome"], ["asym_outcome"],
    ):
        for row in grouped_error_rows(frame, columns):
            row = {"grouping": "+".join(columns), **row}
            group_rows.append(row)

    effect = []
    feature_columns = list(numeric.columns)
    effect.extend(effect_rows(frame, feature_columns, "base_error", "Global24"))
    effect.extend(effect_rows(frame, feature_columns, "a25_error", "A25_final"))
    effect.extend(effect_rows(frame, feature_columns, "asym_error", "AsymA25_final"))
    effect_frame = pd.DataFrame(effect)
    if not effect_frame.empty:
        effect_frame = effect_frame.sort_values(
            ["method", "scope", "absolute_standardized_difference"],
            ascending=[True, True, False],
        )

    available_key_features = [feature for feature in KEY_FEATURES if feature in frame]
    quartiles = pd.DataFrame(quartile_rows(frame, available_key_features))

    display_columns = [
        "sha256", "label", "fold", "Global24__prob", "base_confidence",
        "base_pred", "base_error", "A25__routed", "a25_pred", "a25_error",
        "base_outcome", "a25_outcome", "asym_outcome", "a25_transition",
        "AsymA25__routed", "asym_pred", "asym_error",
        "asym_transition", "route_relationship", "FrozenDedupExpert__prob",
    ] + available_key_features
    display_columns = list(dict.fromkeys(column for column in display_columns if column in frame))
    error_cases = frame[frame["base_error"] | frame["a25_error"] | frame["asym_error"]]
    error_cases = error_cases.sort_values(
        ["a25_error", "asym_error", "base_confidence"], ascending=[False, False, False]
    )
    hard_cases = frame[frame["a25_error"] & frame["asym_error"]].sort_values(
        "base_confidence", ascending=False
    )
    route_swaps = frame[frame["A25__routed"] != frame["AsymA25__routed"]].copy()
    route_swaps["A25_better"] = (~route_swaps["a25_error"] & route_swaps["asym_error"])
    route_swaps["Asym_better"] = (route_swaps["a25_error"] & ~route_swaps["asym_error"])
    route_swaps = route_swaps.sort_values(
        ["A25_better", "Asym_better", "base_confidence"],
        ascending=[False, False, False],
    )

    frame[display_columns].to_csv(args.output / "all_sample_audit.csv", index=False)
    error_cases[display_columns].to_csv(args.output / "error_cases.csv", index=False)
    hard_cases[display_columns].to_csv(args.output / "common_final_errors.csv", index=False)
    swap_columns = display_columns + ["A25_better", "Asym_better"]
    route_swaps[swap_columns].to_csv(args.output / "route_swap_cases.csv", index=False)
    pd.DataFrame(group_rows).to_csv(args.output / "group_error_rates.csv", index=False)
    effect_frame.to_csv(args.output / "feature_error_effects.csv", index=False)
    quartiles.to_csv(args.output / "key_feature_quartile_error_rates.csv", index=False)

    summary = {
        "n": int(len(frame)),
        "base_error_n": int(frame["base_error"].sum()),
        "a25_error_n": int(frame["a25_error"].sum()),
        "asym_error_n": int(frame["asym_error"].sum()),
        "a25_fixed_n": int((frame["a25_transition"] == "fixed").sum()),
        "a25_broken_n": int((frame["a25_transition"] == "broken").sum()),
        "asym_fixed_n": int((frame["asym_transition"] == "fixed").sum()),
        "asym_broken_n": int((frame["asym_transition"] == "broken").sum()),
        "common_final_error_n": int((frame["a25_error"] & frame["asym_error"]).sum()),
        "a25_only_final_error_n": int((frame["a25_error"] & ~frame["asym_error"]).sum()),
        "asym_only_final_error_n": int((~frame["a25_error"] & frame["asym_error"]).sum()),
        "route_swap_n": int((frame["A25__routed"] != frame["AsymA25__routed"]).sum()),
        "route_swap_a25_better_n": int(route_swaps["A25_better"].sum()),
        "route_swap_asym_better_n": int(route_swaps["Asym_better"].sum()),
        "numeric_feature_n": int(len(feature_columns)),
        "available_key_features": available_key_features,
        "interpretation": "descriptive post-hoc associations; not causal feature importance",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top = pd.DataFrame()
    if not effect_frame.empty:
        top = effect_frame[
            (effect_frame["method"] == "A25_final") &
            (effect_frame["scope"] == "all")
        ].head(12)
    lines = [
        "CARE-Droid source error audit v1",
        "",
        f"Samples: {summary['n']}",
        f"Global24 errors: {summary['base_error_n']}",
        f"A25 final errors: {summary['a25_error_n']} "
        f"(fixed={summary['a25_fixed_n']}, broken={summary['a25_broken_n']})",
        f"AsymA25 final errors: {summary['asym_error_n']} "
        f"(fixed={summary['asym_fixed_n']}, broken={summary['asym_broken_n']})",
        f"Common final errors: {summary['common_final_error_n']}",
        f"A25-only final errors: {summary['a25_only_final_error_n']}",
        f"AsymA25-only final errors: {summary['asym_only_final_error_n']}",
        f"Different routing decisions: {summary['route_swap_n']}; "
        f"A25 better={summary['route_swap_a25_better_n']}, "
        f"AsymA25 better={summary['route_swap_asym_better_n']}",
        "",
        "Largest descriptive feature shifts for A25 final errors:",
    ]
    for row in top.to_dict("records"):
        direction = "higher" if row["standardized_mean_difference"] > 0 else "lower"
        lines.append(
            f"{row['feature']}: errors are {direction}; "
            f"SMD={row['standardized_mean_difference']:.3f}; "
            f"wrong median={row['wrong_median']:.4g}; "
            f"correct median={row['correct_median']:.4g}"
        )
    lines.extend([
        "",
        "Boundary: feature shifts are post-hoc associations and must not be described",
        "as causal explanations or used to tune the frozen method on the same folds.",
    ])
    (args.output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Done:", args.output)


if __name__ == "__main__":
    main()
