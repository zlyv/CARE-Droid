#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

OLD_FEATURES = ROOT / "artifacts/p1-features/features.csv"
NEW_FEATURES = ROOT / "artifacts/p54-new-compact-features/features.csv"
OLD_FUTURE_PREDICTIONS = (
    ROOT
    / "artifacts/p51-rawcount70-shared-caller-future-exploratory"
    / "predictions.csv"
)
P61_PREDICTIONS = (
    ROOT
    / "artifacts/p61-frozen-2017-new-external-validation"
    / "predictions.csv"
)
OUTPUT = ROOT / "artifacts/p62a-top5-route-semantic-drift"

TOP5_ROLES = (
    "background_service_control",
    "location_access",
    "camera_access",
    "call_control_or_recording",
    "microphone_access",
)


def norm_sha(value: object) -> str:
    return str(value).strip().casefold()


def parse_label_series(values: pd.Series) -> pd.Series:
    mapping = {
        "0": 0,
        "0.0": 0,
        "benign": 0,
        "goodware": 0,
        "false": 0,
        "1": 1,
        "1.0": 1,
        "malicious": 1,
        "malware": 1,
        "true": 1,
    }
    result = values.astype(str).str.strip().str.casefold().map(mapping)
    if result.isna().any():
        bad = sorted(values[result.isna()].astype(str).unique().tolist())
        raise RuntimeError("unknown labels: " + repr(bad))
    return result.astype(int)


def normalize_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {"sha256", "metadata_year", "metadata_label"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.copy()
    frame["sha256"] = frame["sha256"].map(norm_sha)
    frame["year"] = pd.to_numeric(frame["metadata_year"], errors="raise").astype(int)
    frame["label"] = parse_label_series(frame["metadata_label"])
    if frame["sha256"].duplicated().any():
        raise RuntimeError(f"{path}: duplicate SHA rows")
    return frame


def discover_roles(columns: list[str]) -> list[str]:
    node_roles = {name.removeprefix("role_node__") for name in columns if name.startswith("role_node__")}
    edge_roles = {name.removeprefix("role_edge__") for name in columns if name.startswith("role_edge__")}
    roles = sorted(node_roles | edge_roles)
    missing_top5 = sorted(set(TOP5_ROLES) - set(roles))
    if missing_top5:
        raise RuntimeError("missing TOP-5 roles: " + repr(missing_top5))
    return roles


def add_role_columns(frame: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for role in roles:
        node_name = "role_node__" + role
        edge_name = "role_edge__" + role
        node = (
            pd.to_numeric(output[node_name], errors="coerce").fillna(0.0)
            if node_name in output.columns
            else pd.Series(0.0, index=output.index)
        )
        edge = (
            pd.to_numeric(output[edge_name], errors="coerce").fillna(0.0)
            if edge_name in output.columns
            else pd.Series(0.0, index=output.index)
        )
        total_name = "role_total__" + role
        present_name = "role_present__" + role
        output[total_name] = node.clip(lower=0.0) + edge.clip(lower=0.0)
        output[present_name] = (output[total_name] > 0.0).astype(int)
    top5_present = ["role_present__" + role for role in TOP5_ROLES]
    output["top5_role_count"] = output[top5_present].sum(axis=1).astype(int)
    output["top5_routed"] = (output["top5_role_count"] > 0).astype(int)
    output["top5_roles"] = output.apply(
        lambda row: ";".join(
            role for role in TOP5_ROLES if int(row["role_present__" + role]) == 1
        ),
        axis=1,
    )
    return output


def build_cohorts(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    future = pd.read_csv(OLD_FUTURE_PREDICTIONS, usecols=["year", "sha256"])
    future["sha256"] = future["sha256"].map(norm_sha)
    future["year"] = pd.to_numeric(future["year"], errors="raise").astype(int)
    future = future.drop_duplicates(["year", "sha256"])

    old_source = old.loc[old["year"] == 2017].copy()
    old_source["cohort"] = "old_training"
    old_future = old.merge(future, on=["year", "sha256"], how="inner", validate="one_to_one")
    old_future["cohort"] = "old_future"

    p61 = pd.read_csv(
        P61_PREDICTIONS,
        usecols=["sha256", "package_certificate_disjoint"],
    )
    p61["sha256"] = p61["sha256"].map(norm_sha)
    p61["package_certificate_disjoint"] = pd.to_numeric(
        p61["package_certificate_disjoint"], errors="raise"
    ).astype(int)
    strict_shas = set(
        p61.loc[p61["package_certificate_disjoint"] == 1, "sha256"].unique()
    )
    strict_new = new.loc[new["sha256"].isin(strict_shas)].copy()
    strict_new["cohort"] = "new_external"
    if len(strict_new) != 1087:
        raise RuntimeError(f"expected 1087 strict new APKs, got {len(strict_new)}")

    return pd.concat([old_source, old_future, strict_new], ignore_index=True, sort=False)


def route_tables(
    cohorts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_rows = []
    top5_rows = []
    combination_rows = []
    for (cohort, year), year_part in cohorts.groupby(["cohort", "year"], sort=True):
        label_groups = [("ALL", year_part)]
        label_groups.extend(
            (str(label), part)
            for label, part in year_part.groupby("label", sort=True)
        )
        for label, part in label_groups:
            route_rows.append({
                "cohort": cohort,
                "year": int(year),
                "label": label,
                "apk_count": len(part),
                "routed_apks": int(part["top5_routed"].sum()),
                "route_coverage": float(part["top5_routed"].mean()),
                "mean_top5_role_count": float(part["top5_role_count"].mean()),
            })
            for role in TOP5_ROLES:
                present = part["role_present__" + role]
                total = part["role_total__" + role]
                top5_rows.append({
                    "cohort": cohort,
                    "year": int(year),
                    "label": label,
                    "role": role,
                    "apk_count": len(part),
                    "present_apks": int(present.sum()),
                    "presence_rate": float(present.mean()),
                    "mean_log1p_count": float(np.log1p(total).mean()),
                })
            for combination, combination_part in part.groupby(
                "top5_roles", sort=True, dropna=False
            ):
                combination_rows.append({
                    "cohort": cohort,
                    "year": int(year),
                    "label": label,
                    "top5_roles": combination or "NONE",
                    "top5_role_count": int(
                        combination_part["top5_role_count"].iloc[0]
                    ),
                    "apk_count": len(combination_part),
                    "prevalence": float(len(combination_part) / len(part)),
                })
    return (
        pd.DataFrame(route_rows),
        pd.DataFrame(top5_rows),
        pd.DataFrame(combination_rows),
    )


def role_drift_table(cohorts: pd.DataFrame, roles: list[str]) -> pd.DataFrame:
    rows = []
    for year in sorted(set(cohorts["year"])):
        old_name = "old_training" if year == 2017 else "old_future"
        for label in (0, 1):
            old = cohorts.loc[
                (cohorts["cohort"] == old_name)
                & (cohorts["year"] == year)
                & (cohorts["label"] == label)
            ]
            new = cohorts.loc[
                (cohorts["cohort"] == "new_external")
                & (cohorts["year"] == year)
                & (cohorts["label"] == label)
            ]
            if old.empty or new.empty:
                continue
            for role in roles:
                present_name = "role_present__" + role
                total_name = "role_total__" + role
                old_present = old[present_name].astype(float)
                new_present = new[present_name].astype(float)
                old_log = np.log1p(old[total_name].astype(float).to_numpy())
                new_log = np.log1p(new[total_name].astype(float).to_numpy())
                pooled = np.concatenate([old_log, new_log])
                scale = float(np.std(pooled))
                smd = 0.0 if scale < 1e-12 else float((np.mean(new_log) - np.mean(old_log)) / scale)
                rows.append({
                    "year": int(year),
                    "label": int(label),
                    "role": role,
                    "old_cohort": old_name,
                    "old_apks": len(old),
                    "new_apks": len(new),
                    "old_presence_rate": float(old_present.mean()),
                    "new_presence_rate": float(new_present.mean()),
                    "presence_delta": float(new_present.mean() - old_present.mean()),
                    "old_mean_log1p_count": float(np.mean(old_log)),
                    "new_mean_log1p_count": float(np.mean(new_log)),
                    "log_count_smd": smd,
                })
    output = pd.DataFrame(rows)
    output["abs_presence_delta"] = output["presence_delta"].abs()
    output["abs_log_count_smd"] = output["log_count_smd"].abs()
    return output.sort_values(
        ["year", "label", "abs_presence_delta", "abs_log_count_smd"],
        ascending=[True, True, False, False],
    )


def p61_change_analysis(new_features: pd.DataFrame, roles: list[str]):
    pred = pd.read_csv(P61_PREDICTIONS, low_memory=False)
    pred["sha256"] = pred["sha256"].map(norm_sha)
    numeric = [
        "year", "seed", "label", "prediction", "probability",
        "top5_routed", "package_certificate_disjoint",
    ]
    for name in numeric:
        pred[name] = pd.to_numeric(pred[name], errors="raise")
    pred = pred.loc[pred["package_certificate_disjoint"] == 1].copy()

    keys = ["year", "seed", "sha256", "label"]
    raw = pred.loc[pred["variant"] == "raw_count70_only", keys + ["probability", "prediction", "top5_routed"]].copy()
    raw = raw.rename(columns={
        "probability": "raw_probability",
        "prediction": "raw_prediction",
        "top5_routed": "top5_routed_raw",
    })
    final = pred.loc[pred["variant"] == "top5_selective_shared_caller", keys + ["probability", "prediction", "top5_routed"]].copy()
    final = final.rename(columns={
        "probability": "final_probability",
        "prediction": "final_prediction",
        "top5_routed": "top5_routed_final",
    })
    merged = raw.merge(final, on=keys, how="inner", validate="one_to_one")
    if len(merged) != 5 * 1087:
        raise RuntimeError(f"unexpected strict P61 prediction pairs: {len(merged)}")
    if not np.array_equal(merged["top5_routed_raw"], merged["top5_routed_final"]):
        raise RuntimeError("route flag differs between variants")
    merged["top5_routed"] = merged.pop("top5_routed_raw").astype(int)
    merged = merged.drop(columns=["top5_routed_final"])
    merged["probability_delta"] = merged["final_probability"] - merged["raw_probability"]

    raw_correct = merged["raw_prediction"].astype(int) == merged["label"].astype(int)
    final_correct = merged["final_prediction"].astype(int) == merged["label"].astype(int)
    merged["outcome"] = "UNCHANGED"
    merged.loc[(~raw_correct) & final_correct & (merged["label"] == 0), "outcome"] = "CORRECTED_FP"
    merged.loc[(~raw_correct) & final_correct & (merged["label"] == 1), "outcome"] = "CORRECTED_FN"
    merged.loc[raw_correct & (~final_correct) & (merged["label"] == 0), "outcome"] = "REGRESSED_TN"
    merged.loc[raw_correct & (~final_correct) & (merged["label"] == 1), "outcome"] = "REGRESSED_TP"
    merged["changed"] = (merged["raw_prediction"] != merged["final_prediction"]).astype(int)

    feature_columns = ["sha256", "top5_roles", "top5_role_count"]
    feature_columns += ["role_present__" + role for role in roles]
    feature_columns += ["role_total__" + role for role in roles]
    merged = merged.merge(
        new_features[feature_columns], on="sha256", how="left", validate="many_to_one"
    )
    if merged["top5_role_count"].isna().any():
        raise RuntimeError("missing new feature rows for P61 predictions")

    changed = merged.loc[merged["outcome"] != "UNCHANGED"].copy()
    summary = (
        changed.groupby(["year", "outcome"], as_index=False)
        .agg(
            seed_case_count=("sha256", "size"),
            unique_apks=("sha256", "nunique"),
            mean_probability_delta=("probability_delta", "mean"),
            mean_raw_probability=("raw_probability", "mean"),
            mean_final_probability=("final_probability", "mean"),
        )
    )
    summary["mean_cases_per_seed"] = summary["seed_case_count"] / 5.0

    stability = (
        changed.groupby(["year", "sha256", "label", "outcome"], as_index=False)
        .agg(
            affected_seeds=("seed", "nunique"),
            mean_raw_probability=("raw_probability", "mean"),
            mean_final_probability=("final_probability", "mean"),
            mean_probability_delta=("probability_delta", "mean"),
            top5_roles=("top5_roles", "first"),
            top5_role_count=("top5_role_count", "first"),
        )
        .sort_values(["year", "outcome", "affected_seeds"], ascending=[True, True, False])
    )

    role_rows = []
    for (year, outcome), part in merged.groupby(["year", "outcome"], sort=True):
        for role in roles:
            present = part["role_present__" + role]
            role_rows.append({
                "year": int(year),
                "outcome": outcome,
                "role": role,
                "seed_cases": len(part),
                "unique_apks": int(part["sha256"].nunique()),
                "presence_rate": float(present.mean()),
                "mean_log1p_count": float(np.log1p(part["role_total__" + role]).mean()),
            })
    role_outcomes = pd.DataFrame(role_rows)
    return changed, summary, stability, role_outcomes


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = normalize_features(OLD_FEATURES)
    new = normalize_features(NEW_FEATURES)
    if set(old["sha256"]) & set(new["sha256"]):
        raise RuntimeError("old/new feature SHA overlap")

    old_roles = discover_roles(old.columns.tolist())
    new_roles = discover_roles(new.columns.tolist())
    if old_roles != new_roles:
        raise RuntimeError("old/new semantic role dictionaries differ")
    roles = old_roles
    old = add_role_columns(old, roles)
    new = add_role_columns(new, roles)

    cohorts = build_cohorts(old, new)
    route, top5, combinations = route_tables(cohorts)
    drift = role_drift_table(cohorts, roles)
    changed, change_summary, stability, outcome_roles = p61_change_analysis(new, roles)

    route.to_csv(OUTPUT / "route_coverage.csv", index=False)
    top5.to_csv(OUTPUT / "top5_role_prevalence.csv", index=False)
    combinations.to_csv(OUTPUT / "top5_combination_prevalence.csv", index=False)
    drift.to_csv(OUTPUT / "all_role_drift.csv", index=False)
    changed.to_csv(OUTPUT / "p61_changed_seed_cases.csv", index=False)
    change_summary.to_csv(OUTPUT / "p61_change_summary.csv", index=False)
    stability.to_csv(OUTPUT / "p61_changed_apk_stability.csv", index=False)
    outcome_roles.to_csv(OUTPUT / "p61_outcome_role_prevalence.csv", index=False)

    new_future = route.loc[
        (route["cohort"] == "new_external")
        & route["year"].isin([2018, 2019, 2020])
        & (route["label"] == "ALL")
    ]
    audit = {
        "phase": "P62A_TOP5_ROUTE_AND_SEMANTIC_DRIFT_AUDIT",
        "apk_reparse_required": False,
        "semantic_role_count": len(roles),
        "top5_roles": list(TOP5_ROLES),
        "old_feature_rows": len(old),
        "new_feature_rows": len(new),
        "strict_new_external_rows": int((cohorts["cohort"] == "new_external").sum()),
        "new_2018_2020_mean_route_coverage_macro_by_year": float(
            new_future["route_coverage"].mean()
        ),
        "p61_changed_seed_cases": len(changed),
        "labels_used_for_training_or_selection": False,
        "labels_used_for_posthoc_error_analysis_only": True,
        "notes": [
            "Five model seeds are not treated as independent APKs.",
            "Both seed-case counts and unique APK counts are reported.",
            "This audit does not modify the frozen P61 detector.",
        ],
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 110)
    print("ROUTE COVERAGE")
    print("=" * 110)
    print(route.to_string(index=False))
    print("\n" + "=" * 110)
    print("P61 CHANGED OUTCOMES")
    print("=" * 110)
    print(change_summary.to_string(index=False))
    print("\n" + "=" * 110)
    print("2020 REGRESSED MALWARE APKS")
    print("=" * 110)
    print(
        stability.loc[
            (stability["year"] == 2020) & (stability["outcome"] == "REGRESSED_TP")
        ].head(30).to_string(index=False)
    )
    print("saved =", OUTPUT)


if __name__ == "__main__":
    main()
