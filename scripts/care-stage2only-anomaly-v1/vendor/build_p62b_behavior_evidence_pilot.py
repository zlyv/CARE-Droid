#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = (
    ROOT
    / "artifacts/p61-frozen-2017-new-external-validation"
    / "predictions.csv"
)
FEATURES = ROOT / "artifacts/p54-new-compact-features/features.csv"
LINEAGE = ROOT / "artifacts/p54-new-lineage/package_certificate_lineage.csv"
OUTPUT = ROOT / "artifacts/p62b-behavior-evidence-pilot"

TOP5_ROLES = (
    "background_service_control",
    "location_access",
    "camera_access",
    "call_control_or_recording",
    "microphone_access",
)
YEARS = (2017, 2018, 2019, 2020)
SEED_COUNT = 5

CHANGED_OUTCOMES = (
    "CORRECTED_FP",
    "CORRECTED_FN",
    "REGRESSED_TN",
    "REGRESSED_TP",
)
OUTCOME_PRIORITY = {
    "REGRESSED_TP": 0,
    "REGRESSED_TN": 1,
    "CORRECTED_FN": 2,
    "CORRECTED_FP": 3,
}


def norm_sha(value: object) -> str:
    return str(value).strip().casefold()


def numeric(frame: pd.DataFrame, names: list[str]) -> None:
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors="raise")


def load_pairs() -> pd.DataFrame:
    pred = pd.read_csv(PREDICTIONS, low_memory=False)
    required = {
        "year", "seed", "variant", "sha256", "label",
        "top5_routed", "package_certificate_disjoint",
        "probability", "prediction",
    }
    missing = required - set(pred.columns)
    if missing:
        raise RuntimeError("P61 predictions missing: " + repr(sorted(missing)))
    pred["sha256"] = pred["sha256"].map(norm_sha)
    numeric(
        pred,
        [
            "year", "seed", "label", "top5_routed",
            "package_certificate_disjoint", "probability", "prediction",
        ],
    )
    pred = pred.loc[
        (pred["package_certificate_disjoint"] == 1)
        & pred["year"].isin(YEARS)
    ].copy()

    keys = ["year", "seed", "sha256", "label"]
    raw = pred.loc[
        pred["variant"].eq("raw_count70_only"),
        keys + ["top5_routed", "probability", "prediction"],
    ].rename(columns={
        "top5_routed": "raw_route",
        "probability": "raw_probability",
        "prediction": "raw_prediction",
    })
    final = pred.loc[
        pred["variant"].eq("top5_selective_shared_caller"),
        keys + ["top5_routed", "probability", "prediction"],
    ].rename(columns={
        "top5_routed": "final_route",
        "probability": "final_probability",
        "prediction": "final_prediction",
    })
    pairs = raw.merge(final, on=keys, how="inner", validate="one_to_one")
    if not np.array_equal(pairs["raw_route"], pairs["final_route"]):
        raise RuntimeError("route differs between paired variants")
    pairs["top5_routed"] = pairs.pop("raw_route").astype(int)
    pairs = pairs.drop(columns=["final_route"])

    label = pairs["label"].astype(int)
    raw_correct = pairs["raw_prediction"].astype(int).eq(label)
    final_correct = pairs["final_prediction"].astype(int).eq(label)
    pairs["outcome"] = "STABLE_WRONG"
    pairs.loc[raw_correct & final_correct, "outcome"] = "STABLE_CORRECT"
    pairs.loc[(~raw_correct) & final_correct & label.eq(0), "outcome"] = "CORRECTED_FP"
    pairs.loc[(~raw_correct) & final_correct & label.eq(1), "outcome"] = "CORRECTED_FN"
    pairs.loc[raw_correct & (~final_correct) & label.eq(0), "outcome"] = "REGRESSED_TN"
    pairs.loc[raw_correct & (~final_correct) & label.eq(1), "outcome"] = "REGRESSED_TP"
    pairs["raw_correct"] = raw_correct.astype(int)
    pairs["final_correct"] = final_correct.astype(int)
    return pairs


def load_feature_context() -> pd.DataFrame:
    frame = pd.read_csv(FEATURES, low_memory=False)
    frame["sha256"] = frame["sha256"].map(norm_sha)
    if frame["sha256"].duplicated().any():
        raise RuntimeError("duplicate feature SHA")
    present_names = []
    for role in TOP5_ROLES:
        node_name = "role_node__" + role
        edge_name = "role_edge__" + role
        if node_name not in frame or edge_name not in frame:
            raise RuntimeError("missing TOP-5 feature columns for " + role)
        total = (
            pd.to_numeric(frame[node_name], errors="coerce").fillna(0.0)
            + pd.to_numeric(frame[edge_name], errors="coerce").fillna(0.0)
        )
        present_name = "present__" + role
        frame[present_name] = total.gt(0.0).astype(int)
        present_names.append(present_name)
    frame["top5_role_count"] = frame[present_names].sum(axis=1).astype(int)
    frame["feature_route"] = frame["top5_role_count"].gt(0).astype(int)
    frame["top5_roles"] = frame.apply(
        lambda row: ";".join(
            role for role in TOP5_ROLES if int(row["present__" + role]) == 1
        ),
        axis=1,
    )
    return frame[[
        "sha256", "top5_role_count", "feature_route", "top5_roles",
        *present_names,
    ]]


def aggregate_apks(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, sha, label), part in pairs.groupby(
        ["year", "sha256", "label"], sort=True
    ):
        if part["seed"].nunique() != SEED_COUNT:
            raise RuntimeError(f"{sha}: expected {SEED_COUNT} seeds")
        counts = Counter(part["outcome"])
        changed = {name: counts.get(name, 0) for name in CHANGED_OUTCOMES}
        primary = "STABLE_CORRECT" if counts.get("STABLE_CORRECT", 0) == SEED_COUNT else ""
        if any(changed.values()):
            primary = sorted(
                CHANGED_OUTCOMES,
                key=lambda name: (-changed[name], OUTCOME_PRIORITY[name]),
            )[0]
        elif not primary:
            primary = "STABLE_WRONG"
        rows.append({
            "year": int(year),
            "sha256": sha,
            "label": int(label),
            "primary_outcome": primary,
            "changed_seed_count": int(sum(changed.values())),
            "stable_correct_seed_count": int(counts.get("STABLE_CORRECT", 0)),
            "stable_wrong_seed_count": int(counts.get("STABLE_WRONG", 0)),
            "corrected_FP_seeds": int(changed["CORRECTED_FP"]),
            "corrected_FN_seeds": int(changed["CORRECTED_FN"]),
            "regressed_TN_seeds": int(changed["REGRESSED_TN"]),
            "regressed_TP_seeds": int(changed["REGRESSED_TP"]),
            "mean_raw_probability": float(part["raw_probability"].mean()),
            "mean_final_probability": float(part["final_probability"].mean()),
            "top5_routed": int(part["top5_routed"].iloc[0]),
        })
    return pd.DataFrame(rows)


def role_set(value: object) -> frozenset[str]:
    return frozenset(item for item in str(value).split(";") if item)


def match_distance(target: pd.Series, candidate: pd.Series) -> float:
    target_roles = role_set(target["top5_roles"])
    candidate_roles = role_set(candidate["top5_roles"])
    union = target_roles | candidate_roles
    jaccard_distance = (
        0.0 if not union else 1.0 - len(target_roles & candidate_roles) / len(union)
    )
    return float(
        abs(target["mean_raw_probability"] - candidate["mean_raw_probability"])
        + 0.12 * abs(target["top5_role_count"] - candidate["top5_role_count"])
        + 0.08 * jaccard_distance
    )


def build_selection(apks: pd.DataFrame) -> pd.DataFrame:
    changed = apks.loc[apks["changed_seed_count"].gt(0)].copy()
    controls = apks.loc[
        apks["stable_correct_seed_count"].eq(SEED_COUNT)
        & apks["top5_routed"].eq(1)
    ].copy()
    used_controls: set[str] = set()
    rows = []

    for _, target in changed.sort_values(
        ["year", "primary_outcome", "sha256"]
    ).iterrows():
        target_row = target.to_dict()
        target_row.update({
            "selection_group": "changed_case",
            "matched_to_sha256": "",
            "match_distance": 0.0,
        })
        rows.append(target_row)

        candidates = controls.loc[
            controls["year"].eq(target["year"])
            & controls["label"].eq(target["label"])
            & controls["top5_routed"].eq(target["top5_routed"])
            & ~controls["sha256"].isin(used_controls)
        ].copy()
        if candidates.empty:
            raise RuntimeError(
                f"no stable control for {target['year']} {target['label']} {target['sha256']}"
            )
        candidates["_distance"] = candidates.apply(
            lambda row: match_distance(target, row), axis=1
        )
        control = candidates.sort_values(
            ["_distance", "sha256"]
        ).iloc[0]
        used_controls.add(control["sha256"])
        control_row = control.drop(labels=["_distance"]).to_dict()
        control_row.update({
            "selection_group": "matched_stable_control",
            "matched_to_sha256": target["sha256"],
            "match_distance": float(control["_distance"]),
        })
        rows.append(control_row)

    selected = pd.DataFrame(rows)
    if selected["sha256"].duplicated().any():
        raise RuntimeError("pilot selection contains duplicate SHA")
    return selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    apks = aggregate_apks(pairs)
    context = load_feature_context()
    apks = apks.merge(context, on="sha256", how="left", validate="one_to_one")
    if apks["top5_role_count"].isna().any():
        raise RuntimeError("missing feature context")
    if not np.array_equal(apks["top5_routed"], apks["feature_route"]):
        raise RuntimeError("P61 route differs from feature-derived route")

    selected = build_selection(apks)
    lineage = pd.read_csv(LINEAGE, low_memory=False)
    lineage["sha256"] = lineage["sha256"].map(norm_sha)
    if lineage["sha256"].duplicated().any():
        raise RuntimeError("duplicate lineage SHA")
    lineage_columns = lineage.columns.tolist()
    selected = selected.merge(
        lineage,
        on="sha256",
        how="left",
        validate="one_to_one",
        suffixes=("", "_lineage"),
    )
    if selected["absolute_path"].isna().any():
        raise RuntimeError("selected APK missing absolute path")
    missing_paths = [
        path for path in selected["absolute_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing_paths:
        raise RuntimeError("selected APK paths missing: " + repr(missing_paths[:5]))

    manifest_columns = [
        "selection_group", "matched_to_sha256", "match_distance",
        "primary_outcome", "changed_seed_count",
        "corrected_FP_seeds", "corrected_FN_seeds",
        "regressed_TN_seeds", "regressed_TP_seeds",
        "stable_correct_seed_count", "stable_wrong_seed_count",
        "year", "label", "sha256", "absolute_path",
        "mean_raw_probability", "mean_final_probability",
        "top5_routed", "top5_role_count", "top5_roles",
    ]
    for name in ("lineage_id", "package_name", "certificate_sha256"):
        if name in selected.columns:
            manifest_columns.append(name)
    selected[manifest_columns].to_csv(
        OUTPUT / "pilot_manifest.csv", index=False
    )

    selected_lineage = lineage.loc[
        lineage["sha256"].isin(set(selected["sha256"]))
    ].copy()
    selected_lineage = selected_lineage.sort_values("sha256")
    selected_lineage[lineage_columns].to_csv(
        OUTPUT / "pilot_lineage.csv", index=False
    )
    (OUTPUT / "pilot_sha256.txt").write_text(
        "\n".join(sorted(selected["sha256"])) + "\n", encoding="utf-8"
    )

    summary = (
        selected.groupby(
            ["selection_group", "year", "label", "primary_outcome"],
            as_index=False,
        ).agg(apk_count=("sha256", "nunique"))
    )
    summary.to_csv(OUTPUT / "pilot_summary.csv", index=False)
    changed_count = int(selected["selection_group"].eq("changed_case").sum())
    control_count = int(
        selected["selection_group"].eq("matched_stable_control").sum()
    )
    audit = {
        "phase": "P62B_BEHAVIOR_EVIDENCE_DIAGNOSTIC_SELECTION",
        "formal_training_enabled": False,
        "selection_uses_labels": True,
        "selection_purpose": "extractor diagnosis and manual precision validation only",
        "selected_years": list(YEARS),
        "changed_apks": changed_count,
        "matched_stable_controls": control_count,
        "total_apks": len(selected),
        "matching": (
            "same year, label and TOP-5 route; nearest first-layer probability, "
            "TOP-5 role count and role-set Jaccard distance"
        ),
        "forbidden_use": (
            "This label-aware pilot must not be used to train, tune or report the "
            "formal future-year classifier."
        ),
    }
    (OUTPUT / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 120)
    print("P62-B DIAGNOSTIC PILOT SELECTION")
    print("=" * 120)
    print(summary.to_string(index=False))
    print("\nchanged APKs =", changed_count)
    print("matched stable controls =", control_count)
    print("total APKs =", len(selected))
    print("saved =", OUTPUT)


if __name__ == "__main__":
    main()
