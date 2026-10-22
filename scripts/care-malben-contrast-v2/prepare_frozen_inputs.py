#!/usr/bin/env python3
"""Resolve/derive frozen source inputs and report every missing dependency once."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


ROUTER_CANONICAL = Path(
    "artifacts/care-committee-decision-instability-router-source-v1/"
    "source_sample_scores.csv"
)
COMMITTEE_CANONICAL = Path(
    "artifacts/care-committee-uncertainty-router-source-v1/"
    "source_sample_scores.csv"
)
BASELINE_CANONICAL = Path(
    "artifacts/care-di-budget-bound-selection-v1/retrained_source/"
    "max_delta_2p25/source_predictions.csv"
)
V4_REQUIRED = (
    Path("artifacts/p1-features/features.csv"),
    Path("artifacts/p1-features/feature_schema.json"),
    Path(
        "artifacts/p25-endpoint-cache-full-cap256/"
        "endpoint_centered_tensor_cache.jsonl.gz"
    ),
    Path(
        "artifacts/p72-verified-dependency-source-ablation/"
        "p72_oof_predictions.csv"
    ),
    Path(
        "artifacts/clipped-intra-inter-structure-ablation-2017-v1-full/"
        "multiscale_features.npz"
    ),
    Path(
        "artifacts/global24-a25-route-conditioned-joints7-v1-full/"
        "source_predictions.csv"
    ),
)
MEMBERS = ("p_lr", "p_rf", "p_xgb", "p_svm")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_top_fraction(score: np.ndarray, budget: float = 0.25) -> np.ndarray:
    count = max(1, min(len(score), int(round(float(budget) * len(score)))))
    order = np.argsort(score, kind="mergesort")
    route = np.zeros(len(score), dtype=bool)
    route[order[-count:]] = True
    return route


def group_top_fraction(score: np.ndarray, group: np.ndarray) -> np.ndarray:
    route = np.zeros(len(score), dtype=bool)
    for value in np.unique(group):
        index = np.flatnonzero(group == value)
        route[index] = exact_top_fraction(score[index])
    return route


def valid_router(rows: list[dict]) -> bool:
    required = {"sha256", "label", "fold", "p_stage1", "route_A25", "route_DI"}
    return len(rows) == 799 and bool(rows) and required.issubset(rows[0])


def can_derive_router(rows: list[dict]) -> bool:
    required = {
        "sha256", "label", "fold", "p_stage1", "route_A25", *MEMBERS
    }
    return len(rows) == 799 and bool(rows) and required.issubset(rows[0])


def derive_router(rows: list[dict]) -> list[dict]:
    base = np.asarray([float(row["p_stage1"]) for row in rows], dtype=np.float64)
    members = np.column_stack([
        np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in MEMBERS
    ])
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    decision = (base >= 0.5).astype(np.float64)
    support_by_member = (
        decision[:, None] * members
        + (1.0 - decision[:, None]) * (1.0 - members)
    )
    support = np.mean(support_by_member, axis=1)
    uncertainty = 1.0 - support
    route = group_top_fraction(uncertainty, folds)
    output = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.update({
            "committee_support_for_stage1": float(support[index]),
            "U_DI": float(uncertainty[index]),
            "support_lr": float(support_by_member[index, 0]),
            "support_rf": float(support_by_member[index, 1]),
            "support_xgb": float(support_by_member[index, 2]),
            "support_svm": float(support_by_member[index, 3]),
            "route_DI": int(route[index]),
        })
        output.append(item)
    if int(np.sum(route)) != 200:
        raise RuntimeError("derived DI route must contain exactly 200 source samples")
    return output


def router_candidates(root: Path) -> list[Path]:
    preferred = [ROUTER_CANONICAL, COMMITTEE_CANONICAL]
    found = []
    for path in preferred:
        if path.exists():
            found.append(path)
    artifacts = root / "artifacts"
    if artifacts.exists():
        for path in sorted(artifacts.glob("**/source_sample_scores.csv")):
            if path not in found:
                found.append(path)
    return found


def prepare_router(root: Path) -> tuple[str | None, list[str]]:
    diagnostics = []
    for path in router_candidates(root):
        try:
            rows = read_rows(path)
        except Exception as exc:
            diagnostics.append(f"unreadable router candidate {path}: {exc}")
            continue
        if valid_router(rows):
            if path != ROUTER_CANONICAL:
                ROUTER_CANONICAL.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, ROUTER_CANONICAL)
            return f"resolved frozen DI router from {path}", diagnostics
        if can_derive_router(rows):
            write_rows(ROUTER_CANONICAL, derive_router(rows))
            return f"derived frozen DI router from committee probabilities in {path}", diagnostics
        diagnostics.append(f"incompatible router candidate {path}")
    return None, diagnostics


def valid_baseline(path: Path) -> bool:
    try:
        rows = read_rows(path)
    except Exception:
        return False
    if len(rows) != 799 or not rows:
        return False
    fields = set(rows[0])
    return (
        {"sha256", "label", "fold", "Global24__prob"}.issubset(fields)
        and any(
            "RGCondAttn" in name and "Dedup" in name and "expert_prob" in name
            for name in fields
        )
    )


def prepare_baseline(root: Path) -> tuple[str | None, list[str]]:
    if BASELINE_CANONICAL.exists() and valid_baseline(BASELINE_CANONICAL):
        return f"using frozen anomaly baseline {BASELINE_CANONICAL}", []
    candidates = []
    artifacts = root / "artifacts"
    if artifacts.exists():
        for path in sorted(artifacts.glob("**/source_predictions.csv")):
            if valid_baseline(path):
                candidates.append(path)
    if len(candidates) == 1:
        BASELINE_CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], BASELINE_CANONICAL)
        return f"resolved frozen anomaly baseline from {candidates[0]}", []
    if len(candidates) > 1:
        return None, [
            "canonical max_delta=2.25 baseline is absent and multiple compatible "
            "source_predictions.csv files were found; refusing to guess: "
            + ", ".join(str(path) for path in candidates)
        ]
    return None, ["no compatible 799-row RGCondAttn Dedup baseline was found"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/care-malben-contrast-v2/preflight.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    router_message, router_notes = prepare_router(root)
    baseline_message, baseline_notes = prepare_baseline(root)
    missing = [str(path) for path in V4_REQUIRED if not path.exists()]
    messages = [item for item in (router_message, baseline_message) if item]
    notes = router_notes + baseline_notes
    if router_message is None:
        missing.append(
            "a 799-row committee source score file containing p_stage1, p_lr, "
            "p_rf, p_xgb, p_svm, fold and route_A25"
        )
    if baseline_message is None:
        missing.append(str(BASELINE_CANONICAL))

    audit = {
        "root": str(root),
        "messages": messages,
        "notes": notes,
        "missing": missing,
        "router": str(ROUTER_CANONICAL),
        "baseline": str(BASELINE_CANONICAL),
        "region_cache_optional": True,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for message in messages:
        print("[preflight]", message, flush=True)
    for note in notes:
        print("[preflight note]", note, flush=True)
    if missing:
        print("\nPreflight found all unresolved inputs:", flush=True)
        for item in missing:
            print("  -", item, flush=True)
        raise SystemExit(2)
    print("[preflight] all frozen inputs resolved", flush=True)


if __name__ == "__main__":
    main()
