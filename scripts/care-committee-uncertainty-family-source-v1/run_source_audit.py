#!/usr/bin/env python3
"""Read-only strict-2017 comparison of committee uncertainty definitions."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

import preflight
from uncertainty_family import MEMBERS, exact_binomial_two_sided, exact_fold_top, scores


OUT = Path("artifacts/care-committee-uncertainty-family-source-v1")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def ratio(a: int, b: int) -> float:
    return float(a / b) if b else 0.0


def rank_auc(y: np.ndarray, s: np.ndarray) -> float:
    # Tie-aware Mann-Whitney AUC, diagnostic only.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]: j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main() -> None:
    path = preflight.resolve(); rows = preflight.load(path); preflight.validate(rows)
    sha = np.asarray([r["sha256"].strip().lower() for r in rows])
    y = np.asarray([int(float(r["label"])) for r in rows])
    fold = np.asarray([int(r["fold"]) for r in rows])
    base = np.asarray([float(r["p_stage1"]) for r in rows])
    member = np.column_stack([[float(r[k]) for r in rows] for k in MEMBERS])
    score_map = scores(base, member)
    base_pred = base >= .5; error = base_pred != y
    high_error = error & (np.maximum(base, 1-base) >= .8)

    routes = {name: exact_fold_top(value, fold, sha, .25)
              for name, value in score_map.items()}
    # Use the canonical persisted DI route when present, and prove equivalence.
    if "route_DI" in rows[0]:
        frozen = np.asarray([int(float(r["route_DI"])) == 1 for r in rows])
        if not np.array_equal(frozen, routes["MeanOpposition_DI25"]):
            raise RuntimeError("BLOCKED_BY_CONTRACT: recomputed DI route differs from frozen route")

    expert_key = next((k for k in ("p_frozen_bounded_expert", "p_expert", "p_stage2")
                       if k in rows[0]), None)
    expert = np.asarray([float(r[expert_key]) for r in rows]) if expert_key else None
    repairable = error & ((expert >= .5) == y) if expert is not None else None
    harmful = (~error) & ((expert >= .5) != y) if expert is not None else None

    metrics = []
    for name, route in routes.items():
        item = {
            "router": name, "n": len(y), "routed": int(route.sum()),
            "captured_errors": int((route & error).sum()), "base_errors": int(error.sum()),
            "error_recall_at_25": ratio(int((route & error).sum()), int(error.sum())),
            "routing_precision": ratio(int((route & error).sum()), int(route.sum())),
            "captured_highconf_errors": int((route & high_error).sum()),
            "highconf_errors": int(high_error.sum()),
            "highconf_error_recall": ratio(int((route & high_error).sum()), int(high_error.sum())),
            "error_score_auc": rank_auc(error, score_map[name]),
            "repairable_error_recall": "unavailable",
            "fixed": "unavailable", "broken": "unavailable", "net": "unavailable",
        }
        if expert is not None:
            fixed = int((route & repairable).sum()); broken = int((route & harmful).sum())
            item.update({"repairable_error_recall": ratio(fixed, int(repairable.sum())),
                         "fixed": fixed, "broken": broken, "net": fixed-broken})
        metrics.append(item)

    di = routes["MeanOpposition_DI25"]
    paired = []
    for name, route in routes.items():
        cand_only = int((error & route & ~di).sum())
        di_only = int((error & di & ~route).sum())
        paired.append({"candidate": name, "candidate_only_errors": cand_only,
                       "DI_only_errors": di_only, "capture_delta": cand_only-di_only,
                       "exact_p": exact_binomial_two_sided(cand_only, di_only),
                       "route_overlap": int((route & di).sum()),
                       "route_jaccard": ratio(int((route & di).sum()), int((route | di).sum()))})

    sample = []
    for i, row in enumerate(rows):
        item = {"sha256": sha[i], "fold": int(fold[i]), "label": int(y[i]),
                "p_stage1": base[i], "stage1_error": int(error[i])}
        for j, key in enumerate(MEMBERS): item[key] = member[i, j]
        for name in score_map:
            item["score__"+name] = score_map[name][i]
            item["route__"+name] = int(routes[name][i])
        sample.append(item)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT/"router_metrics.csv", metrics)
    write_csv(OUT/"paired_vs_current_DI.csv", paired)
    write_csv(OUT/"sample_scores_and_routes.csv", sample)
    ranked = sorted(metrics, key=lambda r: (float(r["error_recall_at_25"]), float(r["routing_precision"])), reverse=True)
    lines = ["CARE-Droid strict-2017 committee uncertainty family audit v1", "",
             "Read-only frozen probabilities; exact foldwise 25% budget (200/799).",
             "No future row, Stage-II training, route blending, or threshold search.", ""]
    for r in ranked:
        lines.append(f"{r['router']}: captured={r['captured_errors']}/{r['base_errors']} "
                     f"error_recall={r['error_recall_at_25']:.6f} precision={r['routing_precision']:.6f} "
                     f"highconf_recall={r['highconf_error_recall']:.6f} error_AUC={r['error_score_auc']:.6f}" +
                     ((f" RER={r['repairable_error_recall']:.6f} fixed={r['fixed']} broken={r['broken']} net={r['net']}")
                      if expert is not None else ""))
    best = ranked[0]["router"]
    di_row = next(r for r in metrics if r["router"] == "MeanOpposition_DI25")
    lines += ["", "PRIMARY_CRITERION=Error Recall@25",
              "CURRENT_DI_ERROR_RECALL=" + f"{di_row['error_recall_at_25']:.6f}",
              "BEST_DIAGNOSTIC_SCORE=" + best,
              "Decision rule: do not replace current DI unless a candidate captures more errors at the same budget,"
              " has a favorable exact paired comparison, and is subsequently frozen before any future diagnostic.",
              "Future rows loaded: 0"]
    (OUT/"REPORT.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("COMMITTEE_UNCERTAINTY_FAMILY_SOURCE_AUDIT_DONE")
    print("\n".join(lines))


if __name__ == "__main__": main()

