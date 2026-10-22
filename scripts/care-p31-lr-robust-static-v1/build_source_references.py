#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

import numpy as np

from common import (ACTIVE_ARMS, MEMBERS, OUTPUT, SOURCE_MEMBER, SOURCE_P31, classification_metrics,
    dump_json, grouped_top25, mean_opposition, norm, read_csv, write_csv)
from stage1_core import fit_calibrated_committee
from transforms import feature_names


def route_metrics(y, base, member, sha, folds):
    score = mean_opposition(member, base); route = grouped_top25(score, sha, folds)
    pred = base >= .5; error = pred != y; fn = (y == 1) & (~pred); fp = (y == 0) & pred
    return score, route, {"routed": int(route.sum()), "captured_errors": int(np.sum(route & error)),
        "error_recall25": float(np.sum(route & error) / max(error.sum(), 1)),
        "route_precision": float(np.sum(route & error) / max(route.sum(), 1)),
        "captured_FN": int(np.sum(route & fn)), "total_FN": int(fn.sum()),
        "captured_FP": int(np.sum(route & fp)), "total_FP": int(fp.sum())}


def main():
    rows = read_csv(SOURCE_MEMBER); sha = np.asarray([norm(row["sha256"]) for row in rows])
    y = np.asarray([int(float(row["label"])) for row in rows], int)
    fold = np.asarray([int(float(row["fold"])) for row in rows], int)
    formal_member = np.asarray([[float(row[f"p_P31__{name}"]) for name in MEMBERS] for row in rows], float)
    with np.load(SOURCE_P31, allow_pickle=False) as archive:
        p31_ids = [norm(value) for value in archive["sha256"]]
        p31_values = np.asarray(archive["role_prevalence"], float)
        cache_keys = sorted(archive.files)
    position = {sid: index for index, sid in enumerate(p31_ids)}
    x = p31_values[[position[sid] for sid in sha]]
    aggregate = []; per_fold = []
    for arm_index, arm in enumerate(ACTIVE_ARMS):
        if arm == "L0_RAW":
            member = formal_member.copy()
        else:
            member = np.full((len(y), 4), np.nan)
            for current_fold in range(1, 6):
                train = np.flatnonzero(fold != current_fold); test = np.flatnonzero(fold == current_fold)
                seed = 20261009 + arm_index * 100000 + current_fold * 1000
                member[test] = fit_calibrated_committee(x[train], y[train], x[test], seed, arm,
                    inner_folds=3, jobs=4)
                print(f"P31_LR_ROBUST_SOURCE_STAGE1 arm={arm} fold={current_fold}/5", flush=True)
        if not np.isfinite(member).all():
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete source member probabilities arm={arm}")
        base = member[:, 0]; score, route, route_value = route_metrics(y, base, member, sha, fold)
        target = OUTPUT / "references" / arm / "source_predictions.csv"
        write_csv(target, [{"sha256": sha[i], "label": int(y[i]), "fold": int(fold[i]),
            "Global24__prob": float(base[i]), "Global24__A25__JointS7_ordinary__prob": float(base[i]),
            "A25__score": float(score[i]), "A25__routed": int(route[i]),
            "stage1_arm": arm, "router_anchor": "LR", **{f"p_member_{MEMBERS[j]}": float(member[i,j]) for j in range(4)}}
            for i in range(len(y))])
        metrics = classification_metrics(y, base)
        aggregate.append({"protocol": "source_2017_fixed_oof", "scope": "pooled", "arm": arm,
            **metrics, **route_value, "feature_width": len(feature_names(arm))})
        for current_fold in range(1, 6):
            selected = fold == current_fold
            _, _, local_route_value = route_metrics(y[selected], base[selected], member[selected],
                sha[selected], np.zeros(selected.sum(), int))
            per_fold.append({"protocol": "source_2017_fixed_oof", "fold": current_fold, "arm": arm,
                **classification_metrics(y[selected], base[selected]), **local_route_value})
    l0 = next(row for row in aggregate if row["arm"] == "L0_RAW")
    if abs(l0["BA"] - .8584774436090226) > 1e-12 or l0["errors"] != 113:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal raw P31-LR did not reproduce")
    ranking = [row["arm"] for row in sorted(aggregate, key=lambda row: (-row["BA"], -row["AUC"], row["arm"]))]
    write_csv(OUTPUT / "source_stage1_metrics.csv", aggregate)
    write_csv(OUTPUT / "source_stage1_fold_metrics.csv", per_fold)
    frozen = {"ranking": ranking, "primary_metric": "2017 fixed-fold OOF BA", "future_used": False,
        "selected_source_leader": ranking[0], "selection_is_diagnostic_not_automatic_formal_change": True}
    dump_json(OUTPUT / "source_stage1_ranking_frozen.json", frozen)
    text = json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (OUTPUT / "source_stage1_ranking_hash.txt").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n", encoding="utf-8")
    dump_json(OUTPUT / "p31_cache_audit.json", {"keys": cache_keys, "shape": list(x.shape),
        "row_sum_min": float(x.sum(1).min()), "row_sum_max": float(x.sum(1).max()),
        "raw_intensity_available": any(key not in {"sha256", "role_prevalence"} for key in cache_keys)})
    print("P31_LR_ROBUST_SOURCE_REFERENCES_DONE leader=" + ranking[0], flush=True)


if __name__ == "__main__":
    main()
