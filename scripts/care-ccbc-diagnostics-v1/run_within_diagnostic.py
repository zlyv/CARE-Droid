#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np

from ccbc_evidence import prepare_records
from common import OUTPUT, SEED, binary_auc, dump_json, metrics, norm, read_jsonl_gz, top_budget, write_csv
from diagnostic_assets import (
    future_p31_path, load_attention, source_protocol_check, within_formal_rows,
)
from run_frozen_diagnostic import aligned_p31
from run_source import (
    AUXILIARY_DIRECTION_ARMS, choose_router_lambda, count_router,
    crossfit_evidence, direction_probabilities, full_evidence, mechanism,
    rank_scores,
)
from stage1_adapter import strict_nested_committee


VARIANTS = ("real", "contract_shuffle", "role_shuffle")


def main():
    source_protocol_check(); formal = within_formal_rows()
    sha = np.asarray([norm(row["sha256"]) for row in formal]); year = np.asarray([int(row["year"]) for row in formal], int)
    fold = np.asarray([int(row["fold"]) for row in formal], int); y = np.asarray([int(row["label"]) for row in formal], int)
    stage1 = np.asarray([float(row["stage1_prob"]) for row in formal], float)
    di = np.asarray([float(row["router_score"]) for row in formal], float)
    formal_route = np.asarray([int(row["routed"]) for row in formal], bool)
    formal_candidate = np.asarray([float(row["expert_prob"]) for row in formal], float)
    formal_final = np.asarray([float(row["di25_final_prob"]) for row in formal], float)
    p31 = aligned_p31(future_p31_path(), sha)
    scope = {row["sha256"]: row for row in read_jsonl_gz(OUTPUT / "cache/component_scope_future.jsonl.gz")}
    records = prepare_records([scope[sid] for sid in sha], p31)
    attention_map, critical_map = load_attention(OUTPUT / "cache/future_attention_within.npz")
    attention = [attention_map[sid] for sid in sha]; critical = np.asarray([critical_map[sid] for sid in sha], int)
    fields = ("em", "eb", "raw", "delta", "prob", "route_score", "route")
    values = {name: {key: np.full(len(y), np.nan) for key in fields} for name in VARIANTS}
    auxiliary = {name: {key: np.full(len(y), np.nan) for key in ("em", "eb", "raw", "delta", "prob")} for name in AUXILIARY_DIRECTION_ARMS}
    count_score = np.full(len(y), np.nan); model_audit = []
    for year_value in (2018, 2019, 2020, 2021):
        annual = np.flatnonzero(year == year_value)
        for fold_value in range(1, 6):
            test = np.flatnonzero((year == year_value) & (fold == fold_value))
            train = np.asarray([index for index in annual if fold[index] != fold_value], int)
            nested_seed = SEED + year_value * 10000 + fold_value * 100
            _, nested_rf, nested_di = strict_nested_committee(p31[train], y[train], nested_seed, jobs=int(os.environ.get("CCBC_JOBS", "4")))
            train_records = [records[i] for i in train]; test_records = [records[i] for i in test]
            train_attn = [attention[i] for i in train]; test_attn = [attention[i] for i in test]
            real_train = real_test = None
            for variant_index, variant in enumerate(VARIANTS):
                train_ev = crossfit_evidence(train_records, p31[train], y[train], train_attn, critical[train],
                                             nested_seed + 20000 + variant_index * 100, variant, nested_rf)
                test_ev, _ = full_evidence(train_records, test_records, y[train], p31[train], p31[test],
                                           test_attn, critical[test], year_value * 10 + fold_value,
                                           variant, nested_rf, stage1[test])
                em, eb, raw, delta, probability, audit = direction_probabilities(
                    train_ev, test_ev, y[train], nested_rf, stage1[test], "real",
                    nested_seed + 30000 + variant_index * 100,
                )
                di_tr, di_te, c_tr, c_te = rank_scores(nested_di, di[test], nested_rf, stage1[test], train_ev, test_ev)
                lam = choose_router_lambda(di_tr, c_tr, nested_rf, y[train], sha[train])
                score = (1.0 - lam) * di_te + lam * c_te
                route = top_budget(score, sha[test], int(round(len(test) * .25)))
                for key, val in zip(fields, (em, eb, raw, delta, probability, score, route.astype(float))): values[variant][key][test] = val
                model_audit.append({"year": year_value, "fold": fold_value, "variant": variant, "router_lambda": lam, **audit})
                if variant == "real": real_train, real_test = train_ev, test_ev
            for aux_index, name in enumerate(AUXILIARY_DIRECTION_ARMS):
                em, eb, raw, delta, probability, audit = direction_probabilities(
                    real_train, real_test, y[train], nested_rf, stage1[test], name,
                    nested_seed + 40000 + aux_index * 100,
                )
                for key, val in zip(("em", "eb", "raw", "delta", "prob"), (em, eb, raw, delta, probability)):
                    auxiliary[name][key][test] = val
                model_audit.append({"year": year_value, "fold": fold_value, "variant": name, **audit})
            count_score[test] = count_router(real_train, real_test, nested_rf, stage1[test], y[train], nested_seed + 50000)
            print(f"CCBC same-year {year_value} fold {fold_value}/5 done", flush=True)
    for group in (values, auxiliary):
        for name, item in group.items():
            for key, value in item.items():
                if not np.isfinite(value).all(): raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete {name}/{key}")
    if not np.isfinite(count_score).all(): raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete count route score")
    count_route = np.zeros(len(y), bool)
    for year_value in (2018, 2019, 2020, 2021):
        for fold_value in range(1, 6):
            test = np.flatnonzero((year == year_value) & (fold == fold_value))
            count_route[test] = top_budget(count_score[test], sha[test], int(round(len(test) * .25)))
    real_route = values["real"]["route"].astype(bool)
    arms = {
        "A0_FORMAL_DI25_C43": (formal_route, formal_final, formal_candidate, np.zeros(len(y))),
        "A1_FORMAL_DI25_CCBC_DIRECTION": (formal_route, np.where(formal_route, values["real"]["prob"], stage1), values["real"]["prob"], values["real"]["raw"]),
        "A2_CCBC_ROUTE_FORMAL_C43": (real_route, np.where(real_route, formal_candidate, stage1), formal_candidate, np.zeros(len(y))),
        "A3_CCBC_ROUTE_CCBC_DIRECTION": (real_route, np.where(real_route, values["real"]["prob"], stage1), values["real"]["prob"], values["real"]["raw"]),
        "A4_CONTRACT_SHUFFLE": (values["contract_shuffle"]["route"].astype(bool), np.where(values["contract_shuffle"]["route"].astype(bool), values["contract_shuffle"]["prob"], stage1), values["contract_shuffle"]["prob"], values["contract_shuffle"]["raw"]),
        "A5_ROLE_EDGE_SHUFFLE": (values["role_shuffle"]["route"].astype(bool), np.where(values["role_shuffle"]["route"].astype(bool), values["role_shuffle"]["prob"], stage1), values["role_shuffle"]["prob"], values["role_shuffle"]["raw"]),
        "A6_COUNT_ONLY": (count_route, np.where(count_route, auxiliary["count"]["prob"], stage1), auxiliary["count"]["prob"], auxiliary["count"]["raw"]),
        "A7_DECLARATION_ONLY": (real_route, np.where(real_route, auxiliary["declaration"]["prob"], stage1), auxiliary["declaration"]["prob"], auxiliary["declaration"]["raw"]),
        "A8_EXECUTION_ORIGIN_ONLY": (real_route, np.where(real_route, auxiliary["origin"]["prob"], stage1), auxiliary["origin"]["prob"], auxiliary["origin"]["raw"]),
    }
    metric_rows = []; prediction_rows = []
    for name, (route, final, candidate, raw) in arms.items():
        for scope in [2018, 2019, 2020, 2021, "pooled"]:
            selected = np.ones(len(y), bool) if scope == "pooled" else year == scope
            metric_rows.append({"protocol": "independent_same_year_5fold", "scope": scope, "arm": name,
                                **metrics(y[selected], stage1[selected], final[selected], route[selected]),
                                **mechanism(y[selected], stage1[selected], candidate[selected], raw[selected], route[selected])})
        for year_value in (2018, 2019, 2020, 2021):
            for fold_value in range(1, 6):
                selected = (year == year_value) & (fold == fold_value)
                metric_rows.append({"protocol": "independent_same_year_5fold", "scope": year_value, "fold": fold_value, "arm": name,
                                    **metrics(y[selected], stage1[selected], final[selected], route[selected]),
                                    **mechanism(y[selected], stage1[selected], candidate[selected], raw[selected], route[selected])})
        prediction_rows.extend({
            "protocol": "independent_same_year_5fold", "arm": name, "sha256": sha[i], "year": int(year[i]), "fold": int(fold[i]),
            "label": int(y[i]), "stage1_prob": float(stage1[i]), "routed": int(route[i]),
            "candidate_probability": float(candidate[i]), "direction_raw": float(raw[i]),
            "final_probability": float(final[i]), "final_prediction": int(final[i] >= .5),
        } for i in range(len(y)))
    direction_rows = []; pred = stage1 >= .5
    for name, item in {**values, **auxiliary}.items():
        for scope in [2018, 2019, 2020, 2021, "pooled"]:
            selected = np.ones(len(y), bool) if scope == "pooled" else year == scope
            fn = selected & ~pred; fp = selected & pred
            direction_rows.append({"protocol": "independent_same_year_5fold", "scope": scope, "variant": name,
                                   "FN_vs_TN_AUC": binary_auc((y[fn] == 1).astype(int), item["em"][fn]),
                                   "FP_vs_TP_AUC": binary_auc((y[fp] == 0).astype(int), item["eb"][fp]),
                                   "FN_side_n": int(fn.sum()), "FP_side_n": int(fp.sum())})
    write_csv(OUTPUT / "within_metrics.csv", metric_rows); write_csv(OUTPUT / "within_predictions.csv", prediction_rows)
    write_csv(OUTPUT / "within_direction.csv", direction_rows); write_csv(OUTPUT / "within_model_audit.csv", model_audit)
    dump_json(OUTPUT / "within_diagnostic.json", {
        "protocol": "independent_same_year_fixed_5fold", "not_temporal_generalization": True,
        "future_labels_used_for_training": True, "future_used_for_method_selection": False,
        "source_protocol_hash": source_protocol_check()["protocol_hash"], "metrics": metric_rows,
    })
    print("CCBC_WITHIN_YEAR_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__": main()
