#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ccbc_evidence import prepare_records
from common import (
    MAX_DELTA, OUTPUT, SEED, binary_auc, discover_p31, dump_json,
    load_formal_rows, metrics, norm, read_jsonl_gz, top_budget, write_csv,
)
from diagnostic_assets import (
    SOURCE_OUTPUT, frozen_formal_rows, future_cohort, future_p31_path,
    load_attention, source_protocol_check,
)
from run_source import (
    AUXILIARY_DIRECTION_ARMS, choose_router_lambda, count_router,
    crossfit_evidence, direction_probabilities, full_evidence, mechanism,
    rank_scores, route_metrics,
)


VARIANTS = ("real", "contract_shuffle", "role_shuffle", "permission")


def aligned_p31(path, ids):
    with np.load(path, allow_pickle=False) as z:
        stored = [norm(value) for value in z["sha256"]]; matrix = np.asarray(z["role_prevalence"], float)
    pos = {sid: i for i, sid in enumerate(stored)}
    if any(sid not in pos for sid in ids): raise RuntimeError("BLOCKED_BY_CONTRACT: P31 SHA alignment failed")
    return matrix[[pos[sid] for sid in ids]]


def route_by_year(score, sha, year):
    output = np.zeros(len(score), bool)
    for value in (2018, 2019, 2020, 2021):
        selected = np.flatnonzero(year == value)
        output[selected] = top_budget(np.asarray(score)[selected], np.asarray(sha)[selected], int(round(len(selected) * .25)))
    return output


def main():
    source_protocol_check()
    source = load_formal_rows(); future = frozen_formal_rows(); cohort = future_cohort()
    source_sha = np.asarray([row["sha256"] for row in source]); future_sha = np.asarray([norm(row["sha256"]) for row in future])
    future_year = np.asarray([int(row["year"]) for row in future], int)
    year_by_sha = dict(cohort)
    if any(year_by_sha[sid] != value for sid, value in zip(future_sha, future_year)):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen future year alignment failed")
    source_y = np.asarray([row["label"] for row in source], int); future_y = np.asarray([int(row["label"]) for row in future], int)
    source_stage1 = np.asarray([row["stage1_prob"] for row in source], float)
    source_di = np.asarray([row["router_score"] for row in source], float)
    future_stage1 = np.asarray([float(row["stage1_prob"]) for row in future], float)
    future_di = np.asarray([float(row["router_score"]) for row in future], float)
    formal_route = np.asarray([int(row["routed"]) for row in future], bool)
    formal_candidate = np.asarray([float(row["expert_prob"]) for row in future], float)
    formal_final = np.asarray([float(row["di25_final_prob"]) for row in future], float)
    source_p31 = aligned_p31(discover_p31(), source_sha)
    future_p31 = aligned_p31(future_p31_path(), future_sha)
    source_scope = {row["sha256"]: row for row in read_jsonl_gz(SOURCE_OUTPUT / "cache/component_scope_2017.jsonl.gz")}
    future_scope = {row["sha256"]: row for row in read_jsonl_gz(OUTPUT / "cache/component_scope_future.jsonl.gz")}
    source_records = prepare_records([source_scope[sid] for sid in source_sha], source_p31)
    future_records = prepare_records([future_scope[sid] for sid in future_sha], future_p31)
    source_attention, source_critical = load_attention(SOURCE_OUTPUT / "cache/formal_attention.npz")
    future_attention, future_critical = load_attention(OUTPUT / "cache/future_attention_frozen.npz")
    source_attn = [source_attention[sid] for sid in source_sha]; source_crit = np.asarray([source_critical[sid] for sid in source_sha], int)
    future_attn = [future_attention[sid] for sid in future_sha]; future_crit = np.asarray([future_critical[sid] for sid in future_sha], int)
    values = {}; evidence_real = None; audits = []
    for index, variant in enumerate(VARIANTS):
        train_ev = crossfit_evidence(source_records, source_p31, source_y, source_attn, source_crit,
                                     SEED + 71000 + index * 100, variant, source_stage1)
        test_ev, model = full_evidence(source_records, future_records, source_y, source_p31, future_p31,
                                       future_attn, future_crit, 90 + index, variant, source_stage1, future_stage1)
        direction_variant = "declaration" if variant == "permission" else "real"
        em, eb, raw, delta, probability, audit = direction_probabilities(
            train_ev, test_ev, source_y, source_stage1, future_stage1, direction_variant,
            SEED + 72000 + index * 100,
        )
        di_tr, di_te, c_tr, c_te = rank_scores(source_di, future_di, source_stage1, future_stage1, train_ev, test_ev)
        lam = choose_router_lambda(di_tr, c_tr, source_stage1, source_y, source_sha)
        score = (1.0 - lam) * di_te + lam * c_te; route = route_by_year(score, future_sha, future_year)
        values[variant] = {"train": train_ev, "test": test_ev, "em": em, "eb": eb, "raw": raw,
                           "delta": delta, "prob": probability, "route_score": score, "route": route, "lambda": lam}
        audits.append({"variant": variant, "router_lambda": lam, **audit})
        if variant == "real": evidence_real = test_ev
        print(f"CCBC_FROZEN_VARIANT_DONE {variant}", flush=True)
    real_train, real_test = values["real"]["train"], values["real"]["test"]
    auxiliary = {}
    for index, name in enumerate(AUXILIARY_DIRECTION_ARMS):
        em, eb, raw, delta, probability, audit = direction_probabilities(
            real_train, real_test, source_y, source_stage1, future_stage1, name, SEED + 73000 + index * 100,
        )
        auxiliary[name] = {"em": em, "eb": eb, "raw": raw, "delta": delta, "prob": probability}; audits.append({"variant": name, **audit})
    count_score = count_router(real_train, real_test, source_stage1, future_stage1, source_y, SEED + 74000)
    count_route = route_by_year(count_score, future_sha, future_year)
    real_route = values["real"]["route"]
    arms = {
        "A0_FORMAL_DI25_C43": (formal_route, formal_final, formal_candidate, np.zeros(len(future))),
        "A1_FORMAL_DI25_CCBC_DIRECTION": (formal_route, np.where(formal_route, values["real"]["prob"], future_stage1), values["real"]["prob"], values["real"]["raw"]),
        "A2_CCBC_ROUTE_FORMAL_C43": (real_route, np.where(real_route, formal_candidate, future_stage1), formal_candidate, np.zeros(len(future))),
        "A3_CCBC_ROUTE_CCBC_DIRECTION": (real_route, np.where(real_route, values["real"]["prob"], future_stage1), values["real"]["prob"], values["real"]["raw"]),
        "A4_CONTRACT_SHUFFLE": (values["contract_shuffle"]["route"], np.where(values["contract_shuffle"]["route"], values["contract_shuffle"]["prob"], future_stage1), values["contract_shuffle"]["prob"], values["contract_shuffle"]["raw"]),
        "A5_ROLE_EDGE_SHUFFLE": (values["role_shuffle"]["route"], np.where(values["role_shuffle"]["route"], values["role_shuffle"]["prob"], future_stage1), values["role_shuffle"]["prob"], values["role_shuffle"]["raw"]),
        "A6_COUNT_ONLY": (count_route, np.where(count_route, auxiliary["count"]["prob"], future_stage1), auxiliary["count"]["prob"], auxiliary["count"]["raw"]),
        "A7_DECLARATION_ONLY": (real_route, np.where(real_route, auxiliary["declaration"]["prob"], future_stage1), auxiliary["declaration"]["prob"], auxiliary["declaration"]["raw"]),
        "A8_EXECUTION_ORIGIN_ONLY": (real_route, np.where(real_route, auxiliary["origin"]["prob"], future_stage1), auxiliary["origin"]["prob"], auxiliary["origin"]["raw"]),
    }
    metrics_rows = []; prediction_rows = []
    for name, (route, final, candidate, raw) in arms.items():
        for scope in [2018, 2019, 2020, 2021, "pooled"]:
            selected = np.ones(len(future), bool) if scope == "pooled" else future_year == scope
            metrics_rows.append({"protocol": "frozen_2017_to_future", "scope": scope, "arm": name,
                                 **metrics(future_y[selected], future_stage1[selected], final[selected], route[selected]),
                                 **mechanism(future_y[selected], future_stage1[selected], candidate[selected], raw[selected], route[selected])})
        prediction_rows.extend({
            "protocol": "frozen_2017_to_future", "arm": name, "sha256": future_sha[i], "year": int(future_year[i]),
            "label": int(future_y[i]), "stage1_prob": float(future_stage1[i]), "routed": int(route[i]),
            "candidate_probability": float(candidate[i]), "direction_raw": float(raw[i]),
            "final_probability": float(final[i]), "final_prediction": int(final[i] >= .5),
        } for i in range(len(future)))
    direction_rows = []
    pred = future_stage1 >= .5
    for name, item in {**values, **auxiliary}.items():
        fn = ~pred; fp = pred
        direction_rows.append({
            "protocol": "frozen_2017_to_future", "variant": name,
            "FN_vs_TN_AUC": binary_auc((future_y[fn] == 1).astype(int), item["em"][fn]),
            "FP_vs_TP_AUC": binary_auc((future_y[fp] == 0).astype(int), item["eb"][fp]),
            "FN_side_n": int(fn.sum()), "FN_n": int(np.sum(future_y[fn] == 1)),
            "FP_side_n": int(fp.sum()), "FP_n": int(np.sum(future_y[fp] == 0)),
        })
    write_csv(OUTPUT / "frozen_metrics.csv", metrics_rows); write_csv(OUTPUT / "frozen_predictions.csv", prediction_rows)
    write_csv(OUTPUT / "frozen_direction.csv", direction_rows); write_csv(OUTPUT / "frozen_model_audit.csv", audits)
    dump_json(OUTPUT / "frozen_diagnostic.json", {
        "protocol": "frozen_2017_to_2018_2021", "source_protocol_hash": source_protocol_check()["protocol_hash"],
        "future_used_for_selection": False, "future_rows": 1989,
        "metrics": metrics_rows, "direction": direction_rows,
        "mapping_coverage": {"mean": float(np.mean([row["mapping_coverage"] for row in evidence_real]))},
    })
    print("CCBC_FROZEN_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__": main()
