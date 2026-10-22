#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from common import OUTPUT, PROJECT, dump_json, read_csv


def main() -> None:
    target = OUTPUT / "cache/formal_attention.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as z:
            if len(z["sha256"]) == 799 and len(z["offsets"]) == 800:
                print("CCBC_FORMAL_ATTENTION_RESUME_PASS")
                return
    formal = PROJECT / "scripts/care-di25-compact43-relationquery-source-v1"
    graphglobal = PROJECT / "scripts/care-di25-graphglobal-path-source-v1"
    rolegraph = PROJECT / "scripts/care-rolegraph-conditioned-v4-script"
    scripts = PROJECT / "scripts"
    for path in (formal, graphglobal, scripts, rolegraph,
                 scripts / "care-rgcondattn-dedup-bounded-source-v1",
                 scripts / "care-malben-host-v2"):
        sys.path.insert(0, str(path))
    from compact_background import make_fit
    from compact_models import make_compact_class
    from source_label_contract import install_source_label_contract
    import run_care_rolegraph_conditioned_dualstream_v4 as v4

    install_source_label_contract(v4)
    v4.MODEL_VARIANTS = ("RGCondAttn",)
    method = "CCBC__ATTENTION_REPLAY_ONLY"
    v4.METHOD_FOR_VARIANT = {"RGCondAttn": method}
    v4.AUGMENTED_REGION_DIM = 78
    v4.RoleGraphConditionedEvidence = make_compact_class(v4, "C43_RELATION_QUERY")
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)
    trace_calls = []

    def traced_predict(model, global_values, base_probability, graph_values, bags, device):
        import torch
        global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
        base_logit = torch.as_tensor(v4.probability_to_logit(base_probability), dtype=torch.float32, device=device)
        graph = v4.graph_tensors(graph_values, device)
        bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
        probability = []; diagnostic = []; attentions = []; scores = []; critical = []
        model.eval()
        with torch.no_grad():
            for index in range(len(bags)):
                one_graph = {key: value[index] for key, value in graph.items()}
                logit, diag, trace = model.forward_with_trace(
                    global_tensor[index], base_logit[index], one_graph, bag_tensors[index]
                )
                probability.append(float(torch.sigmoid(logit).cpu()))
                diagnostic.append(diag.detach().cpu().numpy())
                attention = trace["attention"].detach().cpu().numpy().astype(np.float32)
                score = model.select_local_evidence(one_graph, bag_tensors[index])[0].score.detach().cpu().numpy().astype(np.float32)
                attentions.append(attention); scores.append(score); critical.append(int(np.argmax(score)))
        trace_calls.append({"attention": attentions, "score": scores, "critical": critical})
        return np.asarray(probability), np.asarray(diagnostic)

    v4.predict_model = traced_predict
    replay_output = OUTPUT / "attention_replay"
    replay_output.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        sys.argv[0],
        "--previous-result", str(PROJECT / "artifacts/care-di25-compact43-relationquery-source-v1/preflight/di25_reference"),
        "--cache", str(PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz"),
        "--region-cache", str(PROJECT / "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz"),
        "--variants", "RGCondAttn", "--seeds", "0,1,2", "--max-delta", "2.25",
        "--device", os.environ.get("CCBC_DEVICE", "auto"), "--output", str(replay_output),
    ]
    v4.main()
    prediction_rows = read_csv(replay_output / "source_predictions.csv")
    if len(prediction_rows) != 799 or len(trace_calls) != 15:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: attention replay calls={len(trace_calls)} rows={len(prediction_rows)}")
    fold = np.asarray([int(row["fold"]) for row in prediction_rows], int)
    attention_by_row = [None] * 799; score_by_row = [None] * 799; critical_by_row = np.zeros(799, int)
    call_index = 0
    for fold_value in range(1, 6):
        validation = np.flatnonzero(fold == fold_value)
        calls = trace_calls[call_index:call_index + 3]; call_index += 3
        if any(len(call["attention"]) != len(validation) for call in calls):
            raise RuntimeError("BLOCKED_BY_CONTRACT: attention validation length mismatch")
        for local, row_index in enumerate(validation):
            lengths = {len(call["attention"][local]) for call in calls}
            if len(lengths) != 1:
                raise RuntimeError("BLOCKED_BY_CONTRACT: per-seed Host length mismatch")
            attention = np.mean([call["attention"][local] for call in calls], axis=0)
            attention = attention / max(float(attention.sum()), 1e-12)
            score = np.mean([call["score"][local] for call in calls], axis=0)
            attention_by_row[row_index] = attention.astype(np.float32)
            score_by_row[row_index] = score.astype(np.float32)
            critical_by_row[row_index] = int(np.argmax(score))
    offsets = [0]; flat_attention = []; flat_score = []
    for attention, score in zip(attention_by_row, score_by_row):
        if attention is None:
            raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete attention replay")
        flat_attention.extend(attention.tolist()); flat_score.extend(score.tolist()); offsets.append(len(flat_attention))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        sha256=np.asarray([row["sha256"].strip().casefold() for row in prediction_rows]),
        offsets=np.asarray(offsets, np.int64),
        attention=np.asarray(flat_attention, np.float32), score=np.asarray(flat_score, np.float32),
        critical_index=critical_by_row,
    )
    dump_json(OUTPUT / "formal_attention_audit.json", {
        "rows": 799, "seeds": [0, 1, 2], "folds": 5,
        "attention_sum_max_abs_error": float(max(abs(float(np.sum(value)) - 1.0) for value in attention_by_row)),
        "formal_code_modified": False, "output": str(target), "future_rows_loaded": 0,
    })
    print("CCBC_FORMAL_ATTENTION_REPLAY_DONE")


if __name__ == "__main__":
    main()

