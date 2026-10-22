#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(os.environ.get("CCBC_PROJECT", "/home/boyuanjie/projects/tasa-droid")).resolve()
OUTPUT = Path(os.environ["CCBC_OUTPUT"]).resolve()
FORMAL_DIR = PROJECT / "scripts/care-droid-formal-diagnostics-v1"
FORMAL_ROOT = PROJECT / "artifacts/care-droid-formal-diagnostics-v1"


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def norm(value): return str(value or "").strip().casefold()


def install_trace(v4, trace_calls):
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
                logit, diag, trace = model.forward_with_trace(global_tensor[index], base_logit[index], one_graph, bag_tensors[index])
                probability.append(float(torch.sigmoid(logit).cpu()))
                diagnostic.append(diag.detach().cpu().numpy())
                attention = trace["attention"].detach().cpu().numpy().astype(np.float32)
                score = model.select_local_evidence(one_graph, bag_tensors[index])[0].score.detach().cpu().numpy().astype(np.float32)
                attentions.append(attention); scores.append(score); critical.append(int(np.argmax(score)))
        trace_calls.append({"attention": attentions, "score": scores, "critical": critical})
        return np.asarray(probability), np.asarray(diagnostic)
    v4.predict_model = traced_predict


def setup():
    if not FORMAL_DIR.is_dir(): raise RuntimeError(f"BLOCKED_BY_CONTRACT: missing {FORMAL_DIR}")
    sys.path.insert(0, str(FORMAL_DIR))
    from a0_adapter import prepare_imports
    prepare_imports()
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    return v4


def aggregate_calls(rows, trace_calls, group_keys):
    attention = [None] * len(rows); score = [None] * len(rows); critical = np.zeros(len(rows), int)
    cursor = 0
    groups = []
    for key in group_keys:
        indices = [i for i, row in enumerate(rows) if key(row)]
        calls = trace_calls[cursor:cursor + 3]; cursor += 3
        if len(calls) != 3 or any(len(call["attention"]) != len(indices) for call in calls):
            raise RuntimeError(f"BLOCKED_BY_CONTRACT: trace group mismatch index={len(groups)} n={len(indices)} calls={len(calls)}")
        for local, index in enumerate(indices):
            lengths = {len(call["attention"][local]) for call in calls}
            if len(lengths) != 1: raise RuntimeError("BLOCKED_BY_CONTRACT: per-seed Host length mismatch")
            value = np.mean([call["attention"][local] for call in calls], axis=0)
            value = value / max(float(value.sum()), 1e-12)
            local_score = np.mean([call["score"][local] for call in calls], axis=0)
            attention[index] = value.astype(np.float32); score[index] = local_score.astype(np.float32)
            critical[index] = int(np.argmax(local_score))
        groups.append({"n": len(indices)})
    if cursor != len(trace_calls) or any(value is None for value in attention):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: incomplete trace cursor={cursor} calls={len(trace_calls)}")
    offsets = [0]; flat_a = []; flat_s = []
    for a, s in zip(attention, score):
        flat_a.extend(a.tolist()); flat_s.extend(s.tolist()); offsets.append(len(flat_a))
    return np.asarray(offsets, np.int64), np.asarray(flat_a, np.float32), np.asarray(flat_s, np.float32), critical


def frozen(device):
    v4 = setup(); trace_calls = []; install_trace(v4, trace_calls)
    import run_frozen_future as runner
    captured = {}
    runner.write_csv = lambda path, rows: captured.__setitem__("rows", list(rows))
    runner.dump = lambda path, value: captured.__setitem__("audit", value)
    sys.argv = [sys.argv[0], "--device", device]
    runner.main()
    rows = captured.get("rows", [])
    if len(rows) != 1989 or len(trace_calls) != 3:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: frozen trace rows={len(rows)} calls={len(trace_calls)}")
    saved = {norm(row["sha256"]): row for row in read_csv(FORMAL_ROOT / "frozen/future_predictions_label_free.csv")}
    difference = max(abs(float(row["expert_prob"]) - float(saved[norm(row["sha256"])]["expert_prob"])) for row in rows)
    if difference > 5e-5: raise RuntimeError(f"BLOCKED_BY_CONTRACT: frozen attention replay prediction mismatch {difference}")
    offsets, attention, score, critical = aggregate_calls(rows, trace_calls, [lambda row: True])
    target = OUTPUT / "cache/future_attention_frozen.npz"; target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, sha256=np.asarray([norm(row["sha256"]) for row in rows]), offsets=offsets,
                        attention=attention, score=score, critical_index=critical)
    (OUTPUT / "future_attention_frozen_audit.json").write_text(json.dumps({
        "rows": len(rows), "trace_calls": len(trace_calls), "prediction_max_abs_difference": difference,
        "formal_assets_modified": False, "mode": "frozen_2017_to_future",
    }, indent=2) + "\n", encoding="utf-8")
    print("CCBC_FROZEN_FUTURE_ATTENTION_DONE", flush=True)


def within(device):
    v4 = setup()
    import run_within_year as runner
    years = (2018, 2019, 2020, 2021); trace_calls = []; install_trace(v4, trace_calls); captured = {}
    def capture_csv(path, rows):
        name = Path(path).name
        captured["rows" if name == "predictions.csv" else "fold_audit"] = list(rows)
    runner.write_csv = capture_csv; runner.dump = lambda path, value: captured.__setitem__("audit", value)
    sys.argv = [sys.argv[0], "--device", device, "--years", *map(str, years)]
    runner.main(); all_rows = captured.get("rows", [])
    if len(all_rows) != 1989 or len(trace_calls) != 60:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: within trace rows={len(all_rows)} calls={len(trace_calls)}")
    saved_rows = []
    for path in (FORMAL_ROOT / "within_2018_2019/predictions.csv", FORMAL_ROOT / "within_2020_2021/predictions.csv"):
        if path.is_file(): saved_rows.extend(read_csv(path))
    saved = {norm(row["sha256"]): row for row in saved_rows}
    difference = max(abs(float(row["expert_prob"]) - float(saved[norm(row["sha256"])]["expert_prob"])) for row in all_rows)
    if difference > 5e-5: raise RuntimeError(f"BLOCKED_BY_CONTRACT: within attention replay mismatch diff={difference}")
    keys = [lambda row, y=year, f=fold: int(row["year"]) == y and int(row["fold"]) == f for year in years for fold in range(1, 6)]
    offsets, attention, score, critical = aggregate_calls(all_rows, trace_calls, keys)
    all_attention = [attention[offsets[i]:offsets[i + 1]] for i in range(len(all_rows))]
    all_score = [score[offsets[i]:offsets[i + 1]] for i in range(len(all_rows))]
    all_critical = critical.tolist()
    print("CCBC_WITHIN_ATTENTION_GROUP_DONE years=2018,2019,2020,2021", flush=True)
    offsets = [0]; flat_a = []; flat_s = []
    for a, s in zip(all_attention, all_score): flat_a.extend(a.tolist()); flat_s.extend(s.tolist()); offsets.append(len(flat_a))
    target = OUTPUT / "cache/future_attention_within.npz"; target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, sha256=np.asarray([norm(row["sha256"]) for row in all_rows]), offsets=np.asarray(offsets, np.int64),
                        attention=np.asarray(flat_a, np.float32), score=np.asarray(flat_s, np.float32), critical_index=np.asarray(all_critical, int))
    print("CCBC_WITHIN_YEAR_ATTENTION_DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("frozen", "within")); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); frozen(args.device) if args.mode == "frozen" else within(args.device)


if __name__ == "__main__": main()
