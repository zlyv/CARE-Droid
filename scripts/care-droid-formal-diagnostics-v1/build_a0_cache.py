#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np

from common import ROOT, SOURCE_A0_CACHE, load_future_cohort, norm, dump


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def bags(data):
    ids = [norm(x) for x in data["sha256"]]
    values = np.asarray(data["host_values"], np.float32)
    offsets = np.asarray(data["host_offsets"], int)
    return ids, [values[offsets[i]:offsets[i + 1]] for i in range(len(ids))]


def main():
    target = ROOT / "cache/a0_role_2017_2021.npz"
    if target.is_file():
        print("[resume] combined A0 future cache exists")
        return
    resolved = json.loads((ROOT / "resolved_inputs.json").read_text())
    source = load(SOURCE_A0_CACHE); future = load(resolved["future_base_relation_cache"])
    source_ids, source_bags = bags(source); future_ids_all, future_bags_all = bags(future)
    future_pos = {sid: i for i, sid in enumerate(future_ids_all)}
    _, cohort = load_future_cohort(); future_ids = [sid for sid, _ in cohort]
    if any(sid not in future_pos for sid in future_ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: future relation cache coverage changed")
    selected = []
    real = []; empty = []
    future_real = np.asarray(future["real_host_count"], int); future_empty = np.asarray(future["host_was_empty"])
    for sid in future_ids:
        i = future_pos[sid]; bag = np.asarray(future_bags_all[i], np.float32)
        if bag.ndim != 2 or bag.shape[1] != 105:
            raise RuntimeError("BLOCKED_BY_CONTRACT: future C43 bag width is not 105")
        selected.append(np.concatenate([bag, np.zeros((len(bag), 64), np.float32)], axis=1))
        real.append(int(future_real[i])); empty.append(future_empty[i])
    all_bags = source_bags + selected
    offsets = [0]; values = []
    for bag in all_bags:
        if np.asarray(bag).shape[1] != 169:
            raise RuntimeError("BLOCKED_BY_CONTRACT: combined A0 width mismatch")
        values.append(np.asarray(bag, np.float32)); offsets.append(offsets[-1] + len(bag))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, sha256=np.asarray(source_ids + future_ids),
        host_values=np.concatenate(values), host_offsets=np.asarray(offsets, np.int64),
        real_host_count=np.concatenate([np.asarray(source["real_host_count"], int), np.asarray(real, int)]),
        host_was_empty=np.concatenate([np.asarray(source["host_was_empty"]), np.asarray(empty)]))
    dump(ROOT / "cache/a0_cache_audit.json", {
        "rows": len(source_ids) + len(future_ids), "source_rows": len(source_ids),
        "future_rows": len(future_ids), "stored_width": 169,
        "construction": "frozen source A0 cache plus future Compact43+Role31+Relation31 with the same 64 structural zero sidecar",
        "future_labels_loaded": False,
    })
    print("A0_FUTURE_CACHE_PASS")
    print(f"rows={len(source_ids)+len(future_ids)} stored_width=169 future_labels_loaded=false")


if __name__ == "__main__":
    main()
