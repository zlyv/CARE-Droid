"""Isolated experiment paths, immutable input contracts and source-only rows."""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from argument_core import SCHEMA, CHANNELS, EFFECTS, COARSE, MAX_ARRAY, MAX_STRING, MAX_INSTRUCTIONS

MODEL_SEEDS = (900000, 900001, 900002)
MANIFEST_SEEDS = (20261018, 20261019, 20261020)
SPLITS = ("S1", "S2", "S3", "S4", "S5")
NAME = "care_api_argument_effects_2017_v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if read_json(path) == obj:
            return
        raise FileExistsError("nonidentical output: " + str(path))
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        os.close(fd)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this function's own uncommitted temp.


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader(); w.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def status(out, state, stage, **extra):
    record = {
        "status": state, "stage": stage, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra,
    }
    write_json(out / "logs/run_status.json", record, replace=True)
    with (out / "logs/progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def source_rows(reference):
    rows = read_csv(reference / "source_2017/source_oof_predictions.csv")
    if any(int(row["year"]) != 2017 for row in rows):
        raise ValueError("source file contains a non-2017 row")
    result = [r for r in rows if r["arm"] == "B0_W0"]
    if len(result) != 799 * 3:
        raise ValueError("expected 799 B0 rows for each of three Manifest seeds")
    by_seed, labels = {}, {}
    for seed in MANIFEST_SEEDS:
        group = [r for r in result if int(r["manifest_seed"]) == seed]
        lookup = {r["sha256"].lower(): r for r in group}
        if len(group) != 799 or len(lookup) != 799:
            raise ValueError("source SHA multiplicity")
        by_seed[seed] = lookup
        for sid, row in lookup.items():
            label = int(row["label"])
            if sid in labels and labels[sid] != label:
                raise ValueError("inconsistent source label")
            labels[sid] = label
    if any(set(v) != set(labels) for v in by_seed.values()) or sorted(labels.values()).count(0) != 400 or sum(labels.values()) != 399:
        raise ValueError("source cohort contract")
    return labels, by_seed


def fold_representations(reference, source_ids):
    result, membership = {}, {}
    for split in SPLITS:
        expected_train = expected_target = None
        for seed in MODEL_SEEDS:
            path = reference / f"features/w0_repr/{split}_{seed}.npz"
            with np.load(path, allow_pickle=False) as z:
                obj = {k: z[k] for k in z.files}
            tr, te = list(map(str, obj["train_ids"])), list(map(str, obj["target_ids"]))
            if len(tr) != len(set(tr)) or len(te) != len(set(te)) or set(tr) & set(te) or set(tr) | set(te) != set(source_ids):
                raise ValueError("fold membership: " + split)
            if expected_train is not None and (tr != expected_train or te != expected_target):
                raise ValueError("fold order differs by W0 seed")
            expected_train, expected_target = tr, te
            if obj["code_train"].shape != (len(tr), 113) or obj["code_target"].shape != (len(te), 113):
                raise ValueError("W0 interface width")
            if not all(np.isfinite(obj[k]).all() for k in ("code_train", "code_target", "logit_train", "logit_target", "replay")):
                raise ValueError("nonfinite W0 representation")
            if obj["replay"].max() > 1e-6:
                raise ValueError("saved W0 replay failed")
            result[(split, seed)] = obj
        for sid in expected_target:
            if sid in membership:
                raise ValueError("source held out in multiple folds")
            membership[sid] = split
    if set(membership) != set(source_ids):
        raise ValueError("missing held-out source SHA")
    return result, membership


def protocol(project, reference, lineage, out):
    labels, _ = source_rows(reference)
    fold_representations(reference, labels)
    folder = Path(__file__).parent
    sources = sorted(folder.glob("*.py")) + sorted(folder.glob("*.sh"))
    sources += [p for p in (folder / "README.md", folder / "requirements.txt") if p.exists()]
    inputs = [reference / "source_2017/source_oof_predictions.csv", Path(lineage)]
    inputs += [reference / f"features/w0_repr/{split}_{seed}.npz" for split in SPLITS for seed in MODEL_SEEDS]
    inputs += [reference / f"features/frozen_views/{sid}.json.gz" for sid in sorted(labels)]
    paths = inputs + sources
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if out == path.resolve() or out in path.resolve().parents:
            raise ValueError("output would contain a protected input")
    rows = read_csv(lineage)
    lookup = {r["sha256"].lower(): r for r in rows}
    if len(rows) != 799 or set(lookup) != set(labels):
        raise ValueError("lineage must match exactly the frozen 799 source SHAs")
    for sid, row in lookup.items():
        year = row.get("metadata_year", row.get("year"))
        if str(year) != "2017" or not Path(row["absolute_path"]).is_file():
            raise ValueError("missing APK path or non-2017 lineage: " + sid)
    config = {
        "schema": SCHEMA, "project": str(project), "reference": str(reference), "lineage": str(lineage),
        "source_apks": 799, "source_years": [2017], "future_allowed": False,
        "analysis": "static basic-block-local command argument forms; no runtime or malware verdict",
        "channels": CHANNELS, "effects": EFFECTS, "coarse": COARSE,
        "limits": {"string": MAX_STRING, "array": MAX_ARRAY, "method_instructions": MAX_INSTRUCTIONS},
        "feasibility": {"minimum_resolved_apks": 30, "minimum_routed_resolved_apks": 10,
                        "minimum_resolved_event_fraction": 0.20},
        "probe": {"optimizer": "fixed L2 offset logistic, scipy L-BFGS-B",
                  "ridge": 0.1, "maximum_iterations": 500, "bootstrap_repeats": 2000,
                  "feature_selection": "none", "training_rows": "all outer-train APKs",
                  "threshold": 0.5, "null_repeats": 5,
                  "primary": "VALUE against W0/CODE/COARSE and all five EFFECT_SHUFFLE repeats",
                  "secondary": "BOUND against W0/VALUE/PLACEBO and all five BINDING_SHUFFLE repeats",
                  "claim": "fixed-readout conditional analysis of a historically examined source cohort and frozen router; not a newly nested end-to-end evaluation"},
        "assets": [digest(p) for p in paths],
    }
    config["contract_sha256"] = signature(config)
    return config, lookup


def verify_inputs(out):
    config = read_json(out / "protocol.json")
    changed = [item["path"] for item in config["assets"] if digest(item["path"]) != item]
    if changed:
        raise RuntimeError("input/code hashes changed: " + repr(changed))
    return config
