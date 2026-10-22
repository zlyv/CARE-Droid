#!/usr/bin/env python3
"""Synthetic pipeline integration; fixtures are not real APK/research evidence."""
from __future__ import annotations
import contextlib
import csv
import hashlib
import importlib.metadata
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from argument_core import analyze_method, aggregate_events
from argument_io import (MODEL_SEEDS, MANIFEST_SEEDS, SPLITS, write_json, read_json)
from argument_probe import run_probe, BASE_ARMS
import run_experiment


def fixture(folder):
    reference = folder / "reference"
    apk_dir = folder / "synthetic-byte-fixtures"
    apk_dir.mkdir()
    source_dir = reference / "source_2017"
    source_dir.mkdir(parents=True)
    ids, lineage, predictions = [], [], []
    labels = np.arange(799) % 2
    for i, label in enumerate(labels):
        # These files are never passed to an APK parser; no-target extraction
        # still checks SHA and end-to-end file/contract handling.
        content = ("SYNTHETIC_NOT_APK_" + str(i)).encode()
        sid = hashlib.sha256(content).hexdigest()
        ids.append(sid)
        path = apk_dir / (sid + ".fixture")
        path.write_bytes(content)
        lineage.append({"sha256": sid, "metadata_year": 2017, "absolute_path": str(path)})
        write_json(reference / f"features/frozen_views/{sid}.json.gz",
                   {"sha256": sid, "year": 2017, "host_ids": [], "pairs_by_host": {}})
    for ms, fp, fn in zip(MANIFEST_SEEDS, (27, 27, 26), (38, 38, 37)):
        negatives = set(np.flatnonzero(labels == 0)[:fp])
        positives = set(np.flatnonzero(labels == 1)[:fn])
        for i, sid in enumerate(ids):
            pred = int(labels[i])
            if i in negatives or i in positives:
                pred = 1 - pred
            predictions.append({"year": 2017, "arm": "B0_W0", "manifest_seed": ms,
                                "sha256": sid, "label": int(labels[i]), "routed": 0,
                                "stage1_probability": .9 if pred else .1,
                                "expert_probability": .5, "final_probability": .9 if pred else .1})
    with (source_dir / "source_oof_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    lineage_path = folder / "lineage.csv"
    with lineage_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, list(lineage[0]))
        writer.writeheader(); writer.writerows(lineage)
    code = np.zeros((799, 113), np.float32)
    code[:, 0] = labels
    destination = reference / "features/w0_repr"
    destination.mkdir()
    for fold, split in enumerate(SPLITS):
        te = np.flatnonzero(np.arange(799) % 5 == fold)
        tr = np.flatnonzero(np.arange(799) % 5 != fold)
        for seed in MODEL_SEEDS:
            np.savez_compressed(destination / f"{split}_{seed}.npz",
                                train_ids=np.asarray(ids)[tr], target_ids=np.asarray(ids)[te],
                                code_train=code[tr], code_target=code[te],
                                logit_train=np.zeros(len(tr), np.float32),
                                logit_target=np.zeros(len(te), np.float32), replay=np.array([0.]))
    return reference, lineage_path, ids


class PipelineTests(unittest.TestCase):
    def test_all_no_targets_and_identical_contract_resume(self):
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix="care-static-fixture-") as temp:
            folder = Path(temp)
            reference, lineage, ids = fixture(folder)
            out = folder / "new-output"
            args = ["all", "--project", str(project), "--reference", str(reference),
                    "--lineage", str(lineage), "--output", str(out), "--workers", "2"]
            real_version = importlib.metadata.version
            # The no-target path does not import Androguard. Mocking its
            # inventory entry does not claim to test the real DEX backend.
            version = lambda name: {"androguard": "4.1.3", "loguru": "0.7.3"}.get(name) or real_version(name)
            with patch("importlib.metadata.version", side_effect=version), contextlib.redirect_stdout(io.StringIO()):
                run_experiment.main(args)
                before = (out / "protocol.json").read_bytes()
                run_experiment.main(args + ["--resume"])
            self.assertEqual(len(list((out / "cache").glob("*.json.gz"))), 799)
            self.assertEqual(read_json(out / "reports/final_decision.json")["status"], "INSUFFICIENT_STATIC_INPUT")
            self.assertEqual(read_json(out / "logs/run_status.json")["status"], "COMPLETE")
            self.assertEqual(read_json(out / "integrity/after.json")["status"], "PASS")
            self.assertEqual(before, (out / "protocol.json").read_bytes())
            self.assertFalse((out / "models").exists())

    def test_fixed_probe_reports_no_gain_when_cascade_cannot_change(self):
        with tempfile.TemporaryDirectory(prefix="care-probe-fixture-") as temp:
            folder = Path(temp)
            reference, _, ids = fixture(folder)
            out = folder / "SYNTHETIC_PROBE_NOT_RESEARCH"
            for i, sid in enumerate(ids):
                command = ("id", "su", "chmod 600 file", "cp a b")[i % 4]
                events = analyze_method([
                    {"offset": 0, "name": "const-string", "registers": [1], "string": command},
                    {"offset": 2, "name": "invoke-virtual", "registers": [0, 1],
                     "reference": "Ljava/lang/Runtime;->exec(Ljava/lang/String;)Ljava/lang/Process;"},
                ])["events"]
                write_json(out / f"cache/{sid}.json.gz", {
                    "summary": aggregate_events(events),
                    "hosts": [{"status": "ANALYZED", "expected_targets": 1,
                               "unrecovered_targets": 0, "events": events}]})
            config = {"contract_sha256": "SYNTHETIC_FIXTURE",
                      "probe": {"claim": "SYNTHETIC PIPELINE TEST; not an APK experiment"}}
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_probe(reference, out, config)
                repeated = run_probe(reference, out, config)
            self.assertEqual(result, repeated)
            self.assertFalse(result["value_supported"])
            self.assertFalse(result["binding_supported"])
            self.assertEqual(result["status"], "NO_SUPPORTED_GAIN_SOURCE")
            self.assertEqual(len(list((out / "models").glob("*/*/*.json"))), 15 * (len(BASE_ARMS) - 1))
            with (out / "reports/source_predictions.csv").open() as f:
                self.assertEqual(sum(1 for _ in f) - 1, 799 * 3 * len(BASE_ARMS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
