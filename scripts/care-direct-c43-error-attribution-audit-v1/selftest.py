#!/usr/bin/env python3
"""Deterministic unit and end-to-end synthetic regression tests for the source-only audit."""
from __future__ import annotations

import csv
import hashlib
import zipfile
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import run_audit as audit


def labels_with_inversions(pos: int, neg: int, inversions: int) -> list[int]:
    # Descending-score labels start with perfect pos-above-neg ranking.
    seq = [1] * pos + [0] * neg
    for _ in range(inversions):
        i = next((i for i in range(len(seq) - 1) if seq[i:i+2] == [1, 0]), None)
        assert i is not None
        seq[i], seq[i+1] = seq[i+1], seq[i]
    return seq


def write_fake_data(temp: Path) -> tuple[Path, Path]:
    rows = []
    i = 0
    def add(y: int, stage1: int, routed: int, n: int, expert: int = -1) -> None:
        nonlocal i
        for _ in range(n):
            i += 1
            rows.append({"sha256": f"{i:064x}", "label": y,
                         "Global24__prob": .91 if stage1 else .09,
                         "A25__routed": routed, audit.EXPERT_PROB: .91 if expert == 1 else .09,
                         "fold": (i - 1) % 5 + 1})
    # Benign: stage1 TN356, FP44; routed TN60 (8 broken), FP43 (30 fixed).
    add(0, 0, 1, 8, 1)
    add(0, 0, 1, 52, 0)
    add(0, 0, 0, 296, 0)
    add(0, 1, 1, 30, 0)
    add(0, 1, 1, 13, 1)
    add(0, 1, 0, 1, 1)
    # Malware: stage1 FN53, TP346; routed FN41 (19 fixed), TP56 (9 broken).
    add(1, 0, 1, 19, 1)
    add(1, 0, 1, 22, 0)
    add(1, 0, 0, 12, 0)
    add(1, 1, 1, 9, 0)
    add(1, 1, 1, 47, 1)
    add(1, 1, 0, 290, 1)
    assert len(rows) == 799
    # Generate frozen known-rank AUC exactly, while preserving threshold-flip counts.
    # PREDICTED_BENIGN: low 22 pos/52 neg with 425 inversions; high 19 pos/8 neg.
    # PREDICTED_MALWARE: low 9 pos/30 neg with 256 inversions; high 47 pos/13 neg.
    configs = [
        (0, 0, 22, 52, 425), (0, 1, 19, 8, 0),
        (1, 0, 9, 30, 256), (1, 1, 47, 13, 0),
    ]
    for stage, expert, pos, neg, inversions in configs:
        chosen = [r for r in rows if int(r["A25__routed"]) and int(r["Global24__prob"] >= .5) == stage
                  and int(r[audit.EXPERT_PROB] >= .5) == expert]
        assert len(chosen) == pos + neg, (stage, expert, len(chosen))
        p = iter(r for r in chosen if r["label"] == 1)
        n = iter(r for r in chosen if r["label"] == 0)
        seq = labels_with_inversions(pos, neg, inversions)
        scores = np.linspace(.49, .01, len(seq)) if expert == 0 else np.linspace(.99, .51, len(seq))
        for label, score in zip(seq, scores):
            item = next(p if label == 1 else n)
            item[audit.EXPERT_PROB] = float(score)
    source = temp / "source_predictions.csv"
    with source.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arch = temp / "architecture.json"
    arch.write_text(json.dumps({"model": "Direct-C43 test data; not a real model"}), encoding="utf-8")
    return source, arch


def main() -> None:
    assert audit.auc_binary([0, 0, 1, 1], [.1, .2, .8, .9]) == 1
    assert audit.auc_binary([0, 1], [.4, .4]) == .5
    assert labels_with_inversions(2, 2, 3) == [0, 1, 0, 1]
    with tempfile.TemporaryDirectory(prefix="eca_selftest_") as root:
        temp = Path(root)
        project = temp / "project"
        project.mkdir()
        source, arch = write_fake_data(temp)
        # The server runner exports ECA_OUTPUT for the real audit. Never inherit
        # that path into this synthetic temporary-project subprocess: doing so
        # correctly triggers run_audit.safe_path() and blocks the self-test.
        # Isolate every optional input that could have leaked from the caller.
        env = os.environ.copy()
        for name in ("ECA_OUTPUT", "ECA_FOLDS", "ECA_EMBEDDINGS", "ECA_STATIC_FEATURES",
                     "ECA_STAGE1_REFERENCE", "ECA_FEATURE_COLUMNS", "ECA_APK_INDEX"):
            env.pop(name, None)
        env.update({"ECA_PROJECT": str(project),
                    "ECA_OUTPUT": str(project / "outputs" / "care-direct-c43-error-attribution-audit-v1"),
                    "ECA_SOURCE": str(source),
                    "ECA_ARCHITECTURE": str(arch),
                    "ECA_SOURCE_PROB_COL": "Global24__prob", "ECA_SOURCE_ROUTED_COL": "A25__routed",
                    "ECA_EXPERT_PROB_COL": audit.EXPERT_PROB})
        completed = subprocess.run([sys.executable, str(Path(__file__).parent / "run_audit.py")],
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, timeout=40)
        assert completed.returncode == 0, completed.stdout
        out = project / "outputs" / "care-direct-c43-error-attribution-audit-v1"
        result = json.loads((out / "final_decision.json").read_text())
        assert result["source_contract"]["group_counts"]["ROUTED_UNFIXED"] == 35
        assert result["source_contract"]["group_counts"]["ROUTED_BROKEN"] == 17
        assert result["formal_integrity"] == "PASS"
        assert result["future_rows_loaded"] == 0
        assert len(audit.read_csv(out / "manual_review_blinded_template.csv")) == 52
        assert abs(result["routed_auc"]["FN_vs_TN"] - .7556910569105691) < 1e-10
        assert abs(result["routed_auc"]["TP_vs_FP"] - .8450996677740864) < 1e-10
        # Rerunning must refuse overwriting an already completed experiment.
        again = subprocess.run([sys.executable, str(Path(__file__).parent / "run_audit.py")],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=40)
        assert again.returncode != 0 and "refusing to overwrite" in again.stdout
        # Validate an empty blinded review: it must not fabricate attribution.
        review = subprocess.run([sys.executable, str(Path(__file__).parent / "validate_reviews.py")],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=40)
        assert review.returncode == 0, review.stdout
        reviewed = json.loads((out / "manual_review_validation.json").read_text())
        assert reviewed["verified_root_causes"] == 0
        assert reviewed["counts_by_assessment"] == {"UNREVIEWED": 52}
        # Exercise optional source-only embeddings, descriptors, and hash-verified APK.
        with source.open(newline="", encoding="utf-8") as f:
            simulated_rows = list(csv.DictReader(f))
        apk = temp / "frozen_source.apk"
        with zipfile.ZipFile(apk, "w") as z:
            z.writestr("AndroidManifest.xml", b"manifest")
            z.writestr("classes.dex", b"dex\n035\0DexClassLoader")
            z.writestr("lib/arm64-v8a/libtest.so", b"elf-data")
            z.writestr("assets/extra.jar", b"jar-data")
        real_sha = hashlib.sha256(apk.read_bytes()).hexdigest()
        simulated_rows[0]["sha256"] = real_sha
        optional_source = temp / "optional_predictions.csv"
        with optional_source.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(simulated_rows[0]))
            writer.writeheader(); writer.writerows(simulated_rows)
        optional_fold = np.asarray([str(r["fold"]) for r in simulated_rows])
        npz = temp / "source_oof_embeddings.npz"
        rng = np.random.default_rng(23)
        np.savez(npz, sha256=np.asarray([r["sha256"] for r in simulated_rows]),
                 fold=optional_fold, embedding=rng.normal(size=(799, 8)))
        feats = temp / "source_features.csv"
        with feats.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sha256", "year", "host_count", "method_count"])
            writer.writeheader()
            for i, r in enumerate(simulated_rows):
                writer.writerow({"sha256": r["sha256"], "year": 2017,
                                 "host_count": 3 + i % 23, "method_count": 30 + i % 111})
        apkindex = temp / "source_apk_index.csv"
        with apkindex.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sha256", "year", "apk_path"])
            writer.writeheader(); writer.writerow({"sha256": real_sha, "year": 2017, "apk_path": str(apk)})
        stage1ref = temp / "canonical_source_reference.csv"
        with stage1ref.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sha256", "year", "stage1_prob", "routed"])
            writer.writeheader()
            for r in simulated_rows:
                writer.writerow({"sha256": r["sha256"], "year": 2017,
                                 "stage1_prob": r["Global24__prob"], "routed": r["A25__routed"]})
        optional_env = env.copy()
        optional_env.update({"ECA_SOURCE": str(optional_source), "ECA_EMBEDDINGS": str(npz),
                             "ECA_STATIC_FEATURES": str(feats), "ECA_APK_INDEX": str(apkindex),
                             "ECA_STAGE1_REFERENCE": str(stage1ref),
                             "ECA_FEATURE_COLUMNS": "host_count,method_count",
                             "ECA_OUTPUT": str(project / "outputs" / "care-direct-c43-error-attribution-audit-v1-optionals")})
        optional = subprocess.run([sys.executable, str(Path(__file__).parent / "run_audit.py")],
                                  env=optional_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=40)
        assert optional.returncode == 0, optional.stdout
        optdir = Path(optional_env["ECA_OUTPUT"])
        optreport = json.loads((optdir / "final_decision.json").read_text())
        assert optreport["geometry"]["covered_routed"] == 200
        assert optreport["geometry"]["folds_with_embeddings"] == 5
        assert optreport["static_descriptors"]["covered_routed"] == 200
        assert optreport["apk_archive_hints"]["scanned_routed"] == 1
        assert optreport["stage1_reference_identity"]["status"] == "EXACT_SOURCE_PROB_AND_ROUTE_MATCH"
        archive_result = audit.read_csv(optdir / "source_apk_archive_hints.csv")
        assert archive_result[0]["has_native_library"] == "1"
        assert archive_result[0]["dex_contains_loader_class_string"] == "1"
    print("CARE_ERROR_ATTRIBUTION_SELFTEST_PASS")


if __name__ == "__main__":
    main()
