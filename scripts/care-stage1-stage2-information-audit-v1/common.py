#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path("/home/boyuanjie/projects/tasa-droid")
OUTPUT = PROJECT / "outputs/care-stage1-stage2-information-audit-v1"
FORMAL_PREDICTIONS = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv"
P31_CACHE = PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz"
HOST_CACHE = PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz"
FORMAL_METHOD = "P31RF__RFRoute__A0_ROLE_ONLY__Bounded"
STAGE1_FIELD = "Global24__prob"  # Historical name; frozen provenance is calibrated P31-RF only.
ROUTE_FIELD = "A25__routed"
ROUTER_SCORE_FIELD = "A25__score"
SEED = 20261007

ROLE_NAMES = (
    "command_execution", "dynamic_code_loading", "native_library_loading",
    "reflective_invocation", "package_installation", "package_uninstallation",
    "device_administration", "accessibility_control", "sms_send_receive",
    "call_control_or_recording", "account_or_credential_access",
    "contacts_or_calendar_access", "device_identifier_access", "location_access",
    "microphone_access", "camera_access", "clipboard_access",
    "network_transmission_or_upload", "sensitive_file_or_content_write",
    "boot_persistence", "task_or_alarm_scheduling", "background_service_control",
    "overlay_or_ui_deception", "icon_or_component_hiding",
    "process_or_wakelock_control", "cryptography", "encoding_or_compression",
    "url_or_http_setup", "file_read", "database_access", "other",
)


def norm(value):
    return str(value).strip().casefold()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_auc(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def metric(y, probability):
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score

    y = np.asarray(y, int)
    probability = np.asarray(probability, float)
    pred = probability >= 0.5
    return {
        "n": int(len(y)),
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": binary_auc(y, probability),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y, probability)),
        "errors": int(np.sum(pred != y)),
    }


def ece(y, probability, bins=10):
    y = np.asarray(y, int)
    probability = np.asarray(probability, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    details = []
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if not np.any(mask):
            continue
        confidence = float(np.mean(probability[mask]))
        frequency = float(np.mean(y[mask]))
        total += float(np.mean(mask)) * abs(confidence - frequency)
        details.append({"bin": index, "n": int(mask.sum()), "mean_probability": confidence, "positive_rate": frequency})
    return float(total), details

