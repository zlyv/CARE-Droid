#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(os.environ.get("CCBC_PROJECT", Path.cwd())).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("CCBC_OUTPUT", PROJECT / "outputs/care-ccbc-source-v1")).resolve()
SEED = 20261006
MAX_DELTA = 2.25
ROLE_COUNT = 31
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


def norm(value: Any) -> str:
    return str(value or "").strip().casefold()


def ensure_output_path(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(OUTPUT)
    except ValueError as exc:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: write outside CCBC output: {path}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, value: Any) -> None:
    path = ensure_output_path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = ensure_output_path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_output_path(path)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_record(path: Path, purpose: str) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
        "purpose": purpose,
    }


def first_existing(candidates: Iterable[Path], label: str) -> Path:
    candidates = [Path(value) for value in candidates]
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path.resolve()
    raise RuntimeError(
        "BLOCKED_BY_CONTRACT: missing " + label + ": " + ", ".join(str(p) for p in candidates)
    )


def discover_formal_predictions() -> Path:
    explicit = os.environ.get("CCBC_FORMAL_PREDICTIONS")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv",
        PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
    ]
    return first_existing(candidates, "formal P31-RF/A0 predictions")


def discover_formal_source_predictions() -> Path:
    explicit = os.environ.get("CCBC_FORMAL_SOURCE_PREDICTIONS")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/arms/A0_ROLE_ONLY/source_predictions.csv",
        PROJECT / "artifacts/care-droid-formal-v1/formal_predictions.csv",
    ]
    return first_existing(candidates, "formal A0 source predictions")


def discover_p31() -> Path:
    explicit = os.environ.get("CCBC_P31")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        PROJECT / "artifacts/care-litelocal-semantic-source-v1/local_semantic_features_2017.npz",
    ]
    return first_existing(candidates, "P31 source cache")


def discover_endpoint_cache() -> Path:
    explicit = os.environ.get("CCBC_ENDPOINT_CACHE")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        PROJECT / "artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz",
        PROJECT / "artifacts/p25-endpoint-cache-full-cap256/endpoint_centered_tensor_cache.jsonl.gz",
    ]
    for path in (PROJECT / "artifacts").glob("**/*endpoint*cache*2017*.jsonl.gz"):
        candidates.append(path)
    return first_existing(candidates, "2017 endpoint/call-graph cache")


def discover_compact_cache() -> Path:
    explicit = os.environ.get("CCBC_COMPACT_CACHE")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        PROJECT / "artifacts/care-p31rf-c43-haig-host-ablation-source-v1/cache/A0_ROLE_ONLY.npz",
        PROJECT / "artifacts/care-di25-compact43-relationquery-source-v1/cache/compact43_relation_source.npz",
    ]
    return first_existing(candidates, "formal Host cache")


def discover_role_manifest() -> Path:
    explicit = os.environ.get("CCBC_ROLE_MANIFEST")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        PROJECT / "artifacts/care-source-local-assets-v2/local_asset_manifest.json",
        Path("/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json"),
    ]
    return first_existing(candidates, "payload-to-model role manifest")


def discover_formal_script_files() -> list[tuple[Path, str]]:
    roots = [
        PROJECT / "scripts/care-droid-formal-v1",
        PROJECT / "scripts/care-p31-stage1-rf-selection-source-v1",
        PROJECT / "scripts/care-di25-compact43-relationquery-source-v1",
        PROJECT / "scripts/care-di25-graphglobal-path-source-v1",
        PROJECT / "scripts/care-rolegraph-conditioned-v4-script",
    ]
    result: list[tuple[Path, str]] = []
    for root in roots:
        if root.is_dir():
            for path in sorted(root.glob("*.py")):
                result.append((path.resolve(), "frozen/reused formal code"))
            for path in sorted(root.glob("*.sh")):
                result.append((path.resolve(), "frozen/reused formal launcher"))
    return result


def load_formal_rows() -> list[dict[str, Any]]:
    path = discover_formal_predictions()
    rows = read_csv(path)
    if len(rows) != 799:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: formal rows {len(rows)} != 799")
    output = []
    for row in rows:
        sid = norm(row.get("sha256"))
        label = int(float(row.get("label", row.get("y", "nan"))))
        fold = int(float(row["fold"]))
        stage1 = float(row.get("stage1_prob", row.get("Global24__prob", "nan")))
        router = float(row.get("router_score", row.get("A25__score", "nan")))
        routed = int(float(row.get("routed", row.get("A25__routed", 0))))
        candidate = row.get("candidate_prob", row.get("bounded_prob"))
        final = row.get("final_prob")
        delta = row.get("bounded_delta", row.get("delta"))
        if candidate in (None, ""):
            keys = [key for key in row if key.endswith("__expert_prob")]
            if len(keys) != 1:
                raise RuntimeError("BLOCKED_BY_CONTRACT: cannot identify formal expert probability")
            candidate = row[keys[0]]
        if final in (None, ""):
            keys = [key for key in row if key.endswith("__prob") and not key.endswith("__expert_prob")]
            preferred = [key for key in keys if "A0_ROLE_ONLY" in key or "P31RF" in key]
            final = row[(preferred or keys)[-1]]
        if delta in (None, ""):
            keys = [key for key in row if key.endswith("__bounded_delta")]
            if keys:
                preferred = [key for key in keys if "A0_ROLE_ONLY" in key or "P31RF" in key]
                delta = row[(preferred or keys)[-1]]
            else:
                delta = probability_to_logit(float(candidate)) - probability_to_logit(stage1)
        output.append({
            "sha256": sid, "fold": fold, "label": label,
            "stage1_prob": stage1, "router_score": router, "routed": routed,
            "candidate_prob": float(candidate), "formal_final_prob": float(final),
            "formal_bounded_delta": float(delta),
        })
    if len({row["sha256"] for row in output}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: duplicate formal SHA")
    return output


def binary_auc(y: Any, score: Any) -> float:
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


def metrics(y: Any, base: Any, probability: Any, route: Any) -> dict[str, Any]:
    y = np.asarray(y, int)
    base = np.asarray(base, float)
    probability = np.asarray(probability, float)
    route = np.asarray(route, bool)
    bp = base >= .5
    pp = probability >= .5
    base_ok = bp == y
    ok = pp == y
    tn = int(np.sum((y == 0) & ~pp)); fp = int(np.sum((y == 0) & pp))
    fn = int(np.sum((y == 1) & ~pp)); tp = int(np.sum((y == 1) & pp))
    fixed = (~base_ok) & ok
    broken = base_ok & ~ok
    return {
        "BA": float((tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2),
        "AUC": binary_auc(y, probability),
        "F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": fp + fn,
        "fixed": int(fixed.sum()), "broken": int(broken.sum()),
        "net": int(fixed.sum() - broken.sum()), "routed": int(route.sum()),
        "fixed_FN": int(np.sum(fixed & (y == 1))),
        "fixed_FP": int(np.sum(fixed & (y == 0))),
        "broken_TN_to_FP": int(np.sum(broken & (y == 0))),
        "broken_TP_to_FN": int(np.sum(broken & (y == 1))),
        "prediction_flips": int(np.sum(bp != pp)),
    }


def percentile_against(reference: Any, values: Any) -> np.ndarray:
    reference = np.sort(np.asarray(reference, float))
    values = np.asarray(values, float)
    if not len(reference):
        return np.full(len(values), .5)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2.0 * len(reference))


def top_budget(score: Any, sha: Any, count: int) -> np.ndarray:
    score = np.asarray(score, float)
    sha = np.asarray(sha).astype(str)
    order = np.lexsort((sha, -score))
    result = np.zeros(len(score), bool)
    result[order[:int(count)]] = True
    return result


def probability_to_logit(probability: Any) -> np.ndarray:
    probability = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def sigmoid(value: Any) -> np.ndarray:
    value = np.asarray(value, float)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def bootstrap_auc_ci(y: Any, score: Any, seed: int = SEED, repeats: int = 2000) -> dict[str, float]:
    y = np.asarray(y, int); score = np.asarray(score, float)
    observed = binary_auc(y, score)
    if len(np.unique(y)) < 2:
        return {"AUC": observed, "CI_low": float("nan"), "CI_high": float("nan")}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            values.append(binary_auc(y[idx], score[idx]))
    return {
        "AUC": observed,
        "CI_low": float(np.quantile(values, .025)),
        "CI_high": float(np.quantile(values, .975)),
    }


def spearman(x: Any, y: Any) -> float:
    from scipy.stats import spearmanr
    value = float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)
    return 0.0 if not math.isfinite(value) else value


def import_path(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, str(path))
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

