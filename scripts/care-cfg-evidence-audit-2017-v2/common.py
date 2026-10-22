from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path('/home/boyuanjie/projects/tasa-droid')
OUT = ROOT / 'artifacts/care_cfg_evidence_audit_2017_v2'
H1_OUT = ROOT / 'outputs/care_host_bound_cfg_source_future_v3'
CFG_ABL = ROOT / 'outputs/care_cfg_witness_fixed_ablation_v1'
CFG_FIXED = ROOT / 'outputs/care_cfg_witness_fixed_source_future_v2'
MANIFEST_CFG = ROOT / 'outputs/care_manifest_cfg_witness_fixed_source_v2'
SCBE = ROOT / 'outputs/care_scbe_v1'
CFG_ART = ROOT / 'artifacts/care_cfg_witness_all_years_v2'
CFG_CACHE = CFG_ART / 'cache'
CFG_FEATURES = H1_OUT / 'host_cfg_features.npz'
EGO_SOURCE = ROOT / 'artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz'
ENDPOINT_CACHE = ROOT / 'artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz'
LOCAL_ASSET_MANIFEST = Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/local_asset_manifest.json')

SPLITS = [f'S{i}' for i in range(1, 6)]
SEEDS = [900000, 900001, 900002]
MANIFEST_SEEDS = [20261018, 20261019, 20261020]


def add_paths():
    paths = [
        ROOT / 'scripts/care-background-functional-attribution-v1',
        ROOT / 'scripts/care-multiyear-development-v1',
        ROOT / 'scripts/care-scbe-v1',
        ROOT / 'scripts/care-host-bound-cfg-source-future-v3',
        ROOT / 'scripts/care-cfg-witness-all-years-v2',
    ]
    for p in reversed(paths):
        sys.path.insert(0, str(p))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json_new(path: Path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def write_text_new(path: Path, text: str):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def write_csv_new(path: Path, frame: pd.DataFrame):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_gzip_json(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(path: Path):
    with path.open('rb') as f:
        return pickle.load(f)


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y)), 'threshold': float(threshold),
        'BA': float(balanced_accuracy_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'F1': float(f1_score(y, pred, zero_division=0)),
        'AUC': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        'FNR': float(fn / (fn + tp)) if fn + tp else None,
        'FPR': float(fp / (fp + tn)) if fp + tn else None,
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def stratified_paired_bootstrap(y, p_a, p_b, fold, n_boot=5000, seed=20261020):
    y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b); fold = np.asarray(fold)
    rng = np.random.default_rng(seed)
    strata = [np.where((fold == f) & (y == c))[0] for f in sorted(set(fold)) for c in (0, 1)]
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata if len(s)])
        diffs[i] = balanced_accuracy_score(y[idx], p_a[idx] >= .5) - balanced_accuracy_score(y[idx], p_b[idx] >= .5)
    return {
        'delta_BA': float(balanced_accuracy_score(y, p_a >= .5) - balanced_accuracy_score(y, p_b >= .5)),
        'bootstrap_mean': float(np.mean(diffs)),
        'ci95_low': float(np.quantile(diffs, .025)),
        'ci95_high': float(np.quantile(diffs, .975)),
        'n_bootstrap': int(n_boot),
        'stratification': '2017 fixed fold x class',
    }


def qstats(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'std': float(a.std()),
            'min': float(a.min()), 'p05': float(np.quantile(a,.05)), 'p25': float(np.quantile(a,.25)),
            'median': float(np.median(a)), 'p75': float(np.quantile(a,.75)), 'p95': float(np.quantile(a,.95)),
            'max': float(a.max())}


def pearson_spearman(x, y):
    from scipy.stats import pearsonr, spearmanr
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {'pearson_r': None, 'pearson_p': None, 'spearman_r': None, 'spearman_p': None}
    pr = pearsonr(x, y); sr = spearmanr(x, y)
    return {'pearson_r': float(pr.statistic), 'pearson_p': float(pr.pvalue),
            'spearman_r': float(sr.statistic), 'spearman_p': float(sr.pvalue)}


def source_fold_map():
    out = {}
    for split in SPLITS:
        ck = load_pickle(SCBE / f'models/{split}/900000/M0_REPRODUCED.pkl')
        for sha in ck['ids']:
            out[str(sha)] = split
    return out

