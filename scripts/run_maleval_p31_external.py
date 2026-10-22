#!/usr/bin/env python3
"""
MalEval2026 外部数据集 P31 特征提取入口。

作用：
1. 复用 CARE-Droid 原有的 P31 特征提取函数和31类角色映射。
2. 对 MalEval APK 进行真实的 DEX 敏感API扫描。
3. 默认运行20个APK的先导测试，也支持全部样本。
4. 将结果保存在独立输出目录，不修改原始实验数据。
5. 保存每个成功样本的特征缓存，支持中断后重新运行。
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
DATA = Path(
    "/home/boyuanjie/datasets/android/"
    "external_benchmarks/MalEval2026/extracted/apk"
)

EXTRACTOR = PROJ / (
    "scripts/care-litelocal-p31-diagnostics-v1/"
    "extract_future_p31.py"
)

CONFIG = PROJ / (
    "artifacts/care-litelocal-p31-diagnostics-v1/"
    "preflight/resolved_inputs.json"
)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "existing_p31_extractor", EXTRACTOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.suppress()

    cfg = json.loads(CONFIG.read_text())
    mapping_path = Path(cfg["mapping"])

    if not mapping_path.is_absolute():
        mapping_path = PROJ / mapping_path

    mapping = module.yaml.safe_load(
        mapping_path.read_text(encoding="utf-8")
    )

    roles = list(mapping["roles"])
    assert len(roles) == 31

    role_index = {role: i for i, role in enumerate(roles)}
    rules = [
        r for r in mapping["rules"]
        if r.get("rule_kind", "api") == "api"
    ]

    return module, roles, rules, role_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    mode = "all255" if args.all else "pilot20"
    out = PROJ / "outputs/maleval2026_external_eval" / mode
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    extractor, roles, rules, role_index = load_extractor()

    groups = [
        ("benign", 0, 10),
        ("malradar", 1, 5),
        ("new", 1, 5),
    ]

    samples = []

    for group, label, count in groups:
        files = sorted((DATA / group).glob("*.apk"))
        if not args.all:
            assert len(files) >= count, group
            files = files[:count]

        for apk in files:
            samples.append((apk, group, label))

    print(f"Mode: {mode}", flush=True)
    print(f"Total APKs: {len(samples)}", flush=True)

    records = []
    features = []
    audits = []
    success_records = []

    for i, (apk, group, label) in enumerate(samples, 1):
        sha = file_sha256(apk)
        shard = shards / f"{sha}.npz"

        status = "extracted"
        error = ""
        p31 = None
        audit = None

        try:
            if shard.exists():
                try:
                    with np.load(shard, allow_pickle=False) as z:
                        p31 = z["p31"]
                        audit = z["audit"]

                    if p31.shape != (31,):
                        raise ValueError("Invalid P31 shape")

                    status = "cached"

                except Exception:
                    p31 = None
                    audit = None

            if p31 is None:
                p31, audit = extractor.extract(
                    {"absolute_path": str(apk)},
                    rules,
                    role_index,
                )

                tmp = shards / f"{sha}.tmp.npz"
                np.savez_compressed(tmp, p31=p31, audit=audit)
                os.replace(tmp, shard)

            features.append(p31)
            audits.append(audit)

            success_records.append({
                "sha256": sha,
                "apk_path": str(apk),
                "label": label,
                "group": group,
            })

        except Exception as e:
            status = "failed"
            error = f"{type(e).__name__}: {e}"

        records.append({
            "sha256": sha,
            "apk_path": str(apk),
            "label": label,
            "group": group,
            "status": status,
            "error": error,
        })

        print(
            f"[{i}/{len(samples)}] "
            f"{group} {sha[:12]} "
            f"status={status} {error}",
            flush=True,
        )

    with (out / "extraction_status.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sha256", "apk_path", "label",
                "group", "status", "error"
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    if success_records:
        np.savez_compressed(
            out / "p31_features.npz",
            sha256=np.array([
                r["sha256"] for r in success_records
            ]),
            apk_paths=np.array([
                r["apk_path"] for r in success_records
            ]),
            labels=np.array([
                r["label"] for r in success_records
            ], dtype=np.int8),
            groups=np.array([
                r["group"] for r in success_records
            ]),
            role_prevalence=np.asarray(
                features, dtype=np.float32
            ),
            audit_values=np.asarray(audits),
            role_names=np.asarray(roles),
        )

    print(
        f"P31_EXTRACTION_FINISHED "
        f"success={len(success_records)} "
        f"failed={len(samples)-len(success_records)}",
        flush=True,
    )
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
