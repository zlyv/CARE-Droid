#!/usr/bin/env python3
"""Export the minimal code/data handoff for error-guided Entry experiments.

The export intentionally does not copy the full APK cache.  It writes hashes,
schemas, field inventories, small redacted feature/cache samples, split
summaries, and the exact implementation files needed to reproduce the
current Compact/Entry protocol.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_COLUMNS = {
    "sha256", "absolute_path", "source_id", "lineage_id", "metadata_year",
    "metadata_label", "package_name", "certificate_sha256", "version_code",
    "status", "manifest_parse_ok", "error_type", "error_message",
}
CODE_FILES = [
    "scripts/run_pilot_compact_eval.py",
    "scripts/run_2017_rolling_classifier_ensemble_eval.py",
    "scripts/run_dual_view_2017_rolling_eval.py",
    "scripts/run_graph_pool_diagnostic.py",
    "scripts/run_dual_view_error_attribution.py",
    "scripts/run_entry_increment_ablation.py",
    "scripts/run_entry_shortcut_audit.py",
    "scripts/run_entry_present_cohort_eval.py",
    "scripts/run_entry_only_transfer_eval.py",
    "scripts/run_entry_no_shortcut_transfer_eval.py",
    "scripts/run_locked_dual_view_module_ablation.py",
    "scripts/run_semantic_mapping_audit_ag.py",
    "src/tasa_droid/p2/graph_representation.py",
    "src/tasa_droid/p3/a3_mvp.py",
    "src/tasa_droid/p3/a3_graph_model.py",
    "src/tasa_droid/p3/reliability.py",
    "configs/protocol/semantic_mapping_v1.yaml",
    "configs/protocol/sensitive_endpoint_priority_v1.yaml",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve(strict=True).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_role_names(mapping_path: Path) -> list[str]:
    text = mapping_path.read_text(encoding="utf-8", errors="replace")
    block = re.search(r"(?ms)^roles:\s*\n(?P<body>.*?)(?=^rules:\s*$)", text)
    if not block:
        raise ValueError("could not locate roles block in semantic mapping")
    roles = re.findall(r"^\s*-\s*([A-Za-z0-9_]+)\s*$", block.group("body"), re.MULTILINE)
    if len(roles) != 31:
        raise ValueError(f"expected 31 roles, observed {len(roles)}")
    return sorted(roles)


def feature_bundle(
    path: Path, output: Path, sample_count: int, name: str | None = None
) -> dict[str, object]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty feature CSV: {path}")
    fields = list(rows[0])
    name = name or path.stem
    feature_dir = output / "features" / name
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "header.txt").write_text("\n".join(fields) + "\n", encoding="utf-8")

    role_fields = [
        field for field in fields
        if any(token in field.casefold() for token in ("role", "sensitive", "mapped", "api"))
    ]
    (feature_dir / "role_related_fields.txt").write_text(
        "\n".join(role_fields) + "\n", encoding="utf-8"
    )
    sample_fields = [
        field for field in fields
        if field not in {"absolute_path", "package_name", "certificate_sha256"}
    ]
    write_csv(
        feature_dir / "sample.csv",
        sample_fields,
        rows[:sample_count],
    )
    schema_path = path.with_name("feature_schema.json")
    schema_copy = None
    if schema_path.exists():
        schema_copy = feature_dir / "feature_schema.json"
        shutil.copy2(schema_path, schema_copy)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "row_count": len(rows),
        "column_count": len(fields),
        "columns": fields,
        "role_related_field_count": len(role_fields),
        "role_related_fields": role_fields,
        "sample_csv": str((feature_dir / "sample.csv").resolve()),
        "schema_copy": str(schema_copy.resolve()) if schema_copy else None,
    }


def redact_cache_row(row: dict, role_names: list[str]) -> dict[str, object]:
    payload = row.get("payload") or {}
    node_ids = list(payload.get("node_ids") or [])
    node_roles = list(payload.get("node_role_ids") or [])
    base = list(payload.get("node_base_features") or [])
    edge_index = payload.get("edge_index") or [[], []]
    edge_features = list(payload.get("edge_features") or [])
    edge_types = list(payload.get("edge_type_ids") or [])
    role_names_by_node = [
        [role_names[int(role)] for role in roles if 0 <= int(role) < len(role_names)]
        for roles in node_roles[:5]
    ]
    return {
        "sha256_prefix": str(row.get("sha256", ""))[:16],
        "status": row.get("status"),
        "metadata_year": row.get("metadata_year"),
        "metadata_label": row.get("metadata_label"),
        "top_level_keys": sorted(row),
        "payload_keys": sorted(payload),
        "node_count": len(node_ids),
        "edge_count": len(edge_types),
        "node_ids_first5_hashed": node_ids[:5],
        "node_role_ids_first5": node_roles[:5],
        "node_roles_first5": role_names_by_node,
        "node_base_features_first5": base[:5],
        "edge_index_first5": [list(edge_index[0][:5]), list(edge_index[1][:5])],
        "edge_features_first5": edge_features[:5],
        "edge_type_ids_first5_hashed": edge_types[:5],
    }


def cache_bundle(path: Path, output: Path, mapping_path: Path, sample_count: int) -> dict[str, object]:
    role_names = parse_role_names(mapping_path)
    samples = []
    top_level_keys = set()
    payload_keys = set()
    node_lengths = []
    edge_lengths = []
    status_counts: Counter[str] = Counter()
    with gzip.open(path.expanduser().resolve(strict=True), "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            status_counts[str(row.get("status", ""))] += 1
            top_level_keys.update(row)
            payload = row.get("payload") or {}
            payload_keys.update(payload)
            node_lengths.append(len(payload.get("node_ids") or []))
            edge_lengths.append(len(payload.get("edge_type_ids") or []))
            if len(samples) < sample_count:
                samples.append(redact_cache_row(row, role_names))
    cache_dir = output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / "schema_and_samples.json", {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status_counts": dict(status_counts),
        "top_level_keys": sorted(top_level_keys),
        "payload_keys": sorted(payload_keys),
        "role_id_mapping_sorted_index_to_role": {
            str(index): role for index, role in enumerate(role_names)
        },
        "node_count_min": min(node_lengths) if node_lengths else 0,
        "node_count_max": max(node_lengths) if node_lengths else 0,
        "edge_count_min": min(edge_lengths) if edge_lengths else 0,
        "edge_count_max": max(edge_lengths) if edge_lengths else 0,
        "samples": samples,
    })
    (cache_dir / "capabilities.md").write_text(
        """# Entry/A3 缓存能力说明\n\n"
        "缓存不是原始 API 实例表。成功行只保留：\n\n"
        "- 稳定哈希后的节点 ID；\n"
        "- 31 类语义角色 ID；\n"
        "- 4 个节点基础字段：external、mapped_sensitive、tier_scaled、manifest_entry；\n"
        "- 有界 edge_index、edge_features、edge_type_ids。\n\n"
        "它不保留原始方法签名、具体 API 名称、原始字段名或完整调用图。\n"
        "Entry/可靠性特征是在评估每个训练折时由缓存动态计算：可靠性组件使用训练折 APK，\n"
        "不能把测试折标签或测试折统计用于拟合。\n\n"
        "因此可以直接做角色级错误引导，但不能从该缓存恢复：\n"
        "`Runtime.exec -> 某具体方法 -> Entry距离=3` 这样的原始 API 实例解释。\n"
        "可以恢复的是：角色 ID、节点是否标记为入口、角色化池化统计、边类型/边界统计。\n""",
        encoding="utf-8",
    )
    (cache_dir / "capabilities_zh.md").write_text(
        """# Entry/A3 缓存能力说明

缓存不是原始 API 实例表。成功行只保留哈希节点 ID、31 类语义角色 ID、4 个节点基础字段，以及有界边索引和边特征。

当前缓存不保留原始方法签名、具体 API 名称、原始字段名或完整调用图。因此可以做角色级错误引导，但不能恢复 `Runtime.exec -> 某方法 -> Entry 距离` 这类具体 API 实例解释。

Entry/可靠性特征在评估时按外层训练折动态计算，测试标签只能用于事后指标。
""",
        encoding="utf-8",
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sample_file": str((cache_dir / "schema_and_samples.json").resolve()),
        "status_counts": dict(status_counts),
        "sample_count": len(samples),
    }


def split_bundle(paths: list[str], output: Path, feature_sets: dict[str, dict]) -> list[dict[str, object]]:
    result = []
    split_dir = output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for item in paths:
        if "=" not in item:
            raise ValueError(f"split must be NAME=PATH: {item}")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve(strict=True)
        rows = read_csv(path)
        fields = list(rows[0]) if rows else []
        counts = defaultdict(Counter)
        for row in rows:
            split_name = row.get("split_name", "")
            partition = row.get("partition", "")
            counts[split_name][partition] += 1
        write_csv(
            split_dir / f"{name}_sample.csv",
            fields,
            rows[:20],
        )
        result.append({
            "name": name,
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            "columns": fields,
            "split_partition_counts": {
                split: dict(counter) for split, counter in counts.items()
            },
            "sample_csv": str((split_dir / f"{name}_sample.csv").resolve()),
        })
    write_json(split_dir / "split_summary.json", result)
    return result


def copy_input_file(path: Path, destination: Path) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path.expanduser().resolve(strict=True), destination)
    return {
        "source": str(path.expanduser().resolve()),
        "copied_to": str(destination.resolve()),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--extra-features", nargs="*", default=[])
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--split", action="append", default=[], help="NAME=PATH; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--copy-inputs",
        action="store_true",
        help="also copy the full feature/cache/schema/mapping/split inputs, never APKs",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_specs = {
        "primary": feature_bundle(args.features, output, args.sample_count, "primary")
    }
    for item in args.extra_features:
        if "=" not in item:
            raise SystemExit(f"extra features must be NAME=PATH: {item}")
        name, raw_path = item.split("=", 1)
        feature_specs[name] = feature_bundle(
            Path(raw_path), output, args.sample_count, name
        )
    mapping_path = args.mapping.expanduser().resolve(strict=True)
    role_names = parse_role_names(mapping_path)
    cache_spec = cache_bundle(args.cache, output, mapping_path, args.sample_count)
    split_specs = split_bundle(args.split, output, feature_specs)

    full_input_copies = {}
    if args.copy_inputs:
        input_dir = output / "inputs"
        full_input_copies["primary_features"] = copy_input_file(
            args.features, input_dir / "primary_features.csv"
        )
        full_input_copies["schema"] = copy_input_file(
            args.schema, input_dir / "feature_schema.json"
        )
        full_input_copies["cache"] = copy_input_file(
            args.cache, input_dir / "a3_mvp_tensor_cache.jsonl.gz"
        )
        full_input_copies["mapping"] = copy_input_file(
            args.mapping, input_dir / "semantic_mapping_v1.yaml"
        )
        for item in args.extra_features:
            name, raw_path = item.split("=", 1)
            full_input_copies[f"features_{name}"] = copy_input_file(
                Path(raw_path), input_dir / f"{name}_features.csv"
            )
        for item in args.split:
            name, raw_path = item.split("=", 1)
            full_input_copies[f"split_{name}"] = copy_input_file(
                Path(raw_path), input_dir / f"{name}_split_manifest.csv"
            )

    code_dir = output / "code"
    copied = []
    for relative in CODE_FILES:
        source = PROJECT_ROOT / relative
        if not source.exists():
            continue
        destination = code_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"path": relative, "sha256": sha256_file(source)})

    manifest = {
        "schema_version": "1.0.0",
        "purpose": "handoff_for_error_guided_selective_entry_experiment",
        "generated_from_project": str(PROJECT_ROOT),
        "role_count": len(role_names),
        "roles_sorted": role_names,
        "features": feature_specs,
        "schema": {
            "path": str(args.schema.expanduser().resolve(strict=True)),
            "sha256": sha256_file(args.schema),
            "copied_to": str((output / "features" / "primary" / "feature_schema.json").resolve()),
        },
        "cache": cache_spec,
        "splits": split_specs,
        "full_input_copies": full_input_copies,
        "copied_code": copied,
        "protocol_notes": {
            "threshold": 0.5,
            "compact_model": "custom logistic regression; 1500 iterations; learning_rate=0.05; l2=0.001; seeds=0..4",
            "compact_scaling": "training-fold mean/std; constant columns filtered using training rows only",
            "entry_model": "role-stratified bounded graph pooling; graph_weight=0.5 in current Compact+Entry experiments",
            "reliability": "D and one_minus_F are fitted from the outer training partition in current no-T Entry evaluation",
            "entry_cache_policy": "H=4, K_out=8, P_cap=8, E_cap=2000, max_states=20000",
            "test_isolation": "test labels are used only for post-hoc metrics",
        },
        "limitations": [
            "The Entry cache is entry-conditioned; the strict no-shortcut ablation removes explicit availability/size proxies but is not a full-graph entry-free view.",
            "The cache does not preserve raw API names or method signatures; role-level error guidance is possible, exact API-instance attribution is not.",
            "Current 31-role mapping is a curated vocabulary, not an exhaustive Android API taxonomy; unmatched external nodes remain unmapped.",
        ],
    }
    write_json(output / "handoff_manifest.json", manifest)
    (output / "README.md").write_text(
        """# Entry 错误引导实验交接包\n\n"
        "这个目录可以直接交给负责写代码的人。\n\n"
        "## 首先查看\n\n"
        "- `handoff_manifest.json`：所有输入文件路径、SHA256、角色映射和协议参数；\n"
        "- `features/primary/header.txt`：Compact 完整列名；\n"
        "- `features/primary/role_related_fields.txt`：角色/API 相关列名；\n"
        "- `features/primary/sample.csv`：脱敏后的 Compact 样例；\n"
        "- `cache/schema_and_samples.json`：缓存字段、角色 ID 映射和样例；\n"
        "- `cache/capabilities.md`：缓存能做什么、不能做什么；\n"
        "- `splits/split_summary.json`：训练/验证/测试划分摘要；\n"
        "- `code/`：当前 Compact、Entry、消融和底层缓存实现的副本。\n\n"
        "## 建议第一版实现\n\n"
        "1. 只用 2017 rolling split 做 Compact 5-fold/滚动 OOF 预测；\n"
        "2. 按 `role__*`、`role_node__*`、`role_edge__*` 统计 FP/FN；\n"
        "3. 冻结高混淆角色；\n"
        "4. 从 Entry 缓存只保留这些角色的池化/上下文字段；\n"
        "5. 比较 Compact、Full Entry、Random-K、Frequent-K、Error-guided-K；\n"
        "6. 最后才在 2018--2021 上做一次锁定测试。\n""",
        encoding="utf-8",
    )
    (output / "README_zh.md").write_text(
        """# Entry 错误引导实验交接包

先查看 `handoff_manifest.json`、`features/primary/header.txt`、`features/primary/role_related_fields.txt`、`cache/schema_and_samples.json`、`cache/capabilities_zh.md` 和 `splits/split_summary.json`。

当前缓存只能支持角色级错误引导，不包含原始 API 名称、方法签名或具体 API 实例到 Entry 的距离。第一版应使用 2017 rolling 训练折产生 Compact OOF 误差，冻结高混淆角色，再比较 Full Entry、Random-K、Frequent-K 和 Error-guided-K。
""",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
