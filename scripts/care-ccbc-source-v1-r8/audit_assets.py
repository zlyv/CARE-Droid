#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from assets import discover_apk_map, load_endpoint_rows, manifest_records
from common import (
    OUTPUT, PROJECT, ROLE_NAMES, discover_compact_cache, discover_endpoint_cache,
    discover_formal_predictions, discover_p31, discover_role_manifest, dump_json,
    load_formal_rows, sha256_file,
)
from contracts import contract_schema


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = manifest_records()
    dump_json(OUTPUT / "baseline_asset_manifest_before.json", {
        "status": "CAPTURED", "project": str(PROJECT), "output": str(OUTPUT),
        "git_used": False, "assets": records,
    })
    rows = load_formal_rows(); wanted = {row["sha256"] for row in rows}
    apk_map, apk_audit = discover_apk_map(wanted)
    endpoints = load_endpoint_rows(wanted)
    p31_path = discover_p31()
    with np.load(p31_path, allow_pickle=False) as z:
        p31_shape = list(np.asarray(z["role_prevalence"]).shape)
        p31_ids = {str(value).strip().casefold() for value in z["sha256"]}
    compact_path = discover_compact_cache()
    with np.load(compact_path, allow_pickle=False) as z:
        compact_schema = {key: list(np.asarray(z[key]).shape) for key in z.files}
    role_manifest = json.loads(discover_role_manifest().read_text(encoding="utf-8"))
    mapping = role_manifest.get("payload_to_model_role")
    endpoint_example = next(iter(endpoints.values())) if endpoints else {}
    node_ids = endpoint_example.get("node_ids", [])
    node_id_types = sorted({type(value).__name__ for value in node_ids[:100]})
    missing_apk = sorted(wanted - set(apk_map))
    missing_endpoint = sorted(wanted - set(endpoints))
    entry_cache_candidates = sorted({
        str(path.resolve()) for pattern in (
            "**/*entry*cache*.jsonl.gz", "**/*entry*centered*.jsonl.gz",
            "**/*entry*guided*.jsonl.gz", "**/a3_mvp_tensor_cache.jsonl.gz",
            "**/*a3*mvp*cache*.jsonl.gz",
        ) for path in (PROJECT / "artifacts").glob(pattern) if path.is_file()
    })
    dump_json(OUTPUT / "contract_schema.json", contract_schema())
    schema_hash = sha256_file(OUTPUT / "contract_schema.json")
    audit = {
        "formal_rows": len(rows),
        "fold_sizes": {str(fold): sum(row["fold"] == fold for row in rows) for fold in range(1, 6)},
        "class_counts": {"benign": sum(row["label"] == 0 for row in rows), "malware": sum(row["label"] == 1 for row in rows)},
        "formal_predictions": str(discover_formal_predictions()),
        "p31": str(p31_path), "p31_shape": p31_shape, "p31_sha_alignment": p31_ids == wanted,
        "endpoint_cache": str(discover_endpoint_cache()), "endpoint_rows_found": len(endpoints),
        "endpoint_example_keys": sorted(endpoint_example), "endpoint_node_id_types": node_id_types,
        "compact_cache": str(compact_path), "compact_schema": compact_schema,
        "role_manifest": str(discover_role_manifest()),
        "role_permutation_valid": isinstance(mapping, list) and sorted(map(int, mapping)) == list(range(31)),
        "apk_paths_found": len(apk_map), "apk_path_inventory": apk_audit,
        "missing_apk_sha_preview": missing_apk[:20], "missing_apk_count": len(missing_apk),
        "missing_endpoint_sha_preview": missing_endpoint[:20], "missing_endpoint_count": len(missing_endpoint),
        "existing_entry_cache_candidates": entry_cache_candidates,
        "contract_schema_sha256": schema_hash,
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "preflight_inventory.json", audit)
    lines = [
        "# CARE-Droid CCBC 工程与资产审计",
        "",
        "## 已确认事实",
        "",
        f"- 项目根目录：`{PROJECT}`",
        f"- CCBC 独立输出：`{OUTPUT}`",
        f"- 正式预测：`{discover_formal_predictions()}`",
        f"- P31：`{p31_path}`，shape={p31_shape}",
        f"- Endpoint/call-graph cache：`{discover_endpoint_cache()}`，覆盖 {len(endpoints)}/799",
        f"- Compact Host cache：`{compact_path}`",
        f"- Manifest APK 路径：{len(apk_map)}/799",
        f"- Contract schema SHA256：`{schema_hash}`",
        f"- 既有 entry-cache 候选：{len(entry_cache_candidates)}；若为0，则不得通过重建完整图或模糊匹配提高覆盖。",
        "- 未运行 Git 命令；正式资产仅记录哈希，不写入。",
        "",
        "## 拟复用正式模块",
        "",
        "- P31-RF、四成员概率校准和 nested OOF：正式 Stage-I 模块。",
        "- RF-anchored MeanOpposition、fold-wise Top25 与 SHA tie-break：正式 router 模块。",
        "- Compact43、Role31、Relation31、RoleGraph/S7、C43 Relation-Query：正式 Stage-II 模块。",
        "- `forward_with_trace`：只在 CCBC 新目录重放并导出 Host attention，不修改正式模型。",
        "- 正式 expert probability：用于 A2 的既有 C43 方向头，不重新定义。",
        "",
        "## CCBC 可精确实现部分",
        "",
        "- AndroidManifest 结构化组件、framework action/category/data、foreground type 和特殊 BIND 服务职责。",
        "- outer-train benign-only contract–Role 兼容度和平滑 specificity。",
        "- 组件生命周期入口到现有 endpoint graph Host 的多源可达性。",
        "- matched/mismatched/mixed/unassigned、VM/FB/UI 和双条件低容量方向模型。",
        "",
        "## 需由运行时审计决定的部分",
        "",
        "- 组件类到 endpoint node 的哈希身份是否可直接匹配；若需要 DEX 方法枚举，只允许做身份恢复，不重建调用图。",
        "- 现有 endpoint graph 是否保留足够多组件入口；覆盖不足将触发 NOT_SUPPORTED/BLOCKED 判定，而不会使用模糊类名匹配。",
        "- C43 attention 若无现成 cache，将在 CCBC 目录按冻结种子重放。",
        "",
        "## 风险",
        "",
        "- 声明可被恶意 APK 伪装；因此 REAL 必须优于 contract shuffle。",
        "- exported/boot/permission 不能单独构成类别方向。",
        "- 组件数、Host 数和图规模可能形成捷径；必须与 count-only 比较并报告相关性。",
        "- FP 数较少，必须按 fold 报告可评估性和置信区间。",
        "",
    ]
    (OUTPUT / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "reproduce_commands.md").write_text(
        "# Reproduction commands\n\n"
        "```bash\n"
        "cd /home/boyuanjie/projects/tasa-droid\n"
        "bash scripts/care-ccbc-source-v1/launch_tmux.sh\n"
        "# Use the output path printed by the launcher:\n"
        "bash scripts/care-ccbc-source-v1/collect_results.sh outputs/care-ccbc-source-v1\n"
        "```\n\n"
        "The launcher never reuses a non-empty CCBC output directory. Set CCBC_APK_INDEX or "
        "CCBC_APK_ROOTS only when automatic APK discovery is incomplete.\n",
        encoding="utf-8",
    )
    apk_index = OUTPUT / "apk_paths.json"
    dump_json(apk_index, apk_map)
    if len(endpoints) != 799:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: endpoint coverage {len(endpoints)}/799")
    if len(apk_map) != 799:
        raise RuntimeError(
            "BLOCKED_BY_CONTRACT: APK path coverage " + str(len(apk_map)) +
            "/799. Set CCBC_APK_INDEX or CCBC_APK_ROOTS; existing HAIG extraction proves APKs exist but no path may be guessed."
        )
    if p31_shape != [799, 31] or p31_ids != wanted:
        raise RuntimeError("BLOCKED_BY_CONTRACT: P31 lineage mismatch")
    print("CCBC_ASSET_AUDIT_PASS")
    print(f"apk_paths=799 endpoint_rows=799 output={OUTPUT}")


if __name__ == "__main__":
    main()

