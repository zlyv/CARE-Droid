#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import OUTPUT, SEED, MAX_DELTA, dump_json, read_csv, read_json, sha256_file


def finite(value, fallback=float("nan")):
    try:
        value = float(value)
        return value if np.isfinite(value) else fallback
    except Exception:
        return fallback


def by_name(rows, key):
    return {row[key]: row for row in rows}


def majority_noninferior(candidate, baseline, key="net"):
    b = {int(row["fold"]): row for row in baseline}
    eligible = 0; wins = 0
    for row in candidate:
        fold = int(row["fold"])
        if fold in b:
            eligible += 1
            wins += finite(row.get(key), -1e9) >= finite(b[fold].get(key), -1e9)
    return int(wins), int(eligible)


def main() -> None:
    baseline = read_json(OUTPUT / "baseline_reproduction.json")
    mapping = read_json(OUTPUT / "component_mapping_audit.json")
    sufficiency = read_json(OUTPUT / "sufficiency_probe.json")["variants"]
    nulls = read_json(OUTPUT / "matched_null_results.json")
    routers = read_json(OUTPUT / "router_results.json")["routers"]
    end = read_json(OUTPUT / "end_to_end_results.json")
    shortcuts = read_csv(OUTPUT / "shortcut_audit.csv")
    integrity = read_json(OUTPUT / "baseline_integrity_check.json")
    s = by_name(sufficiency, "variant")
    e = by_name(end["aggregate"], "arm")
    r = by_name(routers, "router")
    folds = end["per_fold"]
    fold_map = {}
    for row in folds:
        fold_map.setdefault(row["arm"], []).append(row)

    real = s["real"]; contract_null = s["contract_shuffle"]
    role_null = s["role_shuffle"]; count = s["count"]
    fn_auc = finite(real["FN_vs_TN_AUC"]); fp_auc = finite(real["FP_vs_TP_AUC"])
    real_beats_null = (
        fp_auc > max(finite(contract_null["FP_vs_TP_AUC"]), finite(role_null["FP_vs_TP_AUC"])) + .02
        and fn_auc >= max(finite(contract_null["FN_vs_TN_AUC"]), finite(role_null["FN_vs_TN_AUC"])) - .02
    )
    real_beats_count = fp_auc > finite(count["FP_vs_TP_AUC"]) and fn_auc > finite(count["FN_vs_TN_AUC"])
    fold_fp = [finite(row["FP_vs_TP_AUC"]) for row in real["folds"]]
    fold_fn = [finite(row["FN_vs_TN_AUC"]) for row in real["folds"]]
    fp_consistent = sum(value > .5 for value in fold_fp if np.isfinite(value))
    fn_consistent = sum(value > .5 for value in fold_fn if np.isfinite(value))
    max_size_rho = max((abs(finite(row["spearman_rho"], 0.0)) for row in shortcuts), default=0.0)
    coverage_ok = finite(mapping.get("host_assignment_ratio"), 0.0) >= .30 and finite(mapping.get("entry_mapping_ratio"), 0.0) >= .30
    sufficiency_supported = (
        coverage_ok and fp_auc >= .55 and fn_auc >= .55 and real_beats_null
        and real_beats_count and fp_consistent >= 3 and fn_consistent >= 3 and max_size_rho < .70
    )
    a0 = e["A0_FORMAL_DI25_C43"]; a3 = e["A3_CCBC_ROUTE_CCBC_DIRECTION"]
    a4 = e["A4_WITHIN_STRATUM_CONTRACT_SHUFFLE"]; a5 = e["A5_CONTRACT_ROLE_EDGE_SHUFFLE"]
    a6 = e["A6_COUNT_ONLY"]
    wins, eligible = majority_noninferior(fold_map["A3_CCBC_ROUTE_CCBC_DIRECTION"], fold_map["A0_FORMAL_DI25_C43"])
    end_supported = (
        int(a3["net"]) > int(a0["net"])
        and int(a3["net"]) > max(int(a4["net"]), int(a5["net"]), int(a6["net"]))
        and int(a3["broken"]) <= int(a0["broken"])
        and int(a3["fixed_FP"]) >= int(a0["fixed_FP"])
        and int(a3["fixed_FN"]) >= int(a0["fixed_FN"])
        and wins >= 3
    )
    one_side = (
        (fn_auc >= .55 and fn_consistent >= 3) ^ (fp_auc >= .55 and fp_consistent >= 3)
    )
    if integrity.get("status") != "PASS" or baseline.get("status") != "PASS":
        status = "BLOCKED"
    elif sufficiency_supported and end_supported:
        status = "SUPPORTED"
    elif one_side or (real_beats_null and (fn_auc >= .55 or fp_auc >= .55)):
        status = "PARTIALLY_SUPPORTED"
    else:
        status = "NOT_SUPPORTED"

    protocol = {
        "experiment": "CARE-Droid Component-Contract Bidirectional Counterevidence v1",
        "selection_data": "2017 source only", "source_rows": 799,
        "folds": [160, 160, 160, 160, 159],
        "stage1": "calibrated P31-RF only",
        "router": "RF-anchored MeanOpposition plus optional coarse CCBC; fold Top25%",
        "stage2_baseline": "A0 Role C43 Relation-Query",
        "direction": "separate FN-vs-TN EM and FP-vs-TP EB regularized logistic models",
        "max_delta": MAX_DELTA, "seed": SEED,
        "gamma_M": 1.0, "gamma_B": 1.0,
        "contract_support_minimum": 3, "compatibility_prior_strength": 5.0,
        "router_lambdas": [0.0, .25, .5],
        "future_rows_loaded_during_selection": 0,
        "source_status": status,
    }
    dump_json(OUTPUT / "protocol.json", protocol)
    (OUTPUT / "protocol_hash.txt").write_text(sha256_file(OUTPUT / "protocol.json") + "\n", encoding="utf-8")
    summary = {
        "conclusion": status,
        "baseline_reproduced": baseline.get("status") == "PASS",
        "formal_assets_unchanged": integrity.get("status") == "PASS",
        "component_mapping": mapping,
        "sufficiency": {
            "supported": sufficiency_supported, "FN_vs_TN_AUC": fn_auc,
            "FP_vs_TP_AUC": fp_auc, "real_beats_matched_null": real_beats_null,
            "real_beats_count_only": real_beats_count,
            "FN_folds_above_0_5": fn_consistent, "FP_folds_above_0_5": fp_consistent,
            "max_absolute_size_spearman": max_size_rho,
        },
        "router": {"formal": r["R0_FORMAL_DI25"], "ccbc": r["R1_MEANOPPOSITION_PLUS_CCBC"]},
        "end_to_end": {"A0": a0, "A3": a3, "A4": a4, "A5": a5, "A6": a6,
                         "A3_noninferior_folds": wins, "eligible_folds": eligible,
                         "supported": end_supported},
        "future_diagnostics": "NOT_RUN_SOURCE_SELECTION_COMPLETE",
        "temporal_generalization": "NOT_VALIDATED_ON_UNTOUCHED_2022_PLUS_OR_EXTERNAL_DATA",
        "future_rows_loaded": 0,
    }
    dump_json(OUTPUT / "final_summary.json", summary)
    dump_json(OUTPUT / "future_authorization.json", {
        "authorized": status == "SUPPORTED",
        "reason": "source evidence passed all preregistered dual-side, null, shortcut, fold-consistency and end-to-end criteria" if status == "SUPPORTED" else "source evidence did not pass all preregistered criteria; no future diagnostic is authorized",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "future_may_not_change_protocol": True,
        "future_rows_loaded": 0,
    })
    lines = [
        "# CARE-Droid CCBC source-only report", "",
        "## 已确认事实", "",
        f"- 正式基线复现：{baseline.get('status')}。Stage-I BA={baseline['stage1']['BA']:.6f}；正式系统 BA={baseline['formal']['BA']:.6f}，fixed={baseline['formal']['fixed']}，broken={baseline['formal']['broken']}，net={baseline['formal']['net']}。",
        f"- 正式资产完整性：{integrity.get('status')}。future_rows_loaded=0。", "",
        "## 实际实现", "",
        "- Manifest 结构化契约；outer-train 良性样本拟合契约—Role 兼容关系。",
        "- 组件生命周期入口沿既有调用图进行多源可达性；没有重建调用图或实施 taint/points-to。",
        "- 现有 C43 attention 只负责 Host 定位；VM/FB/UI 负责组件职责反证。",
        "- EM 与 EB 是两个条件化低容量模型，max_delta 固定为 2.25。", "",
        "## 与设想的差异和边界", "",
        "- 调用图只能证明可达性，不能证明某次运行一定由该入口触发。",
        "- 组件类名只用于映射并散列，不进入模型；缺失入口和 mixed/unassigned 保守进入 UI。",
        "- 2018–2021 未参与本次选择；它们已在历史研发中被查看，未来即使运行也只能称 frozen diagnostic。", "",
        "## 组件映射覆盖", "",
        f"- entry mapping={finite(mapping.get('entry_mapping_ratio'), 0):.4f}；Host assignment={finite(mapping.get('host_assignment_ratio'), 0):.4f}。", "",
        "## 信息充分性", "",
        f"- VM / FN-vs-TN AUC={fn_auc:.6f}；FB / FP-vs-TP AUC={fp_auc:.6f}。",
        f"- REAL优于matched null={real_beats_null}；优于count-only={real_beats_count}；最大规模相关绝对值={max_size_rho:.4f}。",
        f"- FN一致fold={fn_consistent}/5；FP一致fold={fp_consistent}/5。", "",
        "## 路由结果", "",
        f"- R0 captured={r['R0_FORMAL_DI25']['captured_errors']}/{r['R0_FORMAL_DI25']['stage1_errors']}；R1 captured={r['R1_MEANOPPOSITION_PLUS_CCBC']['captured_errors']}/{r['R1_MEANOPPOSITION_PLUS_CCBC']['stage1_errors']}。", "",
        "## Source端到端结果", "",
        f"- A0 BA={a0['BA']:.6f}，fixed={a0['fixed']}，broken={a0['broken']}，net={a0['net']}。",
        f"- A3 BA={a3['BA']:.6f}，fixed={a3['fixed']}，broken={a3['broken']}，net={a3['net']}。",
        f"- A3 non-inferior folds={wins}/{eligible}；端到端最低标准={end_supported}。", "",
        "## Frozen future与成本", "",
        "- Future未运行；只有source协议冻结后才允许运行，不得据其调整方法。",
        "- 计算成本见 timing.json；未实际测量的项明确标记 unavailable，不从路由比例推算时间。", "",
        "## 结论", "",
        f"**{status}**", "",
        "该状态由预注册规则自动产生；不因单个pooled指标而升级结论。",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CCBC_SOURCE_CONCLUSION={status}")
    print(f"FN_VS_TN_AUC={fn_auc:.6f} FP_VS_TP_AUC={fp_auc:.6f}")
    print(f"A0 fixed={a0['fixed']} broken={a0['broken']} net={a0['net']}")
    print(f"A3 fixed={a3['fixed']} broken={a3['broken']} net={a3['net']}")
    print("CCBC_SOURCE_PIPELINE_DONE")


if __name__ == "__main__":
    main()
