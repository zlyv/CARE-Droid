"""Publish a bounded Chinese research decision after all frozen comparisons and source hashes pass."""
from __future__ import annotations
import json
from pathlib import Path
from common import *

def main():
    required=['baseline_replay.json','background_tensor_manifest.json','existing_three_background_audit.md','prior_background_experiment_map.md','diagnostic_decision.json','expert_bottleneck_diagnosis.csv','background_response_diagnosis.csv','model_comparisons.csv','ablation_results.csv','paired_intervals.csv','runtime_cost.csv','learned_representation_geometry.csv','within_model_temporal_geometry.csv','background_temporal_response.csv','model_selection_decision.json']
    missing=[s for s in required if not (OUTPUT/s).is_file()]
    if missing:raise RuntimeError('required result missing '+repr(missing))
    before=read_json(OUTPUT/'baseline_asset_manifest_before.json');after=[];changed=[]
    for r in before:
        p=Path(r['path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='missing'));continue
        q=asset(p,r['purpose']);after.append(q)
        if any(r[k]!=q[k] for k in ['size','mtime_ns','sha256']):changed.append(dict(before=r,after=q))
    write_json(OUTPUT/'baseline_asset_manifest_after.json',after)
    write_json(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed,scope='explicitly reused original assets only, not entire server'))
    if changed:raise RuntimeError('protected original assets changed')
    metrics=read_csv(OUTPUT/'model_comparisons.csv');reference={r['arm']:r for r in metrics if r['scope']=='pooled_T2_T4'};decision=read_json(OUTPUT/'model_selection_decision.json');diag=read_csv(OUTPUT/'expert_bottleneck_diagnosis.csv');interval=read_csv(OUTPUT/'paired_intervals.csv')
    route_missed=sum(int(r['unrouted_correctable_errors']) for r in diag);unfixed=sum(int(r['routed_unfixed_errors']) for r in diag);broken=sum(int(r['routed_broken']) for r in diag)
    main_status=decision['status'];labels=[]
    if route_missed:labels.append('ROUTER_BOTTLENECK')
    if unfixed>0:labels.append('EXPERT_DECISION_BOTTLENECK')
    if decision['M1_gate']:labels.append('CONTRASTIVE_LEARNING_CANDIDATE_ON_DEVELOPMENT')
    if main_status.startswith('BACKGROUND_CONDITIONING_SUPPORTED'):labels.append('BACKGROUND_CONDITIONING_CANDIDATE_ON_DEVELOPMENT')
    if not decision['M1_gate'] and not main_status.startswith('BACKGROUND_CONDITIONING_SUPPORTED'):labels.append('NO_RELIABLE_INCREMENT_IDENTIFIED')
    write_json(OUTPUT/'final_decision.json',dict(status=main_status,diagnostic_labels=labels,route_missed_current_C43_correctable=route_missed,routed_unfixed_current_C43=unfixed,routed_broken_current_C43=broken,criteria=decision,formal_integrity='PASS',development_only=True,independent_final_test=False,MalEval2026_used=False,formal_baseline_changed=False,next_action='retain original three backgrounds and Direct-C43 unless a candidate also passes a new source-frozen independent validation; do not tune on these development years'))
    lines=['# CARE-Droid既有三背景与监督对比训练审计','','## 结论',f'状态：**{main_status}**。现有三背景保持不变，2017—2021已是开发数据，任何改善只能作为候选。','',f'纳入清单的{len(before)}份复用资产大小、mtime和SHA256前后一致：PASS。未读MalEval2026，未更改正式脚本或APK。','', '## 实际三背景','','共享32个原型由两类训练Host平衡采样；外部24个原型只来自训练良性Host；内部背景在每个APK的Host袋内动态计算。最终每Host附加外部距离、内部距离、commonness与由前三者确定的residual四个量。原始Direct-C43已在Host编码、注意力和分类输入使用它们，原始训练目标为等权BCE。详细函数、形状及既往TB-FRS区别见配套审计文件。','','## 冻结专家可修复空间','','| 验证年 | Stage-I错误 | 已路由错误 | C43已修复 | 已路由仍未修复 | 漏路由但C43可修复 | 破坏原正确 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in diag:lines.append(f'| {r["tag"]} | {r["StageI_errors"]} | {r["routed_StageI_errors"]} | {r["routed_correctable_errors"]} | {r["routed_unfixed_errors"]} | {r["unrouted_correctable_errors"]} | {r["routed_broken"]} |')
    lines+=['','“漏路由但C43可修复”通过冻结C43对该年全部APK推理计算，不涉及重新提取APK。标签知情Oracle只是理论上限，不能用于部署路由。','','## M0—M5统一上游端到端比较','','| 组 | T2 BA | T3 BA | T4 BA | 合并BA | Fixed | Broken | Net | FN/TN AUC | TP/FP AUC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ['M0','M1','M2','M3','M4','M5']:
        r=reference[arm];by={x['scope']:x for x in metrics if x['arm']==arm};fmt=lambda z:f'{float(z):.4f}' if z not in (None,'') else 'NA'
        lines.append(f'| {arm} | {fmt(by["T2_C1"]["BA"])} | {fmt(by["T3_C1"]["BA"])} | {fmt(by["T4_C1"]["BA"])} | {fmt(r["BA"])} | {r["fixed"]} | {r["broken"]} | {r["net"]} | {fmt(r["FN_vs_TN_AUC"])} | {fmt(r["TP_vs_FP_AUC"])} |')
    lines+=['','M0为原始等权BCE重训，全部九组种子须通过与原checkpoint概率的1e-6重训门槛。M1为普通APK级SupCon；M2在相同正负样本集合上按Role相似度加权；M3在M2上按已有四个背景响应加权；M4置乱新增目标标签；M5在类别内置乱新增背景条件。所有组保留原三背景、同一Stage-I/路由/阈值/训练窗口与三个种子。','','## 配对不确定性','','| 对比 | BA差(pp) | 95%配对bootstrap区间 | Net差 | Holm p |','|---|---:|---:|---:|---:|']
    for r in interval:lines.append(f'| {r["candidate"]}−{r["reference"]} | {float(r["BA_gain_pp"]):+.3f} | [{float(r["BA_CI_low"]):+.3f},{float(r["BA_CI_high"]):+.3f}] | {int(r["net_gain"]):+d} | {float(r["holm_adjusted_p"]):.4g} |')
    lines+=['','这些区间是同一批APK的配对重抽样，未涵盖重叠训练窗口、APK家族依赖和既往开发选择的不确定性；不能称为独立年份显著性。','','## 表示与机制边界','','`background_response_diagnosis.csv`记录四列与类别、Host数量的描述性关联；`learned_representation_geometry.csv`仅在每个独立模型内部比较类内/类间距离和Role条件近邻纯度。T4的跨年距离只在同一个冻结编码器内计算，训练年向量已参与拟合。表征更紧凑本身不算纠错收益。','','## 下一步修改位置','','路由对当前冻结专家的可修复错误没有遗漏，因此需要先看第二层对已路由错误的条件判别。如果M1仅获有限收益，保持现有BCE，不增加对比头。若M2或M3相对上级与置乱/容量对照均达冻结门槛，下一轮只冻结对应训练损失作独立数据验证；仍不改变三背景原型、Host特征、MeanOpposition或阈值。','',f'具体分类：{", ".join(labels)}。不能由相关性推断某个背景量造成误判。','','## 计算成本','','`runtime_cost.csv`逐模型记录真实训练秒数、epoch、有效anchor和正样本对。历史Background Bank单独拟合秒数没有可靠计时，标为未测；本实验复用缓存，没有测APK从零解析的端到端时间。', '']
    write_text(OUTPUT/'final_report.md','\n'.join(lines))
    commands=f'''# 新版本复现：先确保v3脚本/输出目录都不存在并复制独立实现；禁止覆盖v2。
cd /home/boyuanjie/projects/tasa-droid
export ETBC_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_existing_three_background_contrastive_v3
export PYTHONDONTWRITEBYTECODE=1
/home/boyuanjie/envs/tif/bin/python -B scripts/care-existing-three-background-contrastive-v3/audit.py
/home/boyuanjie/envs/tif/bin/python -B scripts/care-existing-three-background-contrastive-v3/diagnose.py
/home/boyuanjie/envs/tif/bin/python -B scripts/care-existing-three-background-contrastive-v3/contrastive_training.py --pilot --tag T2_C1 --seed 900000 --arm M3 --device cuda
# 将run_full.sh中的版本化绝对路径更新为全新v3后执行完整T2-T4对照。
bash scripts/care-existing-three-background-contrastive-v3/run_full.sh
/home/boyuanjie/envs/tif/bin/python -B scripts/care-existing-three-background-contrastive-v3/summarize.py
/home/boyuanjie/envs/tif/bin/python -B scripts/care-existing-three-background-contrastive-v3/geometry.py
/home/boyuanjie/envs/tif/bin/python -B scripts/care-existing-three-background-contrastive-v3/finalize.py

# 这次v2只读结果：
cat {OUTPUT}/final_decision.json
cat {OUTPUT}/final_report.md
'''
    write_text(OUTPUT/'reproduce_commands.md',commands)
    print('ETBC_FINAL',main_status,flush=True)

if __name__=='__main__':main()
