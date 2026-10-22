"""Finalize only owned forensic outputs, preserving diagnostic boundaries and protected original assets."""
from __future__ import annotations
import ast,zipfile
from collections import Counter
from audit_common import *


def main():
    marker=OUTPUT/'completion_manifest.json'
    if marker.exists():raise RuntimeError('completion already exists; no overwrite')
    behavior=read_json(OUTPUT/'behavior_observations.json')
    replay=read_json(OUTPUT/'replay_validation.json')
    selection=read_json(OUTPUT/'selection_intervention_summary.json')
    identity=read_json(OUTPUT/'canonical_identity_audit.json')
    details=read_json(OUTPUT/'private/canonical_identity_details.json')
    code=read_json(OUTPUT/'code_observability.json')
    sources=[]
    for p in [PROJECT/'src/tasa_droid/p2/graph_representation.py',PROJECT/'scripts/care-source-recovery-stage2only-v1/extractor_repo/src/tasa_droid/p2/graph_representation.py']:
        if not p.is_file():continue
        before=asset(p,'current source identity implementation, not historical import provenance proof')
        lines=p.read_text().splitlines()
        anchors=[i for i,l in enumerate(lines) if 'def canonical_signature' in l or 'signature = canonical_signature(method_signature(node))' in l]
        excerpts=[dict(start_line=i+1,text='\n'.join(lines[i:min(len(lines),i+(4 if 'def canonical_signature' in lines[i] else 25))])) for i in anchors]
        after=asset(p,before['purpose'])
        if before!=after:raise RuntimeError('source changed during read')
        sources.append(dict(asset=before,excerpts=excerpts))
    dump(OUTPUT/'identity_source_evidence.json',dict(current_sources=sources,historical_import_provenance_verified=False,observed_frozen_ID_collapses=True,interpretation='current code casefolds signatures before node_records lookup, merging identity keys; frozen selected IDs map to multiple exact DEX signatures/definitions. This is not a measured classification cause.'))
    affected={k:len({r['sha256'] for r in details if r['kind']==k and r['target']=='True'}) for k in sorted({r['kind'] for r in details})}
    recovered=next(r for r in selection['groups'] if r['cohort']=='REGRESSED_FN' and r['policy']=='OLD_CRITICAL_AND_ATTENDED')
    if replay['status']!='PASS' or not selection['all_weight_hashes_unchanged'] or recovered['repaired']!=0:raise RuntimeError('summary evidence differs; inspect before finalization')
    integrity();check=read_json(OUTPUT/'baseline_integrity_check.json')
    decision=dict(status='TARGETED_FORENSICS_COMPLETE_CLASS_CAUSES_UNRESOLVED',audited_target_APKs=73,unique_matched_correct_APKs=34,inspected_APKs=107,probability_replay='PASS',device=replay['device'],formal_integrity=check['status'],protected_assets=check['asset_count'],regressed_FN=10,regressed_FN_rescued_by_old_critical=0,regressed_FN_rescued_by_old_attention=0,regressed_FN_rescued_by_both=0,persistent_targets_all_nine_seed_predictions_wrong=behavior['persistent_all_9_seed_predictions_wrong'],identity_ambiguities=identity['kind_counts'],identity_affected_target_APKs=identity['target_APKs'],identity_affected_correct_APKs=identity['correct_control_APKs'],identity_affected_target_APKs_by_kind=affected,verified_class_error_causes=[],model_training_authorized=False,model_training_performed=False,official_assets_modified=False,APK_execution_performed=False,development_not_independent_final_test=True,next_action='independently review the 10 blinded regression dossiers for exact class-direction code facts; separately specify a no-label corpus identity-scope audit before any identity repair. Do not tune attention, thresholds, route or models from this diagnostic.')
    dump(OUTPUT/'decision.json',decision)
    rows=['# CARE-Droid多年份错误样本定点取证审计','','## 最终结论','','`TARGETED_FORENSICS_COMPLETE_CLASS_CAUSES_UNRESOLVED`。审计完成，但尚未确认具体的良恶类别误判原因；不授权训练、阈值搜索或正式身份修复。','','## 范围与可比性','','检查2019—2021年同一Stage-I/Top25路由下的73个目标：持续FN35、持续FP18、持续Broken-TP3、持续Broken-TN7、旧专家正确而C1退步的FN10。另有34个独立正确对照，共107个APK。2018三个专家相同，不计作三次独立比较。正确对照可能复用，73组不代表73个独立对照。匹配仅用于事后取证，不构成训练或随机实验。2017—2021现为开发域，不能重称独立最终测试集。','','## 已实际验证','','| 检查 | 结果 |','|---|---|',f'| 冻结模型/预处理数值重放 | {len(replay["details"])}个checkpoint，全部PASS，最大概率差{max(r["max_probability_difference"] for r in replay["details"]):.9g}，原容差1e-6未放宽 |',f'| 静态DEX扫描 | {code["unique_APKs_inspected"]}/107完成 |',f'| 已选Host精确DEX方法体 | {code["exact_Host_bodies"]}/{code["selected_Hosts"]}；13个歧义保留UNKNOWN |','| 冻结输入对齐 | 全107 APK的Host顺序、Role presence、端点计数、Relation31重建一致 |',f'| 持续错误的种子稳定性 | {behavior["persistent_all_9_seed_predictions_wrong"]}/63在三个专家×三个种子中全部错误 |','| 10个退步FN | C0三个种子全对，C1三个种子全错；集成最高注意力Host均变化 |','','## Host选择是否是充分解释？','','在同一C1模型中重算C1编码，只替换旧专家的Host索引/注意力，绝不混用旧模型隐藏坐标、scaler或head。权重及buffer前后哈希全部相同。','','| 冻结C1内的计算干预 | 10个退步FN修复 | 63个持续目标修复 | 34个正确对照被破坏 |','|---|---:|---:|---:|','| 换回C0关键Host | 0 | 3 | 4 |','| 换回C0注意力 | 0 | 1 | 3 |','| 同时换回两者 | 0 | 3 | 3 |','','因此，Host选择变化不是这10例退步的充分解释或有效单独修复方式。变化定位于拟合后的预处理/表示/决策链条，但不能进一步唯一归因到背景、编码器、RoleGraph或读出。旧专家正确也不保证其依据就是正确语义。','','20次Role匹配随机选择只作计算对照，不是独立APK重复，也不是语义保持APK变换。有效非退化对照数量各组不同：退步FN10、持续目标42、正确对照25；不得将其期望计数与63/34全组直接比较。无因果或显著性宣称。','','## 找到的具体工程线索：身份歧义','','13个已选Host无法唯一回溯：8个存在不同大小写精确方法签名归一至同一ID，另5个为跨DEX重复精确定义。5个APK受影响：3个持续FN、2个正确对照；10个退步FN无此歧义。分类分别统计可能重叠，详见canonical_identity_audit.json及private明细。','','示例：同一类中的`getIMEI(Context)`和`getImei(Context)`具有不同DEX指令内容，却归一成相同小写签名/冻结Host ID。当前两份graph_representation.py均在node_records查找前对完整签名casefold，并在相同key下合并角色及边关系。精确源码摘录/哈希已存identity_source_evidence.json。历史缓存究竟导入哪份代码的完整溯源尚未证明；不能将当前代码版本直接等同历史运行版本。','','该结果确认的是身份可追踪性缺口，不证明它导致分类错误，更不能解释其余70个目标或概念漂移。未修改提取器、Host集合、冻结缓存或APK；未知的跨DEX运行选择也不作猜测。','','## 尚未回答','','- 关键类别事实是否存在于选中的原始代码，是否足以解释数据集标签？','- 这些事实是否已进入C43，或仅存在于未选代码？','- 滚动训练的预处理/背景/编码/读出中哪个变化导致退步？本轮不做不兼容模块拼接。','- 是否存在概念漂移？本次观察不能确认，也不能排除。','- 训练标签和外部语义证据是否可靠？本轮没有独立标签真值核验。','','权限检查不构成良性证明，缺失/UNKNOWN不构成恶意证明，注意力不是Host真值。未选代码不宣称不存在；静态分支不等于控制依赖或运行因果。','','## 最小后续行动','','先审10个退步FN的盲审三元组：blind_priority_review/含10案例、14个唯一代码成员，隐藏标签、预测和目标位置；应用代码名可能暴露功能，不能宣称身份完全盲化。逐条引用DEX、精确方法和指令偏移，先记录可核实程序事实，再独立判断类别方向。若没有可靠事实，保留UNKNOWN，不再自动增加模型。review_tools.py只能验证引用位置，不会把主观意见升级为已确认根因。','','身份问题另需单独冻结无标签的全语料范围审计；在确认覆盖、历史来源和表示影响之前不修正式提取器。','','## 安全、完整性与成本','','仅CPU冻结推理与静态DEX解析；不安装/执行/改写APK，不训练，不改路由/阈值/正式脚本。GPU工具可见但PyTorch驱动初始化失败，失败输出v1保留；实际在独立v2用CPU完成。任意字符串常量已从盲审文本中脱敏（仅保留权限名）。',f'保护清单中{check["asset_count"]}份实际复用资产（含107个原始APK）前后大小、mtime、SHA256一致：PASS；不是全服务器完整性声明。',f'主审计耗时{read_json(OUTPUT/"phase_timing.json")["total_seconds"]:.2f}秒，另执行了Host选择和身份核查；主审计计时不冒充所有后续步骤总耗时。','','详细证据：replay_validation.json、behavior_observations.json、selection_intervention_summary.json、canonical_identity_audit.json、identity_source_evidence.json、baseline_integrity_check.json。私有标签key和模型轨迹不得混入盲审包。','']
    safe(OUTPUT/'report.md').write_text('\n'.join(rows),encoding='utf-8')
    commands=f'''# 实际路径与复现边界

当前完成输出：{OUTPUT}
全部命令在 /home/boyuanjie/projects/tasa-droid 执行；无Git、无训练、无APK执行。
新完整复现必须选择不存在的新版本目录，不能覆盖v1/v2；示例v3也必须先检查。

```bash
export PYTHONDONTWRITEBYTECODE=1
export MEEA_OUTPUT=/home/boyuanjie/projects/tasa-droid/outputs/care_multiyear_error_evidence_audit_v3
bash scripts/care-multiyear-error-evidence-audit-v1/run.sh --device cpu --workers 2
/home/boyuanjie/envs/tif/bin/python -B scripts/care-multiyear-error-evidence-audit-v1/selection_intervention.py
/home/boyuanjie/envs/tif/bin/python -B scripts/care-multiyear-error-evidence-audit-v1/inspect_identity.py
/home/boyuanjie/envs/tif/bin/python -B scripts/care-multiyear-error-evidence-audit-v1/review_tools.py --priority-packet
/home/boyuanjie/envs/tif/bin/python -B scripts/care-multiyear-error-evidence-audit-v1/finalize_report.py
```

查看本次完成结果（只读）：
```bash
cat {OUTPUT}/decision.json
cat {OUTPUT}/report.md
cat {OUTPUT}/selection_intervention_results.csv
cat {OUTPUT}/canonical_identity_audit.json
cat {OUTPUT}/baseline_integrity_check.json
```

上述是事后诊断，不是模型/策略选择协议。未来年月数据已是开发域，不得重新声明独立最终评估。
'''
    safe(OUTPUT/'reproduce_commands.md').write_text(commands,encoding='utf-8')
    implementation=[]
    for p in sorted(HERE.glob('*.py')):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        if ast.get_docstring(tree) is None:raise RuntimeError('missing module docstring '+str(p))
        implementation.append(asset(p,'independent audit implementation'))
    dump(OUTPUT/'implementation_manifest.json',implementation)
    packet=OUTPUT/'blind_priority_review';z=safe(OUTPUT/'priority_blind_review.zip')
    if z.exists():raise RuntimeError('packet archive exists; no overwrite')
    with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(packet.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(packet))
    dump(marker,dict(status='COMPLETE',priority_zip_sha256=sha256(z),output=str(OUTPUT),model_training_authorized=False))
    print('MEEA_FINALIZED',str(OUTPUT/'report.md'),flush=True)


if __name__=='__main__':main()
