#!/usr/bin/env python3
"""LiteStage-V1 Chinese report/decision. Reads only new result tables and frozen-asset manifest; writes final report in new output; trains nothing, reads no APK."""
from __future__ import annotations
import json
from common import *

def main():
    before=json.loads((OUT/'asset_manifest_before.json').read_text())
    changed=[]
    for r in before:
        if digest(r['path'])!=r:changed.append(r['path'])
    writejson('asset_integrity_report.json',dict(status='PASS' if not changed else 'FAIL',checked=len(before),changed=changed,scope='named historical files; APKs read-only but not all hashed'))
    if changed:raise RuntimeError('historical asset changed '+repr(changed))
    sm=OUT/'stage1_source2017_metrics.csv';tm=OUT/'two_stage_source2017_metrics.csv';fm=OUT/'future_frozen_metrics.csv'
    if not sm.is_file() or not tm.is_file():
        writejson('final_decision.json',dict(status='LITESTAGE_SOURCE_EVALUATION_BLOCKED',reason='source APK feature cohort incomplete or training failed; see run_state and extraction failures',future_used_for_training=False,asset_integrity='PASS'))
        writetext('final_report.md','# CARE-LiteStage-V1 阻碍报告\n\n已核实P31本身不建调用图，完成真实APK提取烟测与成本/依赖审计。但2017完整799 APK特征或训练未完成，不能报告LiteStage完整OOF、第二层增量或未来性能。请查看`run_state.json`与提取失败清单。\n')
        print('LITESTAGE_FINAL_BLOCKED',flush=True);return
    stage=readcsv(sm);two=readcsv(tm);fut=readcsv(fm) if fm.is_file() else []
    selected=json.loads((OUT/'source_selection.json').read_text());s={r['arm']:r for r in stage};t={r['method']:r for r in two}
    coverage=readcsv(OUT/'future_feature_coverage.csv') if (OUT/'future_feature_coverage.csv').is_file() else []
    cost=readcsv(OUT/'feature_extraction_cost.csv') if (OUT/'feature_extraction_cost.csv').is_file() else []
    mean_cost={r['operation']:r['mean_s'] for r in cost if r['scope']=='source_2017'}
    best=selected['global_final_arm'];lite=s['LITE_NESTED_SELECTED'];rf=s['L5_P31_RF'];lite2=t['T7_LITE_C43'];rf2=t['T1_P31_RF_C43']
    futurecomplete=bool(fut) and all(int(r['failed'])==0 for r in coverage)
    if float(lite['BA'])<float(rf['BA'])-.01:status='LITESTAGE_ACCURACY_TOO_LOW'
    elif float(lite2['BA'])>float(rf2['BA']) and int(lite2['net'])>0:status='LITESTAGE_COST_SAVING_NOT_VERIFIED'
    else:status='TWO_STAGE_DOES_NOT_BEAT_STRONG_RF'
    # No matched Host/call-graph construction clock -> end-to-end cost claim disallowed.
    adjunct=['LITESTAGE_COST_SAVING_NOT_VERIFIED','STAGE2_STRUCTURAL_INCREMENT_UNVERIFIED']
    if not futurecomplete:adjunct.append('FUTURE_DIAGNOSTIC_INCOMPLETE')
    report=f'''# CARE-LiteStage-V1 低成本第一层与选择性结构证据\n\n主结论：**{status}**。本轮为2017固定五折开发实验；2018—2021是历史冻结诊断，不是独立盲测。没有新APK加入训练。\n\n## 1. 真实依赖与成本\n\n正式P31读取DEX方法体invoke、按31类Role统计敏感方法比例；不构建完整调用图或Host。L1只读Manifest；L2再解析DEX元信息；L3扫描MethodId引用；L4扫描实际invoke。L3“引用”不等于调用，L4静态调用不等于运行时执行。本轮L4首例与正式P31最大差异6.65e−9。Source每APK平均Manifest解析{mean_cost.get('manifest_s','UNKNOWN')}秒，DEX解析{mean_cost.get('dex_metadata_s','UNKNOWN')}秒，引用表扫描{mean_cost.get('reference_s','UNKNOWN')}秒，invoke扫描{mean_cost.get('invoke_s','UNKNOWN')}秒，详见`feature_extraction_cost.csv`。OS页缓存未严格受控；阶段增量计时不能解释为独立冷启动成本。Host/调用图构建的匹配逐APK成本缺失，**无法证明端到端成本节省**。\n\n## 2. 第一层，强基线不可省略\n\n历史P31-RF：BA={rf['BA']}，错误={rf['errors']}；P31-SVM是另一分支，不能混算。经每个外层训练折内3折选择的LiteStage：BA={lite['BA']}，错误={lite['errors']}；全2017最终冻结候选为{best}。Manifest、DexMeta、RoleRef、RoleInvoke、Presence/Log/TopK各臂均在`stage1_source2017_metrics.csv`与`role_compression_results.csv`。TopK选择只看当前训练子折标签；Top8降低输入维度但仍须扫描相应DEX阶段，不自动降低提取成本。\n\n## 3. 同Fold冻结Direct-C43增量\n\n专家采用外层训练集等BCE训练、与路由无关的SCBE M0 Direct-C43。新Stage-I与该专家同2017 Fold、SHA和标签配对；新LiteStage的四成员MeanOpposition在本次训练集内校准。Top25%时：T6（Lite-only）BA={t['T6_LITE_ONLY']['BA']}；T7（Lite+C43）BA={lite2['BA']}，Fixed={lite2['fixed']}、Broken={lite2['broken']}、Net={lite2['net']}。强基线T0（RF-only）BA={t['T0_P31_RF']['BA']}；T1（RF+C43）BA={rf2['BA']}，Net={rf2['net']}。L1–L4的T2–T5采用各自RF不确定性路由，**不能与MeanOpposition臂把差异完全归因于特征**。10/25/40%预算见`router_budget_comparison.csv`；均为批次相对排序，不能写作固定单APK阈值部署。\n\n## 4. 结构证据边界\n\n`stage2_structural_ablation.csv`复用同Fold、同样容量的RPF RoleGraph query/readout 2×2直接专家，仅可在RPF分支内比较RoleGraph路径；S1纯Role与S2 Role+Host没有与SCBE M0完全匹配的直接训练checkpoint，故不声称完整S0–S5结构增量已证明。Role presence异标签碰撞案例在`role_collision_analysis.csv`，它不证明具体调用边是恶意因果。三背景BFA历史消融与本轮路由/模型分支不同，不可直接混算。\n\n## 5. 冻结未来与最终界限\n\n未来模型只使用2017训练并冻结；各年指标见`future_frozen_metrics.csv`。未来特征成功率见`future_feature_coverage.csv`；缺失APK不悄悄剔除。没有根据2020年或其他未来表现更改臂、阈值或路由。多重候选与历史开发数据使最优单臂结果仍有选择偏差，主要汇报是每外层折内部选择的Lite OOF。现有证据尚不能将输入维度缩减、特征提取时间和端到端平均成本等同，也不能证明调用图关系独立于Role统计。\n'''
    writetext('final_report.md',report)
    writejson('final_decision.json',dict(status=status,secondary=adjunct,selected_global_arm=best,source_lite_BA=float(lite['BA']),source_rf_BA=float(rf['BA']),source_lite_c43_BA=float(lite2['BA']),source_rf_c43_BA=float(rf2['BA']),lite_net=int(lite2['net']),rf_net=int(rf2['net']),future_complete=futurecomplete,stage2_extraction_cost_verified=False,structural_increment_verified=False,asset_integrity='PASS',future_used_for_training=False))
    writetext('reproduce_commands.md','`bash scripts/care-lite-stage-v1/status.sh`\n`bash scripts/care-lite-stage-v1/run.sh --resume`\n')
    print('LITESTAGE_FINAL_DONE',status,flush=True)
if __name__=='__main__':main()
