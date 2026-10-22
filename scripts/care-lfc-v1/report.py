#!/usr/bin/env python3
"""LFC-V1 read-only synthesis, costs and decisions. Inputs: new LFC tables plus frozen LiteStage timings; outputs reports/tables only in LFC_OUT. No APK reads or training."""
from __future__ import annotations
import json
import numpy as np
from core import *

def fnum(v):return float(v) if v not in ('','UNVERIFIED','UNKNOWN',None) else None
def main():
    manifest=json.loads((OUT/'asset_manifest_before.json').read_text())
    changed=[r['path'] for r in manifest if sha(r['path'])!=r]
    put_json('asset_integrity_report.json',dict(status='PASS' if not changed else 'FAIL',checked=len(manifest),changed=changed,scope='named historical files; existing APKs and all shards not individually rehashed'))
    if changed:raise RuntimeError('historical asset changed '+repr(changed))
    source=rows(OUT/'first_stage_metrics.csv');cascade=rows(OUT/'fixed_broken_by_candidate.csv');future=rows(OUT/'future_frozen_metrics.csv')
    if len(source)!=len(MAIN)+len(EXTRA) or len(future)!=4*len(source):raise RuntimeError('summary table coverage')
    s={r['candidate']:r for r in source};c={r['candidate']:r for r in cascade};shards=shard('source')
    cost=[]
    for arm in list(MAIN)+list(EXTRA):
        xs=[]
        for v in shards.values():
            t=v['timing']
            if arm=='ManifestOnly':xs.append(float(t['manifest_s']))
            elif cost_class(arm)=='DEX_REFERENCE_SCAN':xs.append(sum(float(t[k]) for k in ('manifest_s','dex_metadata_s','reference_s')))
            elif arm=='OriginalP31RF':continue
            else:xs.append(float(t['total_s']))
        if arm=='OriginalP31RF':
            prior=[r for r in rows(LITE/'feature_extraction_cost.csv') if r['scope']=='historic_P31_2017' and r['operation']=='total_seconds']
            if len(prior)!=1:raise RuntimeError('historic P31 timing missing')
            cost.append(dict(candidate=arm,extraction_class=cost_class(arm),n=799,mean_s=float(prior[0]['mean_s']),median_s=float(prior[0]['median_s']),p90_s=float(prior[0]['p90_s']),p95_s=float(prior[0]['p95_s']),measurement='historic unmatched environment; not directly comparable',stage2_incremental_s='UNVERIFIED',end_to_end_mean_s='UNVERIFIED'))
        else:
            a=np.asarray(xs,float)
            cost.append(dict(candidate=arm,extraction_class=cost_class(arm),n=len(a),mean_s=float(a.mean()),median_s=float(np.median(a)),p90_s=float(np.quantile(a,.9)),p95_s=float(np.quantile(a,.95)),measurement='same APK prior LiteStage shared-run phase timing; OS cache uncontrolled; not isolated standalone cost',stage2_incremental_s='UNVERIFIED',end_to_end_mean_s='UNVERIFIED'))
    put_csv('first_stage_cost_comparison.csv',cost)
    costs={r['candidate']:r for r in cost}
    pareto=[]
    for arm in list(MAIN)+list(EXTRA):
        q=costs[arm];eligible=arm!='OriginalP31RF';v=fnum(q['mean_s']);ba=fnum(s[arm]['BA'])
        dominated=eligible and any(other!=arm and other!='OriginalP31RF' and fnum(costs[other]['mean_s'])<=v and fnum(s[other]['BA'])>=ba and (fnum(costs[other]['mean_s'])<v or fnum(s[other]['BA'])>ba) for other in list(MAIN)+list(EXTRA))
        pareto.append(dict(candidate=arm,stage1_mean_phase_s=v,stage1_BA=ba,stage1_cost_BA_nondominated='UNVERIFIED_ACROSS_ENVIRONMENTS' if not eligible else int(not dominated),final_BA=c[arm]['BA'],net=c[arm]['net'],end_to_end_mean_s='UNVERIFIED',end_to_end_pareto='UNVERIFIED'))
    put_csv('cost_accuracy_pareto.csv',pareto)
    # The requested whole-system cost curve cannot be drawn without matched Stage-II extraction clocks.
    plot_status={'stage1_cost_vs_BA':'not plotted as strict cold/standalone cost; source phase-cost CSV provided','average_total_cost_vs_final_BA':'UNVERIFIED: matched Host/graph construction clocks absent','average_total_cost_vs_net':'UNVERIFIED: same reason','budget_vs_final_BA':'routing_budget_results.csv'}
    put_json('plot_status.json',plot_status)
    best=float(s['BestLiteStage']['BA']);strict=[]
    for arm in MAIN:
        if arm in ('OriginalP31RF','BestLiteStage'):continue
        r=c[arm];base=s[arm];folds=rows(OUT/'source_25pct_per_apk.csv')
        q=[t for t in folds if t['candidate']==arm]
        net_by=[sum(int(t['fixed'])-int(t['broken']) for t in q if int(t['fold'])==k) for k in range(1,6)]
        if float(base['BA'])>=best-.01 and int(r['net'])>0 and float(r['BA'])>=best and sum(n>=0 for n in net_by)>=4:strict.append(arm)
    status='EXPLORATORY_ONLY' if strict else 'NO_COMPLEMENTARY_LOW_COST_STAGE1_FOUND'
    base_good=[arm for arm in MAIN if arm not in ('OriginalP31RF','BestLiteStage') and int(c[arm]['net'])>0]
    old=c['BestLiteStage']
    lines=['# CARE-Droid LFC-V1：低成本第一层与冻结Direct-C43互补性','',f'主状态：**{status}**。全部候选使用2017固定外层OOF；2018–2021仅为历史开发诊断，不是独立盲测。没有重新训练Direct-C43。',
      '', '## 第一层与冻结专家主表','', '| 第一层 | 成本口径/平均秒 | BA-I | Routed | Fixed | Broken | Net | BA-Final |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in MAIN:
        m=c[arm];q=costs[arm]
        lines.append(f"| {arm} | {float(q['mean_s']):.3f}* | {float(s[arm]['BA']):.4f} | {m['routed']} | {m['fixed']} | {m['broken']} | {m['net']} | {float(m['BA']):.4f} |")
    lines+=['','* 2017既有逐APK提取阶段计时：Manifest/引用/Invoke均来自LiteStage共享提取，页缓存未严格控制；P31来自更早环境，不能直接作等价冷启动对比。Host、调用图、RoleGraph与C43前向的同APK增量耗时缺失，所以端到端成本、节省比例及完整Pareto均 **UNVERIFIED**。',
      '',f"最优LiteStage的外层折内选择第一层BA={best:.4f}。上轮12 Fixed、21 Broken（Net=-9），根本的可观测事实是冻结C43虽然能判对一部分第一层错例，但在不确定性路由的200个APK中误伤更多原本正确样本；这不是由更低第一层BA自动解释的。逐APK及概率区间见`source_25pct_per_apk.csv`、`confidence_bin_analysis.csv`。",
      '',f"在主候选中，Net为正的低成本臂：{', '.join(base_good) if base_good else '无'}。达到预设的第一层接近强基线、正Net、最终BA不低于{best:.4f}且至少4/5折Net非负的候选：{', '.join(strict) if strict else '无'}。即便出现这样的候选，也因多候选并列查看外层OOF，状态仅为探索性，不能据此宣称无偏获胜。",
      '', '## 语义表示、路由与结构结论','',
      'RefPresence/RefCount/RefLog/RefDensity/RefTop16、InvokeCount/Presence/Density共用既有原始缓存，但按合法低成本字段重新拟合。Top16只能降低输入维度，方法引用扫描仍须完成。引用只表示DEX MethodId存在，invoke只表示字节码调用指令，不是运行时行为。各表达结果见`first_stage_metrics.csv`。',
      '主路由为每外层Fold内按第一层不确定度排序的25%批次相对路由；10%/40%为预声明敏感性。路由不读取C43概率或真实标签。既有MeanOpposition只对其兼容的原P31/Lite委员结果作历史附加参考，不能无成本移植到每个新RF。',
      '本轮所有组合使用同Fold、全样本覆盖的冻结C43 H0专家，没有新建Host/RoleGraph模型。因缺乏与这些新第一层完全匹配的Role-only、Host-only及结构破坏专家，不能证明C43增益独立于廉价Role统计或模型容量。',
      '', '## 冻结未来诊断','', '| 年份 | 候选 | BA-I | BA-Final | Fixed | Broken | Net |','|---|---|---:|---:|---:|---:|---:|']
    for yr in (2018,2019,2020,2021):
        for arm in MAIN:
            r=next(v for v in future if int(v['year'])==yr and v['candidate']==arm)
            lines.append(f"| {yr} | {arm} | {float(r['stage1_BA']):.4f} | {float(r['final_BA']):.4f} | {r['fixed']} | {r['broken']} | {r['net']} |")
    lines+=['','四年全部1989 APK有既有LiteStage特征与冻结C43概率，未用未来标签训练、选Role或调阈值。未来比较仍属反复使用过的开发数据。','',
      '## 科研决策','', '只在兼顾第一层质量、可复核提取成本、正向净纠错和结构对照时，才考虑替换正式第一层。本轮不能把Stage-I阶段计时当作完整部署成本；若没有严格候选，应保留强RF并停止此轮低成本互补搜索，不再根据已见外层/未来结果调路由。']
    put_text('final_report.md','\n'.join(lines)+'\n')
    put_json('final_decision.json',dict(status=status,strict_candidates=strict,positive_net_low_cost_arms=base_good,source_2017=799,future_2018_2021=1989,source_BestLiteStage_BA=best,source_BestLiteStage_C43_net=int(old['net']),source_BestLiteStage_C43_final_BA=float(old['BA']),C43_retrained=False,C43_all_source_predictions=True,stage2_incremental_cost_verified=False,structural_information_independent_value_verified=False,future_used_for_selection=False,asset_integrity='PASS',selection_boundary='all candidate systems compared descriptively on same outer OOF; no post-hoc winning-arm unbiased claim'))
    print('LFC_REPORT_DONE',status,flush=True)
if __name__=='__main__':main()
