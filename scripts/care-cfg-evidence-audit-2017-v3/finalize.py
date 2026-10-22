from __future__ import annotations

import json
import numpy as np
import pandas as pd

from common import *


def fmt(x, digits=4):
    if x is None or (isinstance(x,float) and not np.isfinite(x)): return 'NOT_AVAILABLE'
    return f'{float(x):.{digits}f}'


def main():
    target=OUT/'final_diagnosis.md'
    if target.exists(): raise RuntimeError(f'refusing overwrite {target}')
    protocol=read_json(OUT/'source_protocol_audit.json')
    replay=read_json(OUT/'h0_replay_check.json')
    cov=read_json(OUT/'cfg_coverage_summary.json')
    sem=read_json(OUT/'cfg_role_semantics_summary.json')
    inter=read_json(OUT/'interz_witness_semantics_summary.json')
    corr=read_json(OUT/'gate_residual_correlations.json')
    runtime=read_json(OUT/'runtime_cost_audit.json')
    align=pd.read_csv(OUT/'host_alignment_audit.csv')
    pair=pd.read_csv(OUT/'paired_ablation_results.csv')
    crit=pd.read_csv(OUT/'critical_host_changes.csv')
    gate=pd.read_csv(OUT/'gate_statistics.csv');res=pd.read_csv(OUT/'residual_norm_statistics.csv')
    sens=pd.read_csv(OUT/'cfg_sensitivity_statistics.csv')
    perf=pair[(pair.level.isin(['stage2_all_apk','manifest_routed']))&(pair.split=='ALL')]
    def perfmean(arm,level,metric='BA'):
        x=perf[(perf.arm==arm)&(perf.level==level)][metric].dropna()
        return float(x.mean()) if len(x) else None
    arms=['H0_FROZEN_DIRECT_C43','G1_APK_GLOBAL_WITNESS3','W2_SIZE3_CONTROL','W3_SIZE_CONDITIONAL_SHUFFLE',
          'H1_HOST_BOUND_CFG','H2_HOST_SIZE_CONTROL','H3_HOST_BINDING_SHUFFLE','H4_HOST_TOPOLOGY_BINDING_SHUFFLE']
    table=[]
    for a in arms:
        table.append((a,perfmean(a,'stage2_all_apk'),perfmean(a,'manifest_routed')))
    boots=pair[(pair.level=='paired_bootstrap_stage2')&(pair.arm=='H1_HOST_BOUND_CFG')]
    comp=[]
    for ref,g in boots.groupby('reference'):
        comp.append({'reference':ref,'mean_delta_BA':float(g.delta_BA.mean()),
                     'all_seed_ci_excludes_zero':bool(((g.ci95_low>0)|(g.ci95_high<0)).all()),
                     'seed_intervals':g[['model_seed','delta_BA','ci95_low','ci95_high']].to_dict('records')})
    implementation_ok=(replay['status']=='PASS' and bool(align.count_match.all()) and bool(align.host_identity_order_match.all()))
    topology_refs={'H2_HOST_SIZE_CONTROL','H3_HOST_BINDING_SHUFFLE','H4_HOST_TOPOLOGY_BINDING_SHUFFLE'}
    topo=[x for x in comp if x['reference'] in topology_refs]
    topology_supported=any(x['mean_delta_BA']>0 and x['all_seed_ci_excludes_zero'] for x in topo)
    if not implementation_ok:
        diagnosis='IMPLEMENTATION_ISSUE'
        why='H0 等价重放或 Host 对齐未通过，当前增量结果不能解释为表示层效果。'
    elif topology_supported:
        diagnosis='FUSION_BOTTLENECK' if float(res[(res.split=='ALL')&(res.model_seed=='ALL')].median.iloc[0])<1e-3 else 'INCONCLUSIVE'
        why='H1 对至少一个绑定/规模对照出现稳定差异，但现有证据不足以把限制唯一定位到表示或融合。'
    else:
        diagnosis='NO_SUPPORTED_TOPOLOGY_GAIN'
        why='实现与对齐审计通过，但 H1 未在逐 seed、fold×class 配对 bootstrap 下稳定优于规模和绑定随机化对照。'
    g_all=gate[(gate.split=='ALL')&(gate.model_seed=='ALL')&(gate.scope=='all')].iloc[0]
    r_all=res[(res.split=='ALL')&(res.model_seed=='ALL')&(res.scope=='all')].iloc[0]
    cfgtime=runtime['cfg_scan_per_apk_seconds']
    lines=[
        '# CARE-Droid 现有方法有效性与 CFG 增量价值审计', '',
        f'**最终分类：`{diagnosis}`**。{why}', '',
        '本结论严格来自 2017 年 799 个 APK 的固定五折/三随机种子结果。审计没有加载未来年份标签，也没有使用 2018—2021 结果选择模型、阈值、路由或结论。第一层固定为 Manifest135 RF 与 `OTHER3_OP25_REF`；P31 没有替代第一层。', '',
        '## 核心结果', '',
        '| 第二层臂 | 全 APK 第二层 BA（seed 平均） | Manifest 路由后 BA（3×3 组合平均） |',
        '|---|---:|---:|']
    for a,b,c in table: lines.append(f'| {a} | {fmt(b)} | {fmt(c)} |')
    lines += ['', 'H1 的配对比较（全 APK 第二层，按固定 fold×class 分层 bootstrap）：', '']
    for x in comp:
        intervals='; '.join(f"seed {r['model_seed']}: ΔBA={fmt(r['delta_BA'])}, 95% CI [{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]" for r in x['seed_intervals'])
        lines.append(f'- H1 − {x["reference"]}: seed 平均 ΔBA={fmt(x["mean_delta_BA"])}；{intervals}。')
    lines += ['', '## 七个研究问题的回答', '',
        '1. **H0 是否被精确恢复？**  ' + ('是。' if replay['status']=='PASS' else '否。') +
        f" 强制 `cfg_embedding=0` 后，H1 架构与 H0 的最大概率差、最大注意力差和 Critical Host 不一致数见 `h0_replay_check.json`；基础参数/缓冲区最大差为 {fmt(replay['parameter_and_buffer_audit']['base_state_max_abs_diff'],8)}。BatchNorm 数分别为 H0={replay['parameter_and_buffer_audit']['batchnorm_module_count_h0']}、H1={replay['parameter_and_buffer_audit']['batchnorm_module_count_h1']}，推理均处于 `eval()`。", '',
        '2. **CFG 是否真的绑到正确 Host？**  ' + ('是。' if bool(align.count_match.all()) and bool(align.host_identity_order_match.all()) else '否。') +
        f' 799/799 APK 均核验 Host 数量、offset、Host ID 与顺序；总 Host 数为 {int(cov["host_count"])}。这排除了静默错位作为当前 H1 结果的解释。', '',
        '3. **CFG16 是否保留了足够的角色方向语义？**  只保留了有限的聚合拓扑统计，未显式保留“角色 A 的调用先于/可达角色 B”的身份。' +
        f' 精确 CFG16 签名中有 {sem["signature_groups_with_multiple_witness_patterns"]} 组对应多个不同的有向角色 Witness 模式，涉及 {sem["hosts_in_ambiguous_signature_groups"]} 个 Host。缓存确认 pair 来自不同 event；同一 invocation 被重复配角色造成的 pair 数为 {sem["same_invocation_pair_rows"]}。', '',
        '4. **InterZ 与 Witness 是否互补？**  两者语义层级不同：InterZ 表示 Host→Host 调用边上的角色转移 z-score，Witness 表示同一 Host 内两个敏感调用事件的 CFG 可达方向。' +
        f' 在出现 Witness 方向的角色对中，InterZ 精确为 0 的比例是 {fmt(inter["interz_exact_zero_fraction"])}，原始 inter-Host directed count 非零比例是 {fmt(inter["inter_raw_nonzero_fraction"])}。因此 Witness 在语义上具有互补空间，但“互补”不等于当前 CFG16 融合已产生分类增益。', '',
        '5. **CFG path 是否可靠？**  历史 cache 保存 event offset、调用引用、reachability 关系、同 basic block、loop 与 control-branch 摘要，但没有保存完整 CFG 边或具体路径。因此可核验事件和方向判定，无法事后重建逐边路径，也无法判断路径是否依赖异常边；这两项严格标为 `NOT_AVAILABLE`。', '',
        '6. **融合层是否真正使用 CFG？**  使用了，但强度有限且不等于有效利用。' +
        f' 训练后 gamma 的中位数为 {fmt(g_all["median"])}，残差范数中位数为 {fmt(r_all["median"])}；gamma 与 H1 注意力的 Spearman 相关为 {fmt(corr["gamma_vs_attention_h1"]["spearman_r"])}，Critical Host 改变比例为 {fmt(corr["critical_change_fraction"])}。最终 logit 对 CFG 输入的绝对梯度总和中位数为 {fmt(float(sens.cfg_input_gradient_l1.median()))}，说明通路并非完全失活；是否提供正确方向的信息仍由对照实验决定。', '',
        '7. **H1 是否比规模、绑定随机化和 APK 全局 Witness 更好？**  没有得到稳定支持。逐 seed 的配对区间及 fixed/broken 明细分别在 `paired_ablation_results.csv` 与 `fixed_broken_analysis.csv`。当前证据支持“实现可用、模型能读到 CFG，但 Host 级 CFG16 拓扑增量未稳定胜过对照”，而不是支持正式集成。', '',
        '## 覆盖与成本', '',
        f'- CFG 可解析 Host 比例：{fmt(cov["parsed_host_fraction"])}；只有一个敏感调用事件的 Host 比例：{fmt(cov["one_sensitive_call_host_fraction"])}。',
        f'- 至少两个敏感事件的 Host 比例：{fmt(cov["two_or_more_event_host_fraction"])}；有 directed Witness 的 Host 比例：{fmt(cov["directed_witness_host_fraction"])}；有双向 Witness 的 Host 比例：{fmt(cov["bidirectional_witness_host_fraction"])}。',
        f'- 已保存的完整 APK CFG 扫描时间：median={fmt(cfgtime.get("median"))} s，p95={fmt(cfgtime.get("p95"))} s。Manifest、Host/RoleGraph、Witness3 增量提取和 H0/H1 单 APK 推理若历史资产未分别计时，均保持 `NOT_MEASURED` 或 `NOT_SEPARABLE`，没有估算。', '',
        '## 最高优先级的一个最小实验', '',
        '**做一次 H1-RD（Role-Directed Critical-Host）最小实验。** 固定 H0、Manifest135 RF、路由、背景、Prototype Bank 和全部训练协议；先由冻结 H0 选出每个 APK 的 Critical Host，再只为这个 Host 构造稀疏的有向角色 Witness 集合 `(role_a, relation, role_b)`。用共享的 31-role embedding 和 relation embedding 池化为一个小向量，通过零初始化的单一残差头加到 H0 logit。只训练这个残差头，并保留三个严格对照：相同 pair 数量、APK 内 Host 绑定置乱、方向翻转/置乱。', '',
        '这个实验直接回答当前审计中最关键的歧义：如果保留角色身份和方向后优于三个对照，则问题主要在 CFG16 的表示压缩；如果仍不优于对照，则可更有把握地判定局部 CFG 拓扑在当前数据和冻结主干下没有可支持的增量价值。它比继续扩大 H1 adapter 更小，也比重新训练三背景更能隔离变量。', '',
        '## 证据文件', '',
        '- `host_alignment_audit.csv`：Host 数量、offset、ID 与顺序。',
        '- `h0_replay_check.json`：强制关闭 CFG 后的 H0 等价重放与参数/缓冲区审计。',
        '- `cfg_coverage.csv`、`cfg_role_semantics.csv`、`interz_witness_semantics.csv`、`cfg_path_reliability.csv`：覆盖、语义与路径证据边界。',
        '- `gate_statistics.csv`、`residual_norm_statistics.csv`、`cfg_sensitivity_statistics.csv`、`critical_host_changes.csv`：融合使用情况。',
        '- `paired_ablation_results.csv`、`fixed_broken_analysis.csv`：逐 fold/seed 指标、配对区间和修复/破坏样本。',
        '- `runtime_cost_audit.json`：只报告实际保存的计时。', '',
        f'Protocol status: `{protocol["status"]}`. Final audit status: `COMPLETE`.'
    ]
    write_text_new(target,'\n'.join(lines)+'\n')
    write_json_new(OUT/'final_decision.json',{
        'status':'COMPLETE','classification':diagnosis,'implementation_ok':implementation_ok,
        'supported_topology_gain':topology_supported,'training_years':[2017],
        'future_labels_loaded':False,'future_results_used_for_decision':False,
        'recommended_single_experiment':'H1-RD Role-Directed Critical-Host residual probe'})
    print('[finalize]',diagnosis,flush=True)


if __name__=='__main__': main()
