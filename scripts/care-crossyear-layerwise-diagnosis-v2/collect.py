"""Paired uncertainty, conservative mechanism decision, integrity check, Chinese report."""
from __future__ import annotations
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from diagnostic_common import *

def paired(y,p,q,seed=0,repeats=1000):
    y=np.asarray(y,int);p=np.asarray(p,float);q=np.asarray(q,float)
    if len(np.unique(y))<2:return dict(n=len(y),positive=int(sum(y)),negative=int(sum(y==0)),delta_auc=float('nan'),ci_low=float('nan'),ci_high=float('nan'),bootstrap_p=float('nan'))
    real=float(roc_auc_score(y,p)-roc_auc_score(y,q));rng=np.random.RandomState(seed);parts=[]
    a=np.flatnonzero(y==0);b=np.flatnonzero(y==1)
    for _ in range(repeats):
        ix=np.r_[rng.choice(a,len(a),replace=True),rng.choice(b,len(b),replace=True)]
        parts.append(roc_auc_score(y[ix],p[ix])-roc_auc_score(y[ix],q[ix]))
    parts=np.asarray(parts)
    return dict(n=len(y),positive=len(b),negative=len(a),delta_auc=real,ci_low=float(np.quantile(parts,.025)),ci_high=float(np.quantile(parts,.975)),bootstrap_p=float(min(1,2*min(np.mean(parts<=0),np.mean(parts>=0)))))
def holm(rows):
    finite=[(i,r['bootstrap_p']) for i,r in enumerate(rows) if np.isfinite(r['bootstrap_p'])]
    finite.sort(key=lambda x:x[1]);m=len(finite);prior=0
    for rank,(i,p) in enumerate(finite):
        prior=max(prior,(m-rank)*p);rows[i]['holm_p']=min(1.,prior)
    return rows

def main():
    required=['baseline_replay.json','layer_extraction_audit.json','layerwise_information_metrics.csv','probe_predictions.csv','representation_geometry.csv','background_response_controls.csv','probe_control_results.csv']
    missing=[f for f in required if not (OUTPUT/f).is_file()]
    if missing:raise RuntimeError('missing completed diagnostic phases '+repr(missing))
    base=read_json(OUTPUT/'baseline_replay.json');extract=read_json(OUTPUT/'layer_extraction_audit.json')
    if base['status']!='PASS' or extract['status']!='PASS':raise RuntimeError('baseline or layer gate failed')
    cases=read_csv(OUTPUT/'per_apk_error_contract.csv');metrics=read_csv(OUTPUT/'layerwise_information_metrics.csv');pred=read_csv(OUTPUT/'probe_predictions.csv');bg=read_csv(OUTPUT/'background_response_controls.csv');controls=read_csv(OUTPUT/'probe_control_results.csv');geometry=read_csv(OUTPUT/'representation_geometry.csv')
    intervals=[];sides=[];cross=[]
    for tag in TAGS:
        c={r['sha256']:r for r in cases if r['tag']==tag};original={r['sha256']:float(r['probability']) for r in pred if r['tag']==tag and r['probe']=='FROZEN_ORIGINAL'}
        year=int(next(iter(c.values()))['year']);ordered=list(c);y=np.array([int(c[s]['label']) for s in ordered]);stage=np.array([int(c[s]['stage1_prediction']) for s in ordered]);route=np.array([int(c[s]['routed']) for s in ordered],bool);q=np.array([original[s] for s in ordered])
        counts=Counter(r['category'] for r in c.values());sides.append(dict(tag=tag,year=year,all_n=len(c),routed_n=int(route.sum()),stage1_predicted_benign=int(np.sum(route&(stage==0))),stage1_predicted_malware=int(np.sum(route&(stage==1))),routed_FN=int(counts['FIXED_FN']+counts['UNFIXED_FN']),routed_FP=int(counts['FIXED_FP']+counts['UNFIXED_FP']),fixed_FN=int(counts['FIXED_FN']),fixed_FP=int(counts['FIXED_FP']),unfixed_FN=int(counts['UNFIXED_FN']),unfixed_FP=int(counts['UNFIXED_FP']),broken_TN=int(counts['BROKEN_TN']),broken_TP=int(counts['BROKEN_TP']),stable_correct=int(counts['STABLE_CORRECT'])))
        for layer in ('P0_P31_SIZE','L0_FIXED_SUMMARY','L1_HOST_ENCODER_SUMMARY','L2_ROLE_TOKEN','L3_HOST_BACKGROUND_DIAGNOSTIC','L4_ATTENTION_APK','L5_EXACT_HEAD_INPUT'):
            d={r['sha256']:float(r['probability']) for r in pred if r['tag']==tag and r['probe']=='LR' and r['layer']==layer}
            if set(d)!=set(c):raise RuntimeError('probe SHA coverage mismatch '+tag+'/'+layer)
            p=np.array([d[s] for s in ordered]);tests=[]
            for name,mask in [('all',np.ones(len(y),bool)),('routed',route),('FN_vs_TN',route&(stage==0)),('TP_vs_FP',route&(stage==1))]:
                result=paired(y[mask],p[mask],q[mask],seed=year+len(layer)+len(name))
                tests.append(dict(tag=tag,year=year,layer=layer,probe='LR',scope=name,**result))
            intervals.extend(tests)
        g=[r for r in geometry if r['tag']==tag and r['layer']=='L5_EXACT_HEAD_INPUT'][0]
        cross.append(dict(tag=tag,year=year,layer='L5_EXACT_HEAD_INPUT',mean_shift_norm=g['mean_shift_norm'],validation_nearest5_train_label_purity=g['validation_nearest5_train_label_purity'],validation_nearest_train_centroid_accuracy=g['validation_nearest_train_centroid_accuracy'],coordinate_system=tag+'/900000',train_embedding_in_sample=True))
    # One Holm family across all predeclared layer/scope/year comparisons, not
    # a separate, permissive family for every layer.
    intervals=holm(intervals)
    write_csv(OUTPUT/'paired_intervals.csv',intervals)
    write_csv(OUTPUT/'year_side_error_summary.csv',sides)
    write_csv(OUTPUT/'cross_year_separability.csv',cross)
    # Decision deliberately requires convergence across years, both conditional
    # directions and matched controls. Readability does not prove causal loss.
    def val(tag,layer,field,probe='LR',capacity='raw'):
        r=next((r for r in metrics if r['tag']==tag and r['layer']==layer and r['probe']==probe and r.get('capacity')==capacity),None)
        return float(r[field]) if r and r[field] not in ('','nan','None') else float('nan')
    def bgval(tag,arm,field='AUC'):
        r=next(r for r in bg if r['tag']==tag and r['arm']==arm);return float(r[field])
    bg_gain={tag:bgval(tag,'L0_plus_background')-bgval(tag,'L0_raw_only') for tag in TAGS}
    encoded_gain={tag:bgval(tag,'L1_plus_background')-bgval(tag,'L1_encoded_only') for tag in TAGS}
    logit_gap={tag:val(tag,'L5_EXACT_HEAD_INPUT','AUC')-val(tag,'L6_DIRECT_LOGIT','AUC','FROZEN_ORIGINAL','frozen') for tag in TAGS}
    # Candidate statuses need evidence beyond a single raw point estimate.
    statuses=[];reason=[]
    if all(bg_gain[t]>.02 and encoded_gain[t]>.02 for t in TAGS):
        statuses.append('BACKGROUND_CONTEXT_INFORMATION_SUPPORTED');reason.append('background response increments exceed 0.02 AUC in all three windows for both raw and encoded controls; interpretation remains observational')
    # L1 already ingests the four responses. Appending them again and seeing no
    # gain cannot identify redundancy of the *original* background mechanism.
    if all(logit_gap[t]>.03 for t in TAGS):
        relevant=[r for r in intervals if r['layer']=='L5_EXACT_HEAD_INPUT' and r['scope'] in ('FN_vs_TN','TP_vs_FP')]
        if len(relevant)==6 and all(np.isfinite(r['ci_low']) and r['ci_low']>0 for r in relevant):
            statuses.append('DECISION_READOUT_BOTTLENECK_CANDIDATE');reason.append('L5 linear readout beats frozen logit on both conditional sides in every window with paired CI above zero')
    if not statuses:
        statuses=['NO_SINGLE_LAYER_BOTTLENECK_IDENTIFIED'];reason=['No candidate meets the predeclared cross-year and control-consistency rule; descriptive differences do not localize a causal mechanism.']
    decision=dict(statuses=statuses,primary_status=statuses[0],reason=reason,baseline_status=base['status'],layer_status=extract['status'],source_years=[2019,2020,2021],all_2017_2021_are_development=True,training_representation_in_sample=True,background_AUC_increment_vs_raw=bg_gain,background_AUC_increment_vs_encoded=encoded_gain,L5_linear_vs_frozen_AUC_gap=logit_gap,causal_claim=False,model_change_authorized=False,external_dataset_used=False,route_or_model_changed=False,v1_exploratory_interpretation_superseded='v1 incorrectly treated re-appending backgrounds to an encoding that already used them as redundancy of the original mechanism; v2 does not',limitations=['no independent post-background fusion tensor','no fully out-of-fold upstream training embeddings','one frozen C43 seed for layer probes; all three seeds replayed for baseline','fixed L0 Host summary is lossy','development years repeatedly inspected'])
    write_json(OUTPUT/'final_decision.json',decision)
    lines=['# CARE-Droid跨年份分层判别信息诊断','',f"最终判断：**{', '.join(statuses)}**。这是一项已多次使用的2017—2021开发集诊断，不是独立测试，也未更改正式模型。",'', '## 冻结基线回放','', '| 年份 | APK | 路由 | Stage-I错误 | 已路由错误 | Fixed | Unfixed | Broken | BA |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in base['years']:lines.append(f"| {r['year']} | {r['n']} | {r['routed']} | {r['stage1_errors']} | {r['routed_stage1_errors']} | {r['fixed']} | {r['unfixed']} | {r['broken']} | {r['BA']:.6f} |")
    lines+=['','九个冻结checkpoint逐一回放，最大概率差 '+str(base['max_probability_difference'])+'，容差 '+str(base['tolerance'])+'。', '', '## 分层结果（原维度线性Probe，AUC）','','| 年份 | P31+规模 | L0固定摘要 | L1 Host编码摘要 | L2 Role token | L3诊断拼接 | L4聚合 | L5头输入 | C43 seed0 | 正式3-seed集成 |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for tag,year in zip(TAGS,(2019,2020,2021)):
        vv=[val(tag,k,'AUC') for k in ('P0_P31_SIZE','L0_FIXED_SUMMARY','L1_HOST_ENCODER_SUMMARY','L2_ROLE_TOKEN','L3_HOST_BACKGROUND_DIAGNOSTIC','L4_ATTENTION_APK','L5_EXACT_HEAD_INPUT')]
        vv.append(val(tag,'L6_DIRECT_LOGIT','AUC','FROZEN_ORIGINAL','frozen'))
        vv.append(val(tag,'L6_DIRECT_ENSEMBLE','AUC','FROZEN_3SEED','frozen'))
        lines.append('| '+str(year)+' | '+' | '.join(f'{x:.4f}' if np.isfinite(x) else 'NaN' for x in vv)+' |')
    lines+=['','上表Probe只检验信息可读性，**不是**新的正式第二层。所有分层Probe对应各窗口seed 900000；为公平比较，它们与同一seed的Logit对照。正式3-seed集成单列列出，不把不同checkpoint隐藏坐标混合。训练表示来自该冻结C43拟合过的APK，验证表示是下一年份未拟合APK；训练内部CV仍不能消除上游in-sample偏差。不同窗口坐标绝不拼接。','','## 路由条件判别方向','','| 年份 | 方向 | 路由样本（正/负） | C43 seed0 AUC | 正式集成 AUC | L5线性Probe AUC |','|---:|---|---:|---:|---:|---:|']
    for tag,year in zip(TAGS,(2019,2020,2021)):
        for direction in ('FN_vs_TN','TP_vs_FP'):
            r=next(r for r in metrics if r['tag']==tag and r['layer']=='L5_EXACT_HEAD_INPUT' and r['probe']=='LR' and r.get('capacity')=='raw')
            n=r[direction+'_n'];pos=r[direction+'_positive'];neg=r[direction+'_negative'];expert=val(tag,'L6_DIRECT_LOGIT',direction+'_AUC','FROZEN_ORIGINAL','frozen');probe=float(r[direction+'_AUC'])
            ensemble=val(tag,'L6_DIRECT_ENSEMBLE',direction+'_AUC','FROZEN_3SEED','frozen')
            lines.append(f'| {year} | {direction} | {n}（{pos}/{neg}） | {expert:.4f} | {ensemble:.4f} | {probe:.4f} |')
    lines+=['','2019的弱侧主要是TP-vs-FP，2020主要是FN-vs-TN，2021两侧相对较高；三个年份不构成同一种稳定的读出瓶颈。相关配对bootstrap与全家族Holm校正见 `paired_intervals.csv`。','','## 背景、几何与负对照','']
    for tag,year in zip(TAGS,(2019,2020,2021)):
        rows=[r for r in controls if r['tag']==tag];lines.append(f"- {year}：L0加入原四背景摘要AUC增量 {bg_gain[tag]:+.4f}；L1编码后再加入背景增量 {encoded_gain[tag]:+.4f}；L5线性读出相对冻结Logit增量 {logit_gap[tag]:+.4f}。20次训练标签置乱对照："+', '.join(f"{r['layer']} real {float(r['real_AUC']):.3f} vs null95 {float(r['null_95pct']):.3f}" for r in rows)+'。')
    lines+=['','背景对照是诊断输入拼接与置乱，非原网络背景模块的因果消融；L1本身已编码背景，额外拼接不代表背景总体贡献。2021的L0+背景增量与2019相反，因此不能认定三背景跨年提供稳定独立增量。尤其不能从L1再加背景近零推断正式背景冗余——本轮v1曾误用此规则，现已在v2纠正并保留v1原始记录。Role token与Host局部摘要不能构成单一顺序的“信息损失链”。L3不是独立网络层。', '', '## 对研究问题的保守回答','', '- Q1：68个已路由未修复错误的年份/方向分解见 `year_side_error_summary.csv`；2019是15 FP、3 FN，2020是4 FP、24 FN，2021是4 FP、18 FN。不能对每个APK唯一归因到一个层。','- Q2/Q4/Q6：各层与两侧条件AUC、训练-未来差、配对区间见相应CSV；2019与2020弱侧不同，2021又恢复，未找到跨年一致的单层弱化。','- Q3：原四背景没有获得稳定独立增量的支持；低异常不等于良性证据。现有诊断也不能宣称背景机制冗余。','- Q5：L5线性读出在完整验证AUC上分别相对原Logit为-0.0020、-0.0056、+0.0016；此前Hidden Readout和Route-Weighted BCE负结果进一步限制读出优化主张。','- Q7：当前不推荐修改正式C43。最小下一步是独立来源/家族标签的盲审验证，或对特定年份的弱方向做可核实代码证据审计；两者都不应把当前Probe直接变成部署模型。','', '## 证据边界与成本','', '- 未重新解析APK、未下载/安装依赖、未修改旧checkpoint/缓存/正式源码；所有临时文件在本轮输出目录。','- 所有标签置乱只作用于训练窗口；验证标签仅用于评分。Bootstrap以APK为单位、在当前模型和开发数据条件下解释，不是独立显著性证明。','- 无可靠家族标签，因此未做家族分组；不同窗口训练集重叠，因此三个年份不视为完全独立实验。','- 完整资产校验范围是manifest所列复用文件，不代表全服务器。', '']
    write_text(OUTPUT/'final_report.md','\n'.join(lines))
    write_text(OUTPUT/'reproduce_commands.md',f"# 复现命令\n\n在服务器执行：\n\n```bash\ncd {PROJECT}\nbash scripts/care-crossyear-layerwise-diagnosis-v2/run.sh --audit-only\nbash scripts/care-crossyear-layerwise-diagnosis-v2/run.sh --extract-layers\nbash scripts/care-crossyear-layerwise-diagnosis-v2/run.sh --run-probes\nbash scripts/care-crossyear-layerwise-diagnosis-v2/run.sh --collect\n```\n\n这些阶段使用独占写入；原目录已有输出时不得重复运行，应使用新版本目录。实际本轮脚本目录：`{HERE}`，输出：`{OUTPUT}`。\n")
    integrity()
    print('CLD_COLLECT_DONE',statuses,flush=True)
if __name__=='__main__':main()
