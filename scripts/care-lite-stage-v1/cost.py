#!/usr/bin/env python3
"""LiteStage-V1 measured extraction cost and cost–accuracy bounds. Reads new per-APK timings, historical P31 timing and new metrics; writes only new CSV/MD; trains nothing, reads no APK."""
from __future__ import annotations
import json
import numpy as np
from common import *

def stat(name,v):
    v=np.asarray(v,float)
    if not len(v):return dict(operation=name,n=0,mean_s=None,median_s=None,p90_s=None,p95_s=None,max_s=None)
    return dict(operation=name,n=len(v),mean_s=float(v.mean()),median_s=float(np.median(v)),p90_s=float(np.quantile(v,.9)),p95_s=float(np.quantile(v,.95)),max_s=float(v.max()))
def main():
    source,missing=load_shards('source');future,fmissing=load_shards('future')
    rows=[]
    for scope,values in [('source_2017',list(source.values())),('future_2018_2021',list(future.values()))]:
        if not values:continue
        timing=[v['timing'] for v in values];fields=['manifest_s','dex_metadata_s','reference_s','invoke_s','total_s']
        for field in fields:rows.append(dict(scope=scope,cache_condition='OS page-cache uncontrolled; first-pass source and later future',cost_type='actual APK parse/scan',**stat(field,[t[field] for t in timing])))
        rows.append(dict(scope=scope,cache_condition='same APKs',cost_type='observed APK size bytes',**stat('apk_bytes',[v['apk_bytes'] for v in values])))
    # Old P31 audit time has its own historic extraction environment and is not a matched cold-cache measurement.
    with np.load(P31,allow_pickle=False) as z:
        names=[str(x) for x in z['audit_names']];audit=np.asarray(z['audit_values'],float)
    if 'total_seconds' in names:rows.append(dict(scope='historic_P31_2017',cache_condition='historic uncontrolled',cost_type='prior full P31 scan; not matched to current run',**stat('total_seconds',audit[:,names.index('total_seconds')])))
    writecsv('feature_extraction_cost.csv',rows)
    metrics_rows=readcsv(OUT/'stage1_source2017_metrics.csv') if (OUT/'stage1_source2017_metrics.csv').exists() else []
    two=readcsv(OUT/'two_stage_source2017_metrics.csv') if (OUT/'two_stage_source2017_metrics.csv').exists() else []
    costs={r['operation']:float(r['mean_s']) for r in rows if r['scope']=='source_2017' and r['mean_s']}
    m=costs.get('manifest_s',0);d=costs.get('dex_metadata_s',0);r=costs.get('reference_s',0);i=costs.get('invoke_s',0)
    stage={'L1':m,'L2':m+d,'L3':m+d+r,'L4':m+d+i,'L5':None}
    est=[]
    for arm,prefix in [('L1_MANIFEST','L1'),('L2_DEXMETA','L2'),('L3_COUNT','L3'),('L4_COUNT','L4')]:
        val=next((x for x in metrics_rows if x['arm']==arm),None)
        if val:est.append(dict(arm=arm,stage1_extraction_mean_s=stage[prefix],stage1_BA=float(val['BA']),stage2_incremental_extraction_s='UNVERIFIED',route_fraction=.25,end_to_end_mean_s='UNVERIFIED'))
    est.append(dict(arm='L5_P31_RF',stage1_extraction_mean_s='HISTORIC_UNCONTROLLED',stage1_BA=next((float(x['BA']) for x in metrics_rows if x['arm']=='L5_P31_RF'),None),stage2_incremental_extraction_s='UNVERIFIED',route_fraction=.25,end_to_end_mean_s='UNVERIFIED'))
    writecsv('end_to_end_cost.csv',est)
    pareto=[]
    for a in est:
        pareto.append(dict(arm=a['arm'],stage1_extraction_mean_s=a['stage1_extraction_mean_s'],stage1_BA=a['stage1_BA'],end_to_end_cost='UNVERIFIED',pareto_status='NOT_DETERMINABLE_WITHOUT_MATCHED_STAGE2_EXTRACTION_COST'))
    writecsv('cost_accuracy_pareto.csv',pareto)
    writetext('cost_interpretation.md','# 成本边界\n\n此表是本轮真实逐APK增量计时，源域阶段在同一次共享解析中分别测量。OS页缓存状态未受控，不能称为严格冷缓存；未来仅对最终所需层级提取。DEX构造本身解析CODE_ITEM，即使不遍历指令也非“仅读取头”。L4从DEX类方法体扫描invoke，不构建图。旧P31历史计时环境不同。第二层Host/RoleGraph构建没有本轮同APK逐项实时计时，`T_total=T_stage1+route*T_stage2_incremental`只保留为公式，不能报告具体端到端节省或可靠Pareto结论。输入维度下降不等于提取成本下降。峰值内存没有按APK隔离测量，标UNKNOWN。\n')
    print('LITESTAGE_COST_DONE source',len(source),'future',len(future),flush=True)
if __name__=='__main__':main()
