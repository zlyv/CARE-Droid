#!/usr/bin/env python3
"""Audit SCBE novelty/assets; in CARE-Droid write only outputs/care_scbe_v1, never read APK bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/home/boyuanjie/projects/tasa-droid')
OUT=ROOT/'outputs/care_scbe_v1'
OLD=ROOT/'outputs/care_multiyear_development_v1'
PY=Path('/home/boyuanjie/envs/tif/bin/python')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    s=Path(path).stat()
    return {'path':str(Path(path).resolve()),'sha256':h.hexdigest(),'size':s.st_size,'mtime_ns':s.st_mtime_ns}

def atomic_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,ensure_ascii=False,indent=2)

def assets():
    paths=[PY, OLD/'data.npz',OLD/'feature_rows.pkl',OLD/'sample_manifest.csv',
        ROOT/'scripts/care-multiyear-development-v1/myd_common.py',
        ROOT/'scripts/care-multiyear-development-v1/train_stage2.py',
        ROOT/'scripts/care-multiyear-development-v1/protocol.py',
        ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py',
        ROOT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py',
        ROOT/'scripts/run_care_joint_dualprototype_ablation.py',
        ROOT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py',
        ROOT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py',
        ROOT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv',
        ROOT/'outputs/care_background_prototype_audit_v1/final_decision.json']
    paths += [OLD/'predictions'/w/'router.npz' for w in ('T1','T2','T3','T4')]
    paths += [OLD/'models/stage2'/w/(str(seed)+'.pkl') for w in ('C0R_2017','T2_C1','T3_C1','T4_C1') for seed in (900000,900001,900002)]
    return paths

def verify():
    prior=json.loads((OUT/'asset_manifest_before.json').read_text())
    changed=[]
    for row in prior:
        try:current=digest(row['path'])
        except OSError as e:changed.append({'path':row['path'],'error':str(e)});continue
        if current!=row:changed.append({'path':row['path'],'before':row,'after':current})
    return {'status':'PASS' if not changed else 'FAIL','checked':len(prior),'changed':changed,
            'scope':'named frozen code/data/route/checkpoints and Python binary; no original APK bytes, not a whole-repository integrity claim'}

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--audit-only',action='store_true');g.add_argument('--verify',action='store_true')
    a=p.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if a.verify:
        result=verify();print('SCBE_ASSET_'+result['status'],result['checked'],flush=True)
        if result['status']!='PASS':raise RuntimeError('historical asset changed')
        return
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'tmp').mkdir()
    manifest=[digest(x) for x in assets()]
    atomic_new(OUT/'asset_manifest_before.json',manifest)
    atomic_new(OUT/'asset_integrity_report.json',{'status':'PREFLIGHT_PASS','checked':len(manifest),
        'baseline_check_at_end_required':True,'new_APKs_used':False,'formal_assets_modified':False})
    formula='''# 正式三背景公式与张量接口\n\n`haig_background.split`：169维Host分为Compact107、Role31、Relation31；训练窗口拟合107维均值与标准差。`fit_prototype_state`：训练APK每个最多16 Host，良恶平衡Host池KMeans得Shared32、良性Host池得External24。`haig_background.profiles`生成31维Role profile；`hard`优先同主Role再回退全局最近。\n\nShared距离为RMS(x,shared_center)，硬分配用RMS/shared_scale；External硬分配用RMS(x,external_center)。原External标量是`log1p(RMS_assigned/external_scale)`，越小越近。Commonness=`shared_commonness[assigned]*exp(-shared_RMS/shared_scale)`，其中训练Shared commonness=`clip(2*min(P(cluster|benign),P(cluster|malware)),0,1)`。Internal为当前APK内部主导Shared簇构造的局部背景距离，不改变。Residual=`(1-clip(Commonness,0,1))*sqrt(max(External,0)*max(Internal,0))`，严格由三列确定。\n\nHost编码输入为`[Compact107, External, Internal, Commonness, Residual, Relation31]`，总宽142；`haig_models.make_class`实际仅把前111维送入region_encoder，Relation31进入relation_key，并以`core[:,-1]`读取Residual。Direct模型用APK级BCE，固定0.5阈值，路由外保留Stage-I。SCBE不会假装正式模型已有32维/24维响应；将从相同标准化107维和冻结Bank计算诊断向量，最终仍写回原有External标量位和对应确定性Residual位。\n'''
    atomic_new(OUT/'formal_background_equations.md',formula)
    novelty='''# SCBE与历史方法的区别及边界\n\n已有Commonness是单个Shared硬分配中心的两类训练APK prevalence 与距离函数；原Residual只是External、Internal、Commonness的确定性组合。TB-FRS研究软分配、有符号/二阶统计；三背景SupCon与Role-aware SupCon研究额外表示损失；Layerwise Probe研究既有量的可读信息；Host删除归因检验局部贡献。它们都没有拟合`E_hat=g(S32,Role31)`并构造逐External中心的交叉拟合条件残差与APK级类别支持方向。因此SCBE不与原Residual在数学上等价。\n\n但这不证明SCBE有增量、也不证明背景重叠是FN根因。上一轮Prototype审计状态为`NO_VERIFIED_BACKGROUND_OVERLAP_BOTTLENECK`。SCBE的32/24响应为从正式Bank推导出的新诊断向量，非旧网络已有显式输出。训练期全窗口Bank含被交叉拟合APK的Host，故即使条件预测器与w排除其标签，也有Bank自来源依赖；必须在报告中保留此限制。\n'''
    atomic_new(OUT/'method_novelty_audit.md',novelty)
    atomic_new(OUT/'tensor_interface_audit.json',{'raw_host':169,'compact':107,'role':31,'relation':31,
        'shared_prototypes':32,'external_prototypes':24,'formal_host_encoder':142,
        'formal_core_encoder':111,'SCBE_output_host_encoder':142,'formal_external_scalar_index':107,
        'formal_internal_scalar_index':108,'formal_commonness_scalar_index':109,
        'formal_residual_scalar_index':110,'new_explicit_S_E_vectors_in_formal_model':False})
    print('SCBE_PREFLIGHT_PASS',len(manifest),flush=True)

if __name__=='__main__':main()
