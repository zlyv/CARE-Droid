"""Freeze old-asset hashes, document the actual three backgrounds, and replay historical Direct-C43."""
from __future__ import annotations
import inspect,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from common import *

TAGS=('T2_C1','T3_C1','T4_C1')
SEEDS=(900000,900001,900002)

def main():
    if OUTPUT.exists():raise RuntimeError('experiment output exists; choose a fresh version')
    mc,v4,args,data=original()
    if read_json(BASE/'final_decision.json')['status']!='DEVELOPMENT_COMPLETE':raise RuntimeError('development source incomplete')
    import haig_background as hb
    import train_stage2 as original_train
    paths={BASE/'data.npz':'frozen full development input',BASE/'feature_rows.pkl':'frozen global feature rows',BASE/'dataset_audit.json':'frozen data contract',BASE/'final_decision.json':'completed development decision',BASE/'per_apk_predictions.csv':'frozen per APK cascade',PROJECT/'scripts/run_care_joint_dualprototype_ablation.py':'original prototype fitting',PROJECT/'scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py':'original four background columns',PROJECT/'scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py':'original RGCondAttn',PROJECT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py':'original Direct head'}
    for p in SOURCE.glob('*.py'):paths[p]='original multi-year code'
    for tag in TAGS:
        for seed in SEEDS:paths[BASE/'models/stage2'/tag/(str(seed)+'.pkl')]='saved original C1 checkpoint'
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and str(PROJECT) in str(p) and Path(p).suffix=='.py' and HERE not in Path(p).resolve().parents:paths[Path(p).resolve()]='loaded original dependency'
    manifest=[asset(p,purpose) for p,purpose in sorted(paths.items(),key=lambda t:str(t[0]))]
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'baseline_asset_manifest_before.json',manifest)
    cases=read_csv(BASE/'per_apk_predictions.csv')
    by={(r['window']+'_'+r['arm'],r['sha256']):r for r in cases}
    shapes=[];rows=[];geom=[];background=[];maxdiff=0
    for tag in TAGS:
        saved={seed:checkpoint(BASE/'models/stage2'/tag/(str(seed)+'.pkl'),mc,data) for seed in SEEDS}
        for seed in SEEDS:
            obj=saved[seed];s=obj['background']['state'];meta=obj['metadata'];shapes.append(dict(tag=tag,seed=seed,shared_centers=np.shape(s['shared_centers']),benign_centers=np.shape(s['benign_centers']),shared_role_profiles=np.shape(obj['background']['sp']),benign_role_profiles=np.shape(obj['background']['bp']),commonness=np.shape(s['commonness']),mean=np.shape(obj['background']['mean']),std=np.shape(obj['background']['std']),graph_scaler_keys=sorted(obj['graph_scaler']),epochs=obj['audit']['selected_epochs'],train_n=len(meta['train_sha']),validation_n=len(meta['validation_sha'])))
            if np.shape(s['shared_centers'])!=(32,107) or np.shape(s['benign_centers'])!=(24,107):raise RuntimeError('actual background shapes differ from declared')
            ix={sid:i for i,sid in enumerate(data['ids'])};vi=np.array([ix[sid] for sid in meta['validation_sha']]);raw=original_train.raw_bags(data,vi);bags=original_train.transform_saved(v4,raw,obj['background'],args)
            for j,b in enumerate(bags):
                if b.ndim!=2 or b.shape[1]!=142:raise RuntimeError('C43 transformed Host width mismatch')
                q=b[:,107:111];sid=meta['validation_sha'][j]
                background.append(dict(tag=tag,seed=seed,sha256=sid,year=int(data['years'][vi[j]]),label=int(data['y'][vi[j]]),host_n=len(b),external=float(q[:,0].mean()),internal=float(q[:,1].mean()),commonness=float(q[:,2].mean()),residual=float(q[:,3].mean())))
            model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',__import__('argparse').Namespace(**meta['args']));model.load_state_dict(obj['state_dict']);model.eval()
            graph=v4.transform_role_graphs(data['s7'][vi],obj['graph_scaler'])
            p,_=v4.predict_model(model,np.asarray(obj['global_validation'],np.float32),np.full(len(vi),.5),graph,bags,v4.torch.device('cpu'))
            diff=float(np.max(np.abs(p-obj['probability'])));maxdiff=max(maxdiff,diff)
            if diff>1e-6:raise RuntimeError(f'frozen replay failed {tag}/{seed} diff={diff}')
            print('ETBC_REPLAY',tag,seed,len(vi),diff,flush=True)
        obj0=saved[SEEDS[0]]
        ix={sid:i for i,sid in enumerate(data['ids'])};vi=np.array([ix[sid] for sid in obj0['metadata']['validation_sha']])
        for j,sid in enumerate(obj0['metadata']['validation_sha']):
            r=by.get((tag.replace('_C1','')+'_U3_FULL_ROLLING_C1',sid))
            if r is None:raise RuntimeError('frozen prediction row missing '+sid)
            stage=int(r['stage1_prediction']);y=int(r['label']);route=int(r['routed']);ex=float(r['expert_probability']);ep=int(ex>=.5)
            seedavg=float(np.mean([saved[seed]['probability'][j] for seed in SEEDS]))
            if abs(ex-seedavg)>1e-6:raise RuntimeError('ensemble replay differs '+sid)
            category=('FIXED_FN' if y==1 else 'FIXED_FP') if stage!=y and ep==y else ('UNFIXED_FN' if y==1 else 'UNFIXED_FP') if stage!=y else ('BROKEN_TN' if y==0 else 'BROKEN_TP') if ep!=y else 'STABLE_CORRECT'
            rows.append(dict(tag=tag,sha256=sid,year=int(r['year']),label=y,stage1_probability=float(r['stage1_probability']),stage1_prediction=stage,stage1_correct=int(stage==y),routed=route,expert_probability=ex,expert_prediction=ep,expert_correct=int(ep==y),category=category,router_missed_correctable=int(not route and stage!=y and ep==y)))
    write_json(OUTPUT/'baseline_replay.json',dict(status='PASS',checkpoints=len(shapes),max_probability_difference=maxdiff,tolerance=1e-6,full_validation_expert_inference=True,stage1_router_and_threshold_frozen=True))
    write_json(OUTPUT/'background_tensor_manifest.json',dict(checkpoints=shapes,raw_host_width=169,prototype_fit_width=107,model_host_width=142,background_host_columns=dict(external=107,internal=108,commonness=109,residual=110),relation31_columns=[111,142],classification_fused_width=113,training_label_roles=dict(shared='balanced benign+malicious samples; commonness uses both labels',external='benign-only training samples',internal='same APK Host bag, no fitted corpus bank')))
    write_csv(OUTPUT/'expert_capability.csv',rows)
    write_csv(OUTPUT/'background_geometry.csv',background)
    # Pair correlation is descriptive only, computed within each independently fitted model and year.
    for tag in TAGS:
        b=[r for r in background if r['tag']==tag and r['seed']==SEEDS[0]];a=np.array([[r[k] for k in ['external','internal','commonness','residual']] for r in b]);corr=np.corrcoef(a,rowvar=False)
        for j,k in enumerate(['external','internal','commonness','residual']):
            for z,h in enumerate(['external','internal','commonness','residual']):
                geom.append(dict(tag=tag,coordinate_1=k,coordinate_2=h,pearson=float(corr[j,z]) if np.isfinite(corr[j,z]) else None,n=len(b)))
    write_csv(OUTPUT/'representation_geometry.csv',geom)
    cols=Counter(r['category'] for r in rows if r['routed']);miss=sum(r['router_missed_correctable'] for r in rows)
    write_json(OUTPUT/'diagnostic_decision.json',dict(status='BACKGROUND_DISCRIMINATIVE_VALUE_UNPROVEN_CONTRASTIVE_PILOT_ALLOWED',full_validation_APKs=len(rows),routed_category_counts=dict(cols),router_missed_correctable=miss,background_scalar_correlations_descriptive_only=True,training_authorized='small pilot M0-M2 first; M3 only after its paired mechanism is frozen',legacy_2018_2021_are_development=True))
    # The core mechanistic distinction is four already-present columns, not three new trainable vectors.
    audit='''# 实际三背景源码审计\n\n共享背景：`run_care_joint_dualprototype_ablation.py:166`在训练APK中按类别平衡抽Host，MiniBatchKMeans拟合32个107维中心；各原型的commonness同时使用两类训练标签出现率。它不是无监督或纯良性背景。\n\n外部良性背景：同一函数仅对训练良性APK的Host拟合24个107维中心，并拟合距离缩放；不把距离视为良性概率。\n\n内部背景：`internal_dominant_distance:231`按该APK的共享原型分配和覆盖率构造若干袋内中心，每个APK动态计算，无跨APK可训练原型。\n\n真正投入模型的量：`haig_background.py:5,12,22,27,29`拆分169维原始Host，先训练集拟合107维标准化，再用Role profile约束硬分配，输出每Host4列：外部距离、内部距离、共享commonness、由三者确定的residual。Rel31另接于末端，形成142维输入。Residual=(1-commonness)*sqrt(external*internal)，与前三列存在确定性冗余，不能当独立背景。\n\n运行时`haig_models.py:12`的select_local_evidence实际将前111维送入Host encoder，将后31维Relation单独送入查询关系；前111维的末4列同时直接参与分数与标量汇总。`graphglobal_path_models.py`合并Role、图全局、关键和受注意Host形成113维分类表示。`model_variants.py`的Direct覆写取消bounded输出与Stage-I logit，三背景仍在C43中。`train_stage2.py:69-83`训练只用等权BCE；无背景辅助损失。原型不可训练，三背景四列为训练期或APK级确定性特征。\n\n本次只比较训练目标，M1-M3不得新增任何原型库、改变Role映射或将低异常视为良性证书。\n'''
    write_text(OUTPUT/'existing_three_background_audit.md',audit)
    prior='''# 历史背景实验边界\n\n- `run_care_joint_dualprototype_ablation.py`建立共享/良性原型与内部距离，并有原始列关闭的消融；与当前Role约束C43的完整训练协议不同。\n- `care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py`加入Role profile约束，当前C43沿用。\n- `outputs/care-tb-frs-source-v1/final_decision.json`检验软分配及有符号一、二阶残差，止于source信息probe，不证明背景对比损失有效。\n- `outputs/care-direct-c43-joint-contribution-audit-v1/`研究已有信息联合贡献；本轮不把它重新训练为新三背景。\n- 本轮SupCon只改APK级训练约束，不伪称首次引入三背景。\n'''
    write_text(OUTPUT/'prior_background_experiment_map.md',prior)
    write_json(OUTPUT/'pilot_protocol.json',dict(source='Khosla et al., NeurIPS 2020, Supervised Contrastive Learning',M0='historical frozen C1 direct reference and one new matched-budget replay',M1='same Direct model plus APK-level SupCon',M2='same SupCon pair set weighted by fixed Role similarity',M3='same M2 pair set additionally weighted by training-only four-column background response similarity',M4='capacity-matched projection head, fixed label-permutation pairing',M5='same M3 with train-only background responses permuted within labels',lambda_con=0.05,temperature=0.1,projection_width=16,role_weight='0.5+0.5*cosine(Role31)',background_weight='0.5+0.5*exp(-L2(zscore_Bi-zscore_Bj)/2)',pair_sets='all different APKs in mini-batch; no Host pseudo labels',seeds=list(SEEDS),pilot='T2 seed900000, one epoch on a fixed 64-APK train subset, never used for performance selection',full_gate='replay PASS, finite pilot gradients/loss, original background outputs unchanged, route and model predictions aligned',candidate_rule=dict(pooled_BA_gain_pp=0.5,pooled_net_gain=6,positive_years=2,controls_must_not_match=True)))
    print('ETBC_AUDIT_DONE',len(rows),maxdiff,cols,flush=True)

if __name__=='__main__':main()
