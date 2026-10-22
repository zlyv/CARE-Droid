"""Export real frozen-model tensors and clearly named label-free diagnostic summaries."""
from __future__ import annotations
import argparse, time
import numpy as np
from diagnostic_common import *

LAYERS=('P0_P31_SIZE','L0_FIXED_SUMMARY','L1_HOST_ENCODER_SUMMARY','L2_ROLE_TOKEN','L2_GRAPHGLOBAL_TOKEN','L3_HOST_BACKGROUND_DIAGNOSTIC','L4_ATTENTION_APK','L5_EXACT_HEAD_INPUT','L6_DIRECT_LOGIT')
def summary(x):
    x=np.asarray(x,np.float32)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise RuntimeError('invalid or empty Host bag')
    return np.concatenate([x.mean(0),x.std(0),x.max(0)]).astype(np.float32)

def one(v4,model,g,graphs,bags,raw,p31,ids,device):
    torch=v4.torch;graph=v4.graph_tensors(graphs,device);gt=torch.as_tensor(g,dtype=torch.float32,device=device)
    holders=[];out={key:[] for key in LAYERS};background=[];host_n=[];attention=[];logits=[]
    def hook(module,values,result):holders.append(result.detach())
    handle=model.region_encoder.register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            for i in range(len(ids)):
                bag=np.asarray(bags[i],np.float32);r=np.asarray(raw[i],np.float32)
                if bag.shape[1]!=142 or r.shape[1]!=169 or len(bag)!=len(r):raise RuntimeError('Host input alignment mismatch')
                onegraph={k:v[i] for k,v in graph.items()};holders.clear()
                logit,_,trace=model.forward_with_trace(gt[i],torch.zeros((),device=device),onegraph,torch.as_tensor(bag,device=device))
                if len(holders)!=1:raise RuntimeError('Host encoder hook fired wrong number of times')
                local=holders[0].cpu().numpy();fused=trace['correction_input'].cpu().numpy();att=trace['attention'].cpu().numpy()
                if fused.shape!=(113,) or local.shape!=(len(bag),24) or abs(float(att.sum())-1)>1e-5:raise RuntimeError('frozen layer shape invalid')
                b=bag[:,107:111];bs=summary(b);role=fused[25:57]
                # GraphGlobal affects the query but is not an independent output
                # of the Direct trace. This is its actual module output.
                gg=model.graphglobal_encoder(onegraph['global'].reshape(-1)).cpu().numpy()
                out['P0_P31_SIZE'].append(np.r_[p31[i],np.log1p(len(bag)),(p31[i]>0).astype(np.float32)])
                out['L0_FIXED_SUMMARY'].append(np.r_[summary(r),p31[i],np.log1p(len(bag))])
                out['L1_HOST_ENCODER_SUMMARY'].append(summary(local))
                out['L2_ROLE_TOKEN'].append(role)
                out['L2_GRAPHGLOBAL_TOKEN'].append(gg)
                out['L3_HOST_BACKGROUND_DIAGNOSTIC'].append(np.r_[summary(local),bs])
                out['L4_ATTENTION_APK'].append(np.r_[role,trace['critical'].cpu().numpy(),trace['attended'].cpu().numpy(),fused[-8:]])
                out['L5_EXACT_HEAD_INPUT'].append(fused)
                out['L6_DIRECT_LOGIT'].append(np.array([float(logit.cpu())],np.float32))
                background.append(bs);host_n.append(len(bag));attention.append(float(np.max(att)));logits.append(float(logit.cpu()))
                if (i+1)%250==0:print('CLD_EXTRACT',len(ids),i+1,flush=True)
    finally:handle.remove()
    result={key:np.asarray(value,np.float32) for key,value in out.items()}
    for key,value in result.items():
        if len(value)!=len(ids) or not np.isfinite(value).all():raise RuntimeError('nonfinite layer '+key)
    result.update(ids=np.asarray(ids),background=np.asarray(background,np.float32),host_n=np.asarray(host_n,np.int32),max_attention=np.asarray(attention,np.float32))
    return result

def main():
    if read_json(OUTPUT/'baseline_replay.json')['status']!='PASS':raise RuntimeError('baseline replay gate failed')
    start=time.monotonic();mc,v4,args,data=original();device=v4.torch.device('cpu')
    from train_stage2 import raw_bags
    audits=[];manifest=[];layers=[]
    for tag in TAGS:
        obj,a,tr,va,glob,bags,graphs,bg=inputs(mc,v4,data,tag,900000)
        model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',a).to(device);model.load_state_dict(obj['state_dict']);model.eval()
        train=one(v4,model,glob[0],graphs[0],bags[0],raw_bags(data,tr),data['p31'][tr],data['ids'][tr],device)
        valid=one(v4,model,glob[1],graphs[1],bags[1],raw_bags(data,va),data['p31'][va],data['ids'][va],device)
        p=1/(1+np.exp(-valid['L6_DIRECT_LOGIT'][:,0].astype(np.float64)))
        diff=float(np.max(np.abs(p-obj['probability'])))
        if diff>1e-6:raise RuntimeError('logit replay mismatch '+tag+' '+str(diff))
        for name,x,idx in [('train',train,tr),('validation',valid,va)]:
            write_npz(OUTPUT/'embeddings'/tag/(name+'.npz'),**x,y=np.asarray(data['y'][idx],np.int8),year=np.asarray(data['years'][idx],np.int16),index=np.asarray(idx,np.int32))
            for key in LAYERS:manifest.append(dict(tag=tag,seed=900000,split=name,layer=key,n=len(idx),width=x[key].shape[1],checkpoint=str(BASE/'models/stage2'/tag/'900000.pkl'),path=str(OUTPUT/'embeddings'/tag/(name+'.npz')),coordinate_system=tag+'/900000',train_representation_in_sample=(name=='train'),is_exact_network_tensor=key in ('L2_ROLE_TOKEN','L2_GRAPHGLOBAL_TOKEN','L4_ATTENTION_APK','L5_EXACT_HEAD_INPUT','L6_DIRECT_LOGIT'),note='L3 is a diagnostic concatenation, not a distinct frozen network tensor' if key.startswith('L3') else 'L0 is a lossy label-free summary of variable-length raw Host bags' if key.startswith('L0') else ''))
        audits.append(dict(tag=tag,checkpoint_seed=900000,train_n=len(tr),validation_n=len(va),train_max_year=int(max(data['years'][tr])),validation_year=int(data['years'][va[0]]),logit_probability_max_difference=diff,empty_host_marked_train=int(np.sum(data['host_was_empty'][tr])),empty_host_marked_validation=int(np.sum(data['host_was_empty'][va])),actual_min_bag_size=min(int(train['host_n'].min()),int(valid['host_n'].min())),all_finite=True))
        print('CLD_LAYER_DONE',tag,diff,flush=True)
    write_csv(OUTPUT/'layer_embedding_manifest.csv',manifest)
    write_json(OUTPUT/'layer_extraction_audit.json',dict(status='PASS',windows=audits,seconds=time.monotonic()-start,checkpoint_seed_for_embeddings=900000,other_seeds_replayed_only=True,no_cross_window_coordinate_pooling=True))
    layer_info=[dict(name='L0_FIXED_SUMMARY',source='prepared raw Host 169; P31 31',operation='label-free mean/std/max + P31 + log Host count',network_tensor=False,background='raw Host features prior to four-response transform',shape='APK x 539'),dict(name='L1_HOST_ENCODER_SUMMARY',source='haig_models.py Model.region_encoder',operation='forward hook; mean/std/max of 24-D Host embedding',network_tensor=False,background='four responses enter 111-D encoder core',shape='APK x 72'),dict(name='L2_ROLE_TOKEN',source='graphglobal_path_models.py encode_role_graph',operation='RoleGraph pooled token in exact head input',network_tensor=True,background='none direct',shape='APK x 32'),dict(name='L2_GRAPHGLOBAL_TOKEN',source='graphglobal_path_models.py graphglobal_encoder',operation='module output; query path only',network_tensor=True,background='none direct',shape='APK x 32'),dict(name='L3_HOST_BACKGROUND_DIAGNOSTIC',source='haig_background.py cols + haig_models.py region_encoder',operation='diagnostic concatenation of Host encoding summary and four background response summaries',network_tensor=False,background='existing external/internal/commonness/residual, not a new layer',shape='APK x 84'),dict(name='L4_ATTENTION_APK',source='graphglobal_path_models.py select_local_evidence',operation='role token + critical Host + attended Host + 8 scalar summary',network_tensor=False,background='indirectly through Host code, attention score and 4 scalar statistics',shape='APK x 88'),dict(name='L5_EXACT_HEAD_INPUT',source='graphglobal_path_models.py delta_from_selected_evidence',operation='exact fused vector before delta_head MLP',network_tensor=True,background='indirectly and 4 scalar statistics',shape='APK x 113'),dict(name='L6_DIRECT_LOGIT',source='model_variants.py DirectStage2.forward_with_trace',operation='direct raw logit; sigmoid replay checked',network_tensor=True,background='through preceding network',shape='APK x 1')]
    write_json(OUTPUT/'layer_manifest.json',dict(status='PASS',actual_model='haig_models.Model inherited GraphGlobalPathHostAnomaly, patched by model_variants.DirectStage2',raw_host_width=169,model_host_width=142,host_encoder_core_width=111,background_columns=[107,108,109,110],relation31_columns=[111,142],shared_prototypes=[32,107],external_benign_prototypes=[24,107],internal_background='per-APK dynamic, not a third corpus prototype bank',layers=layer_info,unavailable_independent_tensor='No separate post-background-fusion tensor exists; four columns enter Host encoder and scalar summaries.'),)
if __name__=='__main__':main()
