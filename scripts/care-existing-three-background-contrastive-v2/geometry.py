"""Within-model validation representation geometry; never compare independently trained coordinate systems."""
from __future__ import annotations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from common import *
from contrastive_training import inputs,TAGS,ARMS

def features(v4,model,g,graph,bags,device):
    torch=v4.torch;gt=torch.as_tensor(g,dtype=torch.float32,device=device);gg=v4.graph_tensors(graph,device);bt=[torch.as_tensor(b,dtype=torch.float32,device=device) for b in bags];model.eval();z=[]
    with torch.no_grad():
        for i in range(len(bags)):
            one={k:v[i] for k,v in gg.items()};_,_,trace=model.forward_with_trace(gt[i],torch.zeros((),device=device),one,bt[i]);z.append(trace['correction_input'].detach().cpu().numpy())
    z=np.asarray(z,np.float32)
    if z.ndim!=2 or z.shape[1]!=113 or not np.isfinite(z).all():raise RuntimeError('invalid fused representation')
    return z

def main():
    mc,v4,args,data=original();device=v4.torch.device('cuda' if v4.torch.cuda.is_available() else 'cpu');geom=[];drift=[];background=[]
    lookup={(r['tag'],r['sha256']):r for r in read_csv(OUTPUT/'expert_capability.csv')}
    for tag in TAGS:
        obj,a,tr,va,glob,bags,graphs,bg=inputs(mc,v4,data,tag,900000)
        if tag=='T4_C1':
            for split,indices,baglist in [('training',tr,bags[0]),('validation',va,bags[1])]:
                for year in sorted(set(data['years'][indices])):
                    selected=[b for b,i in zip(baglist,indices) if data['years'][i]==year]
                    q=np.stack([np.array([b[:,j].mean() for j in range(107,111)]) for b in selected])
                    background.append(dict(tag=tag,split=split,year=int(year),APKs=len(selected),external_mean=float(q[:,0].mean()),internal_mean=float(q[:,1].mean()),commonness_mean=float(q[:,2].mean()),residual_mean=float(q[:,3].mean()),all_responses_from_one_frozen_T4_background=True))
        mask=np.asarray([int(lookup[(tag,str(data['ids'][i]))]['routed']) for i in va],bool)
        if mask.sum()<10:raise RuntimeError('too few routed validation APKs')
        labels=np.asarray(data['y'][va])[mask];role=np.asarray(data['p31'][va])[mask]
        for arm in ARMS:
            model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',a).to(device)
            if arm=='M0':state=obj['state_dict']
            else:
                ck=OUTPUT/'full'/tag/'900000'/(arm+'_checkpoint.pt')
                state=v4.torch.load(ck,map_location='cpu',weights_only=False)['model_state']
            model.load_state_dict(state)
            selected_graph={k:v[mask] for k,v in graphs[1].items()};selected_bags=[x for x,m in zip(bags[1],mask) if m];z=features(v4,model,glob[1][mask],selected_graph,selected_bags,device)
            d=euclidean_distances(z);np.fill_diagonal(d,np.inf);same=labels[:,None]==labels[None,:];np.fill_diagonal(same,False)
            intra=float(np.mean(d[same]));inter=float(np.mean(d[~same&np.isfinite(d)]));nn=np.argsort(d,axis=1)[:,:5];purity=float(np.mean(labels[nn]==labels[:,None]))
            rs=cosine_similarity(role);cond=np.zeros(len(z),bool);cp=[]
            for i in range(len(z)):
                candidates=np.flatnonzero((rs[i]>=.8)&(np.arange(len(z))!=i))
                if len(candidates):cond[i]=True;near=candidates[np.argsort(d[i,candidates])[:min(5,len(candidates))]];cp.append(np.mean(labels[near]==labels[i]))
            geom.append(dict(tag=tag,seed=900000,arm=arm,routed=len(z),same_label_distance=intra,opposite_label_distance=inter,separation_ratio=inter/max(intra,1e-8),nearest5_label_purity=purity,role_matched_APKs=int(cond.sum()),role_matched_label_purity=float(np.mean(cp)) if cp else None,collapse_flag=bool(np.std(z,axis=0).mean()<1e-3),coordinate_comparison='within this tag/seed/model only'))
            # A single frozen T4 encoder applied to its own historical training
            # years and 2021; training-vs-validation fit relation is disclosed.
            if tag=='T4_C1' and arm in ['M0','M3']:
                train_z=features(v4,model,glob[0],graphs[0],bags[0],device)
                val_z=features(v4,model,glob[1],graphs[1],bags[1],device)
                for year in sorted(set(data['years'][tr])):
                    old=train_z[data['years'][tr]==year]
                    drift.append(dict(arm=arm,tag=tag,reference_training_year=int(year),validation_year=2021,train_n=len(old),validation_n=len(val_z),centroid_L2=float(np.linalg.norm(old.mean(0)-val_z.mean(0))),train_vs_validation='descriptive in one frozen model; train samples were fitted, not independent',cross_model_coordinates_compared=False))
            print('ETBC_GEOMETRY',tag,arm,len(z),flush=True)
    write_csv(OUTPUT/'learned_representation_geometry.csv',geom);write_csv(OUTPUT/'within_model_temporal_geometry.csv',drift);write_csv(OUTPUT/'background_temporal_response.csv',background)

if __name__=='__main__':main()
