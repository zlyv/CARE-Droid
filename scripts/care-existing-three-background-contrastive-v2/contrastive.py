"""APK-level auxiliary SupCon on the original Direct-C43 fused representation; no Host pseudo labels."""
from __future__ import annotations
import argparse,math,time
from pathlib import Path
import numpy as np
from common import *

SEEDS=(900000,900001,900002)
TAGS=('T2_C1','T3_C1','T4_C1')
ARMS=('M0','M1','M2','M3','M4','M5')

def inputs(mc,v4,data,tag,seed):
    from train_stage2 import raw_bags,transform_saved
    obj=checkpoint(BASE/'models/stage2'/tag/(str(seed)+'.pkl'),mc,data)
    m=obj['metadata'];index={str(s):i for i,s in enumerate(data['ids'])}
    tr=np.array([index[s] for s in m['train_sha']]);va=np.array([index[s] for s in m['validation_sha']])
    if np.max(data['years'][tr])>=np.min(data['years'][va]) or set(m['train_sha'])&set(m['validation_sha']):raise RuntimeError('temporal or SHA leakage')
    fr=data['feature_rows'];x,z,names=v4.audit.compact_matrix_variant(v4.VARIANT,[fr['rows'][i] for i in tr],[fr['rows'][i] for i in va],fr['compact_by_sha'],fr['schema'],fr['canonical_names'])
    gi,_=v4.evidence.split_view_indices(names);gx=np.asarray(x[:,gi],np.float32);gz=np.asarray(z[:,gi],np.float32)
    if gx.shape!=(len(tr),24) or gz.shape!=(len(va),24) or np.max(np.abs(gz-np.asarray(obj['global_validation'],np.float32)))>1e-6:raise RuntimeError('frozen global preprocessing not reproduced')
    raw_tr=raw_bags(data,tr);raw_va=raw_bags(data,va);a=argparse.Namespace(**m['args']);bags=[transform_saved(v4,b,obj['background'],a) for b in [raw_tr,raw_va]]
    bg=np.stack([np.stack([b[:,107].mean(),b[:,108].mean(),b[:,109].mean(),b[:,110].mean()]) for b in bags[0]])
    bg_mean=bg.mean(0);bg_std=bg.std(0);bg_std[bg_std<1e-6]=1
    bg=(bg-bg_mean)/bg_std
    if not np.isfinite(bg).all():raise RuntimeError('background response not finite')
    rg=[v4.transform_role_graphs(data['s7'][ix],obj['graph_scaler']) for ix in [tr,va]]
    return obj,a,tr,va,(gx,gz),bags,rg,bg

def supcon(torch,nn,z,y,role,bg,arm,temperature=.1):
    if arm=='M0':return z.sum()*0,0,0
    n=z.shape[0]
    if n<3:return z.sum()*0,0,0
    diag=torch.eye(n,device=z.device,dtype=torch.bool);same=(y[:,None]==y[None,:])&~diag
    different=(y[:,None]!=y[None,:])
    valid=same.any(1)&different.any(1)
    if not bool(valid.any()):return z.sum()*0,0,0
    weight=torch.ones((n,n),device=z.device)
    if arm in ['M2','M3','M5']:
        norm=nn.functional.normalize(role,dim=1,eps=1e-8)
        weight=weight*(0.5+0.5*torch.clamp(norm@norm.T,min=0,max=1))
    if arm in ['M3','M5']:
        dist=torch.cdist(bg,bg,p=2)
        weight=weight*(0.5+0.5*torch.exp(-dist/2))
    sim=(nn.functional.normalize(z,dim=1,eps=1e-8)@nn.functional.normalize(z,dim=1,eps=1e-8).T)/temperature
    logit=sim+torch.log(weight.clamp_min(1e-8))
    logit=logit.masked_fill(diag,-1e9)
    logprob=logit-torch.logsumexp(logit,dim=1,keepdim=True)
    pweight=weight*same.float();den=pweight.sum(1).clamp_min(1e-8)
    loss=-(pweight*logprob).sum(1)/den
    return loss[valid].mean(),int(valid.sum().item()),int(same.sum().item())

def one_arm(mc,v4,data,tag,seed,arm,device,pilot=False):
    torch=v4.torch;nn=v4.nn;obj,args,tr,va,glob,bags,graphs,bg=inputs(mc,v4,data,tag,seed)
    if pilot:
        local=np.r_[np.flatnonzero(data['y'][tr]==0)[:32],np.flatnonzero(data['y'][tr]==1)[:32]]
        if len(local)<64:raise RuntimeError('pilot class balance unavailable')
        tr=tr[local];glob=(glob[0][local],glob[1]);bags=( [bags[0][int(i)] for i in local],bags[1]);graphs=({k:v[local] for k,v in graphs[0].items()},graphs[1]);bg=bg[local]
    label=np.asarray(data['y'][tr],int);role=np.asarray(data['p31'][tr],np.float32)
    # Deliberately use the saved original background state, RoleGraph scaler,
    # global preprocessing and original model initialization for every arm.
    v4.seed_everything(seed);model=v4.RoleGraphConditionedEvidence(24,'RGCondAttn',args).to(device)
    # The auxiliary head must not shift the original model's dropout RNG stream.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type=='cuda' else []):
        torch.manual_seed(seed+17)
        proj=nn.Sequential(nn.Linear(113,32),nn.ReLU(),nn.Linear(32,16)).to(device)
    opt=torch.optim.Adam(list(model.parameters())+([] if arm=='M0' else list(proj.parameters())),lr=args.learning_rate,weight_decay=args.weight_decay)
    g=torch.as_tensor(glob[0],dtype=torch.float32,device=device);graph=v4.graph_tensors(graphs[0],device);bt=[torch.as_tensor(b,dtype=torch.float32,device=device) for b in bags[0]]
    y=torch.as_tensor(label,dtype=torch.long,device=device);roles=torch.as_tensor(role,dtype=torch.float32,device=device);bg_t=torch.as_tensor(bg,dtype=torch.float32,device=device)
    # Original train_stage2.fit passes seed+3301 into v4.train_epochs,
    # whose own sampler adds 811. Both offsets are required for M0 replay.
    rng=np.random.RandomState(seed+3301+811);epochs=1 if pilot else int(obj['audit']['selected_epochs']);checks=[];t=time.monotonic();proj_grad_seen=False
    for epoch in range(epochs):
        model.train();proj.train();order=rng.permutation(len(tr));tot=0.;anchors=0;positive_pairs=0;steps=0
        for start in range(0,len(order),int(args.batch_size)):
            batch=order[start:start+int(args.batch_size)];opt.zero_grad(set_to_none=True)
            logits=[];fused=[]
            for idx in batch:
                i=int(idx);one_graph={key:val[i] for key,val in graph.items()}
                logit,diagnostic,trace=model.forward_with_trace(g[i],torch.zeros((),device=device),one_graph,bt[i]);logits.append(logit);fused.append(trace['correction_input'])
            logits=torch.stack(logits);fused=torch.stack(fused)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y[batch].float())
            if arm!='M0':
                yy=y[batch]
                if arm=='M4':
                    # A paired capacity/control loss: labels only for this auxiliary
                    # objective are permuted, preserving batch class counts.
                    yy=yy[torch.as_tensor(np.random.RandomState(seed+epoch*100000+start+2901).permutation(len(batch)).copy(),device=device)]
                bb=bg_t[batch]
                if arm=='M5':
                    # Break APK/background correspondence within true class,
                    # preserving label and coarse background distributions.
                    perm=np.arange(len(batch));prng=np.random.RandomState(seed+epoch*100000+start+777)
                    for cls in [0,1]:
                        ids=np.flatnonzero(label[batch]==cls);perm[ids]=prng.permutation(ids)
                    bb=bb[torch.as_tensor(perm.copy(),device=device)]
                aux,nanchor,npair=supcon(torch,nn,proj(fused),yy,roles[batch],bb,arm)
                loss=loss+0.05*aux;anchors+=nanchor;positive_pairs+=npair
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward()
            if arm!='M0' and any(p.grad is not None and torch.any(p.grad!=0) for p in proj.parameters()):proj_grad_seen=True
            nn.utils.clip_grad_norm_(list(model.parameters())+([] if arm=='M0' else list(proj.parameters())),args.grad_clip)
            opt.step();tot+=float(loss.detach().cpu())*len(batch);steps+=1
        checks.append(dict(epoch=epoch+1,loss=tot/len(tr),valid_anchors=anchors,positive_pairs=positive_pairs,steps=steps))
        if pilot or (epoch+1)%5==0:print('ETBC_TRAIN',tag,seed,arm,epoch+1,epochs,checks[-1],flush=True)
    elapsed=time.monotonic()-t
    if arm!='M0' and not proj_grad_seen:raise RuntimeError('projection head received no gradient')
    model.eval();gv=v4.graph_tensors(graphs[1],device);btv=[torch.as_tensor(b,dtype=torch.float32,device=device) for b in bags[1]];pg=torch.as_tensor(glob[1],dtype=torch.float32,device=device)
    pp=[]
    with torch.no_grad():
        for i in range(len(va)):
            gg={k:v[i] for k,v in gv.items()};logit,_,trace=model.forward_with_trace(pg[i],torch.zeros((),device=device),gg,btv[i]);pp.append(float(torch.sigmoid(logit).cpu()))
    p=np.asarray(pp)
    if not np.isfinite(p).all() or len(p)!=len(va):raise RuntimeError('invalid validation probabilities')
    if pilot:return dict(tag=tag,seed=seed,arm=arm,status='MECHANICAL_PILOT_PASS',n=64,epoch=1,loss=checks[-1]['loss'],valid_anchors=checks[-1]['valid_anchors'],positive_pairs=checks[-1]['positive_pairs'],projection_gradient=proj_grad_seen,validation_finite=True,seconds=elapsed,params_model=sum(q.numel() for q in model.parameters()),params_projection=sum(q.numel() for q in proj.parameters()))
    return dict(tag=tag,seed=seed,arm=arm,probability=p,train_checks=checks,seconds=elapsed,params_model=sum(q.numel() for q in model.parameters()),params_projection=sum(q.numel() for q in proj.parameters()),train_APKs=len(tr),validation_APKs=len(va),validation_sha=list(obj['metadata']['validation_sha']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--tag',choices=TAGS,default='T2_C1');p.add_argument('--seed',type=int,default=900000);p.add_argument('--arm',choices=ARMS,default='M1');p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if not (OUTPUT/'baseline_replay.json').is_file() or read_json(OUTPUT/'baseline_replay.json')['status']!='PASS':raise RuntimeError('baseline replay gate failed')
    mc,v4,args,data=original();torch=v4.torch;device=torch.device(a.device)
    if a.device=='cuda':
        torch.cuda.init()
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('less than 2 GiB free GPU')
    result=one_arm(mc,v4,data,a.tag,a.seed,a.arm,device,a.pilot)
    if a.pilot:
        write_json(OUTPUT/('pilot_'+a.arm+'.json'),result)
    else:
        pred=result.pop('probability');sha=result.pop('validation_sha');dest=OUTPUT/'full'/a.tag/str(a.seed)
        dest.mkdir(parents=True,exist_ok=True)
        with (dest/(a.arm+'_predictions.npz')).open('xb') as f:np.savez_compressed(f,sha256=np.asarray(sha),probability=pred)
        write_json(dest/(a.arm+'_audit.json'),result)
    print('ETBC_ARM_DONE',a.tag,a.seed,a.arm,'PILOT' if a.pilot else 'FULL',flush=True)

if __name__=='__main__':main()
