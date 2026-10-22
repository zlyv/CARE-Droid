#!/usr/bin/env python3
"""Conservative instruction CFG, postdominance and local permission def-use.

Accepted proofs are scoped to the decoded method under ordinary DEX instruction
semantics, with possible uncaught exceptions explicitly ending at EXIT. Methods
with catch handlers, reachable cycles, unresolved opcodes or targets are UNKNOWN.
No absence claim or runtime feasibility claim is produced.
"""
from __future__ import annotations
from collections import deque
import hashlib,json

EXIT=-1
PERMISSION_APIS={
 'landroid/content/context;->checkselfpermission(ljava/lang/string;)i':1,
 'landroidx/core/content/contextcompat;->checkselfpermission(landroid/content/context;ljava/lang/string;)i':1,
 'landroid/support/v4/content/contextcompat;->checkselfpermission(landroid/content/context;ljava/lang/string;)i':1}
def may_throw(name):
    return (name.startswith(('invoke-','iget','iput','sget','sput','aget','aput','new-','filled-new-array','monitor-','const-string','const-class','const-method-handle','const-method-type','check-cast','instance-of','array-length','fill-array-data'))
            or ((name.startswith(('div-','rem-'))) and ('int' in name or 'long' in name)))
def opcode_supported(name):
    prefixes=('move','const','return','invoke-','if-','goto','neg-','not-','cmp','int-to-','long-to-','float-to-','double-to-',
              'add-','sub-','mul-','div-','rem-','and-','or-','xor-','shl-','shr-','ushr-','rsub-',
              'aget','aput','iget','iput','sget','sput','new-','filled-new-array','monitor-')
    return name in ('nop','throw','packed-switch','sparse-switch','fill-array-data','check-cast','instance-of','array-length') or name.startswith(prefixes)
def build_cfg(instructions,tries_size=0,max_nodes=4096):
    if tries_size:return dict(status='UNKNOWN',reason='TRY_CATCH_EXCEPTION_EDGES_UNRESOLVED')
    rows={int(r['offset']):r for r in instructions if not r.get('payload')}
    if not rows:return dict(status='UNKNOWN',reason='NO_CODE')
    if len(rows)>max_nodes:return dict(status='UNKNOWN',reason='INSTRUCTION_CAP')
    if len(rows)!=sum(not r.get('payload') for r in instructions):return dict(status='UNKNOWN',reason='DUPLICATE_INSTRUCTION_OFFSET')
    normal={};succ={EXIT:set()};branch={}
    for off,r in rows.items():
        name=r['name'];nxt=off+int(r['length'])
        if r.get('decode_error') or not opcode_supported(name):return dict(status='UNKNOWN',reason='UNSUPPORTED_OR_UNDECODED_OPCODE',offset=off)
        if name.startswith('if-'):
            if 'target' not in r:return dict(status='UNKNOWN',reason='BRANCH_TARGET_UNRESOLVED',offset=off)
            normal[off]={nxt,int(r['target'])};branch[off]={'TAKEN':int(r['target']),'FALLTHROUGH':nxt}
        elif name.startswith('goto'):
            if 'target' not in r:return dict(status='UNKNOWN',reason='GOTO_TARGET_UNRESOLVED',offset=off)
            normal[off]={int(r['target'])}
        elif name in ('packed-switch','sparse-switch'):
            if r.get('switch_error') or 'switch_targets' not in r:return dict(status='UNKNOWN',reason='SWITCH_PAYLOAD_UNRESOLVED',offset=off)
            normal[off]=set(map(int,r['switch_targets']))|{nxt}
            branch[off]={f'CASE_{k}':int(t) for k,t in zip(r.get('switch_keys',[]),r['switch_targets'])};branch[off]['DEFAULT']=nxt
        elif name.startswith('return') or name=='throw':normal[off]={EXIT}
        else:normal[off]={nxt}
        if any(t!=EXIT and t not in rows for t in normal[off]):return dict(status='UNKNOWN',reason='INVALID_TARGET_OR_FALLTHROUGH',offset=off)
        succ[off]=set(normal[off])
        if may_throw(name):succ[off].add(EXIT)
    entry=min(rows);reachable=set();q=[entry]
    while q:
        n=q.pop()
        if n in reachable:continue
        reachable.add(n);q.extend(succ[n]-reachable)
    pred={n:set() for n in reachable}
    for a in reachable:
        for b in succ[a]:pred[b].add(a)
    indeg={n:len(pred[n]) for n in reachable};q=deque(sorted(n for n in reachable if indeg[n]==0));order=[]
    while q:
        a=q.popleft();order.append(a)
        for b in sorted(succ[a]):
            indeg[b]-=1
            if indeg[b]==0:q.append(b)
    if len(order)!=len(reachable):return dict(status='UNKNOWN',reason='REACHABLE_LOOP_OR_CYCLE')
    index={n:i for i,n in enumerate(order)};bits={n:1<<i for n,i in index.items()};dom={};post={};desc={}
    for n in order:
        if n==entry:dom[n]=bits[n]
        else:
            pp=list(pred[n]);v=dom[pp[0]]
            for p in pp[1:]:v &= dom[p]
            dom[n]=v|bits[n]
    for n in reversed(order):
        if n==EXIT:post[n]=bits[n];desc[n]=bits[n]
        else:
            ss=list(succ[n]);v=post[ss[0]];reach=0
            for s in ss:v &= post[s];reach |= desc[s]
            post[n]=v|bits[n];desc[n]=reach|bits[n]
    payload={'edges':sorted([a,b] for a in reachable for b in succ[a]),'entry':entry}
    return dict(status='PARSED',scope='ACYCLIC_CFG_WITH_UNCAUGHT_EXCEPTION_EXIT',rows=rows,normal=normal,succ=succ,
                branch=branch,entry=entry,reachable=reachable,order=order,pred=pred,bits=bits,dom=dom,post=post,desc=desc,
                cfg_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest())
def controlling_branches(cfg,event):
    if cfg['status']!='PARSED' or event not in cfg['reachable']:return []
    bit=cfg['bits'][event];proofs=[]
    for b,labels in cfg['branch'].items():
        if b not in cfg['reachable']:continue
        successors=set(labels.values())
        if len(successors)<2:continue
        good=[s for s in successors if cfg['desc'][s]&bit];bad=[s for s in successors if not(cfg['desc'][s]&bit)]
        # This is a strict subset of PDG control dependence: additionally require exclusive reachability.
        for s in good:
            if bad and cfg['post'][s]&bit and not(cfg['post'][b]&bit):
                proofs.append(dict(branch_offset=b,branch_opcode=cfg['rows'][b]['name'],
                  allowing_successor_offset=s,allowing_successor_labels=[k for k,v in labels.items() if v==s],
                  blocking_successor_offsets=sorted(bad),event_postdominates_allowing_successor=True,
                  blocked_successors_cannot_reach_event=True,branch_dominates_event=bool(cfg['dom'][event]&cfg['bits'][b]),
                  bypass_path_possible=not bool(cfg['dom'][event]&cfg['bits'][b]),cfg_sha256=cfg['cfg_sha256']))
    return proofs

def merge_states(states):
    if not states:return {}
    result=dict(states[0])
    for state in states[1:]:
        result={k:v for k,v in result.items() if state.get(k)==v}
    return result
def no_destination(name):
    return name.startswith(('invoke-','return','if-','goto','aput','iput','sput','monitor-','filled-new-array')) or name in ('nop','throw','packed-switch','sparse-switch','fill-array-data')
def symbolic_states(cfg,max_entries=300000):
    """Unique forward symbolic values at joins; unsupported transformations kill values."""
    if cfg['status']!='PARSED':return {}
    ins={};outs={};used=0
    for off in cfg['order']:
        if off==EXIT:continue
        state=dict(cfg.get('initial_symbols',{})) if off==cfg['entry'] else merge_states([outs[p] for p in cfg['pred'][off] if p!=EXIT]);ins[off]=state
        r=cfg['rows'][off];name=r['name'];regs=r.get('registers',[]);new=dict(state)
        if regs and not no_destination(name):
            dst=regs[0]
            new.pop(dst,None)
            wide_result=('wide' in name or name.endswith(('-long','-double','-long/2addr','-double/2addr')) or name.endswith(('to-long','to-double')))
            if wide_result:new.pop(dst+1,None)
            if name in ('move','move/from16','move/16','move-object','move-object/from16','move-object/16') and len(regs)==2 and regs[1] in state:
                value=state[regs[1]]
                typed_ok=(('object' in name and value[0] in ('PERMISSION_STRING','THIS')) or ('object' not in name and value[0] in ('PERMISSION_RESULT','INT')))
                if typed_ok:
                    if value[0]=='PERMISSION_STRING':value=(value[0],value[1],value[2]+(off,))
                    elif value[0]=='PERMISSION_RESULT':value=value[:6]+(value[6]+(off,),)
                    new[dst]=value
            elif name.startswith('const-string') and str(r.get('string','')).startswith('android.permission.'):
                new[dst]=('PERMISSION_STRING',r['string'],(off,))
            elif name in ('const/4','const/16','const','const/high16') and r.get('literal') is not None:
                new[dst]=('INT',int(r['literal']),off)
            elif name=='move-result':
                # DEX move-result must immediately follow its producing invoke on every predecessor.
                preds=list(cfg['pred'][off])
                if len(preds)==1:
                    prev=cfg['rows'][preds[0]];ref=prev.get('reference','')
                    if prev['offset']+prev['length']==off and ref in PERMISSION_APIS:
                        preg=prev.get('registers',[]);pos=PERMISSION_APIS[ref]
                        if prev['name'].startswith('invoke-static'):pos=1
                        permission=ins[preds[0]].get(preg[pos]) if len(preg)>pos else None
                        if permission and permission[0]=='PERMISSION_STRING':
                            dispatch=prev.get('permission_dispatch','UNVERIFIED')
                            receiver=ins[preds[0]].get(preg[0]) if preg else None
                            if prev.get('framework_target_when_receiver_this') and receiver==('THIS',):dispatch='VERIFIED_FRAMEWORK_DISPATCH'
                            new[dst]=('PERMISSION_RESULT',permission[1],preds[0],off,ref,dispatch,permission[2]+(preds[0],off))
        outs[off]=new;used+=len(new)
        if used>max_entries:return {'__ERROR__':'DEF_USE_STATE_CAP'}
    return ins
def permission_guard(cfg,proof,states):
    b=proof['branch_offset'];r=cfg['rows'][b];name=r['name'];regs=r.get('registers',[]);state=states.get(b,{})
    if '__ERROR__' in states:return dict(status='PARTIAL',reason=states['__ERROR__'])
    if name not in ('if-eqz','if-nez','if-eq','if-ne'):return None
    values=[state.get(reg) for reg in regs];permission=None;constant_offset=None
    if name.endswith('z') and len(values)==1 and values[0] and values[0][0]=='PERMISSION_RESULT':permission=values[0]
    elif len(values)==2:
        for a,c in ((values[0],values[1]),(values[1],values[0])):
            if a and a[0]=='PERMISSION_RESULT' and c and c[0]=='INT' and c[1]==0:permission=a;constant_offset=c[2]
    if not permission:return None
    granted_taken=name in ('if-eqz','if-eq');labels=proof['allowing_successor_labels']
    allowing_granted=('TAKEN' in labels and granted_taken) or ('FALLTHROUGH' in labels and not granted_taken)
    # Only comparisons against PERMISSION_GRANTED=0 have an interpreted direction.
    if permission[5]!='VERIFIED_FRAMEWORK_DISPATCH':return dict(status='PARTIAL',reason='PERMISSION_TARGET_DISPATCH_UNVERIFIED',permission=permission[1])
    if proof['bypass_path_possible']:return dict(status='PARTIAL',reason='LOCAL_PERMISSION_BRANCH_HAS_BYPASS',permission=permission[1])
    return dict(status='VERIFIED_PERMISSION_GUARD',permission=permission[1],permission_check_offset=permission[2],
                move_result_offset=permission[3],permission_api=permission[4],allows_permission_granted=allowing_granted,
                allows_permission_denied=not allowing_granted,branch_offset=b,permission_guard_bypass=False,
                def_use_scope='UNIQUE_SYMBOLIC_VALUE_ON_ALL_NORMAL_PREDECESSORS',
                def_use_instruction_offsets=sorted(set(permission[6]+(b,)+((constant_offset,) if constant_offset is not None else ()))),
                interpretation='Framework permission status gates this call; user consent for data purpose is not established.')
def analyze_event(cfg,offset,states=None):
    if cfg['status']!='PARSED':return dict(evidence_tier='UNKNOWN',reason=cfg['reason'],controls=[],permissions=[])
    if offset not in cfg['reachable']:return dict(evidence_tier='UNKNOWN',reason='CALL_NOT_ENTRY_REACHABLE',controls=[],permissions=[])
    proofs=controlling_branches(cfg,offset)
    if not proofs:return dict(evidence_tier='PARTIAL',reason='NO_STRICT_LOCAL_CONTROL_PROOF',controls=[],permissions=[])
    states=states if states is not None else symbolic_states(cfg)
    permissions=[p for proof in proofs if (p:=permission_guard(cfg,proof,states)) is not None]
    tier='VERIFIED_PERMISSION_GUARD' if any(p['status']=='VERIFIED_PERMISSION_GUARD' for p in permissions) else 'VERIFIED_CONTROL'
    return dict(evidence_tier=tier,reason='STRICT_LOCAL_POSTDOMINANCE_AND_EXCLUSIVE_REACHABILITY',controls=proofs,permissions=permissions)
