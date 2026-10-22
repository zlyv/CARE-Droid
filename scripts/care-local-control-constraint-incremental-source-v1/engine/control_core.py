#!/usr/bin/env python3
"""Conservative instruction CFG, postdominance and local permission def-use.

Accepted proofs use an over-approximated exceptional CFG. All potentially
throwing instructions retain EXIT, even inside catch-all regions. Cyclic CFGs
use strong postdominance: an infinite path avoiding the event invalidates proof.
No absence claim, termination claim or runtime feasibility claim is produced.
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
def _fixed_point(nodes,initial,dependencies,dependents,combine,queue_order,max_updates=2500000):
    values=dict(initial);queue=deque(queue_order);pending=set(queue);updates=0
    while queue:
        n=queue.popleft();pending.remove(n)
        value=combine(n,[values[s] for s in dependencies[n]])
        if value==values[n]:continue
        values[n]=value;updates+=1
        if updates>max_updates:raise RuntimeError('CFG_FIXED_POINT_WORK_CAP')
        for p in sorted(dependents[n]):
            if p not in pending:queue.append(p);pending.add(p)
    return values,updates

def build_cfg(instructions,tries_size=0,max_nodes=4096,exception_regions=None,exception_error=None):
    if exception_error:return dict(status='UNKNOWN',reason='EXCEPTION_TABLE_DECODE_FAILED',detail=str(exception_error))
    if tries_size and exception_regions is None:return dict(status='UNKNOWN',reason='EXCEPTION_TABLE_MISSING')
    regions=exception_regions or []
    if len(regions)!=int(tries_size):return dict(status='UNKNOWN',reason='EXCEPTION_REGION_COUNT_MISMATCH')
    rows={int(r['offset']):r for r in instructions if not r.get('payload')}
    if not rows:return dict(status='UNKNOWN',reason='NO_CODE')
    if len(rows)>max_nodes:return dict(status='UNKNOWN',reason='INSTRUCTION_CAP')
    if len(rows)!=sum(not r.get('payload') for r in instructions):return dict(status='UNKNOWN',reason='DUPLICATE_INSTRUCTION_OFFSET')
    code_end=max(int(r['offset'])+int(r['length']) for r in instructions)
    boundaries={int(r['offset']) for r in instructions}|{code_end};previous_end=-1
    for region in sorted(regions,key=lambda x:int(x['start_offset_bytes'])):
        start=int(region['start_offset_bytes']);end=int(region['end_offset_bytes'])
        if start not in rows or end not in boundaries or end<=start or start<previous_end:
            return dict(status='UNKNOWN',reason='INVALID_OR_OVERLAPPING_TRY_RANGE')
        previous_end=end
        handlers=region.get('handlers',[])
        if not handlers or any(int(h['target_offset_bytes']) not in rows for h in handlers):
            return dict(status='UNKNOWN',reason='INVALID_EXCEPTION_HANDLER_TARGET')
        # DEX permits ignoring the caught exception, in which case there need
        # not be move-exception at handler entry. Do not reject such handlers.
    normal={};succ={EXIT:set()};exceptional={EXIT:set()};branch={}
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
        elif name.startswith('return'):normal[off]={EXIT}
        elif name=='throw':normal[off]=set()
        else:normal[off]={nxt}
        if any(t!=EXIT and t not in rows for t in normal[off]):return dict(status='UNKNOWN',reason='INVALID_TARGET_OR_FALLTHROUGH',offset=off)
        succ[off]=set(normal[off])
        exceptional[off]=set()
        if may_throw(name) or name=='throw':
            exceptional[off].add(EXIT)
            for region in regions:
                # monitor-exit throws with its PC advanced. Include both PC
                # interpretations conservatively; never omit the advanced one.
                pcs=(off,nxt) if name=='monitor-exit' else (off,)
                if any(int(region['start_offset_bytes'])<=pc<int(region['end_offset_bytes']) for pc in pcs):
                    exceptional[off].update(int(h['target_offset_bytes']) for h in region['handlers'])
            succ[off].update(exceptional[off])
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
    cyclic=len(order)!=len(reachable)
    # DFS finishing order also accelerates monotone equations on cyclic graphs.
    visited=set();finished=[];stack=[(entry,False)]
    while stack:
        n,closing=stack.pop()
        if closing:finished.append(n);continue
        if n in visited:continue
        visited.add(n);stack.append((n,True))
        for s in sorted(succ[n],reverse=True):
            if s not in visited:stack.append((s,False))
    index={n:i for i,n in enumerate(sorted(reachable))};bits={n:1<<i for n,i in index.items()}
    universe=(1<<len(bits))-1
    def intersection(values):
        value=values[0]
        for other in values[1:]:value &= other
        return value
    def post_step(n,values):return bits[n] if not values else bits[n]|intersection(values)
    def desc_step(n,values):
        value=bits[n]
        for v in values:value |= v
        return value
    def dom_step(n,values):return bits[n] if n==entry else bits[n]|intersection(values)
    try:
        # Singleton initialization is essential: a greatest fixed point would
        # incorrectly assume loop termination and prove events after a loop.
        post,post_updates=_fixed_point(reachable,bits,succ,pred,post_step,finished)
        desc,desc_updates=_fixed_point(reachable,bits,succ,pred,desc_step,finished)
        dom,dom_updates=_fixed_point(reachable,{n:bits[n] if n==entry else universe for n in reachable},
                                     pred,succ,dom_step,list(reversed(finished)))
    except RuntimeError as exc:return dict(status='UNKNOWN',reason=str(exc))
    payload={'edges':sorted([a,b] for a in reachable for b in succ[a]),'entry':entry,
             'exception_regions':regions,'strong_postdominance':'least_fixed_point_all_maximal_paths'}
    return dict(status='PARSED',scope='EXCEPTIONAL_CFG_STRONG_POSTDOMINANCE',rows=rows,normal=normal,succ=succ,
                branch=branch,entry=entry,reachable=reachable,order=order,pred=pred,bits=bits,dom=dom,post=post,desc=desc,
                exceptional=exceptional,exception_regions=regions,has_reachable_cycle=cyclic,
                exceptional_edge_count=sum(len(exceptional[n]) for n in reachable),try_region_count=len(regions),
                fixed_point_updates=dict(post=post_updates,reachability=desc_updates,dominance=dom_updates),
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
                proofs[-1].update(exception_paths_included=True,infinite_paths_considered=True,
                                  postdominance_semantics='ALL_MAXIMAL_PATHS_LEAST_FIXED_POINT')
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
    if cfg.get('has_reachable_cycle'):return {'__ERROR__':'CYCLIC_PERMISSION_DEF_USE_UNSUPPORTED'}
    ins={};outs={};used=0
    for off in cfg['order']:
        if off==EXIT:continue
        # A throwing instruction may not complete its writes. Never forward
        # normal-completion values onto exception edges; erase all symbols.
        state=dict(cfg.get('initial_symbols',{})) if off==cfg['entry'] else merge_states(
            [{} if off in cfg['exceptional'][p] else outs[p] for p in cfg['pred'][off] if p!=EXIT]);ins[off]=state
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
                def_use_scope='UNIQUE_SYMBOLIC_VALUE_ON_ALL_CFG_PREDECESSORS_EXCEPTION_VALUES_KILLED',
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
