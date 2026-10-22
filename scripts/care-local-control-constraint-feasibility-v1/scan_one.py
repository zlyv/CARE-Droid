#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,re,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
from common import *
from control_core import build_cfg,symbolic_states,analyze_event,PERMISSION_APIS
from dex_adapter import decode_method,matching_roles,alignment,framework_permission_target

def scan(apk_path,sid,mapping,frozen,scan_identity):
    import yaml
    try:
        from loguru import logger
        logger.remove()
    except ImportError:pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    start=time.monotonic();apk=APK(str(apk_path));vocab=sorted(mapping['roles']);role_id={v:i for i,v in enumerate(vocab)}
    units=[];class_info={};signature_counts=Counter();method_defs={};methods=[];method_failures=[]
    for dex_name in sorted(apk.get_dex_names()):
        raw=apk.get_file(dex_name);dex=DEX(raw);dsha=hashlib.sha256(raw).hexdigest();units.append(dex)
        for cls in dex.get_classes():
            exact_cls=str(cls.get_name());c=canonical(exact_cls);info=dict(super=canonical(cls.get_superclassname()),final=bool(int(cls.get_access_flags())&0x10),methods=set())
            if c in class_info:
                info['final']=False;info['ambiguous']=True
                info['methods'].update(class_info[c]['methods'])
            for method in cls.get_methods():
                exact_name=str(method.get_name());exact_desc=re.sub(r'\s+','',str(method.get_descriptor()))
                exact_sig=exact_cls+'->'+exact_name+exact_desc;sig=canonical(exact_sig);hid=str(stable_id(sig))
                signature_counts[hid]+=1;method_defs.setdefault(hid,[]).append(dict(dex_name=dex_name,dex_sha256=dsha,exact_signature=exact_sig))
                info['methods'].add(canonical(exact_name+exact_desc));methods.append((method,dex_name,dsha,exact_cls,exact_name,exact_desc,sig,hid))
            class_info[c]=info
    events=[];method_rows=[];recovered_pairs=set();recovered_hosts=set();invokes_by_host={};host_set=set(frozen['host_ids']);role_cache={};unresolved_by_host=Counter()
    for method,dex_name,dsha,exact_cls,exact_name,exact_desc,sig,hid in methods:
        if hid in host_set and signature_counts[hid]==1:recovered_hosts.add(hid)
        try:instructions,meta=decode_method(method)
        except Exception as exc:
            if hid in host_set:method_failures.append(dict(host_node_id=hid,method_signature=sig,reason='DEX_INSTRUCTION_DECODE_FAILED',error=repr(exc)))
            continue
        sensitive=[]
        for r in instructions:
            ref=r.get('reference','')
            if hid in host_set and r['name'].startswith('invoke-') and not ref:unresolved_by_host[hid]+=1
            if ref:
                if hid in host_set:invokes_by_host.setdefault(hid,[]).append(ref)
                if ref not in role_cache:role_cache[ref]=matching_roles(ref,mapping)
                roles=role_cache[ref]
                if roles or hid+':'+str(stable_id(ref)) in frozen['pairs']:sensitive.append((r,ref,roles))
                r['framework_target_when_receiver_this']=framework_permission_target(canonical(exact_cls),ref,r['name'],class_info)
        if not sensitive and hid not in host_set:continue
        cfg=build_cfg(instructions,meta['tries_size']) if hid in host_set else dict(status='NOT_ANALYZED',reason='OUTSIDE_FROZEN_HOST_SCOPE')
        if cfg['status']=='PARSED' and not int(method.get_access_flags())&0x8:
            cfg['initial_symbols']={meta['registers_size']-meta['ins_size']:('THIS',)}
        states=symbolic_states(cfg) if cfg['status']=='PARSED' and any(r.get('reference') in PERMISSION_APIS for r in instructions) else {}
        method_rows.append(dict(host_node_id=hid,method_signature=sig,dex_sha256=dsha,dex_name=dex_name,
           instruction_count=len(instructions),sensitive_call_count=len(sensitive),unresolved_invoke_count=unresolved_by_host[hid],frozen_host=hid in host_set,
           cfg_status=cfg['status'],cfg_reason=cfg.get('reason',''),cfg_sha256=cfg.get('cfg_sha256','')))
        for r,ref,roles in sensitive:
            eid=hashlib.sha256(json.dumps([sid,dex_name,dsha,exact_cls,exact_name,exact_desc,r['offset']],separators=(',',':')).encode()).hexdigest()
            e=dict(event_id=eid,sha256=sid,dex_name=dex_name,dex_sha256=dsha,class_descriptor=exact_cls,method_name=exact_name,
              method_descriptor=exact_desc,method_signature=sig,host_node_id=int(hid),endpoint_node_id=stable_id(ref),
              invoke_offset_bytes=r['offset'],invoke_opcode=r['name'],sensitive_reference=ref,roles=roles,role_ids=[role_id[v] for v in roles])
            e['alignment_status']=alignment(e,frozen,signature_counts);e['frozen_aligned']=e['alignment_status']=='EXACT_FROZEN_HOST_ENDPOINT_EDGE'
            if e['frozen_aligned']:recovered_pairs.add(hid+':'+str(e['endpoint_node_id']))
            # Scope excludes any non-frozen event from new formal candidate observations.
            if e['frozen_aligned']:
                proof=analyze_event(cfg,r['offset'],states);e.update(proof)
                cited={r['offset']}
                for c in proof['controls']:cited.add(c['branch_offset'])
                for p in proof['permissions']:
                    if p.get('permission_check_offset') is not None:cited.update([p['permission_check_offset'],p['move_result_offset']])
                    cited.update(p.get('def_use_instruction_offsets',[]))
                e['instruction_evidence']=[dict(offset=x['offset'],name=x['name'],output=x['output']) for x in instructions if x['offset'] in cited]
                e['cfg_status']=cfg['status'];e['cfg_sha256']=cfg.get('cfg_sha256','')
            else:e.update(evidence_tier='UNKNOWN',reason='OUTSIDE_FROZEN_HOST_SCOPE',controls=[],permissions=[])
            events.append(e)
    missing=[];events_by_pair=defaultdict(list)
    for e in events:events_by_pair[str(e['host_node_id'])+':'+str(e['endpoint_node_id'])].append(e)
    for pair in sorted(set(frozen['pairs'])-recovered_pairs):
        host,target=pair.split(':');reason='FROZEN_HOST_METHOD_NOT_FOUND'
        if signature_counts.get(host,0)>1:reason='AMBIGUOUS_CANONICAL_HOST_ID'
        if host in recovered_hosts:
            reason='DEX_METHOD_DECODE_FAILED' if any(r['host_node_id']==host for r in method_failures) else 'FROZEN_EDGE_HAS_NO_EXACT_ROLE_MATCHED_INVOKE'
            if unresolved_by_host[host] and reason!='DEX_METHOD_DECODE_FAILED':reason='UNRESOLVED_INVOKES_PRESENT_IN_FROZEN_HOST'
            if signature_counts.get(host,0)>1:reason='AMBIGUOUS_CANONICAL_HOST_ID'
            if any(e['alignment_status']=='FROZEN_ROLE_SET_MISMATCH' for e in events_by_pair.get(pair,[])):reason='FROZEN_ROLE_SET_MISMATCH'
            for ref in invokes_by_host.get(host,[]):
                cls,tail=ref.split('->',1);seen=set()
                while cls in class_info and cls not in seen:
                    seen.add(cls);cls=class_info[cls]['super']
                    if str(stable_id(cls+'->'+tail))==target:reason='INHERITED_DECLARATION_ALIGNMENT_CANDIDATE_UNVERIFIED'
        missing.append(dict(pair_key=pair,host_node_id=host,endpoint_node_id=target,reason=reason))
    return dict(schema=SCHEMA,scan_identity_sha256=scan_identity,status='OK',sha256=sid,labels_used=False,future_rows_loaded=0,
      total_methods=len(methods),dex_count=len(units),events=events,methods=method_rows,method_failures=method_failures,
      frozen_pairs_total=len(frozen['pairs']),frozen_pairs_recovered=len(recovered_pairs),frozen_hosts_total=len(frozen['host_ids']),
      frozen_hosts_recovered=len(recovered_hosts),missing_frozen_pairs=missing,wall_seconds=time.monotonic()-start)
def main():
    p=argparse.ArgumentParser();p.add_argument('--apk',required=True);p.add_argument('--sha',required=True);p.add_argument('--frozen',required=True);p.add_argument('--mapping',required=True);p.add_argument('--output',required=True);p.add_argument('--status',required=True);p.add_argument('--scan-identity',required=True);a=p.parse_args()
    t=time.monotonic()
    try:
        import yaml
        result=scan(Path(a.apk),a.sha,yaml.safe_load(Path(a.mapping).read_text(encoding='utf-8')),json.loads(Path(a.frozen).read_text(encoding='utf-8')),a.scan_identity)
        dump_gz(Path(a.output),result);status=dict(sha256=a.sha,status='OK',scan_identity_sha256=a.scan_identity,wall_seconds=time.monotonic()-t)
    except Exception as exc:status=dict(sha256=a.sha,status='ERROR',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-t,scan_identity_sha256=a.scan_identity)
    dump(Path(a.status),status)
    if status['status']!='OK':raise SystemExit(1)
if __name__=='__main__':main()
