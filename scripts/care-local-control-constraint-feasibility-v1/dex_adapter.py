#!/usr/bin/env python3
from __future__ import annotations
import re
from common import canonical,stable_id

REF=re.compile(r'L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+',re.I)
def method_reference(text):
    hits=REF.findall(re.sub(r'\s+','',str(text)));return canonical(hits[-1]) if hits else None
def matching_roles(reference,mapping):
    if '->' not in reference:return []
    cls,tail=reference.split('->',1);name=tail.split('(',1)[0];roles=set()
    for rule in sorted(mapping.get('rules',[]),key=lambda r:str(r.get('rule_id',''))):
        if rule.get('rule_kind','api')!='api':continue
        cp=[canonical(v) for v in rule.get('class_prefixes',[])];cc=[canonical(v) for v in rule.get('class_contains',[])]
        mn=[canonical(v) for v in rule.get('method_names',[])];mp=[canonical(v) for v in rule.get('method_prefixes',[])]
        if (any(cls.startswith(v) for v in cp) or any(v in cls for v in cc)) and (name in mn or any(name.startswith(v) for v in mp)):
            roles.add(str(rule['role']))
    return sorted(roles)
def decode_method(method):
    code=method.get_code()
    if code is None:return [],dict(tries_size=0,registers_size=0,ins_size=0)
    objects=list(method.get_instructions_idx());by={int(off):obj for off,obj in objects};result=[]
    for off,obj in objects:
        off=int(off);name=str(obj.get_name()).lower();row=dict(offset=off,name=name,length=int(obj.get_length()),output=str(obj.get_output()),registers=[])
        row['payload']='payload' in name or name in ('packed-switch-payload','sparse-switch-payload','fill-array-data-payload')
        if row['payload']:result.append(row);continue
        try:
            ops=obj.get_operands();row['registers']=[int(v[1]) for v in ops if int(v[0])==0]
            literals=[int(v[1]) for v in ops if int(v[0])==1]
            if literals:row['literal']=literals[0]
            if name.startswith('const-string'):
                strings=[str(v[2]) for v in ops if int(v[0])==257 and len(v)>2];row['string']=strings[0] if strings else ''
            if name.startswith('invoke-'):row['reference']=method_reference(row['output']) or ''
            if name.startswith(('if-','goto')):row['target']=off+2*int(obj.get_ref_off())
            if name in ('packed-switch','sparse-switch','fill-array-data'):
                payload_off=off+2*int(obj.get_ref_off());payload=by.get(payload_off)
                if payload is None:row['decode_error']='PAYLOAD_NOT_FOUND'
                elif name in ('packed-switch','sparse-switch'):
                    expected='packed-switch-payload' if name=='packed-switch' else 'sparse-switch-payload'
                    if str(payload.get_name()).lower()!=expected:row['switch_error']='WRONG_PAYLOAD_KIND'
                    else:
                        row['switch_targets']=[off+2*int(x) for x in payload.get_targets()];row['switch_keys']=list(map(int,payload.get_keys()))
                        if len(row['switch_targets'])!=len(row['switch_keys']):row['switch_error']='KEY_TARGET_LENGTH_MISMATCH'
                elif str(payload.get_name()).lower()!='fill-array-data-payload':row['decode_error']='WRONG_ARRAY_PAYLOAD_KIND'
        except Exception as exc:row['decode_error']=type(exc).__name__+':'+str(exc)
        result.append(row)
    return result,dict(tries_size=int(code.get_tries_size()),registers_size=int(code.get_registers_size()),ins_size=int(code.get_ins_size()))
def alignment(event,frozen,signature_counts):
    host=str(event['host_node_id']);target=str(event['endpoint_node_id']);pair=host+':'+target
    if signature_counts.get(host,0)>1:return 'AMBIGUOUS_CANONICAL_HOST_ID'
    if pair in frozen['pairs']:
        if set(event['role_ids'])!=set(frozen['pairs'][pair]):return 'FROZEN_ROLE_SET_MISMATCH'
        return 'EXACT_FROZEN_HOST_ENDPOINT_EDGE'
    nodes=frozen['nodes'];summary=frozen['summary']
    if target not in nodes or not nodes[target]['endpoint']:
        return 'ENDPOINT_NOT_SELECTED_CAP_PRESENT' if summary.get('endpoint_truncated') else 'ENDPOINT_NOT_SELECTED_OR_MAPPING_MISMATCH'
    if host not in nodes:
        return 'HOST_NOT_RETAINED_CAP_PRESENT' if summary.get('node_truncated') or summary.get('edge_truncated') else 'HOST_NOT_RETAINED_OR_NOT_RESOLVED_BY_OLD_GRAPH'
    if host not in frozen['host_ids']:
        return 'HOST_PRESENT_WITHOUT_FROZEN_SENSITIVE_EDGE'
    return 'FROZEN_HOST_DIFFERENT_ENDPOINT_OR_EDGE_NOT_RETAINED'

FRAMEWORK_CONTEXTS={'landroid/content/context;','landroid/content/contextwrapper;','landroid/view/contextthemewrapper;',
                    'landroid/app/activity;','landroid/app/service;','landroid/app/application;'}
def framework_permission_target(class_name,reference,opcode,class_info):
    # Restrict permission semantics to a known framework target and a resolvable this receiver.
    if reference!='landroid/content/context;->checkselfpermission(ljava/lang/string;)i':return False
    info=class_info.get(class_name)
    if not info or info.get('ambiguous'):return False
    if opcode.startswith('invoke-super'):current=info['super']
    elif opcode.startswith('invoke-virtual') and info['final']:current=class_name
    else:return False
    seen=set()
    while current in class_info:
        if current in seen:return False
        seen.add(current);c=class_info[current]
        if c.get('ambiguous'):return False
        if 'checkselfpermission(ljava/lang/string;)i' in c['methods']:return False
        current=c['super']
    return current in FRAMEWORK_CONTEXTS
