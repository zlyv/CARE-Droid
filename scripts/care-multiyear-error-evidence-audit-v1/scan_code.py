"""Bounded static DEX instruction extraction for exact frozen Host hashes; no APK installation or execution."""
from __future__ import annotations
import argparse,gzip,re,traceback,zipfile
from collections import Counter,defaultdict
from audit_common import *

def reference(output):
    matches=re.findall(r'(l[^,\s]+;->[^\s(]+\([^)]*\)[^,\s]+)',canonical(output))
    return matches[-1] if matches else None

def redact(name,output):
    # Retain permission names for review; never emit arbitrary embedded tokens,
    # URLs or credentials from APK string constants into logs/dossiers.
    if name.startswith('const-string'):
        permissions=re.findall(r'android\.permission\.[A-Za-z_]+',output)
        if permissions:return '[PERMISSION_LITERAL '+','.join(permissions)+']'
        return '[STRING_LITERAL_REDACTED sha256='+hashlib.sha256(output.encode()).hexdigest()+' length='+str(len(output))+']'
    return output

def scan(job):
    start=time.monotonic();sid=job['sha256'];path=Path(job['apk_path']);out=dict(member_id=job['member_id'],status='UNKNOWN',methods=[],static_only=True,labels_and_predictions_excluded=True)
    if not path.is_file():return dict(out,status='RAW_APK_MISSING')
    if sha256(path)!=sid:return dict(out,status='RAW_ASSET_SHA_MISMATCH')
    try:
        from loguru import logger
        logger.remove()
    except ImportError:pass
    from androguard.core.dex import DEX
    wanted={r['host_node_id']:r for r in job['hosts'] if r['host_node_id']!='EMPTY_PLACEHOLDER'};found=defaultdict(list);units=[];method_count=0
    with zipfile.ZipFile(path) as apk:
        dex_names=sorted(n for n in apk.namelist() if re.fullmatch(r'classes(?:\d+)?\.dex',n))
        total=sum(apk.getinfo(n).file_size for n in dex_names)
        if total>512*1024*1024 or any(apk.getinfo(n).file_size>128*1024*1024 for n in dex_names):return dict(out,status='UNKNOWN_DEX_RESOURCE_BOUND',uncompressed_dex_bytes=total)
        for name in dex_names:
            raw=apk.read(name);dsha=hashlib.sha256(raw).hexdigest();dex=DEX(raw);units.append(dex)
            for cls in dex.get_classes():
                for m in cls.get_methods():
                    method_count+=1;sig=str(cls.get_name())+'->'+str(m.get_name())+re.sub(r'\s+','',str(m.get_descriptor()));hid=stable_id(sig)
                    if hid in wanted:found[hid].append((m,name,dsha,sig))
    hosts=[]
    for hid,h in sorted(wanted.items(),key=lambda z:z[1]['slot']):
        matches=found[hid];record=dict(slot=h['slot'],host_node_id=hid,role_indices=h['role_indices'],frozen_endpoint_ids=h['endpoint_ids'],frozen_endpoint_roles=h['endpoint_roles'])
        if len(matches)!=1:
            record.update(status='CANONICAL_HOST_AMBIGUOUS' if len(matches)>1 else 'FROZEN_HOST_METHOD_NOT_FOUND',candidate_count=len(matches));hosts.append(record);continue
        method,dex_name,dsha,sig=matches[0];record.update(status='EXACT_CANONICAL_HOST',method_signature=sig,dex_name=dex_name,dex_sha256=dsha)
        if method.get_code() is None:record.update(status='HOST_HAS_NO_DEX_BODY');hosts.append(record);continue
        try:
            iterator=method.get_instructions_idx();instructions=[];invokes=[];offsets=[];branches=[]
            for offset,ins in iterator:
                name=str(ins.get_name());output=str(ins.get_output());ref=reference(output) if name.startswith('invoke-') else None
                line=dict(offset_bytes=int(offset),opcode=name,operands=redact(name,output));instructions.append(line)
                if name.startswith('if-') or 'switch' in name:branches.append(int(offset))
                if name.startswith('invoke-'):
                    eid=stable_id(ref) if ref else None;exact=bool(eid in set(h['endpoint_ids']))
                    invokes.append(dict(offset_bytes=int(offset),opcode=name,canonical_reference=ref or '',endpoint_node_id=eid,exact_frozen_endpoint=exact,roles=h['endpoint_roles'].get(eid,[])))
                    if exact:offsets.append(int(offset))
            # Preserve a full bounded method, or explicitly say it is incomplete.
            # For long methods retain neighborhoods around matching frozen invokes.
            trunc=len(instructions)>2000
            if trunc:
                selected=set()
                for i,row in enumerate(instructions):
                    if row['offset_bytes'] in offsets:selected.update(range(max(0,i-12),min(len(instructions),i+13)))
                selected.update(range(min(30,len(instructions))));evidence=[instructions[i] for i in sorted(selected)][:2000]
            else:evidence=instructions
            record.update(instruction_count=len(instructions),instruction_evidence=evidence,instruction_evidence_truncated=trunc,invoke_records=invokes,branch_offsets=branches,exact_frozen_endpoint_ids_recovered=sorted({r['endpoint_node_id'] for r in invokes if r['exact_frozen_endpoint']}),method_access_flags=int(method.get_access_flags()),tries_size=int(method.get_code().get_tries_size()),reason_boundary='branches and invoke co-location are code facts only; no control/data-flow or class-direction proof')
        except Exception as e:record.update(status='METHOD_DECODE_ERROR',error=repr(e))
        hosts.append(record)
    out.update(status='STATIC_CODE_EXTRACTED',methods=hosts,dex_count=len(units),indexed_method_count=method_count,selected_host_count=len(wanted),exact_host_bodies=sum(r['status']=='EXACT_CANONICAL_HOST' and 'instruction_evidence' in r for r in hosts),unresolved_host_count=sum(r['status']!='EXACT_CANONICAL_HOST' for r in hosts),wall_seconds=time.monotonic()-start,literals_redacted=True,all_original_APK_bytes_unchanged=True,unverified_inherited_targets_not_guessed=True)
    return out

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--job',required=True);a=cli.parse_args();job=read_json(a.job);start=time.monotonic()
    try:out=scan(job)
    except Exception as e:out=dict(member_id=job['member_id'],status='STATIC_SCAN_ERROR',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start)
    path=safe(OUTPUT/'blind/code'/(job['member_id']+'.json.gz'))
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(plain(out),f,ensure_ascii=False)
    dump(OUTPUT/'private/scan_status'/(job['sha256']+'.json'),dict(sha256=job['sha256'],member_id=job['member_id'],status=out['status'],wall_seconds=out.get('wall_seconds',time.monotonic()-start),selected_hosts=out.get('selected_host_count',0),exact_host_bodies=out.get('exact_host_bodies',0)))
    print('MEEA_STATIC_CODE',job['member_id'],out['status'],flush=True)

if __name__=='__main__':main()
