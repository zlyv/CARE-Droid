"""Resolve the observed ambiguous frozen Host IDs to exact DEX definitions; never guess or repair official identities."""
from __future__ import annotations
import gzip,zipfile
from collections import defaultdict,Counter
from audit_common import *

def main():
    if (OUTPUT/'canonical_identity_audit.json').exists():raise RuntimeError('identity audit exists; no overwrite')
    try:
        from loguru import logger
        logger.remove()
    except ImportError:pass
    from androguard.core.dex import DEX
    members=read_csv(OUTPUT/'private/cohort_key.csv');queries={}
    for r in members:
        with gzip.open(OUTPUT/'blind/code'/(r['member_id']+'.json.gz'),'rt',encoding='utf-8') as f:d=json.load(f)
        wanted={m['host_node_id'] for m in d.get('methods',[]) if m['status']=='CANONICAL_HOST_AMBIGUOUS'}
        if wanted:queries[r['sha256']]=(r,wanted)
    details=[]
    for sid,(r,wanted) in queries.items():
        found=defaultdict(list)
        with zipfile.ZipFile(r['apk_path']) as apk:
            for name in sorted(n for n in apk.namelist() if re.fullmatch(r'classes(?:\d+)?\.dex',n)):
                dex=DEX(apk.read(name))
                for cls in dex.get_classes():
                    for method in cls.get_methods():
                        sig=str(cls.get_name())+'->'+str(method.get_name())+re.sub(r'\s+','',str(method.get_descriptor()));hid=stable_id(sig)
                        if hid in wanted:
                            code=method.get_code();instructions=[(i.get_name(),i.get_output()) for i in method.get_instructions()] if code else []
                            found[hid].append(dict(dex_name=name,exact_signature=sig,canonical_signature=canonical(sig),bytecode_instruction_content_sha256=signature(instructions),instruction_count=len(instructions),method_index=int(method.get_method_idx())))
        for hid,candidates in found.items():
            distinct={a['exact_signature'] for a in candidates};normalized={a['canonical_signature'] for a in candidates}
            kind='CASE_NORMALIZATION_COLLAPSE' if len(distinct)>1 and len(normalized)==1 else 'DUPLICATE_EXACT_DEFINITION' if len(distinct)==1 else 'OTHER_IDENTITY_AMBIGUITY'
            details.append(dict(sha256=sid,member_id=r['member_id'],target=r['target'],category=r['category'],host_node_id=hid,kind=kind,distinct_exact_signatures=len(distinct),distinct_body_hashes=len({a['bytecode_instruction_content_sha256'] for a in candidates}),candidates=candidates))
        print('MEEA_IDENTITY',r['member_id'],len(found),flush=True)
    dump(OUTPUT/'private/canonical_identity_details.json',details)
    summary=dict(status='FROZEN_IDENTITY_AMBIGUITY_INSPECTED',APK_count=len(queries),Host_count=len(details),kind_counts=dict(Counter(r['kind'] for r in details)),target_APKs=len({r['sha256'] for r in details if r['target']=='True'}),correct_control_APKs=len({r['sha256'] for r in details if r['target']=='False'}),all_affected_target_categories=sorted({r['category'] for r in details if r['target']=='True'}),classification_error_causality_confirmed=False,official_identity_repair_performed=False,scope='only ambiguous selected Hosts among inspected cases/controls; not a whole-corpus estimate')
    dump(OUTPUT/'canonical_identity_audit.json',summary);integrity();print('MEEA_IDENTITY_AUDIT_DONE',flush=True)

if __name__=='__main__':main()
