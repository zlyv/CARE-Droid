"""Package priority blinded code dossiers and validate reviewer citations without converting opinions into confirmed causes."""
from __future__ import annotations
import argparse,gzip,shutil
from audit_common import *

def priority():
    dest=OUTPUT/'blind_priority_review'
    if dest.exists():raise RuntimeError('priority packet exists; no overwrite')
    keys=read_csv(OUTPUT/'private/blind_review_key.csv');wanted={r['case_id'] for r in keys if r['target_category']=='REGRESSED_FN'};forms=read_csv(OUTPUT/'blind/review_form.csv');members=set()
    for cid in sorted(wanted):
        path=OUTPUT/'blind/cases'/(cid+'.json');d=read_json(path)
        shutil.copy2(path,safe(dest/'cases'/path.name))
        members.update(m['member_id'] for m in d['members'])
    for mid in sorted(members):
        for ext in ['.json.gz','.failure.json']:
            path=OUTPUT/'blind/code'/(mid+ext)
            if path.exists():shutil.copy2(path,safe(dest/'code'/path.name))
    write_csv(dest/'review_form.csv',[r for r in forms if r['case_id'] in wanted])
    safe(dest/'README.md').write_text('# Blinded priority code review\n\n10 case triples, selected before semantic review. Labels, predictions and target-member identities are withheld. Application code can reveal function, so identity blinding is incomplete. Cite member, exact method and DEX offset. Distinguish a verifiable code fact from a benign/malicious judgment; permission checks, branch presence and weak anomaly are not benign proof. Unknown is valid. Do not inspect private keys until the review is frozen. A completed form records a reviewer claim, not an independently confirmed classifier-error mechanism.\n',encoding='utf-8')
    dump(OUTPUT/'priority_packet_manifest.json',dict(case_count=len(wanted),unique_code_members=len(members),packet=str(dest),private_keys_not_copied=True,manual_claims_not_automatic_causes=True))
    print('MEEA_PRIORITY_PACKET',len(wanted),len(members),flush=True)

def validate(form):
    rows=read_csv(form);expected={r['case_id'] for r in read_csv(OUTPUT/'blind/review_form.csv')};seen=set();checked=[]
    for r in rows:
        cid=r['case_id']
        if cid not in expected or cid in seen:raise RuntimeError('unknown or duplicate case ID '+cid)
        seen.add(cid);complete=bool(r.get('verified_program_fact','').strip());status='PENDING';reason='no reviewer fact recorded'
        if complete:
            case=read_json(OUTPUT/'blind/cases'/(cid+'.json'));letter=r.get('supporting_code_member','');member=next((m for m in case['members'] if m['member']==letter),None)
            if not member:status='INVALID_CITATION';reason='member must be A/B/C from that case'
            else:
                path=OUTPUT/'blind/code'/(member['member_id']+'.json.gz')
                if not path.exists():status='INVALID_CITATION';reason='selected code was unavailable'
                else:
                    with gzip.open(path,'rt',encoding='utf-8') as f:code=json.load(f)
                    try:offset=int(r.get('dex_offset_bytes',''))
                    except (TypeError,ValueError):offset=None
                    valid=[m for m in code.get('methods',[]) if m.get('method_signature')==r.get('method_signature') and any(i['offset_bytes']==offset for i in m.get('instruction_evidence',[]))]
                    status='REVIEWER_CLAIM_WITH_LOCATABLE_CODE' if len(valid)==1 else 'INVALID_CITATION';reason='citation exists; semantic/class-causal inference still requires independent checking' if len(valid)==1 else 'method/offset absent or ambiguous in bounded dossier'
            if status=='INVALID_CITATION':raise RuntimeError(cid+': '+reason)
        checked.append(dict(case_id=cid,status=status,reason=reason,direction=r.get('direction','UNKNOWN'),scope=r.get('scope','UNKNOWN')))
    dest=OUTPUT/'review_validation.json'
    if dest.exists():raise RuntimeError('review validation exists; use a new independently versioned review, no overwrite')
    dump(dest,dict(status='REVIEW_VALIDATED_FOR_LOCATION_ONLY',reviewed_claims=sum(r['status']=='REVIEWER_CLAIM_WITH_LOCATABLE_CODE' for r in checked),causes_confirmed=0,training_authorized=False,missing_cases=sorted(expected-seen),details=checked))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--priority-packet',action='store_true');g.add_argument('--validate-form');a=p.parse_args()
    if a.priority_packet:priority()
    else:validate(a.validate_form)
