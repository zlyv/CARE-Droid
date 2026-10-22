#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
from common import *

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
    sys.dont_write_bytecode=True
    import androguard,yaml
    assets=discover();selected=route_rows(assets['route']);wanted={r['sha256'] for r in selected};apks,audit=apk_paths(wanted)
    endpoints=endpoint_rows(assets['endpoint'],wanted);protocol_path=Path(__file__).with_name('protocol.json');protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol_sha=sha256(protocol_path)
    parent_files=[BASELINE_OUTPUT/name for name in ('protocol.json','selection_label_free.csv','protocol_identity.json','coverage.json','final_decision.json','baseline_integrity_check.json')]
    for p in parent_files:
        if not p.is_file():raise RuntimeError('BLOCKED: completed v1 audit required: '+str(p))
    parent=json.loads((BASELINE_OUTPUT/'protocol.json').read_text(encoding='utf-8'))
    validate_parent_protocol(protocol,parent)
    validate_parent_selection(selected,list(csv.DictReader((BASELINE_OUTPUT/'selection_label_free.csv').open(encoding='utf-8-sig'))))
    parent_integrity=json.loads((BASELINE_OUTPUT/'baseline_integrity_check.json').read_text(encoding='utf-8'))
    if parent_integrity['status']!='PASS':raise RuntimeError('BLOCKED: v1 formal integrity was not PASS')
    engine={name:sha256(Path(__file__).with_name(name)) for name in ('common.py','control_core.py','dex_adapter.py','scan_one.py')}
    identity=dict(protocol_sha256=protocol_sha,engine_source_sha256=engine)
    token=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    existing=OUTPUT/'protocol.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))!=protocol:raise RuntimeError('BLOCKED: output belongs to a different protocol')
    dump(existing,protocol);dump(OUTPUT/'protocol_identity.json',dict(**identity,scan_identity_sha256=token,schema=SCHEMA))
    # Verify the exact normalization/hash contract against the local formal builder.
    bases=[PROJECT/'scripts/care-source-recovery-stage2only-v1/extractor_repo/src/tasa_droid/p2',PROJECT/'src/tasa_droid/p2']
    if os.environ.get('LCC_FORMAL_P2_ROOT'):bases.insert(0,Path(os.environ['LCC_FORMAL_P2_ROOT']))
    base=next((p for p in bases if (p/'graph_representation.py').is_file() and (p/'endpoint_centered_view.py').is_file()),None)
    if base is None:raise RuntimeError('BLOCKED: formal graph signature implementation missing; set LCC_FORMAL_P2_ROOT to existing p2 directory')
    graph=load_module('lcc_formal_graph',base/'graph_representation.py');endpoint=load_module('lcc_formal_endpoint',base/'endpoint_centered_view.py')
    hostrel_path=first([PROJECT/'scripts/care-di25-compact43-relationquery-source-v1/hostrel_features.py'],'formal Host reconstruction')
    hostrel=load_module('lcc_formal_hostrel',hostrel_path)
    for sig in ('LExample/Foo;->Bar( I Ljava/lang/String; )V','Landroid/Telephony/Foo;->getID()I'):
        if canonical(sig)!=graph.canonical_signature(sig) or stable_id(sig)!=endpoint._stable_id(graph.canonical_signature(sig)):
            raise RuntimeError('BLOCKED: frozen signature/hash contract mismatch')
    mapping=yaml.safe_load(assets['mapping'].read_text(encoding='utf-8'))
    if len(mapping['roles'])!=31:raise RuntimeError('Role vocabulary is not frozen Role31')
    from dex_adapter import matching_roles
    for ref in ('Ljava/net/Socket;->connect(Ljava/net/SocketAddress;)V','Landroid/content/ContentResolver;->insert()V'):
        cls,tail=canonical(ref).split('->');name=tail.split('(')[0]
        expected=sorted({r['role'] for r in mapping['rules'] if r.get('rule_kind','api')=='api' and graph.api_rule_matches(r,cls,name)})
        if matching_roles(canonical(ref),mapping)!=expected:raise RuntimeError('Role API matching differs from frozen builder')
    write_csv(OUTPUT/'selection_label_free.csv',selected)
    inventory=dict(python=sys.executable,python_version=sys.version,androguard_version=getattr(androguard,'__version__','unknown'),
      project=str(PROJECT),output=str(OUTPUT),assets={k:str(v) for k,v in assets.items()},apk_path_sources=audit,
      source_rows=799,selected=200,labels_used=False,future_rows_loaded=0,gpu_required=False,network_required=False,
      graph_contract_root=str(base),protocol_sha256=protocol_sha,scan_identity_sha256=token,parent_output=str(BASELINE_OUTPUT),
      parent_gate_thresholds_unchanged=True,whole_apk_event_alignment_is_not_frozen_edge_recovery=True)
    dump(OUTPUT/'environment_inventory.json',inventory)
    protected=[(p,k) for k,p in assets.items()]+[(base/'graph_representation.py','formal signature/Role matching'),(base/'endpoint_centered_view.py','formal endpoint hash contract'),(hostrel_path,'formal C43 Host reconstruction')]
    protected += [(apks[sid],'original source APK '+sid) for sid in sorted(wanted)]
    protected += [(p,'read-only v1 result '+p.name) for p in parent_files]
    parent_token=json.loads((BASELINE_OUTPUT/'protocol_identity.json').read_text(encoding='utf-8'))['scan_identity_sha256']
    for sid in sorted(wanted):
        p=BASELINE_OUTPUT/'cache/scans'/parent_token[:16]/f'{sid}.json.gz'
        if not p.is_file():raise RuntimeError('BLOCKED: v1 matched scan cache missing '+str(p))
        protected.append((p,'read-only v1 event cache '+sid))
    manifest_path=OUTPUT/'baseline_asset_manifest_before.json'
    fresh=[]
    for i,(p,purpose) in enumerate(protected,1):
        rec=file_record(p,purpose);fresh.append(rec)
        if purpose.startswith('original source APK ') and rec['sha256']!=purpose.split()[-1]:raise RuntimeError('APK SHA mismatch '+str(p))
        if i%25==0:print(f'LCC_ASSET_BEFORE {i}/{len(protected)}',flush=True)
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old!=fresh:raise RuntimeError('BLOCKED: protected assets changed since first run')
    else:dump(manifest_path,fresh)
    for r in selected:
        import numpy as np
        sid=r['sha256'];view=frozen_view(endpoints[sid]);actual={str(int(r['host_id'])) for r in hostrel.reconstruct_host_relations(endpoints[sid]['payload'],np.arange(31))}
        if set(view['host_ids'])!=actual:raise RuntimeError('BLOCKED: frozen C43 Host set differs for '+sid)
        dump(OUTPUT/'cache/frozen'/f'{sid}.json',view)
    dump(OUTPUT/'apk_paths.json',{k:str(v) for k,v in apks.items()})
    dump(OUTPUT/'preflight.json',dict(status='PASS',selected=200,zero_host_apks=sum(not frozen_view(endpoints[s])['pairs'] for s in wanted),
       labels_used=False,future_rows_loaded=0,protocol_sha256=protocol_sha,scan_identity_sha256=token,
       parent_gate_thresholds_unchanged=True,parent_routed_selection_equal=True))
    print('LCC_PREFLIGHT_PASS selected=200 labels_used=false future_rows_loaded=0',flush=True)
if __name__=='__main__':main()
