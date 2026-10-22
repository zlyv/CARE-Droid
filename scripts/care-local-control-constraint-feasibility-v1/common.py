#!/usr/bin/env python3
from __future__ import annotations
import csv, gzip, hashlib, json, os, re
from pathlib import Path

PROJECT=Path(os.environ.get('LCC_PROJECT','/home/boyuanjie/projects/tasa-droid')).resolve()
OUTPUT=Path(os.environ.get('LCC_OUTPUT',str(PROJECT/'outputs/care-local-control-constraint-feasibility-v1'))).resolve()
SCHEMA='lcc-fa-v1.0'
def canonical(x):return re.sub(r'\s+','',str(x or '')).casefold()
def stable_id(x):return int.from_bytes(hashlib.sha256(canonical(x).encode()).digest()[:8],'big')
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def safe(path):
    path=Path(path).resolve();root=(PROJECT/'outputs').resolve()
    if root not in OUTPUT.parents or not OUTPUT.name.startswith('care-local-control-constraint-feasibility-v'):
        raise RuntimeError('unsafe output root '+str(OUTPUT))
    if path!=OUTPUT and OUTPUT not in path.parents:raise RuntimeError('write outside LCC output '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);return path
def dump(path,obj):
    path=safe(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def dump_gz(path,obj):
    path=safe(path);temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,separators=(',',':'))
    temp.replace(path)
def load_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def write_csv(path,rows,fields=None):
    rows=list(rows);fields=fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status'];path=safe(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    temp.replace(path)
def file_record(path,purpose):
    path=Path(path).resolve();s=path.stat()
    return dict(absolute_path=str(path),size_bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha256(path),purpose=purpose)
def first(paths,name):
    for p in paths:
        p=Path(p)
        if p.is_file():return p.resolve()
    raise RuntimeError('BLOCKED: missing '+name+'; checked='+str([str(p) for p in paths]))
def discover():
    return {
      'route':first([PROJECT/'outputs/care-p31-classifier-family-stage2-v1/source_stage2_predictions.csv'],'frozen SVM route'),
      'endpoint':first([os.environ.get('LCC_ENDPOINT',''),'/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz',PROJECT/'artifacts/care-source-local-assets-v2/endpoint_centered_tensor_cache_2017.jsonl.gz'],'endpoint cache'),
      'mapping':first([PROJECT/'scripts/care-source-recovery-stage2only-v1/extractor_repo/configs/protocol/semantic_mapping_v1.yaml',PROJECT/'configs/protocol/semantic_mapping_v1.yaml'],'frozen Role mapping')}
def route_rows(path):
    rows=[]
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if canonical(row.get('arm'))=='svm':
                # Do not access label, probabilities, correctness or Direct-C43 output fields.
                rows.append(dict(sha256=str(row['sha256']).strip().lower(),fold=int(float(row['fold'])),routed=int(float(row['routed']))))
    if len(rows)!=799 or len({r['sha256'] for r in rows})!=799:raise RuntimeError('source membership mismatch')
    selected=[r for r in rows if r['routed']]
    if len(selected)!=200 or any(sum(r['fold']==f for r in selected)!=40 for f in range(1,6)):raise RuntimeError('frozen routed membership mismatch')
    return selected
def endpoint_rows(path,wanted):
    result={}
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);sid=str(r.get('sha256','')).lower()
            if sid in wanted and r.get('status')=='OK':
                if sid in result:raise RuntimeError('duplicate endpoint cache SHA '+sid)
                result[sid]={'payload':r['payload'],'summary':r.get('summary',{})}
    if set(result)!=wanted:raise RuntimeError('missing frozen endpoint rows')
    return result
def frozen_view(record):
    p=record['payload'];ids=p['node_ids'];roles=p['node_role_ids'];base=p['node_base_features'];edges=p['edge_index']
    if not(len(ids)==len(roles)==len(base)) or len(edges)!=2 or len(edges[0])!=len(edges[1]):raise RuntimeError('endpoint schema mismatch')
    if len(set(map(int,ids)))!=len(ids):raise RuntimeError('duplicate frozen node ID')
    nodes={str(int(n)):dict(roles=list(map(int,roles[i])),endpoint=bool(base[i][4]>.5),external=bool(base[i][0]>.5)) for i,n in enumerate(ids)}
    pairs={}
    for a,b in zip(*edges):
        a,b=int(a),int(b)
        if not(0<=a<len(ids) and 0<=b<len(ids)):raise RuntimeError('endpoint index out of bounds')
        if base[b][4]>.5 and roles[b]:
            pairs[str(int(ids[a]))+':'+str(int(ids[b]))]=list(map(int,roles[b]))
    return dict(nodes=nodes,pairs=pairs,summary=record['summary'],host_ids=sorted(set(k.split(':')[0] for k in pairs)))
def apk_paths(wanted):
    indexes=[]
    if os.environ.get('LCC_APK_INDEX'):indexes.append(Path(os.environ['LCC_APK_INDEX']))
    indexes += [PROJECT/'outputs/care-ccbc-source-v3/apk_paths.json',PROJECT/'outputs/care-icde-source-v1/apk_asset_audit.json']
    found={};audit=[]
    for p in indexes:
        if not p.is_file():continue
        d=json.loads(p.read_text(encoding='utf-8'));pairs=[]
        if isinstance(d,dict):
            pairs += [(k,v) for k,v in d.items() if isinstance(v,str) and len(k)==64]
            for name in ('records','rows','samples','data'):
                if isinstance(d.get(name),list):
                    pairs += [(r.get('sha256'),r.get('absolute_path') or r.get('apk_path') or r.get('path')) for r in d[name]]
        elif isinstance(d,list):pairs += [(r.get('sha256'),r.get('absolute_path') or r.get('apk_path') or r.get('path')) for r in d]
        n=0
        for sid,raw in pairs:
            sid=str(sid or '').lower();a=Path(str(raw or ''))
            if not a.is_absolute():a=PROJECT/a
            if sid in wanted and sid not in found and a.is_file():found[sid]=a.resolve();n+=1
        audit.append(dict(path=str(p.resolve()),matched=n))
    for raw in os.environ.get('LCC_APK_ROOTS','').split(os.pathsep):
        if not raw:continue
        root=Path(raw)
        if root.is_dir():
            for p in root.rglob('*.apk'):
                if p.stem.lower() in wanted:found.setdefault(p.stem.lower(),p.resolve())
    if set(found)!=wanted:raise RuntimeError('BLOCKED: routed APK paths '+str(len(found))+'/200 missing='+repr(sorted(wanted-set(found))[:10]))
    return found,audit
