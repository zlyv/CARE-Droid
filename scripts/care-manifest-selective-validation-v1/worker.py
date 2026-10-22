#!/usr/bin/env python3
"""Isolated static extraction timer. Reads one APK, never executes it. Writes only one fresh MSV_OUT cost result."""
import argparse, importlib.util, time, resource, sys
import yaml
from core import *

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--row',required=True);ap.add_argument('--kind',choices=['ManifestOnly','RefPresence31','P31'],required=True);ap.add_argument('--result',required=True);a=ap.parse_args()
    row=json.loads(Path(a.row).read_text());cfg=yaml.safe_load(lc.MAPPING.read_text());roles={r:i for i,r in enumerate(cfg['roles'])};rules=[r for r in cfg['rules'] if r.get('rule_kind','api')=='api']
    assert len(roles)==31
    if a.kind=='P31':
        m=load('msv_p31',ROOT/'scripts/care-litelocal-p31-diagnostics-v1/extract_future_p31.py');m.suppress();t=time.perf_counter();v,detail=m.extract(row,rules,roles);elapsed=time.perf_counter()-t
        value={'p31':v.tolist(),'audit':detail.tolist()}
    else:
        m=load('msv_lite',ROOT/'scripts/care-lite-stage-v1/extract.py');t=time.perf_counter();v=m.extract(row,rules,roles,'L1' if a.kind=='ManifestOnly' else 'L3');elapsed=time.perf_counter()-t
        value={'permissions':v['permissions'],'manifest':v['manifest'],'role_ref_count':v['role_ref_count']}
    dump(a.result,dict(status='PASS',kind=a.kind,sha256=row['sha256'],extract_s=elapsed,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,values=value))
if __name__=='__main__':main()
