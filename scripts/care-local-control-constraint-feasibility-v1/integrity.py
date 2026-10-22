#!/usr/bin/env python3
import json
from pathlib import Path
from common import *
def main():
    before=json.loads((OUTPUT/'baseline_asset_manifest_before.json').read_text());after=[];changed=[]
    for i,old in enumerate(before,1):
        p=Path(old['absolute_path'])
        if not p.is_file():changed.append(dict(path=str(p),reason='MISSING'));continue
        now=file_record(p,old['purpose']);after.append(now)
        if any(old[k]!=now[k] for k in ('sha256','size_bytes','mtime_ns')):changed.append(dict(before=old,after=now))
        if i%25==0:print(f'LCC_ASSET_AFTER {i}/{len(before)}',flush=True)
    dump(OUTPUT/'baseline_asset_manifest_after.json',after)
    dump(OUTPUT/'baseline_integrity_check.json',dict(status='PASS' if not changed else 'FAIL',asset_count=len(before),changed=changed))
    if changed:raise RuntimeError('protected asset integrity failed')
    print('LCC_INTEGRITY_PASS',flush=True)
if __name__=='__main__':main()
