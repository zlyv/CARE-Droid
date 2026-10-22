#!/usr/bin/env python3
"""Verify protected assets after the run and set the terminal run status."""
import json
from semantic_common import *
def main():
 before=json.loads((OUT/'audit/asset_integrity_before.json').read_text()); changed=[]
 for old in before['assets']:
  try: now=digest(old['path'])
  except Exception as e: changed.append({'path':old['path'],'error':repr(e)}); continue
  if now!=old: changed.append({'before':old,'after':now})
 status='PASS' if not changed else 'FAIL'; atomic_json(OUT/'audit/asset_integrity_after.json',{'status':status,'checked':len(before['assets']),'changed':changed})
 if changed: raise RuntimeError('protected asset changed')
 atomic_json(OUT/'logs/run_status.json',{'status':'COMPLETED','stage':'done','future_years':[2018,2019,2020,2021]}); print('EXPERIMENT_COMPLETED',flush=True)
if __name__=='__main__': main()
