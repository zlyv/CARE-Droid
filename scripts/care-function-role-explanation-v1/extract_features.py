#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,sys,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from common import APK_PATH_CANDIDATES,OUTPUT,dump_json,read_json,write_csv
from taxonomy import FAMILIES,METHOD_REF,classify

NS="{http://schemas.android.com/apk/res/android}"
def attr(e,n):return str(e.get(NS+n,e.get(n,"")) or "").strip()
def framework(v):
    v=str(v).strip().casefold();return v if v.startswith(("android.","com.android.")) else ""
def manifest_tokens(root):
    out=set()
    for e in root.iter():
        k=str(e.tag).split("}")[-1]
        if k in {"activity","activity-alias","service","receiver","provider"}:out.add("component:"+("activity" if k=="activity-alias" else k))
        elif k in {"action","category"}:
            v=framework(attr(e,"name"));
            if v:out.add(k+":"+v)
        elif k=="uses-feature":
            v=framework(attr(e,"name"));
            if v:out.add("feature:"+v)
        elif k=="data":
            s=attr(e,"scheme").casefold();m=attr(e,"mimeType").casefold()
            if s:out.add("scheme:"+(s if s in {"http","https","tel","sms","geo","content","file","mailto"} else "custom"))
            if m:out.add("mime:"+m.split("/",1)[0])
    return sorted(out)
def instructions(method):
    g=getattr(method,"get_instructions_idx",None)
    if callable(g):return g()
    code=getattr(method,"get_code",lambda:None)();bc=getattr(code,"get_bc",lambda:None)() if code else None
    return enumerate(getattr(bc,"get_instructions",lambda:[])()) if bc else []
def one(item):
    sid,path,target=item;started=time.monotonic()
    try:
        try:
            from loguru import logger;logger.remove()
        except Exception:pass
        from androguard.core.apk import APK
        from androguard.core.dex import DEX
        apk=APK(path);tokens=manifest_tokens(apk.get_android_manifest_xml());counts=Counter();invokes=0
        for raw in apk.get_all_dex():
            dex=DEX(raw)
            for cls in dex.get_classes():
                for method in cls.get_methods():
                    for _,ins in instructions(method):
                        name=str(getattr(ins,"get_name",lambda:"")()).casefold()
                        if not name.startswith("invoke-"):continue
                        invokes+=1;out=str(getattr(ins,"get_output",lambda:"")());ms=METHOD_REF.findall(out)
                        if ms:
                            for family in classify(ms[-1]):counts[family]+=1
        row={"sha256":sid,"status":"OK","manifest_tokens":tokens,"family_counts":dict(counts),"invoke_count":invokes,"active_family_count":len(counts),"wall_seconds":time.monotonic()-started}
    except Exception as exc:row={"sha256":sid,"status":"ERROR","error_type":type(exc).__name__,"error":str(exc)}
    Path(target).write_text(json.dumps(row,separators=(",",":"),ensure_ascii=False)+"\n",encoding="utf-8");return row
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--jobs",type=int,default=4);args=ap.parse_args()
    path_file=next((p for p in APK_PATH_CANDIDATES if p.is_file()),None)
    if path_file is None:raise RuntimeError("BLOCKED_BY_CONTRACT: no frozen 2017 APK path map")
    paths=read_json(path_file);shards=OUTPUT/"cache/function_features";shards.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for sid,path in sorted(paths.items()):
        t=shards/(sid+".json")
        if t.is_file():
            try:
                if json.loads(t.read_text(encoding="utf-8")).get("status")=="OK":continue
            except Exception:pass
        tasks.append((sid,path,str(t)))
    done=0;failed=0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for f in as_completed([pool.submit(one,x) for x in tasks]):
            r=f.result();done+=1;failed+=r.get("status")!="OK"
            if done<=5 or done%25==0 or done==len(tasks):print(f"FCBE_FEATURES {done}/{len(tasks)} failed={failed}",flush=True)
    rows=[json.loads((shards/(sid+".json")).read_text(encoding="utf-8")) for sid in sorted(paths)]
    failures=[r for r in rows if r.get("status")!="OK"]
    if failures:raise RuntimeError(f"BLOCKED_BY_CONTRACT: function extraction failures={len(failures)}")
    write_csv(OUTPUT/"function_feature_inventory.csv",[{"sha256":r["sha256"],"manifest_token_count":len(r["manifest_tokens"]),"active_family_count":r["active_family_count"],"invoke_count":r["invoke_count"],**{k:r["family_counts"].get(k,0) for k in FAMILIES}} for r in rows])
    dump_json(OUTPUT/"feature_extraction_summary.json",{"status":"PASS","rows":len(rows),"failed":0,"path_source":str(path_file),"future_rows_loaded":0})
    print("FCBE_FEATURE_EXTRACTION_DONE rows="+str(len(rows)),flush=True)
if __name__=="__main__":main()
