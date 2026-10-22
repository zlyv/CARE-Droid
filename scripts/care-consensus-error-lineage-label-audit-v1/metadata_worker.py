#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path

def safe(callable_object,default=None):
    try:return callable_object()
    except Exception:return default
def file_sha(path):
    d=hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda:h.read(1<<20),b""):d.update(block)
    return d.hexdigest()
def certificates(apk):
    result=set()
    for name in ("get_certificates_der_v3","get_certificates_der_v2","get_certificates_der_v1"):
        getter=getattr(apk,name,None)
        if callable(getter):
            for value in safe(getter,[]) or []:
                try:result.add(hashlib.sha256(bytes(value)).hexdigest())
                except Exception:pass
    return sorted(result)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--apk",required=True);parser.add_argument("--sha256",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();path=Path(args.apk).resolve();output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    actual=file_sha(path)
    if actual!=args.sha256:raise RuntimeError(f"APK SHA mismatch expected={args.sha256} actual={actual}")
    try:
        from loguru import logger;logger.remove()
    except Exception:pass
    from androguard.core.apk import APK
    apk=APK(str(path));package=str(safe(apk.get_package,"") or "").strip().casefold()
    dex_hashes=[];dex_sizes=[];names=[]
    with zipfile.ZipFile(path,"r") as archive:
        names=sorted(info.filename.casefold() for info in archive.infolist())
        for info in archive.infolist():
            leaf=Path(info.filename).name.casefold()
            if leaf.startswith("classes") and leaf.endswith(".dex"):
                blob=archive.read(info);dex_hashes.append(hashlib.sha256(blob).hexdigest());dex_sizes.append(len(blob))
    status=path.stat()
    record={"schema_version":"1.0.0","status":"OK","sha256":args.sha256,"apk_size":status.st_size,"mtime_ns":status.st_mtime_ns,
        "package_sha256":hashlib.sha256(package.encode()).hexdigest() if package else None,
        "certificate_sha256":certificates(apk),"dex_sha256":sorted(dex_hashes),"dex_sizes":sorted(dex_sizes),
        "dex_count":len(dex_hashes),"zip_entry_count":len(names),
        "zip_layout_sha256":hashlib.sha256("\n".join(names).encode()).hexdigest()}
    temp=output.with_suffix(output.suffix+".tmp");temp.write_text(json.dumps(record,sort_keys=True),encoding="utf-8");temp.replace(output)
if __name__=="__main__":main()
