#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math,re,zipfile
from pathlib import Path

METHOD_REFERENCE=re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+",re.I)
DYNAMIC_MARKERS=("ldalvik/system/dexclassloader;->","ldalvik/system/pathclassloader;->","ldalvik/system/inmemorydexclassloader;->","ldalvik/system/basedexclassloader;->","ldalvik/system/dexfile;->load")
REFLECTION_MARKERS=("ljava/lang/reflect/","ljava/lang/class;->forname","ljava/lang/class;->getdeclared","ljava/lang/class;->getmethod","ljava/lang/class;->getconstructor")
NATIVE_LOAD_MARKERS=("ljava/lang/system;->loadlibrary","ljava/lang/system;->load(","ljava/lang/runtime;->loadlibrary","ljava/lang/runtime;->load(")
MEDIA_EXT=(".png",".jpg",".jpeg",".gif",".webp",".mp3",".mp4",".ogg",".wav",".arsc",".ttf",".otf")
PAYLOAD_EXT=(".apk",".jar",".dex",".odex",".vdex",".zip",".bin",".dat",".payload")

def norm(value):return str(value or "").strip().casefold()
def safe(callable_object,default=None):
    try:return callable_object()
    except Exception:return default
def sha256_file(path):
    d=hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda:h.read(1<<20),b""):d.update(block)
    return d.hexdigest()
def entropy(blob):
    if not blob:return 0.0
    counts=[0]*256
    for value in blob:counts[value]+=1
    n=len(blob);return -sum((c/n)*math.log2(c/n) for c in counts if c)
def component_name(package,name):
    value=norm(name);package=norm(package)
    if value.startswith("."):value=package+value
    elif "." not in value and package:value=package+"."+value
    return "l"+value.replace(".","/")+";" if value else ""
def instructions(method):
    getter=getattr(method,"get_instructions_idx",None);iterator=getter() if callable(getter) else enumerate(safe(getattr(method,"get_instructions",lambda:()),()) or ())
    for _,ins in iterator:yield norm(safe(getattr(ins,"get_name",lambda:"") ,"")),norm(safe(getattr(ins,"get_output",lambda:"") ,""))
def main():
    p=argparse.ArgumentParser();p.add_argument("--apk",required=True);p.add_argument("--sha256",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    path=Path(a.apk).resolve();out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);actual=sha256_file(path)
    if actual!=a.sha256:raise RuntimeError(f"APK SHA mismatch expected={a.sha256} actual={actual}")
    try:
        from loguru import logger;logger.remove()
    except Exception:pass
    from androguard.core.apk import APK
    from androguard.core.dex import DEX
    apk=APK(str(path));package=norm(safe(apk.get_package,""));zip_entries=0;native_count=0;native_bytes=0;embedded=0;encrypted=0;high_entropy=0;high_entropy_max=0.0
    dex_entry_count=0;abi=set();payload_names=[]
    with zipfile.ZipFile(path,"r") as archive:
        for info in archive.infolist():
            if info.is_dir():continue
            zip_entries+=1;name=info.filename.casefold();leaf=Path(name).name
            is_primary=bool(re.fullmatch(r"classes(?:\d+)?\.dex",leaf)) and "/" not in name
            if is_primary:dex_entry_count+=1
            if name.startswith("lib/") and name.endswith(".so"):
                native_count+=1;native_bytes+=int(info.file_size)
                parts=name.split("/");
                if len(parts)>=3:abi.add(parts[1])
            suspicious_ext=name.endswith(PAYLOAD_EXT) and not is_primary
            encrypted+=int(bool(info.flag_bits&1))
            sample=b""
            if suspicious_ext or (info.file_size>=65536 and (name.startswith("assets/") or name.startswith("res/raw/")) and not name.endswith(MEDIA_EXT)):
                try:
                    with archive.open(info,"r") as h:sample=h.read(131072)
                except Exception:sample=b""
                magic=sample[:4]
                magic_code=magic.startswith(b"dex\n") or magic.startswith(b"PK\x03\x04") or magic==b"\x7fELF"
                if suspicious_ext or magic_code:
                    embedded+=1;payload_names.append(hashlib.sha256(name.encode()).hexdigest()[:16])
                value=entropy(sample);high_entropy_max=max(high_entropy_max,value)
                if info.file_size>=65536 and value>=7.5:high_entropy+=1
    class_names=set();method_count=0;native_methods=0;instruction_count=0;invoke_count=0;dynamic=0;reflection=0;native_load=0;indirect=0;parse_errors=0
    try:
        for raw in apk.get_all_dex():
            try:dex=DEX(raw)
            except Exception:parse_errors+=1;continue
            for cls in dex.get_classes():
                class_names.add(norm(cls.get_name()))
                for method in cls.get_methods():
                    method_count+=1;flags=norm(safe(getattr(method,"get_access_flags_string",lambda:"") ,""));native_methods+=int("native" in flags.split())
                    if safe(getattr(method,"get_code",lambda:None)) is None:continue
                    try:
                        for opcode,output in instructions(method):
                            instruction_count+=1
                            if opcode.startswith("invoke-"):
                                invoke_count+=1;indirect+=int(opcode in {"invoke-custom","invoke-custom/range","invoke-polymorphic","invoke-polymorphic/range"})
                                refs=METHOD_REFERENCE.findall(output);reference=norm(refs[-1]) if refs else output
                                dynamic+=int(any(marker in reference for marker in DYNAMIC_MARKERS))
                                reflection+=int(any(marker in reference for marker in REFLECTION_MARKERS))
                                native_load+=int(any(marker in reference for marker in NATIVE_LOAD_MARKERS))
                    except Exception:parse_errors+=1
            dex=None
    except Exception:parse_errors+=1
    components=[]
    for getter in ("get_activities","get_services","get_receivers","get_providers"):
        components.extend(component_name(package,value) for value in (safe(getattr(apk,getter,lambda:[]),[]) or []))
    components=[value for value in components if value];missing=sum(value not in class_names for value in components);component_n=len(components);status=path.stat()
    record={"schema_version":"1.0.0","status":"OK","sha256":a.sha256,"apk_size":status.st_size,"mtime_ns":status.st_mtime_ns,
      "zip_entry_count":zip_entries,"dex_count":dex_entry_count,"native_binary_count":native_count,"native_binary_bytes":native_bytes,"abi_count":len(abi),
      "embedded_code_payload_count":embedded,"encrypted_zip_entry_count":encrypted,"high_entropy_payload_count":high_entropy,"high_entropy_max":high_entropy_max,
      "class_count":len(class_names),"method_count":method_count,"native_method_count":native_methods,"instruction_count":instruction_count,"invoke_count":invoke_count,
      "dynamic_loader_invoke_count":dynamic,"reflection_invoke_count":reflection,"native_load_invoke_count":native_load,"indirect_invoke_count":indirect,"dex_parse_error_count":parse_errors,
      "manifest_component_count":component_n,"manifest_component_missing_count":missing,"manifest_component_missing_rate":missing/max(1,component_n),
      "has_native_surface":int(native_count>0 or native_methods>0),"has_multidex":int(dex_entry_count>1),"has_embedded_code_payload":int(embedded>0),
      "has_dynamic_loader":int(dynamic>0),"has_reflection":int(reflection>0),"has_native_loader":int(native_load>0),"has_indirect_dispatch":int(indirect>0),
      "has_encrypted_zip_entry":int(encrypted>0),"has_high_entropy_payload":int(high_entropy>0),
      "has_manifest_code_gap":int(component_n>=2 and missing/component_n>=.25),"payload_name_hashes":payload_names[:20]}
    record["blindspot_family_count"]=sum(record[key] for key in ("has_native_surface","has_multidex","has_embedded_code_payload","has_dynamic_loader","has_reflection","has_indirect_dispatch","has_encrypted_zip_entry","has_high_entropy_payload","has_manifest_code_gap"))
    temp=out.with_suffix(out.suffix+".tmp");temp.write_text(json.dumps(record,sort_keys=True),encoding="utf-8");temp.replace(out)
if __name__=="__main__":main()
