"""Build an exclusive new upload archive; used locally, never by the server pipeline."""
from pathlib import Path
import ast,hashlib,json,zipfile

def main():
    root=Path(__file__).resolve().parent
    target=root.parent/(root.name+'-package-r1.zip')
    if target.exists(): raise RuntimeError('archive already exists; never overwrite')
    files=sorted(p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md','.json'))
    for p in files:
        if p.suffix=='.py': ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        if p.suffix=='.sh' and b'\r' in p.read_bytes(): raise RuntimeError('CRLF shell file: '+str(p))
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files: z.write(p,arcname=root.name+'/'+p.name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None: raise RuntimeError('archive CRC failure')
        if len(z.namelist())!=len(files) or any('\\' in n or '..' in n.split('/') for n in z.namelist()): raise RuntimeError('archive paths failed')
        for p in files:
            if z.read(root.name+'/'+p.name)!=p.read_bytes(): raise RuntimeError('archived contents differ')
    print(json.dumps(dict(path=str(target.resolve()),bytes=target.stat().st_size,files=len(files),sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
if __name__=='__main__': main()
