from __future__ import annotations
import ast, csv, importlib, inspect, sys
from pathlib import Path
import numpy as np
from pilot_common import *

def load_runtime(arm):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(FAMILY))
    adapter=importlib.import_module('model_adapter'); adapter.prepare_imports()
    sys.path.insert(0,str(PROJECT/'scripts/care-p31-stage1-rf-selection-source-v1'))
    contract=importlib.import_module('source_label_contract')
    tree=ast.parse(inspect.getsource(contract.install_source_label_contract))
    body=tree.body[0].body
    statements=[n for n in body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    expected=ast.parse('v4_module.audit.future.label_of = source_label_of').body[0]
    if len(statements)!=1 or ast.dump(statements[0])!=ast.dump(expected): raise RuntimeError('source label wrapper changed; review required')
    v4=importlib.import_module('run_care_rolegraph_conditioned_dualstream_v4')
    contract.install_source_label_contract(v4); adapter.install_a0(v4,METHOD)
    sys.path.insert(0,str(DIRECT_SCRIPT))
    variant=importlib.import_module('model_variants'); variant.install_variant(v4,'U2_DIRECT_STAGE2')
    v4.METHOD_FOR_VARIANT={'RGCondAttn': METHOD+'__'+arm}
    # Reuse only the cache branch. Never fall through to an extractor/cache rebuild.
    v4.get_source_host_bags=read_only_bags
    v4.audit.load_inputs=wrap_source_loader(v4.audit.load_inputs,set(reference()[0]))
    return v4,adapter,contract

def wrap_source_loader(original_loader,wanted):
    def source_only_loader(args):
        # Some old CSVs contain other years. Supply only fixed-source SHA rows.
        namespace=original_loader.__globals__; reader=namespace.get('read_csv')
        if reader is None: raise RuntimeError('original source loader reader changed; review required')
        target=Path(args.features).resolve()
        def source_reader(path):
            if Path(path).resolve()!=target: return reader(path)
            rows=[]
            with Path(path).open(encoding='utf-8-sig',newline='') as f:
                for r in csv.DictReader(f):
                    if norm(r.get('sha256','')) in wanted: rows.append(r)
            return rows
        namespace['read_csv']=source_reader
        try: return original_loader(args)
        finally: namespace['read_csv']=reader
    return source_only_loader

def read_only_bags(rows,endpoint_cache,args):
    if args.rebuild_region_cache: raise RuntimeError('cache rebuild prohibited')
    expected=np.asarray([norm(r['sha256']) for r in rows])
    with np.load(args.region_cache,allow_pickle=False) as a:
        ids=np.asarray([norm(v) for v in a['sha256']]); values=np.asarray(a['host_values'],np.float32); offsets=np.asarray(a['host_offsets'],np.int64)
        if not np.array_equal(ids[:len(expected)],expected) or values.ndim!=2 or values.shape[1]!=169 or offsets.shape!=(len(ids)+1,):
            raise RuntimeError('frozen A0 cache prefix/shape mismatch; no rebuild permitted')
        if offsets[0]!=0 or offsets[-1]!=len(values) or np.any(np.diff(offsets)<0) or not np.isfinite(values).all(): raise RuntimeError('invalid cached Host offsets/values')
        bags=[values[offsets[i]:offsets[i+1]] for i in range(len(expected))]
    audit=[dict(sha256=expected[i],metadata_year=str(rows[i].get('metadata_year','')).strip(),host_region_n=len(bags[i]),
        real_host_region_n='cached_prefix',ego_region_n='not_loaded',host_was_empty='cached_prefix',hostego_total_region_n=len(bags[i])) for i in range(len(rows))]
    return bags,audit,True

def arguments(arm,device='cuda'):
    from common import ENDPOINT_SOURCE,SOURCE_A0_CACHE
    return ['--previous-result',str(FAMILY_OUTPUT/'references/SVM'),'--cache',str(ENDPOINT_SOURCE),
        '--region-cache',str(SOURCE_A0_CACHE),'--variants','RGCondAttn','--seeds','0,1,2','--max-delta','2.25',
        '--device',device,'--output',str(OUTPUT/'source'/arm),'--keep-weight','1','--fix-weight','1','--keep-consistency','0']

def parsed_args(v4,argv):
    # Execute only the original argument definitions, never the original main body.
    node=ast.parse(inspect.getsource(v4.main)).body[0]; prefix=[]
    for s in node.body:
        if isinstance(s,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in s.targets): break
        prefix.append(s)
    scope=dict(v4.__dict__); exec(compile(ast.Module(body=prefix,type_ignores=[]),'<original argument definitions>','exec'),scope)
    return scope['parser'].parse_args(argv)

def validate_loaded_inputs(result):
    manifest,rows=result[0],result[1]; ids=np.asarray([norm(r['sha256']) for r in manifest]); row_ids=np.asarray([norm(r['sha256']) for r in rows])
    ids,y,folds,base,route,expert=reference(ids)
    if not np.array_equal(ids,row_ids) or any(norm(r.get('metadata_year','2017'))!='2017' for r in rows): raise RuntimeError('source order/year contract')
    if not np.array_equal(np.asarray([r['label'] for r in manifest],int),y) or not np.array_equal(np.asarray([r['fold'] for r in manifest],int),folds): raise RuntimeError('source manifest label/fold changed')
    return ids,y,folds,base,route,expert
