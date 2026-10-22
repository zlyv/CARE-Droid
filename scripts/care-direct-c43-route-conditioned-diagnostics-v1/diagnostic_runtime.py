from __future__ import annotations
import importlib,sys
from diag_common import *

def direct_installer(adapter,variant):
    install=adapter.install_a0
    def wrapped(v,method):install(v,method);variant.install_variant(v,'U2_DIRECT_STAGE2')
    return wrapped

def direct_parser(builder):
    def wrapped():
        p=builder();p.set_defaults(keep_weight=1.,fix_weight=1.,keep_consistency=0.);return p
    return wrapped

def create_runtime(mode):
    core,rt,w,ta=parent_modules();rt.family_module('common');rt.family_module('stage1_core')
    adapter=rt.family_module('model_adapter');adapter.prepare_imports()
    sys.path.insert(0,str(PROJECT/'scripts/care-p31-stage1-rf-selection-source-v1'))
    contract=importlib.import_module('source_label_contract')
    v4=importlib.import_module('run_care_rolegraph_conditioned_dualstream_v4')
    contract.install_source_label_contract(v4);adapter.install_fold_safe_compact76(v4.audit)
    variant=module('model_variants',PROJECT/'scripts/care-svm-c43-remove-bounded-correction-v1/model_variants.py')
    if mode=='frozen':
        fc=module('future_common',OLD_FUTURE/'future_common.py');fc.OUTPUT=OUTPUT/'runner-frozen'
        runner=module('_rctd_frozen_runner',OLD_FUTURE/'run_future.py')
    elif mode=='within':
        rt.family_module('run_frozen_arm')
        runner=rt.family_module('run_within_arm',alias='_rctd_within_runner');runner.OUTPUT=OUTPUT/'runner-within'
        runner.install_a0=direct_installer(adapter,variant)
        runner.build_parser=direct_parser(runner.build_parser)
    else:raise ValueError(mode)
    return core,rt,w,ta,v4,adapter,runner
