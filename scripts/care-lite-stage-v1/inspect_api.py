#!/usr/bin/env python3
"""LiteStage API smoke: reads one existing APK and prints available androguard interfaces; writes nothing, trains nothing."""
import csv
from pathlib import Path
from loguru import logger
logger.remove()
from androguard.core.apk import APK
from androguard.core.dex import DEX
LINEAGE=Path('/mnt/HDD1/boyuanjie/tasa-droid-storage/care-source-local-assets-v2/source_lineage_2017.csv')
with LINEAGE.open() as f:path=Path(next(csv.DictReader(f))['absolute_path'])
print('APK_PRESENT',path.is_file(),flush=True)
apk=APK(str(path))
print('APK_METHODS',[x for x in dir(apk) if x.startswith('get_') and ('permission' in x or 'activit' in x or 'service' in x or 'receiver' in x or 'provider' in x or 'sdk' in x or 'dex' in x)],flush=True)
print('APK_INFO',len(apk.get_permissions()),len(apk.get_activities()),len(apk.get_services()),len(apk.get_receivers()),len(apk.get_providers()),flush=True)
blob=next(iter(apk.get_all_dex()))
vm=DEX(blob)
print('DEX_METHODS',[x for x in dir(vm) if 'method' in x and ('get_' in x or 'methods' in x)],flush=True)
print('COUNTS',len(vm.get_classes()),len(vm.get_methods()),flush=True)
m=vm.get_methods()[0]
print('METHOD_TYPE',type(m),[x for x in dir(m) if x.startswith('get_')][:35],flush=True)
