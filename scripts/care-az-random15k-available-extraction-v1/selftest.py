#!/usr/bin/env python3
"""Dependency and path self-test; opens no APK and writes no files."""
from common import COMPACT, EDGE, ENDPOINT, PRIOR, PRIORITY, RESOLVED

required = [COMPACT, ENDPOINT, EDGE, PRIORITY, RESOLVED,
            PRIOR / 'dataset_manifest.csv', PRIOR / 'p31_shards']
missing = [str(p) for p in required if not p.exists()]
if missing: raise RuntimeError('missing required assets: ' + repr(missing))
print('AZ_AVAILABLE_EXTRACTION_SELFTEST_PASS')

