#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from common import OUT, PROJECT, SCRIPTS, save_json

HERE = Path(__file__).resolve().parent
COMPACT = SCRIPTS / 'care-di25-compact43-relationquery-source-v1'
GRAPHGLOBAL = SCRIPTS / 'care-di25-graphglobal-path-source-v1'
V4DIR = SCRIPTS / 'care-rolegraph-conditioned-v4-script'
for path in (HERE, COMPACT, GRAPHGLOBAL, SCRIPTS, V4DIR,
             SCRIPTS / 'care-rgcondattn-dedup-bounded-source-v1',
             SCRIPTS / 'care-malben-host-v2'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_background import make_fit
from compact_models import make_compact_class
from source_label_contract import install_source_label_contract
import run_care_rolegraph_conditioned_dualstream_v4 as v4

METHOD = 'P31Mean__DI25__C43RelationQuery__Bounded'


def load_path_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    resolved_path = OUT / 'preflight/resolved_inputs.json'
    if not resolved_path.is_file():
        raise RuntimeError('BLOCKED_BY_CONTRACT: run preflight.py first')
    resolved = json.loads(resolved_path.read_text(encoding='utf-8'))
    runner_path = Path(resolved['future_runner'])
    if str(runner_path.parent) not in sys.path:
        sys.path.insert(0, str(runner_path.parent))

    # Import after v4 so the future module shares the already-patched v4 module.
    future = load_path_module('care_p31_c43_generic_future', runner_path)
    install_source_label_contract(v4)
    v4.MODEL_VARIANTS = ('RGCondAttn',)
    v4.METHOD_FOR_VARIANT = {'RGCondAttn': METHOD}
    v4.AUGMENTED_REGION_DIM = 78
    v4.RoleGraphConditionedEvidence = make_compact_class(v4, 'C43_RELATION_QUERY')
    v4.v3.fit_scaled_dualproto_bags = make_fit(v4.v3)
    # Some generic future runners keep their own module-level v4 reference.
    if hasattr(future, 'v4'):
        future.v4 = v4

    output = OUT / 'arm/P31_C43_RQ'
    output.mkdir(parents=True, exist_ok=True)
    source_text = runner_path.read_text(encoding='utf-8', errors='replace')
    argv = [
        sys.argv[0],
        '--features', resolved['eval_features'],
        '--previous-result', resolved['previous_result'],
        '--future-inter-cache', resolved['future_inter_cache'],
        '--region-cache', resolved['c43_future_region_cache'],
        '--years', '2018,2019,2020,2021',
        '--variants', 'RGCondAttn',
        '--max-delta', '2.25',
        '--output', str(output),
    ]
    if '--device' in source_text:
        argv += ['--device', args.device]
    if '--seeds' in source_text:
        argv += ['--seeds', args.seeds]
    sys.argv = argv
    future.main()

    prediction_candidates = [output / 'future_predictions_diagnostic.csv'] + sorted(output.glob('**/future_predictions_diagnostic.csv'))
    prediction = next((p for p in prediction_candidates if p.is_file()), None)
    if prediction is None:
        raise RuntimeError(f'BLOCKED_BY_CONTRACT: generic future runner completed but no future_predictions_diagnostic.csv under {output}')
    save_json(OUT / 'arm/architecture.json', {
        'experiment': 'P31 Mean Committee -> year-wise P31-DI25 -> C43-RQ future diagnostic',
        'stage1_representation': 'P31 only; FUSED63 prohibited',
        'stage1_training': 'all 799 labeled 2017 source APKs only',
        'stage1_probability': 'mean calibrated LR/RF/XGB/SVM',
        'router': 'MeanOpposition Top25% independently per future year; label-free',
        'stage2': 'C43_RELATION_QUERY, same model family as strict source experiment',
        'relation_enters_attention': True,
        'relation_enters_delta': False,
        'max_delta_logit': 2.25,
        'future_labels': 'diagnostic evaluation only; P31 predictions/routes frozen before unsealing',
        'future_model_selection_allowed': False,
        'prediction_file': str(prediction),
    })
    print('P31_C43_FUTURE_ARM_DONE')
    print(f'prediction_file={prediction}')


if __name__ == '__main__':
    main()
