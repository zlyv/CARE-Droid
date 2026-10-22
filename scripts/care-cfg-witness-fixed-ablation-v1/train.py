#!/usr/bin/env python3
"""Train only the three 27-dimensional fixed-capacity arms; reads no APKs and writes only new checkpoints."""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from cfgwa_common import (ARMS, OUT, SEEDS, SPLITS, atomic_json,
                          augment_features, preflight, verify_assets,
                          write_pickle)


def auxiliary_model(v4, args):
    """Extend only the correction readout while preserving formal Global24 paths."""
    torch = v4.torch
    Base = v4.RoleGraphConditionedEvidence

    class AuxiliaryReadout(Base):
        def __init__(self):
            super().__init__(24, 'RGCondAttn', args)
            hidden = int(args.hidden_dim)
            self.delta_head = torch.nn.Sequential(
                torch.nn.Linear(116, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(float(args.dropout)),
                torch.nn.Linear(hidden, 1),
            )

        def delta_from_selected_evidence(self, global_features, base_logit, cache, graphglobal):
            flat = global_features.reshape(-1)
            if flat.numel() != 27:
                raise RuntimeError(f'Expected Global24+Aux3 width 27, got {flat.numel()}')
            role_slot = cache.role_token + self.graphglobal_delta_switch * self.global_to_delta(graphglobal)
            fused = torch.cat([
                flat[:24], flat[24:], base_logit.reshape(1), role_slot,
                cache.critical, cache.attended, cache.scalar_summary,
            ])
            if fused.numel() != 116:
                raise RuntimeError(f'Auxiliary correction width {fused.numel()} != 116')
            raw_delta = self.delta_head(fused).reshape(())
            bounded_delta = self.max_delta * torch.tanh(raw_delta)
            return raw_delta, bounded_delta, fused

        def forward_with_trace(self, global_features, base_logit, graph, regions):
            """Preserve the frozen Direct-C43 direct-logit semantics with Aux3 appended."""
            cache, graphglobal = self.select_local_evidence(graph, regions)
            zero_base = base_logit.reshape(()) * 0.0
            raw_logit, _, fused = self.delta_from_selected_evidence(
                global_features, zero_base, cache, graphglobal)
            diagnostic = torch.cat([
                raw_logit.reshape(1),
                cache.score[torch.argmax(cache.score)].reshape(1),
                torch.mean(cache.score).reshape(1),
                cache.scalar_summary[2].reshape(1),
                torch.max(cache.attention).reshape(1),
                cache.scalar_summary[2].reshape(1),
                cache.scalar_summary[-4:],
                cache.graph_entropy.reshape(1),
            ])
            trace = {
                'attention': cache.attention,
                'critical': cache.critical,
                'attended': cache.attended,
                'correction_input': fused,
                'raw_delta': raw_logit,
                'bounded_delta': raw_logit,
            }
            return raw_logit, diagnostic, trace

    return AuxiliaryReadout()


def train_one(pack, seed, epochs, global_train, global_val, labels):
    v4, args, dev = pack['v4'], pack['args'], pack['dev']; torch = v4.torch
    neutral_train = np.full(len(global_train), .5); neutral_val = np.full(len(global_val), .5)
    weights, correct = v4.fixkeep_weights(labels, neutral_train, args)
    if not np.array_equal(weights, np.ones(len(labels), np.float32)) or args.keep_consistency != 0:
        raise RuntimeError('equal BCE contract changed')
    v4.seed_everything(seed)
    model = auxiliary_model(v4, args).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    global_tensor = torch.as_tensor(global_train, dtype=torch.float32, device=dev)
    graph = v4.graph_tensors(pack['graph_train'], dev)
    bags = [torch.as_tensor(x, dtype=torch.float32, device=dev) for x in pack['bags_train']]
    if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    loss = v4.train_epochs(model, optimizer, global_tensor, neutral_train, graph, bags,
                           labels, weights, correct, epochs, args, seed+3301)
    training_seconds = time.perf_counter()-started; started = time.perf_counter()
    probability, _ = v4.predict_model(model, global_val, neutral_val,
                                      pack['graph_val'], pack['bags_val'], dev)
    inference_seconds = time.perf_counter()-started
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    cost = {'training_seconds':training_seconds,'inference_seconds':inference_seconds,
            'parameters':sum(v.numel() for v in model.parameters()),
            'gpu_peak_mib':float(torch.cuda.max_memory_allocated(dev))/1024**2 if dev.type=='cuda' else 0.,
            'final_BCE':float(loss)}
    del model, optimizer, global_tensor, graph, bags
    if dev.type == 'cuda': torch.cuda.empty_cache()
    if not np.isfinite(probability).all(): raise RuntimeError('nonfinite probability')
    return np.asarray(probability), state, cost


def run(which, device, resume):
    from bfa_core import formal_split, load_data_contract
    if verify_assets()['status'] != 'PASS': raise RuntimeError('assets changed before training')
    data, _ = load_data_contract(); splits = SPLITS[:5] if which == 'source' else SPLITS[5:]
    for split in splits:
        for seed in SEEDS:
            pack = formal_split(data, split, seed, device)
            labels = np.asarray(data['y'][pack['train']], int); epochs = int(pack['old']['selected_epochs'])
            for arm in ARMS[1:]:
                path = OUT/'models'/split/str(seed)/(arm+'.pkl')
                if path.is_file():
                    if not resume: raise RuntimeError('checkpoint exists without --resume')
                    print('CFGWA_RESUME',split,seed,arm,flush=True); continue
                aux_train, aux_val, protocol = augment_features(data, pack['train'], pack['val'], arm, seed)
                global_train = np.concatenate([pack['global_train'], aux_train], 1)
                global_val = np.concatenate([pack['global_val'], aux_val], 1)
                if global_train.shape[1] != 27 or global_val.shape[1] != 27: raise RuntimeError('global width is not 27')
                probability, state, cost = train_one(pack, seed, epochs, global_train, global_val, labels)
                obj = {'arm':arm,'split':split,'seed':seed,'ids':list(map(str,data['ids'][pack['val']])),
                       'probability':probability,'state_dict':state,'selected_epochs':epochs,
                       'cost':cost,'feature_protocol':protocol,'global_width':27,
                       'Stage1_changed':False,'Router_changed':False,'Host_RoleGraph_backgrounds_changed':False}
                write_pickle(path,obj)
                print('CFGWA_ARM_DONE',split,seed,arm,'epochs='+str(epochs),flush=True)
            print('CFGWA_SPLIT_SEED_DONE',split,seed,flush=True)


def smoke(device):
    from bfa_core import formal_split, load_data_contract
    data,_ = load_data_contract(); pack = formal_split(data,'S1',SEEDS[0],device)
    witness_val = None
    for arm in ARMS[1:]:
        a,b,p = augment_features(data,pack['train'],pack['val'],arm,SEEDS[0])
        if a.shape != (len(pack['train']),3) or b.shape != (len(pack['val']),3): raise RuntimeError('feature shape')
        if not np.isfinite(a).all() or not np.isfinite(b).all(): raise RuntimeError('nonfinite feature')
        if arm == 'W1_WITNESS3': witness_val = b
    if witness_val is None: raise RuntimeError('missing witness smoke features')
    v4, dev = pack['v4'], pack['dev']
    smoke_global = np.concatenate([pack['global_val'], witness_val], 1)
    model = auxiliary_model(v4, pack['args']).to(dev)
    probability, _ = v4.predict_model(
        model, smoke_global, np.full(len(smoke_global), .5),
        pack['graph_val'], pack['bags_val'], dev)
    if len(probability) != len(pack['val']) or not np.isfinite(probability).all():
        raise RuntimeError('auxiliary readout smoke prediction failed')
    parameters = int(sum(value.numel() for value in model.parameters()))
    del model
    if dev.type == 'cuda': v4.torch.cuda.empty_cache()
    atomic_json(OUT/'smoke_test.json',{'status':'PASS','split':'S1','baseline_replay_max_difference':pack['replay_diff'],
                'train_n':len(pack['train']),'val_n':len(pack['val']),'arms':ARMS[1:],
                'auxiliary_global_width':27,'auxiliary_model_parameters':parameters,
                'auxiliary_prediction_n':len(probability)})
    print('CFGWA_SMOKE_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true'); group.add_argument('--smoke-test',action='store_true')
    group.add_argument('--train-source',action='store_true'); group.add_argument('--train-rolling',action='store_true')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if args.preflight: preflight(args.resume); print('CFGWA_PREFLIGHT_PASS',flush=True); return
    if verify_assets()['status']!='PASS': raise RuntimeError('run preflight first')
    if args.device=='cuda':
        import torch
        probe=torch.zeros(1,device='cuda'); del probe
    if args.smoke_test: smoke(args.device)
    elif args.train_source: run('source',args.device,args.resume)
    else: run('rolling',args.device,args.resume)


if __name__=='__main__': main()
