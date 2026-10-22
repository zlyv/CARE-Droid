#!/usr/bin/env python3
"""Shared, read-only utilities for the M1 feature-sufficiency audit."""
from __future__ import annotations

import csv, hashlib, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path.cwd()
M1 = PROJECT / "scripts/care-di25-hostego-within-year-v1"
MECH = PROJECT / "scripts/care-di25-hostego-stage2-mechanism-v1"
MATRIX = M1 / "within_matrix"
for p in (HERE, MECH, M1, MATRIX, MATRIX / "vendor"):
    sys.path.insert(0, str(p))

import run_m1_within_year as frozen
import run_within_year_arm as within
import run_instrumented_m1 as previous
from host_ego_encoder import unpack, repack_augmented
from instrumented_model import instrumented_class

v4 = within.v4
OUT = Path("artifacts/care-di25-hostego-dmd-bounded-v1/feature_sufficiency_audit")
RUN = Path("artifacts/care-di25-hostego-v1/stage2_mechanism_diagnosis/instrumented_rerun")
MANIFEST = RUN / "checkpoint_manifest.csv"
HISTORY = RUN / "per_seed_apk_internal.csv"
ROUTER = Path("artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv")
YEARS = (2018, 2019, 2020, 2021)
FOLDS = (1, 2, 3, 4, 5)
SEEDS = (900000, 900001, 900002)
EXPECTED = {(y, f, s) for y in YEARS for f in FOLDS for s in SEEDS}


def norm(x): return str(x or "").strip().lower()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); a = value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def torch_load(path):
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")


def resolve_path(raw, base, fallback):
    candidates = []
    if raw:
        value = Path(str(raw))
        candidates += [value, base / value]
    candidates.append(Path(fallback))
    for p in candidates:
        if p.is_file(): return p.resolve()
    return candidates[0].resolve()


def manifest_records():
    if not MANIFEST.is_file(): raise RuntimeError("BLOCKED_BY_CONTRACT: checkpoint_manifest.csv missing")
    rows = read_csv(MANIFEST); result = []
    for r in rows:
        y, f, s = int(r["year"]), int(r["fold"]), int(r["seed"])
        cp_default = RUN / "checkpoints" / str(y) / f"fold_{f}" / f"seed_{s}.pt"
        vi_default = cp_default.with_name(f"seed_{s}_validation_inputs.npz")
        cp = resolve_path(r.get("checkpoint_path"), MANIFEST.parent, cp_default)
        vi = resolve_path(r.get("validation_inputs_path"), MANIFEST.parent, vi_default)
        result.append({**r, "year": y, "fold": f, "seed": s,
                       "checkpoint_path_resolved": cp, "validation_inputs_path_resolved": vi})
    return result


def checkpoint_args(state):
    args = frozen.parser().parse_args([])
    aliases = {"global_dim":"global_dim", "hidden_dim":"hidden_dim", "embedding_dim":"embedding_dim",
               "graph_dim":"graph_dim", "graph_heads":"graph_heads", "graph_layers":"graph_layers",
               "dropout":"dropout"}
    for source, dest in aliases.items():
        if source in state: setattr(args, dest, state[source])
    args.max_delta = float(state.get("max_delta_logit", state.get("max_delta", 2.25)))
    return args


def build_model(state, device):
    args = checkpoint_args(state)
    v4.AUGMENTED_REGION_DIM = within.future.ORIGINAL_AUGMENTED_DIM
    Model = instrumented_class(within.anomaly_bounded, v4)
    model = Model(int(state["global_dim"]), "RGCondAttn", args).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval(); model.requires_grad_(False)
    return model, args


def unpack_bags(values, offsets):
    values = np.asarray(values, np.float32); offsets = np.asarray(offsets, np.int64)
    return [values[int(offsets[i]):int(offsets[i+1])] for i in range(len(offsets)-1)]


def pack_bags(bags):
    offsets = [0]
    for b in bags: offsets.append(offsets[-1] + len(b))
    values = np.concatenate(bags, axis=0) if bags else np.empty((0, 0), np.float32)
    return np.asarray(values, np.float32), np.asarray(offsets, np.int64)


def load_saved_inputs(path):
    with np.load(path, allow_pickle=False) as z:
        required = {"sha256", "global_values", "base_probability", "graph_node", "graph_edge",
                    "graph_global", "bag_values", "bag_offsets"}
        if not required.issubset(z.files):
            raise RuntimeError("validation input schema missing: " + ",".join(sorted(required-set(z.files))))
        return {k: np.asarray(z[k]) for k in z.files}


def forward_r4(model, saved, device):
    ids = [norm(x) for x in saved["sha256"].astype(str)]
    x = np.asarray(saved["global_values"], np.float32)
    base = np.asarray(saved["base_probability"], np.float64)
    graphs = {"node":np.asarray(saved["graph_node"]), "edge":np.asarray(saved["graph_edge"]),
              "global":np.asarray(saved["graph_global"])}
    bags = unpack_bags(saved["bag_values"], saved["bag_offsets"])
    gt = torch.as_tensor(x, dtype=torch.float32, device=device)
    bl = torch.as_tensor(v4.probability_to_logit(base), dtype=torch.float32, device=device)
    gr = v4.graph_tensors(graphs, device)
    features=[]; raw=[]; bounded=[]; probability=[]
    captured=[]
    def hook(_module, args): captured.append(args[0].detach().cpu().numpy().copy())
    handle = model.delta_head.register_forward_pre_hook(hook)
    before = state_hash(model)
    with torch.no_grad():
        for i, bag in enumerate(bags):
            one = {k:value[i] for k,value in gr.items()}
            logit, _ = model(gt[i], bl[i], one, torch.as_tensor(bag, dtype=torch.float32, device=device))
            c = model.last_internal_capture
            features.append(captured.pop()); raw.append(float(c["raw_delta"])); bounded.append(float(c["bounded_delta"]))
            probability.append(float(torch.sigmoid(logit).cpu()))
    handle.remove(); after = state_hash(model)
    if before != after: raise RuntimeError("BLOCKED_BY_CONTRACT: frozen parameters changed during forward")
    return {"sha256":np.asarray(ids), "R4":np.asarray(features,np.float32), "raw_delta":np.asarray(raw),
            "bounded_delta":np.asarray(bounded), "probability":np.asarray(probability),
            "base_probability":base, "parameter_hash_before":before, "parameter_hash_after":after}


def fit_preprocessing(data, train, validation, args, seed):
    """Recreate deterministic inputs only; this function has no optimizer."""
    xtr, xva = within.fold_global24(data, train, validation)
    base_train = within.inner_oof_base(data, train, int(args.split_seed)+int(data.years[validation[0]])*10+int(data.folds[validation[0]]))
    rawtr = within.subset_bags(data.bags, train); rawva = within.subset_bags(data.bags, validation)
    host_train=[unpack(x)[0] for x in rawtr]; host_val=[unpack(x)[0] for x in rawva]
    module=sys.modules[within.future.ORIGINAL_ANOMALY_FIT.__module__]
    mean,std=module.care_v1.fit_equal_apk_scaler(host_train,np.arange(len(host_train),dtype=np.int64))
    scaled_train=[(np.asarray(x,np.float32)-mean)/std for x in host_train]
    scaled_val=[(np.asarray(x,np.float32)-mean)/std for x in host_val]
    proto=module.v2.fit_prototype_state(scaled_train,data.labels[train],args,int(seed)+1707)
    aug_train,_=module.v2.augment_bags(scaled_train,proto,"JointDualPrototype",args)
    aug_val,_=module.v2.augment_bags(scaled_val,proto,"JointDualPrototype",args)
    train_bags=repack_augmented(aug_train,rawtr); val_bags=repack_augmented(aug_val,rawva)
    gs=v4.fit_role_graph_scaler(data.s7[train])
    train_graph=v4.transform_role_graphs(data.s7[train],gs); val_graph=v4.transform_role_graphs(data.s7[validation],gs)
    return {"train":{"sha256":np.asarray([data.ids[i] for i in train]),"global_values":xtr,"base_probability":base_train,
                     "graphs":train_graph,"bags":train_bags,"label":data.labels[train]},
            "validation":{"sha256":np.asarray([data.ids[i] for i in validation]),"global_values":xva,"base_probability":data.base[validation],
                          "graphs":val_graph,"bags":val_bags,"label":data.labels[validation]}}


def input_dict(part):
    values, offsets = pack_bags(part["bags"])
    return {"sha256":np.asarray(part["sha256"]), "global_values":np.asarray(part["global_values"],np.float32),
            "base_probability":np.asarray(part["base_probability"],np.float64),
            "graph_node":np.asarray(part["graphs"]["node"]), "graph_edge":np.asarray(part["graphs"]["edge"]),
            "graph_global":np.asarray(part["graphs"]["global"]), "bag_values":values, "bag_offsets":offsets,
            "label":np.asarray(part["label"],np.int64)}


def load_full_data(args):
    data=within.load_data(args)
    data.bags,_=previous.load_packed_full(data.ids,args.future_host,args.future_ego)
    router={norm(r["sha256"]):r for r in read_csv(ROUTER)}
    if set(router)!=set(data.ids): raise RuntimeError("router/data cohort mismatch")
    data.route=np.asarray([int(float(router[x]["route_DI"]))==1 for x in data.ids],bool)
    return data

