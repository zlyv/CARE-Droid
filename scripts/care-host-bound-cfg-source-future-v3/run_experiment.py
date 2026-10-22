#!/usr/bin/env python3
"""Host-bound CFG incremental probe with 2017-only fitting and frozen future evaluation."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/boyuanjie/projects/tasa-droid")
HERE = ROOT / "scripts/care-host-bound-cfg-source-future-v3"
OUT = ROOT / "outputs/care_host_bound_cfg_source_future_v3"
BFA = ROOT / "scripts/care-background-functional-attribution-v1"
SCBE_SCRIPT = ROOT / "scripts/care-scbe-v1"
MYD_SCRIPT = ROOT / "scripts/care-multiyear-development-v1"
MYD_OUT = ROOT / "outputs/care_multiyear_development_v1"
SCBE_OUT = ROOT / "outputs/care_scbe_v1"
CFG_ART = ROOT / "artifacts/care_cfg_witness_all_years_v2"
ENDPOINT = ROOT / "artifacts/care-droid-formal-diagnostics-v1/cache/endpoint_cache_2788.jsonl.gz"
EGO_SOURCE = ROOT / "artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz"
MANIFEST = ROOT / "outputs/care_manifest_four_member_disagreement_v1/run_20261019_pilot01"
GLOBAL_CFG_SOURCE = ROOT / "outputs/care_cfg_witness_fixed_ablation_v1/per_apk_predictions.csv"
GLOBAL_CFG_FUTURE = ROOT / "outputs/care_cfg_witness_fixed_source_future_v2/per_apk_predictions.csv"
HOSTREL_SCRIPT = ROOT / "scripts/care-di25-compact43-relationquery-source-v1"

sys.path[:0] = [str(HERE), str(BFA), str(SCBE_SCRIPT), str(MYD_SCRIPT), str(HOSTREL_SCRIPT)]

SEEDS = (900000, 900001, 900002)
SOURCE_SPLITS = ("S1", "S2", "S3", "S4", "S5")
YEARS = (2017, 2018, 2019, 2020, 2021)
TRAINED_ARMS = (
    "H1_HOST_BOUND_CFG",
    "H2_HOST_SIZE_CONTROL",
    "H3_HOST_BINDING_SHUFFLE",
    "H4_HOST_TOPOLOGY_BINDING_SHUFFLE",
)
REPORT_ARMS = (
    "H0_FROZEN_DIRECT_C43",
    "G1_APK_GLOBAL_WITNESS3",
) + TRAINED_ARMS
CFG_DIM = 16
BASE_PREFIX = 4


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            hasher.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": hasher.hexdigest()}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=encode) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, values):
    values = list(values)
    fields = list(dict.fromkeys(key for row in values for key in row)) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_pickle(path: Path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def asset_paths():
    paths = [
        HERE / "run_experiment.py",
        HERE / "run.sh",
        BFA / "bfa_core.py",
        SCBE_SCRIPT / "train.py",
        MYD_SCRIPT / "train_stage2.py",
        MYD_SCRIPT / "myd_common.py",
        MYD_SCRIPT / "metrics.py",
        ROOT / "scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_models.py",
        ROOT / "scripts/care-p31rf-c43-haig-host-ablation-source-v1/haig_background.py",
        ROOT / "scripts/care-di25-graphglobal-path-source-v1/graphglobal_path_models.py",
        HOSTREL_SCRIPT / "hostrel_features.py",
        ROOT / "scripts/care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py",
        MYD_OUT / "data.npz",
        MYD_OUT / "feature_rows.pkl",
        MYD_OUT / "sample_manifest.csv",
        ENDPOINT,
        EGO_SOURCE,
        CFG_ART / "cfg_witness_apk.csv",
        CFG_ART / "final_decision.json",
        CFG_ART / "baseline_integrity_check.json",
        MANIFEST / "protocol.json",
        MANIFEST / "per_apk_predictions.csv",
        MANIFEST / "asset_integrity_check.json",
        GLOBAL_CFG_SOURCE,
        GLOBAL_CFG_FUTURE,
    ]
    paths += [CFG_ART / "cache" / f"{sid}.json.gz" for sid in source_and_future_ids()]
    paths += [SCBE_OUT / "models" / split / str(seed) / "M0_REPRODUCED.pkl"
              for split in SOURCE_SPLITS + ("T1",) for seed in SEEDS]
    return paths


def source_and_future_ids():
    with np.load(MYD_OUT / "data.npz", allow_pickle=False) as archive:
        return list(map(str, archive["ids"]))


def verify_assets(items):
    changed = []
    for item in items:
        try:
            current = digest(Path(item["path"]))
        except OSError as exc:
            changed.append({"path": item["path"], "error": repr(exc)})
            continue
        if current != item:
            changed.append({"before": item, "after": current})
    return {
        "status": "PASS" if not changed else "FAIL",
        "checked": len(items),
        "changed": changed,
        "scope": "named code, 2017-only checkpoints, Manifest135 predictions, endpoint cache, and 2,788 CFG caches; no APK bytes",
    }


def preflight():
    if OUT.exists():
        raise RuntimeError("output exists; refusing to overwrite")
    required = asset_paths()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing required assets: " + repr(missing[:20]))
    for path in (
        CFG_ART / "baseline_integrity_check.json",
        MANIFEST / "asset_integrity_check.json",
    ):
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("upstream integrity is not PASS: " + str(path))
    OUT.mkdir(parents=True)
    (OUT / "models").mkdir()
    (OUT / "logs").mkdir()
    manifest = [digest(path) for path in required]
    atomic_json(OUT / "asset_manifest_before.json", manifest)
    atomic_json(OUT / "preflight.json", {
        "status": "PASS",
        "source_rows": 799,
        "future_rows": 1989,
        "training_years": [2017],
        "evaluation_years": list(YEARS),
        "stage1": "Manifest135 RF",
        "router": "OTHER3_OP25_REF using Manifest probabilities only",
        "APK_files_opened": 0,
        "new_APKs_used": False,
        "backgrounds_changed": False,
        "prototype_bank_changed": False,
        "rolling_training_used": False,
        "cfg_feature_dimension": CFG_DIM,
        "arms": list(REPORT_ARMS),
    })
    print(f"HBCFG_PREFLIGHT_PASS assets={len(manifest)}", flush=True)


def endpoint_host_ids(payload):
    from hostrel_features import reconstruct_host_relations

    records = reconstruct_host_relations(payload, np.arange(31, dtype=np.int64))
    return [str(record["host_id"]) for record in records]


def endpoint_map(wanted):
    result = {}
    with gzip.open(ENDPOINT, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sha256", "")).strip().lower()
            if sid in wanted and row.get("status") == "OK":
                result[sid] = endpoint_host_ids(row["payload"])
    if set(result) != wanted:
        raise RuntimeError(f"endpoint Host coverage {len(result)}/{len(wanted)}")
    return result


def fraction(values, key):
    selected = [row.get(key) for row in values if row.get(key) is not None]
    return float(sum(bool(value) for value in selected) / max(len(selected), 1))


def host_features(host, events, pairs):
    parsed = float(host.get("cfg_status") == "PARSED")
    instruction_count = max(float(host.get("instruction_count", 0)), 0.)
    aligned = max(float(host.get("aligned_sensitive_calls", 0)), 0.)
    roles = max(float(host.get("distinct_roles", 0)), 0.)
    cycle = float(bool(host.get("has_reachable_cycle")))
    loop_fraction = float(sum(bool(row.get("in_loop")) for row in events) / max(len(events), 1))
    entry_fraction = float(sum(bool(row.get("entry_reachable")) for row in events) / max(len(events), 1))
    relation = Counter(str(row.get("relation")) for row in pairs)
    total = max(len(pairs), 1)
    witness = set()
    for pair in pairs:
        left, right = int(pair["role_a"]), int(pair["role_b"])
        if pair.get("relation") in ("A_TO_B", "BIDIRECTIONAL"):
            witness.add((left, right))
        if pair.get("relation") in ("B_TO_A", "BIDIRECTIONAL"):
            witness.add((right, left))
    controlled_event_fraction = float(sum(float(row.get("verified_control_branch_count", 0)) > 0
                                          for row in events) / max(len(events), 1))
    real = np.asarray([
        parsed,
        np.log1p(instruction_count),
        np.log1p(aligned),
        np.log1p(roles),
        cycle,
        loop_fraction,
        entry_fraction,
        np.log1p(len(witness)),
        (relation["A_TO_B"] + relation["B_TO_A"]) / total,
        relation["BIDIRECTIONAL"] / total,
        relation["NEITHER"] / total,
        relation["UNKNOWN"] / total,
        fraction(pairs, "same_basic_block"),
        fraction(pairs, "shared_verified_control_branch"),
        fraction(pairs, "different_verified_control_sets"),
        controlled_event_fraction,
    ], np.float32)
    base = np.asarray([
        parsed,
        np.log1p(instruction_count),
        np.log1p(aligned),
        np.log1p(roles),
        np.log1p(max(float(host.get("potential_event_pairs", 0)), 0.)),
        aligned / max(instruction_count, 1.),
        roles / max(aligned, 1.),
        max(float(host.get("potential_event_pairs", 0)), 0.) / max(aligned * max(aligned - 1., 0.) / 2., 1.),
    ], np.float32)
    control = np.concatenate([base, np.square(base)]).astype(np.float32)
    if real.shape != (CFG_DIM,) or control.shape != (CFG_DIM,):
        raise RuntimeError("Host CFG feature width changed")
    return real, control


def build_feature_cache():
    from bfa_core import load_data_contract

    data, _ = load_data_contract()
    ids = list(map(str, data["ids"]))
    wanted = set(ids)
    host_order = endpoint_map(wanted)
    source_ids = ids[:799]
    with np.load(EGO_SOURCE, allow_pickle=False) as archive:
        if list(map(str, archive["sha256"])) != source_ids:
            raise RuntimeError("source Ego SHA order changed")
        ego_offsets = np.asarray(archive["ego_offsets"], int)
        ego_host_ids = list(map(str, archive["host_ids"]))

    real_values = []
    control_values = []
    offsets = [0]
    parse_ok = 0
    real_hosts = 0
    missing_rows = 0
    source_exact = 0
    for index, sid in enumerate(ids):
        ordered = host_order[sid]
        expected_real = int(data["real_host_count"][index])
        stored = int(data["host_offsets"][index + 1] - data["host_offsets"][index])
        if len(ordered) != expected_real:
            raise RuntimeError(f"Host count mismatch {sid}: endpoint={len(ordered)} data={expected_real}")
        if stored != (expected_real if expected_real else 1):
            raise RuntimeError(f"Host placeholder mismatch {sid}: stored={stored} real={expected_real}")
        if index < 799:
            left, right = int(ego_offsets[index]), int(ego_offsets[index + 1])
            if ego_host_ids[left:right] != ordered:
                raise RuntimeError("source exact Host identity/order mismatch: " + sid)
            source_exact += len(ordered)

        cache_path = CFG_ART / "cache" / f"{sid}.json.gz"
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            scan = json.load(handle)
        if scan.get("status") != "OK" or str(scan.get("sha256")) != sid:
            raise RuntimeError("CFG cache identity/status mismatch: " + sid)
        host_rows = {}
        for row in scan.get("hosts", []):
            key = str(row["host_node_id"])
            if key in host_rows:
                raise RuntimeError("duplicate CFG Host ID: " + sid + "/" + key)
            host_rows[key] = row
        event_map = defaultdict(list)
        pair_map = defaultdict(list)
        for row in scan.get("events", []):
            event_map[str(row["host_node_id"])].append(row)
        for row in scan.get("pairs", []):
            pair_map[str(row["host_node_id"])].append(row)
        if not ordered:
            real_values.append(np.zeros(CFG_DIM, np.float32))
            control_values.append(np.zeros(CFG_DIM, np.float32))
            missing_rows += 1
        else:
            for host_id in ordered:
                row = host_rows.get(host_id)
                if row is None:
                    real_values.append(np.zeros(CFG_DIM, np.float32))
                    control_values.append(np.zeros(CFG_DIM, np.float32))
                    missing_rows += 1
                    continue
                real, control = host_features(row, event_map[host_id], pair_map[host_id])
                real_values.append(real)
                control_values.append(control)
                real_hosts += 1
                parse_ok += int(row.get("cfg_status") == "PARSED")
        offsets.append(len(real_values))
        if (index + 1) % 100 == 0 or index + 1 == len(ids):
            print(f"HBCFG_FEATURES {index+1}/{len(ids)} hosts={len(real_values)}", flush=True)

    real = np.asarray(real_values, np.float32)
    control = np.asarray(control_values, np.float32)
    offsets = np.asarray(offsets, np.int64)
    if real.shape != (int(data["host_offsets"][-1]), CFG_DIM) or control.shape != real.shape:
        raise RuntimeError(f"feature/data Host shape mismatch real={real.shape} data={int(data['host_offsets'][-1])}")
    if not np.array_equal(offsets, np.asarray(data["host_offsets"], np.int64)):
        raise RuntimeError("feature offsets differ from frozen Host offsets")
    if not np.isfinite(real).all() or not np.isfinite(control).all():
        raise RuntimeError("nonfinite Host CFG feature")
    temp = OUT / "host_cfg_features.tmp.npz"
    np.savez_compressed(temp, sha256=np.asarray(ids), host_offsets=offsets, real=real, control=control)
    temp.replace(OUT / "host_cfg_features.npz")
    audit = {
        "status": "PASS",
        "APKs": len(ids),
        "stored_host_rows": len(real),
        "real_CFG_host_rows_recovered": real_hosts,
        "CFG_parsed_host_rows": parse_ok,
        "missing_or_empty_placeholder_rows": missing_rows,
        "source_exact_Host_identity_order_matches": source_exact,
        "labels_used": False,
        "APK_files_opened": 0,
        "feature_width": CFG_DIM,
        "candidate_columns": [
            "parsed", "log_instruction_count", "log_aligned_sensitive_calls", "log_distinct_roles",
            "has_cycle", "loop_event_fraction", "entry_reachable_fraction", "log_ordered_witness_pairs",
            "directed_pair_fraction", "bidirectional_pair_fraction", "neither_pair_fraction",
            "unknown_pair_fraction", "same_basic_block_fraction", "shared_control_branch_fraction",
            "different_control_sets_fraction", "controlled_event_fraction",
        ],
        "size_control": "8 label-free Host size/coverage values concatenated with their elementwise squares",
    }
    atomic_json(OUT / "feature_audit.json", audit)
    print("HBCFG_FEATURE_CACHE_PASS", flush=True)


def load_feature_cache(data):
    with np.load(OUT / "host_cfg_features.npz", allow_pickle=False) as archive:
        if list(map(str, archive["sha256"])) != list(map(str, data["ids"])):
            raise RuntimeError("Host CFG feature SHA order changed")
        if not np.array_equal(archive["host_offsets"], data["host_offsets"]):
            raise RuntimeError("Host CFG feature offsets changed")
        return np.asarray(archive["real"], np.float32), np.asarray(archive["control"], np.float32)


def feature_bags(values, data, indices):
    offsets = np.asarray(data["host_offsets"], int)
    return [np.asarray(values[offsets[index]:offsets[index + 1]], np.float32) for index in indices]


def equal_apk_scaler(bags):
    apk_mean = np.asarray([bag.mean(0) for bag in bags], np.float64)
    mean = apk_mean.mean(0)
    apk_second = np.asarray([np.mean(np.square(bag.astype(np.float64)), 0) for bag in bags])
    variance = np.maximum(apk_second.mean(0) - np.square(mean), 0.)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


def stable_rng(seed, sid, token):
    raw = hashlib.sha256(f"{seed}:{sid}:{token}".encode()).digest()
    return np.random.RandomState(int.from_bytes(raw[:4], "big"))


def strata(real):
    parsed = (real[:, 0] > .5).astype(int)
    calls = np.minimum(np.rint(np.expm1(np.maximum(real[:, 2], 0))).astype(int), 2)
    roles = np.minimum(np.rint(np.expm1(np.maximum(real[:, 3], 0))).astype(int), 2)
    return parsed * 9 + calls * 3 + roles


def shuffled_bag(real, seed, sid, topology_only):
    result = real.copy()
    code = strata(real)
    rng = stable_rng(seed, sid, "topology" if topology_only else "binding")
    for value in np.unique(code):
        index = np.flatnonzero(code == value)
        if len(index) > 1:
            donor = rng.permutation(index)
            if np.array_equal(donor, index):
                donor = np.roll(index, 1)
            if topology_only:
                result[index, BASE_PREFIX:] = real[donor, BASE_PREFIX:]
            else:
                result[index] = real[donor]
    return result


def prepare_cfg_bags(data, indices, arm, seed, real_values, control_values, scaler=None):
    ids = list(map(str, data["ids"][indices]))
    real = feature_bags(real_values, data, indices)
    if arm == "H2_HOST_SIZE_CONTROL":
        bags = feature_bags(control_values, data, indices)
    elif arm == "H3_HOST_BINDING_SHUFFLE":
        bags = [shuffled_bag(bag, seed, sid, False) for bag, sid in zip(real, ids)]
    elif arm == "H4_HOST_TOPOLOGY_BINDING_SHUFFLE":
        bags = [shuffled_bag(bag, seed, sid, True) for bag, sid in zip(real, ids)]
    elif arm == "H1_HOST_BOUND_CFG":
        bags = real
    else:
        raise KeyError(arm)
    if scaler is None:
        mean, std = equal_apk_scaler(bags)
    else:
        mean = np.asarray(scaler["mean"], np.float32)
        std = np.asarray(scaler["std"], np.float32)
    output = [((bag - mean) / std).astype(np.float32) for bag in bags]
    if any(bag.shape[1] != CFG_DIM or not np.isfinite(bag).all() for bag in output):
        raise RuntimeError("invalid standardized Host CFG bag")
    changed = None
    if arm in ("H3_HOST_BINDING_SHUFFLE", "H4_HOST_TOPOLOGY_BINDING_SHUFFLE"):
        changed = float(sum(np.any(a != b, axis=1).sum() for a, b in zip(real, bags)) / max(sum(len(b) for b in bags), 1))
    return output, {"mean": mean, "std": std, "changed_host_fraction": changed}


def host_bound_model(v4, args):
    import torch
    from torch import nn
    from graphglobal_path_models import LocalEvidenceCache

    Base = v4.RoleGraphConditionedEvidence

    class HostBoundCFG(Base):
        def __init__(self):
            super().__init__(24, "RGCondAttn", args)
            dim = int(args.embedding_dim)
            hidden = int(args.hidden_dim)
            # The zero-initialized final affine layer preserves the frozen H0
            # function at step zero.  It deliberately has no trailing ReLU:
            # d(linear)/dz remains nonzero at z=0, so the adapter can learn.
            self.cfg_adapter = nn.Sequential(nn.Linear(CFG_DIM, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.cfg_gate = nn.Linear(2 * dim, 1)
            nn.init.zeros_(self.cfg_adapter[2].weight)
            nn.init.zeros_(self.cfg_adapter[2].bias)
            nn.init.zeros_(self.cfg_gate.weight)
            nn.init.constant_(self.cfg_gate.bias, -2.0)

        def select_local_evidence(self, graph, regions):
            if regions.ndim != 2 or regions.shape[1] != 142 + CFG_DIM:
                raise RuntimeError(f"Host-bound CFG expects N x {142+CFG_DIM}, got {tuple(regions.shape)}")
            base = regions[:, :142]
            cfg = regions[:, 142:]
            core, relation = base[:, :111], base[:, 111:]
            role, graph_entropy = self.encode_role_graph(graph["node"], graph["edge"])
            graphglobal = self.graphglobal_encoder(graph["global"].reshape(-1))
            host = self.region_encoder(core)
            cfg_embedding = self.cfg_adapter(cfg)
            gamma = torch.sigmoid(self.cfg_gate(torch.cat([host, cfg_embedding], 1))).reshape(-1)
            local = host + gamma[:, None] * cfg_embedding
            residual = torch.log1p(torch.clamp(core[:, -1], min=0))
            score = self.local_risk(local).reshape(-1) + nn.functional.softplus(self.residual_beta_raw) * residual
            query = self.role_query(role) + self.global_to_query(graphglobal)
            score = score + torch.mv(self.local_key(local), query) / math.sqrt(local.shape[1])
            score = score + torch.mv(self.relation_key(relation), query) / math.sqrt(local.shape[1])
            attention = torch.softmax(score, 0)
            critical_index = torch.argmax(score)
            summary = torch.cat([
                score[critical_index].reshape(1),
                torch.mean(score).reshape(1),
                (-torch.sum(attention * torch.log(attention.clamp_min(1e-8))) / math.log(max(len(attention), 2))).reshape(1),
                torch.max(attention).reshape(1),
                torch.stack([torch.mean(core[:, -4]), torch.mean(core[:, -3]), torch.mean(core[:, -2]), torch.max(core[:, -1])]),
            ])
            return LocalEvidenceCache(
                role_token=role,
                critical=local[critical_index],
                attended=torch.sum(attention[:, None] * local, 0),
                scalar_summary=summary,
                attention=attention,
                score=score,
                graph_entropy=graph_entropy,
            ), graphglobal

    return HostBoundCFG()


def initialize_from_m0(model, old):
    result = model.load_state_dict(old["state_dict"], strict=False)
    expected = {
        "cfg_adapter.0.weight", "cfg_adapter.0.bias", "cfg_adapter.2.weight", "cfg_adapter.2.bias",
        "cfg_gate.weight", "cfg_gate.bias",
    }
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError(f"M0 adapter load mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in expected
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def append_cfg(base_bags, cfg_bags):
    if len(base_bags) != len(cfg_bags):
        raise RuntimeError("base/CFG bag count mismatch")
    output = []
    for base, cfg in zip(base_bags, cfg_bags):
        if len(base) != len(cfg) or base.shape[1] != 142 or cfg.shape[1] != CFG_DIM:
            raise RuntimeError(f"Host alignment mismatch base={base.shape} cfg={cfg.shape}")
        output.append(np.concatenate([base, cfg], 1).astype(np.float32))
    return output


def train_adapter(pack, arm, seed, labels, cfg_train, cfg_val, epochs):
    v4, args, dev = pack["v4"], pack["args"], pack["dev"]
    torch = v4.torch
    v4.seed_everything(seed + 81001)
    model = host_bound_model(v4, args).to(dev)
    trainable = initialize_from_m0(model, pack["old"])
    if trainable <= 0:
        raise RuntimeError("no trainable CFG adapter parameters")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_bags = append_cfg(pack["bags_train"], cfg_train)
    val_bags = append_cfg(pack["bags_val"], cfg_val)
    global_tensor = torch.as_tensor(pack["global_train"], dtype=torch.float32, device=dev)
    graph_tensor = v4.graph_tensors(pack["graph_train"], dev)
    bag_tensors = [torch.as_tensor(value, dtype=torch.float32, device=dev) for value in train_bags]
    neutral_train = np.full(len(labels), .5)
    weights = np.ones(len(labels), np.float32)
    correct = np.zeros(len(labels), np.float32)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.perf_counter()
    loss = v4.train_epochs(model, optimizer, global_tensor, neutral_train, graph_tensor, bag_tensors,
                           labels, weights, correct, epochs, args, seed + 83001)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability, _ = v4.predict_model(model, pack["global_val"], np.full(len(pack["val"]), .5),
                                      pack["graph_val"], val_bags, dev)
    inference_seconds = time.perf_counter() - started
    adapter_state = {name: value.detach().cpu() for name, value in model.state_dict().items()
                     if name.startswith("cfg_adapter.") or name.startswith("cfg_gate.")}
    cost = {
        "trainable_parameters": trainable,
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "gpu_peak_mib": float(torch.cuda.max_memory_allocated(dev)) / 1024 ** 2 if dev.type == "cuda" else 0.,
        "final_BCE": float(loss),
    }
    del model, optimizer, global_tensor, graph_tensor, bag_tensors
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(probability, float), adapter_state, cost


def fit_source(device, resume):
    from bfa_core import formal_split, load_data_contract

    data, _ = load_data_contract()
    real, control = load_feature_cache(data)
    for split in SOURCE_SPLITS + ("T1",):
        for seed in SEEDS:
            pack = formal_split(data, split, seed, device)
            labels = np.asarray(data["y"][pack["train"]], int)
            epochs = int(pack["old"]["selected_epochs"])
            for arm in TRAINED_ARMS:
                path = OUT / "models" / split / str(seed) / f"{arm}.pkl"
                if path.is_file():
                    if not resume:
                        raise RuntimeError("checkpoint exists without --resume: " + str(path))
                    print("HBCFG_RESUME", split, seed, arm, flush=True)
                    continue
                cfg_train, scaler = prepare_cfg_bags(data, pack["train"], arm, seed, real, control)
                cfg_val, _ = prepare_cfg_bags(data, pack["val"], arm, seed, real, control, scaler)
                probability, state, cost = train_adapter(pack, arm, seed, labels, cfg_train, cfg_val, epochs)
                write_pickle(path, {
                    "arm": arm,
                    "split": split,
                    "seed": seed,
                    "ids": list(map(str, data["ids"][pack["val"]])),
                    "probability": probability,
                    "adapter_state": state,
                    "feature_scaler": scaler,
                    "selected_epochs": epochs,
                    "cost": cost,
                    "training_years": [2017],
                    "base_checkpoint": str((SCBE_OUT / "models" / split / str(seed) / "M0_REPRODUCED.pkl").resolve()),
                    "backgrounds_changed": False,
                    "prototype_bank_changed": False,
                    "base_backbone_frozen": True,
                })
                print("HBCFG_MODEL_DONE", split, seed, arm, f"epochs={epochs}", flush=True)
    print("HBCFG_SOURCE_AND_T1_TRAINING_DONE", flush=True)


def predict_target(v4, args, dev, old, checkpoint, global_target, graph_target, base_bags, cfg_bags):
    model = host_bound_model(v4, args).to(dev)
    initialize_from_m0(model, old)
    combined_state = dict(old["state_dict"])
    combined_state.update(checkpoint["adapter_state"])
    model.load_state_dict(combined_state, strict=True)
    model.eval()
    probability, _ = v4.predict_model(model, global_target, np.full(len(global_target), .5),
                                      graph_target, append_cfg(base_bags, cfg_bags), dev)
    del model
    if dev.type == "cuda":
        v4.torch.cuda.empty_cache()
    return np.asarray(probability, float)


def infer_future(device):
    from bfa_core import load_data_contract
    from myd_common import runtime
    from train import global24
    from train_stage2 import raw_bags, transform_saved

    data, _ = load_data_contract()
    real, control = load_feature_cache(data)
    years = np.asarray(data["years"], int)
    train_index = np.flatnonzero(years == 2017)
    target_index = np.flatnonzero(years > 2017)
    v4, _, args = runtime()
    args.device = device
    args.keep_weight = args.fix_weight = 1.
    args.keep_consistency = 0.
    dev = v4.resolve_device(device)
    _, global_target = global24(v4, data, train_index, target_index)
    raw_target = raw_bags(data, target_index)
    for seed in SEEDS:
        old = load_pickle(SCBE_OUT / "models" / "T1" / str(seed) / "M0_REPRODUCED.pkl")
        base_bags = transform_saved(v4, raw_target, old["background"], args)
        graph_target = v4.transform_role_graphs(data["s7"][target_index], old["graph_scaler"])
        base_model = v4.RoleGraphConditionedEvidence(24, "RGCondAttn", args).to(dev)
        base_model.load_state_dict(old["state_dict"])
        base_model.eval()
        base_probability, _ = v4.predict_model(base_model, global_target, np.full(len(target_index), .5),
                                               graph_target, base_bags, dev)
        write_pickle(OUT / "models" / "FUTURE" / str(seed) / "H0_FROZEN_DIRECT_C43.pkl", {
            "arm": "H0_FROZEN_DIRECT_C43", "seed": seed,
            "ids": list(map(str, data["ids"][target_index])), "probability": np.asarray(base_probability, float),
        })
        del base_model
        if dev.type == "cuda":
            v4.torch.cuda.empty_cache()
        for arm in TRAINED_ARMS:
            checkpoint = load_pickle(OUT / "models" / "T1" / str(seed) / f"{arm}.pkl")
            cfg_bags, _ = prepare_cfg_bags(data, target_index, arm, seed, real, control,
                                           checkpoint["feature_scaler"])
            probability = predict_target(v4, args, dev, old, checkpoint, global_target,
                                         graph_target, base_bags, cfg_bags)
            write_pickle(OUT / "models" / "FUTURE" / str(seed) / f"{arm}.pkl", {
                "arm": arm, "seed": seed, "ids": list(map(str, data["ids"][target_index])),
                "probability": probability, "training_years": [2017], "evaluation_years": [2018, 2019, 2020, 2021],
            })
            print("HBCFG_FUTURE_DONE", seed, arm, flush=True)
    print("HBCFG_ALL_FUTURE_INFERENCE_DONE", flush=True)


def global_witness_map():
    result = {}
    for row in rows(GLOBAL_CFG_SOURCE):
        if row["arm"] == "W1_WITNESS3" and row["split"].startswith("S"):
            result[(2017, row["sha256"])] = float(row["expert_probability"])
    for row in rows(GLOBAL_CFG_FUTURE):
        if row["arm"] == "W1_WITNESS3":
            result[(int(row["test_year"]), row["sha256"])] = float(row["expert_probability"])
    if len(result) != 2788:
        raise RuntimeError(f"global Witness expert inventory {len(result)} != 2788")
    return result


def manifest_groups():
    result = defaultdict(list)
    with (MANIFEST / "per_apk_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["anchor"] == "RF" and row["strategy"] == "OTHER3_OP25_REF":
                result[(int(row["year"]), int(row["seed"]))].append(row)
    expected = {(year, seed) for year in YEARS for seed in (20261018, 20261019, 20261020)}
    if set(result) != expected:
        raise RuntimeError("Manifest RF group inventory changed")
    for group in result.values():
        group.sort(key=lambda row: row["sha256"])
    return result


def evaluate():
    from bfa_core import load_data_contract
    from metrics import cascade

    data, _ = load_data_contract()
    ids = list(map(str, data["ids"]))
    years = np.asarray(data["years"], int)
    expert = {arm: {} for arm in REPORT_ARMS if arm != "G1_APK_GLOBAL_WITNESS3"}
    source_position = {sid: index for index, sid in enumerate(ids)}
    for split in SOURCE_SPLITS:
        split_indices = None
        for seed in SEEDS:
            old = load_pickle(SCBE_OUT / "models" / split / str(seed) / "M0_REPRODUCED.pkl")
            if split_indices is None:
                split_indices = [source_position[sid] for sid in old["ids"]]
            for sid, probability in zip(old["ids"], old["probability"]):
                expert["H0_FROZEN_DIRECT_C43"].setdefault((2017, sid), []).append(float(probability))
            for arm in TRAINED_ARMS:
                checkpoint = load_pickle(OUT / "models" / split / str(seed) / f"{arm}.pkl")
                for sid, probability in zip(checkpoint["ids"], checkpoint["probability"]):
                    expert[arm].setdefault((2017, sid), []).append(float(probability))
    future_index = np.flatnonzero(years > 2017)
    future_ids = list(map(str, data["ids"][future_index]))
    for seed in SEEDS:
        for arm in ("H0_FROZEN_DIRECT_C43",) + TRAINED_ARMS:
            checkpoint = load_pickle(OUT / "models" / "FUTURE" / str(seed) / f"{arm}.pkl")
            if checkpoint["ids"] != future_ids:
                raise RuntimeError("future SHA order mismatch")
            for position, (index, sid) in enumerate(zip(future_index, future_ids)):
                expert[arm].setdefault((int(years[index]), sid), []).append(float(checkpoint["probability"][position]))
    for arm, mapping in expert.items():
        if len(mapping) != 2788 or any(len(values) != 3 for values in mapping.values()):
            raise RuntimeError(f"expert inventory incomplete {arm}: rows={len(mapping)}")
        for key in list(mapping):
            mapping[key] = float(np.mean(mapping[key]))
    expert["G1_APK_GLOBAL_WITNESS3"] = global_witness_map()

    groups = manifest_groups()
    performance = []
    per_apk = []
    for (year, manifest_seed), group in sorted(groups.items()):
        labels = np.asarray([int(row["label"]) for row in group], int)
        base = np.asarray([float(row["stage1_probability"]) for row in group], float)
        route = np.asarray([int(row["routed"]) == 1 for row in group], bool)
        group_ids = [row["sha256"] for row in group]
        for arm in REPORT_ARMS:
            probability = np.asarray([expert[arm][(year, sid)] for sid in group_ids], float)
            metric = cascade(labels, base, probability, route)
            performance.append({
                "year": year, "manifest_seed": manifest_seed, "arm": arm,
                "test_n": len(group), "route_n": int(route.sum()), "route_fraction": float(route.mean()),
                "training_years": "2017", **metric,
            })
            final = np.where(route, probability, base)
            for index, sid in enumerate(group_ids):
                per_apk.append({
                    "year": year, "manifest_seed": manifest_seed, "arm": arm, "sha256": sid,
                    "label": int(labels[index]), "routed": int(route[index]),
                    "stage1_probability": float(base[index]), "expert_probability": float(probability[index]),
                    "final_probability": float(final[index]),
                    "final_correct": int(int(final[index] >= .5) == int(labels[index])),
                })
    means = []
    for key in sorted({(row["year"], row["arm"]) for row in performance}):
        group = [row for row in performance if (row["year"], row["arm"]) == key]
        item = {"year": key[0], "arm": key[1], "manifest_seed_count": len(group)}
        for field in group[0]:
            if field in ("year", "manifest_seed", "arm", "training_years"):
                continue
            try:
                item[field] = float(np.mean([float(row[field]) for row in group]))
            except (TypeError, ValueError):
                pass
        means.append(item)
    lookup = {(row["year"], row["arm"]): row for row in means}
    comparisons = []
    for reference in ("H0_FROZEN_DIRECT_C43", "G1_APK_GLOBAL_WITNESS3",
                      "H2_HOST_SIZE_CONTROL", "H3_HOST_BINDING_SHUFFLE", "H4_HOST_TOPOLOGY_BINDING_SHUFFLE"):
        for year in YEARS:
            candidate = lookup[(year, "H1_HOST_BOUND_CFG")]
            control = lookup[(year, reference)]
            comparisons.append({
                "scope": str(year), "candidate": "H1_HOST_BOUND_CFG", "reference": reference,
                "BA_gain": candidate["BA"] - control["BA"],
                "AUC_gain": candidate["AUC"] - control["AUC"],
                "Net_gain": candidate["Net"] - control["Net"],
            })
        future = [row for row in comparisons if row["reference"] == reference and row["scope"] in {"2018", "2019", "2020", "2021"}]
        comparisons.append({
            "scope": "FUTURE_MACRO_2018_2021", "candidate": "H1_HOST_BOUND_CFG", "reference": reference,
            "BA_gain": float(np.mean([row["BA_gain"] for row in future])),
            "AUC_gain": float(np.mean([row["AUC_gain"] for row in future])),
            "Net_gain": float(np.sum([row["Net_gain"] for row in future])),
            "positive_BA_years": int(sum(row["BA_gain"] > 0 for row in future)),
        })
    source = {row["reference"]: row for row in comparisons if row["scope"] == "2017"}
    source_supported = all(source[reference]["BA_gain"] > 0 for reference in (
        "H0_FROZEN_DIRECT_C43", "H2_HOST_SIZE_CONTROL", "H3_HOST_BINDING_SHUFFLE",
        "H4_HOST_TOPOLOGY_BINDING_SHUFFLE"))
    status = "HOST_BOUND_CFG_SOURCE_SUPPORTED" if source_supported else "HOST_BOUND_CFG_SOURCE_NOT_SUPPORTED"

    atomic_csv(OUT / "performance_per_manifest_seed.csv", performance)
    atomic_csv(OUT / "performance_seed_mean.csv", means)
    atomic_csv(OUT / "comparisons.csv", comparisons)
    with gzip.open(OUT / "per_apk_predictions.csv.gz", "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(per_apk[0]))
        writer.writeheader()
        writer.writerows(per_apk)
    costs = []
    for split in SOURCE_SPLITS + ("T1",):
        for seed in SEEDS:
            for arm in TRAINED_ARMS:
                checkpoint = load_pickle(OUT / "models" / split / str(seed) / f"{arm}.pkl")
                costs.append({"split": split, "seed": seed, "arm": arm, **checkpoint["cost"],
                              "shuffle_changed_host_fraction": checkpoint["feature_scaler"].get("changed_host_fraction")})
    atomic_csv(OUT / "training_cost.csv", costs)

    manifest = json.loads((OUT / "asset_manifest_before.json").read_text(encoding="utf-8"))
    integrity = verify_assets(manifest)
    atomic_json(OUT / "baseline_integrity_check.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError("protected assets changed")
    atomic_json(OUT / "protocol.json", {
        "status": "FROZEN_HOST_BOUND_CFG_PROBE",
        "training_years": [2017], "evaluation_years": list(YEARS),
        "future_years_are_development": True,
        "stage1": "Manifest135 RF",
        "router": "OTHER3_OP25_REF; Manifest probabilities only",
        "backgrounds": "unchanged 2017-fitted External/Internal/Commonness/Residual",
        "prototype_bank_changed": False,
        "base_Direct_C43_backbone_frozen": True,
        "trainable_module": "identical 16->48->24 CFG adapter plus scalar gate for H1-H4",
        "source_selection_only": True,
        "future_results_used_for_selection": False,
        "rolling_training_used": False,
        "new_APKs_used": False,
        "arms": list(REPORT_ARMS),
    })
    atomic_json(OUT / "final_decision.json", {
        "status": status,
        "source_2017_gate_passed": source_supported,
        "source_comparisons": source,
        "future_years_are_development": True,
        "future_results_used_for_decision": False,
        "formal_model_changed": False,
        "next_action": "Independent expanded-data validation before integration." if source_supported else
                       "Retain H0/G1; Host-bound CFG did not beat every source control.",
    })
    report = [
        "# Manifest135-RF + Host-bound CFG source/future experiment",
        "",
        f"Final source-only decision: **{status}**.",
        "",
        "All adapters, scalers and model parameters were fitted with 2017 only. 2018--2021 are frozen development evaluations and do not affect the decision.",
        "The original Direct-C43 backbone, External/Internal/Commonness backgrounds, Residual and Prototype Bank remain frozen.",
        "",
        "| Year | Arm | Route | BA | AUC | Fixed | Broken | Net |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for year in YEARS:
        for arm in REPORT_ARMS:
            row = lookup[(year, arm)]
            report.append(f"| {year} | {arm} | {row['route_fraction']:.3f} | {row['BA']:.4f} | {row['AUC']:.4f} | {row['Fixed']:.1f} | {row['Broken']:.1f} | {row['Net']:.1f} |")
    report += ["", "## H1 comparisons", ""]
    for row in comparisons:
        if row["scope"] in ("2017", "FUTURE_MACRO_2018_2021"):
            report.append(f"- {row['scope']} vs {row['reference']}: BA {row['BA_gain']:+.5f}, AUC {row['AUC_gain']:+.5f}, Net {row['Net_gain']:+.1f}.")
    report += ["", "Static CFG reachability is possible program structure, not runtime execution or malicious intent.", ""]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("HBCFG_EVALUATION_DONE status=" + status, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "features", "train", "future", "evaluate", "all"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.phase == "preflight":
        preflight()
        return
    if not OUT.is_dir() or not (OUT / "asset_manifest_before.json").is_file():
        raise RuntimeError("run preflight first")
    if args.phase in ("features", "all") and not (OUT / "host_cfg_features.npz").is_file():
        build_feature_cache()
    if args.phase in ("train", "all"):
        fit_source(args.device, args.resume)
    if args.phase in ("future", "all"):
        infer_future(args.device)
    if args.phase in ("evaluate", "all"):
        evaluate()


if __name__ == "__main__":
    main()
