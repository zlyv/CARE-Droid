#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from audit_common import *
from transforms import apply_record


FROZEN_MULTISCALE = {
    "null_samples": 64,
    "swaps_per_edge": 10,
    "null_seed": 20260827,
    "z_clip": 5.0,
    "progress_every": 100,
}


def _builder_defaults(module):
    path = Path(module.__file__).resolve(strict=True)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    flags = {
        "--null-samples": "null_samples",
        "--swaps-per-edge": "swaps_per_edge",
        "--null-seed": "null_seed",
        "--z-clip": "z_clip",
        "--progress-every": "progress_every",
    }
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        flag_node = node.args[0]
        if not isinstance(flag_node, ast.Constant) or flag_node.value not in flags:
            continue
        defaults = [keyword.value for keyword in node.keywords if keyword.arg == "default"]
        if len(defaults) != 1:
            raise RuntimeError("BLOCKED_BY_CONTRACT: builder default is not explicit for " + str(flag_node.value))
        values[flags[flag_node.value]] = ast.literal_eval(defaults[0])
    missing = sorted(set(flags.values()) - set(values))
    if missing:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen builder defaults missing " + repr(missing))
    return path, source, values


def install_frozen_multiscale_args(v4, args):
    """Restore builder-only parameters omitted by the frozen inference CLI."""
    builder_path, builder_source, builder_values = _builder_defaults(v4.v3.future_s7.clipped)
    builder_checks = {
        name: float(builder_values[name]) == float(expected)
        for name, expected in FROZEN_MULTISCALE.items()
    }
    if not all(builder_checks.values()):
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen builder protocol mismatch " + repr({
            "observed": builder_values, "expected": FROZEN_MULTISCALE,
        }))
    audit_path = Path(args.source_multiscale_features).with_name("audit.json")
    metadata_checks = None
    if audit_path.is_file():
        metadata = json.loads(audit_path.read_text(encoding="utf-8"))
        null_model = metadata.get("null_model") or {}
        recorded_samples = int(null_model.get("null_samples_per_apk", -1))
        recorded_clip = list(null_model.get("z_clip") or [])
        recorded_swaps = str(null_model.get("successful_swaps_target_per_sample", ""))
        metadata_checks = {
            "null_samples": recorded_samples == FROZEN_MULTISCALE["null_samples"],
            "swaps_per_edge": recorded_swaps.startswith(str(FROZEN_MULTISCALE["swaps_per_edge"]) + " x "),
            "z_clip": recorded_clip == [-FROZEN_MULTISCALE["z_clip"], FROZEN_MULTISCALE["z_clip"]],
        }
        if not all(metadata_checks.values()):
            raise RuntimeError("BLOCKED_BY_CONTRACT: frozen multiscale metadata mismatch " + repr(metadata_checks))
    for name, expected in FROZEN_MULTISCALE.items():
        if hasattr(args, name) and name != "progress_every":
            actual = getattr(args, name)
            if float(actual) != float(expected):
                raise RuntimeError(f"BLOCKED_BY_CONTRACT: {name} changed: {actual!r} != {expected!r}")
        setattr(args, name, expected)
    dump_json(OUTPUT / "transformed_s7_protocol.json", {
        "status": "FROZEN",
        "source_multiscale_features": str(Path(args.source_multiscale_features).resolve()),
        "source_audit": str(audit_path.resolve()),
        "source_audit_available": audit_path.is_file(),
        "builder_path": str(builder_path),
        "builder_sha256": hashlib.sha256(builder_source.encode("utf-8")).hexdigest(),
        "parameters": FROZEN_MULTISCALE,
        "builder_defaults": builder_values,
        "builder_checks": builder_checks,
        "metadata_checks": metadata_checks,
        "labels_used": False,
        "future_rows_loaded": 0,
    })


class AuditState:
    def __init__(self, v4, ids, folds, route, variant_s7, repeats, output):
        self.v4 = v4; self.ids = ids; self.folds = folds; self.route = route
        self.variant_s7 = variant_s7; self.repeats = repeats; self.output = output
        self.next_fold = 1
        self.predictions = {(name, repeat): np.full(len(ids), np.nan) for name in TRANSFORMS for repeat in range(repeats)}
        self.original = np.full(len(ids), np.nan)

    def predict(self, model, global_values, base_probability, graph_values, bags, device):
        v4 = self.v4; torch = v4.torch
        global_tensor = torch.as_tensor(global_values, dtype=torch.float32, device=device)
        graph = v4.graph_tensors(graph_values, device)
        bag_tensor = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in bags]
        probability = []; model.eval()
        with torch.no_grad():
            for index in range(len(bags)):
                one = {key: value[index] for key, value in graph.items()}
                logit, _, _ = model.forward_with_trace(global_tensor[index], torch.as_tensor(v4.probability_to_logit(base_probability[index]), dtype=torch.float32, device=device), one, bag_tensor[index])
                probability.append(float(torch.sigmoid(logit).cpu()))
        return np.asarray(probability)

    @staticmethod
    def permute_bags(bags, ids, repeat):
        output = []
        for bag, sid in zip(bags, ids):
            bag = np.asarray(bag)
            raw = __import__("hashlib").sha256(f"{sid}|bag-order|{repeat}|20261012".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(raw[:8], "little"))
            output.append(bag[rng.permutation(len(bag))] if len(bag) else bag)
        return output

    def fit_predict(self, global_train, global_test, s7_train, s7_test, raw_train_bags, raw_test_bags,
                    base_train, base_test, labels, variant, seeds, args, device):
        v4 = self.v4; torch = v4.torch; fold = self.next_fold; self.next_fold += 1
        train_index = np.flatnonzero(self.folds != fold); test_index = np.flatnonzero(self.folds == fold)
        if len(test_index) != len(global_test) or len(train_index) != len(global_train):
            raise RuntimeError("BLOCKED_BY_CONTRACT: fold order mismatch")
        original_seed = []; transformed_seed = {(name, repeat): [] for name in TRANSFORMS for repeat in range(self.repeats)}
        scaler = v4.fit_role_graph_scaler(s7_train)
        original_graph = v4.transform_role_graphs(s7_test, scaler)
        for seed in seeds:
            selected_epochs, _ = v4.select_epochs(global_train, s7_train, raw_train_bags, base_train, labels, variant, args, device, int(seed))
            train_bags, test_bags, _ = v4.v3.fit_scaled_dualproto_bags(raw_train_bags, raw_test_bags, labels, args, int(seed) + 1707)
            train_graph = v4.transform_role_graphs(s7_train, scaler)
            global_tensor = torch.as_tensor(global_train, dtype=torch.float32, device=device)
            train_graph_tensor = v4.graph_tensors(train_graph, device)
            train_bag_tensor = [torch.as_tensor(value, dtype=torch.float32, device=device) for value in train_bags]
            weights, correct = v4.fixkeep_weights(labels, base_train, args); v4.seed_everything(seed)
            model = v4.RoleGraphConditionedEvidence(global_train.shape[1], variant, args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
            v4.train_epochs(model, optimizer, global_tensor, base_train, train_graph_tensor, train_bag_tensor, labels, weights, correct, selected_epochs, args, int(seed) + 3301)
            original_seed.append(self.predict(model, global_test, base_test, original_graph, test_bags, device))
            for name in TRANSFORMS:
                for repeat in range(self.repeats):
                    transformed_graph = v4.transform_role_graphs(self.variant_s7[(name, repeat)][test_index], scaler)
                    bags = self.permute_bags(test_bags, self.ids[test_index], repeat) if name == "T0_ORDER" else test_bags
                    transformed_seed[(name, repeat)].append(self.predict(model, global_test, base_test, transformed_graph, bags, device))
            del model, train_graph_tensor, train_bag_tensor
            if device.type == "cuda": torch.cuda.empty_cache()
        original = np.mean(np.asarray(original_seed), axis=0); self.original[test_index] = original
        for key, values in transformed_seed.items(): self.predictions[key][test_index] = np.mean(np.asarray(values), axis=0)
        # Preserve the direct-model runner's expected return shape.
        diagnostic = np.zeros((len(test_index), len(v4.DIAGNOSTIC_NAMES)), dtype=float)
        return original, diagnostic, [{"equivalence_proxy_audit": True, "fold": fold}]

    def save(self):
        payload = {"ids": self.ids, "folds": self.folds, "route": self.route.astype(np.uint8), "original": self.original}
        for (name, repeat), value in self.predictions.items(): payload[f"{name}__r{repeat}"] = value
        np.savez_compressed(self.output / "proxy_predictions.npz", **payload)


def transformed_s7(v4, rows, cache, args, route, original_values):
    """Rebuild transformed S7 while retaining the frozen null-model protocol."""
    raw, _ = v4.v3.audit.future.reconstruct_structure(
        rows, cache, int(v4.v3.audit.exp.ROLE_VOCABULARY_SIZE)
    )
    if raw.shape != (len(rows), 585):
        raise RuntimeError("bad transformed source S3 shape: " + repr(raw.shape))
    selected = np.flatnonzero(route)
    selected_rows = [rows[int(index)] for index in selected]
    pair_raw = np.asarray(raw[selected, 120:], dtype=np.float64)
    _, _, inter_z, _ = v4.v3.future_s7.clipped.build_multiscale_features(
        selected_rows, cache, pair_raw, args
    )
    inter_clip = np.clip(inter_z, -float(args.z_clip), float(args.z_clip))
    result = np.asarray(original_values, dtype=np.float64).copy()
    result[selected] = np.concatenate([raw[selected], inter_clip], axis=1)
    if result.shape != (len(rows), 1546) or not np.isfinite(result).all():
        raise RuntimeError("bad transformed source S7: " + repr(result.shape))
    return np.asarray(result, np.float32)


def build_variant_s7(v4, rows, endpoint_cache, args, ids, folds, route, repeats, original_values):
    result = {}; edits = []
    checkpoint_dir = OUTPUT / "checkpoints" / "variant_s7_payload_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    row_ids = np.asarray([norm(row.get("sha256", row.get("sample_id", ""))) for row in rows])
    if len(row_ids) != len(ids) or not np.array_equal(row_ids, ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: source row order differs from frozen prediction order")
    for name in TRANSFORMS:
        for repeat in range(repeats):
            checkpoint = checkpoint_dir / f"{name}__r{repeat}.npy"
            edit_checkpoint = checkpoint_dir / f"{name}__r{repeat}.edits.json"
            if checkpoint.exists() and edit_checkpoint.exists():
                values = np.load(checkpoint, allow_pickle=False)
                cached_edits = json.loads(edit_checkpoint.read_text(encoding="utf-8"))
                if len(values) != len(ids) or len(cached_edits) != int(np.sum(route)):
                    raise RuntimeError("BLOCKED_BY_CONTRACT: invalid transformed-S7 checkpoint " + str(checkpoint))
                result[(name, repeat)] = np.asarray(values, np.float32)
                edits.extend(cached_edits)
                print(f"CESA_TRANSFORM_RESUME name={name} repeat={repeat}", flush=True)
                continue
            cache = dict(endpoint_cache)
            one_edits = []
            for index in np.flatnonzero(route):
                sid = ids[index]; original = endpoint_cache.get(sid)
                if original is None: raise RuntimeError("BLOCKED_BY_CONTRACT: endpoint missing " + sid)
                transformed, audit = apply_record(original, sid, name, repeat)
                cache[sid] = transformed
                one_edits.append({"sha256": sid, "fold": int(folds[index]), "transform": name, "repeat": repeat, **audit})
            values = transformed_s7(v4, rows, cache, args, route, original_values)
            np.save(checkpoint, values, allow_pickle=False)
            edit_checkpoint.write_text(json.dumps(one_edits, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            result[(name, repeat)] = values
            edits.extend(one_edits)
            print(f"CESA_TRANSFORM_READY name={name} repeat={repeat}", flush=True)
    write_csv(OUTPUT / "transformation_edits.csv", edits)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda"); parser.add_argument("--jobs", type=int, default=4); parser.add_argument("--repeats", type=int, default=3)
    cli = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(FAMILY_SCRIPT)); sys.path.insert(0, str(DIRECT_SCRIPT))
    from common import ENDPOINT_SOURCE, SOURCE_A0_CACHE
    from model_adapter import install_a0, prepare_imports
    from nested_stage1 import NestedStage1
    from model_variants import install_variant
    prepare_imports()
    stage1_script = SCRIPTS / "care-p31-stage1-rf-selection-source-v1"
    if str(stage1_script) not in sys.path: sys.path.insert(0, str(stage1_script))
    from source_label_contract import install_source_label_contract
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_source_label_contract(v4); install_a0(v4, REPLAY_METHOD); install_variant(v4, "U2_DIRECT_STAGE2")
    ids, y, folds, base, route, frozen = load_reference()
    holder = {}
    original_s7_loader = v4.v3.load_source_s7
    def capture_s7(rows, endpoint_cache, args):
        values, audit = original_s7_loader(rows, endpoint_cache, args)
        if not holder:
            install_frozen_multiscale_args(v4, args)
            holder["variant_s7"] = build_variant_s7(
                v4, rows, endpoint_cache, args, ids, folds, route, cli.repeats,
                values,
            )
        return values, audit
    v4.v3.load_source_s7 = capture_s7
    nested = NestedStage1("SVM", OUTPUT / "audits/nested.json", cli.jobs); v4.inner_oof_global24 = nested.inner_oof
    # Delay state creation until transformed S7 has been generated inside main().
    original_fit = v4.fit_predict_variant
    state_holder = {}
    def fit_proxy(*args, **kwargs):
        if "state" not in state_holder:
            state_holder["state"] = AuditState(v4, ids, folds, route, holder["variant_s7"], cli.repeats, OUTPUT)
        return state_holder["state"].fit_predict(*args, **kwargs)
    v4.fit_predict_variant = fit_proxy
    out = OUTPUT / "replay"
    sys.argv = [sys.argv[0], "--previous-result", str(FAMILY_OUTPUT / "references/SVM"), "--cache", str(ENDPOINT_SOURCE),
                "--region-cache", str(SOURCE_A0_CACHE), "--variants", "RGCondAttn", "--seeds", "0,1,2", "--max-delta", "2.25",
                "--device", cli.device, "--output", str(out), "--keep-weight", "1", "--fix-weight", "1", "--keep-consistency", "0"]
    v4.main(); state = state_holder["state"]; state.save()
    if state.next_fold != 6 or len(nested.calls) != 5: raise RuntimeError("BLOCKED_BY_CONTRACT: incomplete five-fold replay")
    max_abs = float(np.max(np.abs(state.original - frozen)))
    original_pred = state.original >= .5; frozen_pred = frozen >= .5
    checks = {"folds==5": state.next_fold == 6, "nested_calls==5": len(nested.calls) == 5,
              "routed_decisions_exact": bool(np.array_equal(original_pred[route], frozen_pred[route])), "max_abs<=1e-4": max_abs <= 1e-4}
    dump_json(OUTPUT / "replay_validation.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "max_abs": max_abs})
    if not all(checks.values()): raise RuntimeError("BLOCKED_BY_CONTRACT: frozen Direct-C43 replay mismatch " + repr(checks))
    print("C43_EQUIVALENCE_PROXY_REPLAY_DONE", flush=True)


if __name__ == "__main__": main()
