#!/usr/bin/env python3
"""Fit frozen M1 on 2017 only and emit label-free 2018--2021 predictions."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "future_matrix"
for path in (HERE, MATRIX, MATRIX / "vendor"):
    sys.path.insert(0, str(path))

import run_future_arm as matrix
from host_ego_encoder import ApkEgoMeta, make_fit_adapter, pack
from host_ego_models import anomaly_classes

v4 = matrix.v4
ROOT = Path("artifacts/care-di25-hostego-future-v1")
M0_SOURCE = Path("artifacts/care-di25-hostego-v1/source/M0_host_anomaly_bounded/source_predictions.csv")
M1_SOURCE = Path("artifacts/care-di25-hostego-v1/source/M1_hostego_anomaly_bounded/source_predictions.csv")


def normal(value: object) -> str:
    return str(value).strip().lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_packed(rows: list[dict], host_path: Path, ego_path: Path):
    expected = [normal(row["sha256"]) for row in rows]
    with np.load(host_path, allow_pickle=False) as h, np.load(ego_path, allow_pickle=False) as e:
        ids = [normal(value) for value in h["sha256"].astype(str)]
        eids = [normal(value) for value in e["sha256"].astype(str)]
        if ids != expected or eids != expected:
            raise RuntimeError(f"Host/Ego cache order differs: {host_path} / {ego_path}")
        host_values = np.asarray(h["host_values"], np.float32)
        host_offsets = np.asarray(h["host_offsets"], np.int64)
        real_count = np.asarray(h["real_host_count"], np.int64)
        ego_offsets = np.asarray(e["ego_offsets"], np.int64)
        host_ids = e["host_ids"].astype(str)
        roles = np.asarray(e["host_primary_role"], np.int64)
        status = np.asarray(e["ego_status"], np.int64)
        in_offsets = np.asarray(e["incoming_offsets"], np.int64)
        out_offsets = np.asarray(e["outgoing_offsets"], np.int64)
        in_pos = np.asarray(e["incoming_host_positions"], np.int64)
        out_pos = np.asarray(e["outgoing_host_positions"], np.int64)
        bags, metadata, audit = [], [], []
        duplicate_n = self_n = 0
        for i, sha in enumerate(expected):
            hb, he = int(host_offsets[i]), int(host_offsets[i+1])
            eb, ee = int(ego_offsets[i]), int(ego_offsets[i+1])
            host = host_values[hb:he]
            nreal = int(real_count[i])
            if ee-eb != nreal or (nreal and len(host) != nreal) or (not nreal and len(host) != 1):
                raise RuntimeError("Host/Ego missing-region alignment failed: " + sha)
            incoming, outgoing, edges = [], [], set()
            for center in range(eb, ee):
                inc_abs = in_pos[in_offsets[center]:in_offsets[center+1]].astype(int).tolist()
                out_abs = out_pos[out_offsets[center]:out_offsets[center+1]].astype(int).tolist()
                duplicate_n += len(inc_abs)-len(set(inc_abs)) + len(out_abs)-len(set(out_abs))
                self_n += inc_abs.count(center) + out_abs.count(center)
                inc = [value-eb for value in inc_abs]
                out = [value-eb for value in out_abs]
                # Neighbor positions are absolute rows in the ragged Ego host
                # arrays. Validate against this APK slice before converting
                # them to packed-bag-local coordinates.
                if any(value < eb or value >= ee for value in inc_abs + out_abs):
                    raise RuntimeError("cross-APK Ego neighbor: " + sha)
                incoming.append(inc); outgoing.append(out)
                edges.update((center-eb, value) for value in out)
            bags.append(pack(host, sorted(edges)))
            meta = ApkEgoMeta(
                sha,
                host_ids[eb:ee].tolist() if nreal else ["__missing_host__"],
                roles[eb:ee].astype(int).tolist() if nreal else [-1],
                incoming if nreal else [[]], outgoing if nreal else [[]],
                status[eb:ee].astype(int).tolist() if nreal else [0],
                [1] * nreal if nreal else [1], [0] * nreal if nreal else [0], nreal,
            )
            metadata.append(meta)
            audit.append({"sha256": sha, "real_host_count": nreal,
                          "nontrivial_ego_count": int(sum(meta.status)),
                          "has_any_nontrivial_ego": int(any(meta.status))})
    if duplicate_n or self_n:
        raise RuntimeError(f"future Host/Ego cache has duplicate={duplicate_n}, self={self_n}")
    return bags, metadata, audit


def parser():
    p = matrix.parser()
    for action in p._actions:
        if action.dest in {"representation", "head", "output"}:
            action.required = False
    p.set_defaults(representation="anomaly", head="bounded",
                   output=ROOT / "m1_hostego_anomaly")
    p.add_argument("--source-host", type=Path, default=Path(
        "artifacts/care-source-local-assets-v2/host_regions_2017.npz"))
    p.add_argument("--source-ego", type=Path, default=Path(
        "artifacts/care-source-local-assets-v2/ego_1hop_regions_2017.npz"))
    p.add_argument("--future-host", type=Path, default=Path(
        "artifacts/care-future-local-assets-v2/host_regions_2018_2021.npz"))
    p.add_argument("--future-ego", type=Path, default=Path(
        "artifacts/care-future-local-assets-v2/ego_1hop_regions_2018_2021.npz"))
    return p


def main() -> None:
    args = parser().parse_args()
    if args.representation != "anomaly" or args.head != "bounded":
        raise SystemExit("frozen M1 requires anomaly/bounded")
    if float(args.max_delta) != 2.25:
        raise SystemExit("frozen M1 requires max_delta=2.25")
    seeds = v4.parse_ints(args.seeds)
    if seeds != [900000, 900001, 900002]:
        raise SystemExit("frozen future deployment seeds must be 900000,900001,900002")
    protected_before = {str(path): sha256_file(path) for path in (M0_SOURCE, M1_SOURCE)}
    data = matrix.load_common(args)
    source_bags, _, source_usage = load_packed(data.source_rows, args.source_host, args.source_ego)
    target_bags, _, target_usage = load_packed(data.target_rows, args.future_host, args.future_ego)

    x_source, x_target, feature_names = v4.audit.compact_matrix_variant(
        v4.VARIANT, data.source_rows, data.target_rows,
        data.compact_by_sha, data.schema, data.names)
    global_index, _ = v4.evidence.split_view_indices(feature_names)
    global_source = np.asarray(x_source[:, global_index], dtype=np.float32)
    global_target = np.asarray(x_target[:, global_index], dtype=np.float32)
    if global_source.shape != (799, 24) or global_target.shape != (1989, 24):
        raise RuntimeError("Global24 feature contract failed")

    _, M1Class = anomaly_classes(matrix.anomaly_bounded, v4)
    v4.AUGMENTED_REGION_DIM = matrix.ORIGINAL_AUGMENTED_DIM
    v4.RoleGraphConditionedEvidence = M1Class
    v4.v3.fit_scaled_dualproto_bags = make_fit_adapter(matrix.ORIGINAL_ANOMALY_FIT)
    device = v4.resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bounded, diagnostic, audits = v4.fit_predict_variant(
        global_source, global_target, data.source_s7, data.target_s7,
        source_bags, target_bags, data.base_source, data.base_target,
        data.source_y, "RGCondAttn", seeds, args, device)
    elapsed = time.perf_counter() - started
    bounded = np.asarray(bounded, dtype=np.float64)
    diagnostic = np.asarray(diagnostic, dtype=np.float64)
    if bounded.shape != (1989,) or diagnostic.shape[0] != 1989:
        raise RuntimeError("M1 future output is incomplete")
    direct = diagnostic[:, -1]
    if not np.isfinite(bounded).all() or not np.isfinite(direct).all():
        raise RuntimeError("M1 future output contains non-finite values")
    parameter_counts = sorted({int(row["parameter_n"]) for row in audits})
    if parameter_counts != [34613]:
        raise RuntimeError(f"M1 parameter count changed: {parameter_counts}")

    rows = []
    for i, item in enumerate(data.future_input):
        rows.append({
            "sha256": normal(item["sha256"]), "year": int(item["year"]),
            "p_expert_bounded": float(bounded[i]),
            "p_expert_direct_same_correction_head": float(direct[i]),
            "real_host_count": int(target_usage[i]["real_host_count"]),
            "nontrivial_ego_count": int(target_usage[i]["nontrivial_ego_count"]),
            "has_any_nontrivial_ego": int(target_usage[i]["has_any_nontrivial_ego"]),
            "diagnostic_only": True,
        })
    write_csv(args.output / "future_m1_predictions_label_free.csv", rows)
    write_csv(args.output / "fit_audit.csv", audits)
    write_csv(args.output / "source_ego_usage.csv", source_usage)
    write_csv(args.output / "future_ego_usage.csv", target_usage)
    protected_after = {str(path): sha256_file(path) for path in (M0_SOURCE, M1_SOURCE)}
    if protected_before != protected_after:
        raise RuntimeError("locked source M0/M1 prediction changed during future fit")
    protocol = {
        "method": "DI25 + HostEgo + Anomaly + Bounded (M1)",
        "source_train_n": 799, "future_n": 1989,
        "source_only_training": True,
        "future_labels_loaded_by_model": False,
        "future_labels_used_for_training_routing_selection_or_early_stopping": False,
        "router": "pre-existing label-free DI25 YearTop25",
        "max_delta_logit": 2.25,
        "seeds": seeds,
        "parameter_count": 34613,
        "training_seconds_including_epoch_selection": elapsed,
        "M0_retrained": False,
        "source_predictions_modified": False,
        "direct_output": "parameter-free sigmoid(raw correction head); secondary diagnostic only",
        "protected_source_sha256_before": protected_before,
        "protected_source_sha256_after": protected_after,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M1_FUTURE_LABEL_FREE_PREDICTIONS_FROZEN seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
