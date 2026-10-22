#!/usr/bin/env python3
"""Strict per-year five-fold OOF M1 Host+Ego/Anomaly/Bounded experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MATRIX = HERE / "within_matrix"
for path in (HERE, MATRIX, MATRIX / "vendor"):
    sys.path.insert(0, str(path))

import run_within_year_arm as within
from host_ego_encoder import ApkEgoMeta, make_fit_adapter, pack
from host_ego_models import anomaly_classes


v4 = within.v4
ROOT = Path("artifacts/care-di25-hostego-within-year-v1")
VERSION = "hostego-anomaly-bounded-sameyear-v1"
PROTECTED = (
    Path("artifacts/care-route-stage2-within-year-matrix-v1/arms/anomaly_bounded/within_year_expert_predictions.csv"),
    Path("artifacts/care-di25-hostego-v1/source/M1_hostego_anomaly_bounded/source_predictions.csv"),
)


def normal(value: object) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_packed(expected: list[str], host_path: Path, ego_path: Path):
    with np.load(host_path, allow_pickle=False) as h, np.load(ego_path, allow_pickle=False) as e:
        ids = [normal(value) for value in h["sha256"].astype(str)]
        eids = [normal(value) for value in e["sha256"].astype(str)]
        lookup = {value: i for i, value in enumerate(ids)}
        elookup = {value: i for i, value in enumerate(eids)}
        if len(lookup) != 1989 or len(elookup) != 1989 or set(lookup) != set(expected) or set(elookup) != set(expected):
            raise RuntimeError("Host/Ego cache cohort differs from same-year cohort")
        host_values = np.asarray(h["host_values"], np.float32)
        host_offsets = np.asarray(h["host_offsets"], np.int64)
        real_count = np.asarray(h["real_host_count"], np.int64)
        ego_offsets = np.asarray(e["ego_offsets"], np.int64)
        status = np.asarray(e["ego_status"], np.int64)
        in_offsets = np.asarray(e["incoming_offsets"], np.int64)
        out_offsets = np.asarray(e["outgoing_offsets"], np.int64)
        in_pos = np.asarray(e["incoming_host_positions"], np.int64)
        out_pos = np.asarray(e["outgoing_host_positions"], np.int64)
        bags, usage = [], []
        duplicate_n = self_n = 0
        for sha in expected:
            i, ei = lookup[sha], elookup[sha]
            if i != ei:
                raise RuntimeError("Host/Ego cache internal order differs: " + sha)
            hb, he = int(host_offsets[i]), int(host_offsets[i + 1])
            eb, ee = int(ego_offsets[ei]), int(ego_offsets[ei + 1])
            host = host_values[hb:he]
            nreal = int(real_count[i])
            if ee - eb != nreal or (nreal and len(host) != nreal) or (not nreal and len(host) != 1):
                raise RuntimeError("missing-region Host/Ego alignment failed: " + sha)
            edge_set: set[tuple[int, int]] = set()
            for center in range(eb, ee):
                incoming = in_pos[in_offsets[center]:in_offsets[center + 1]].astype(int).tolist()
                outgoing = out_pos[out_offsets[center]:out_offsets[center + 1]].astype(int).tolist()
                duplicate_n += len(incoming) - len(set(incoming)) + len(outgoing) - len(set(outgoing))
                self_n += incoming.count(center) + outgoing.count(center)
                if any(value < eb or value >= ee for value in incoming + outgoing):
                    raise RuntimeError("cross-APK Ego neighbor: " + sha)
                edge_set.update((center - eb, value - eb) for value in outgoing)
            bags.append(pack(host, sorted(edge_set)))
            apk_status = status[eb:ee]
            usage.append({
                "sha256": sha,
                "real_host_count": nreal,
                "nontrivial_ego_count": int(np.sum(apk_status == 1)),
                "has_any_nontrivial_ego": int(np.any(apk_status == 1)),
            })
    if duplicate_n or self_n:
        raise RuntimeError(f"Host/Ego cache has duplicate={duplicate_n}, self={self_n}")
    return bags, usage


def parser():
    p = within.future.parser()
    for action in p._actions:
        if action.dest in {"representation", "head", "output"}:
            action.required = False
    p.set_defaults(
        representation="anomaly", head="bounded",
        resolved=Path("artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"),
        future_inter_cache=Path("artifacts/care-route-stage2-future-matrix-v1/shared/future_interz.npz"),
        output=ROOT / "m1_hostego_anomaly_bounded",
    )
    p.add_argument("--router-inputs", type=Path, default=Path(
        "artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"))
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
        raise SystemExit("same-year comparison requires seeds 900000,900001,900002")
    protected_before = {str(path): sha256_file(path) for path in PROTECTED}
    device = v4.resolve_device(args.device)
    data = within.load_data(args)
    data.bags, usage = load_packed(data.ids, args.future_host, args.future_ego)

    _, M1Class = anomaly_classes(within.anomaly_bounded, v4)
    v4.AUGMENTED_REGION_DIM = within.future.ORIGINAL_AUGMENTED_DIM
    v4.RoleGraphConditionedEvidence = M1Class
    v4.v3.fit_scaled_dualproto_bags = make_fit_adapter(within.future.ORIGINAL_ANOMALY_FIT)
    v4.predict_model = within.future.ORIGINAL_V4_PREDICT

    args.output.mkdir(parents=True, exist_ok=True)
    bounded = np.full(1989, np.nan, dtype=np.float64)
    direct = np.full(1989, np.nan, dtype=np.float64)
    checkpoint = args.output / "within_year_m1_checkpoint.csv"
    audit_checkpoint = args.output / "fit_audit_checkpoint.csv"
    audits = read_csv(audit_checkpoint) if audit_checkpoint.is_file() else []
    if checkpoint.is_file():
        saved = read_csv(checkpoint)
        by_sha = {normal(row["sha256"]): row for row in saved}
        if len(saved) != len(by_sha):
            raise RuntimeError("duplicate SHA in M1 checkpoint")
        versions = {row.get("version", "") for row in saved}
        if versions == {VERSION}:
            for i, sha in enumerate(data.ids):
                if sha in by_sha:
                    bounded[i] = float(by_sha[sha]["p_expert_bounded"])
                    direct[i] = float(by_sha[sha]["p_expert_direct_secondary"])
            print(f"[resume] restored {int(np.sum(np.isfinite(bounded)))}/1989 M1 predictions", flush=True)
        else:
            print("[invalidate] M1 checkpoint version differs", flush=True)
            audits = []

    started = time.perf_counter()
    for year in sorted(set(data.years.tolist())):
        for fold in range(1, 6):
            validation = np.flatnonzero((data.years == year) & (data.folds == fold))
            train = np.flatnonzero((data.years == year) & (data.folds != fold))
            if len(validation) == 0 or len(np.unique(data.labels[train])) != 2:
                raise RuntimeError(f"invalid same-year M1 split: {year} fold {fold}")
            if np.isfinite(bounded[validation]).all() and np.isfinite(direct[validation]).all():
                print(f"[resume] M1 {year} fold {fold}/5 already complete", flush=True)
                continue
            x_train, x_validation = within.fold_global24(data, train, validation)
            base_train = within.inner_oof_base(
                data, train, int(args.split_seed) + year * 10 + fold)
            probability, diagnostic, fold_audits = v4.fit_predict_variant(
                x_train, x_validation,
                data.s7[train], data.s7[validation],
                within.subset_bags(data.bags, train), within.subset_bags(data.bags, validation),
                base_train, data.base[validation], data.labels[train],
                "RGCondAttn", seeds, args, device,
            )
            probability = np.asarray(probability, dtype=np.float64)
            diagnostic = np.asarray(diagnostic, dtype=np.float64)
            if probability.shape != (len(validation),) or diagnostic.shape[0] != len(validation):
                raise RuntimeError(f"incomplete M1 fold output: {year}/{fold}")
            counts = {int(row["parameter_n"]) for row in fold_audits}
            if counts != {34613}:
                raise RuntimeError(f"M1 parameter count changed in {year}/{fold}: {counts}")
            bounded[validation] = probability
            direct[validation] = diagnostic[:, -1]
            audits.extend({"year": year, "outer_fold": fold, **row} for row in fold_audits)

            complete = np.flatnonzero(np.isfinite(bounded) & np.isfinite(direct))
            rows = [{
                "sha256": data.ids[i], "year": int(data.years[i]), "fold": int(data.folds[i]),
                "p_expert_bounded": float(bounded[i]),
                "p_expert_direct_secondary": float(direct[i]), "version": VERSION,
            } for i in complete]
            audit_temporary = audit_checkpoint.with_suffix(".tmp")
            write_csv(audit_temporary, audits)
            audit_temporary.replace(audit_checkpoint)
            temporary = checkpoint.with_suffix(".tmp")
            write_csv(temporary, rows)
            temporary.replace(checkpoint)
            print(f"M1 same-year {year} fold {fold}/5 done", flush=True)

    elapsed = time.perf_counter() - started
    if not np.isfinite(bounded).all() or not np.isfinite(direct).all():
        raise RuntimeError("incomplete same-year M1 OOF predictions")
    audit_parameter_counts = {int(float(row["parameter_n"])) for row in audits if row.get("parameter_n") not in (None, "")}
    if audit_parameter_counts != {34613}:
        raise RuntimeError(f"incomplete or changed M1 parameter audit: {audit_parameter_counts}")
    rows = [{
        "sha256": data.ids[i], "year": int(data.years[i]), "label": int(data.labels[i]),
        "fold": int(data.folds[i]), "p_expert_bounded": float(bounded[i]),
        "p_expert_direct_secondary": float(direct[i]),
        "real_host_count": int(usage[i]["real_host_count"]),
        "nontrivial_ego_count": int(usage[i]["nontrivial_ego_count"]),
        "has_any_nontrivial_ego": int(usage[i]["has_any_nontrivial_ego"]),
        "same_year_oof": True, "version": VERSION,
    } for i in range(1989)]
    write_csv(args.output / "within_year_m1_predictions.csv", rows)
    write_csv(args.output / "fit_audit.csv", audits)
    write_csv(args.output / "ego_usage.csv", usage)
    protected_after = {str(path): sha256_file(path) for path in PROTECTED}
    if protected_before != protected_after:
        raise RuntimeError("locked M0/source-M1 prediction changed during same-year M1 run")
    protocol = {
        "method": "DI25 + HostEgo + Anomaly + Bounded (M1)",
        "protocol": "independent per-year five-fold OOF",
        "years": [2018, 2019, 2020, 2021],
        "temporal_generalization_result": False,
        "labels_used_only_within_each_year_outer_train": True,
        "outer_validation_labels_used_for_training_or_epoch_selection": False,
        "M0_retrained": False,
        "max_delta_logit": 2.25,
        "seeds": seeds,
        "parameter_count": 34613,
        "training_seconds_this_run": elapsed,
        "protected_prediction_sha256_before": protected_before,
        "protected_prediction_sha256_after": protected_after,
        "feature_contract_version": "source-frozen-compact76-foldsafe-v1",
        "version": VERSION,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DI25_HOSTEGO_WITHIN_YEAR_M1_DONE seconds={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
