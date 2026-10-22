#!/usr/bin/env python3
"""Run the three predeclared Host-complement arms; never retrain locked M0."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from audit_common import (
    ARM_NAMES, ASSET_MANIFEST, EGO, ENDPOINT, HOST, ROOT, ROUTER,
    hash_manifest, load_m1_modules, normal, protected_paths, read_csv,
    router_contract, write_csv, write_json,
)
from host_complement_model import make_model_class
from host_complement_training import (
    POSITION_DIM, RAW_DIM, load_extended_bags, make_fit_adapter, unpack,
)


VERSION = "host-complement-neural-v1"
DIAG_NAMES = [
    "bounded_delta", "critical_score", "mean_score", "attention_entropy",
    "max_attention", "context_entropy", "prototype_mean_1",
    "prototype_mean_2", "prototype_mean_3", "prototype_max_residual",
    "graph_entropy", "raw_delta_sigmoid", "mean_alpha", "median_alpha",
    "max_alpha", "mean_beta", "median_beta", "max_beta",
    "multi_role_host_fraction", "noego_fallback_max_abs_diff",
]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("dry-run", "full"), required=True)
    p.add_argument("--arm", choices=("A", "B", "AB", "all"), default="all")
    p.add_argument("--dry-year", type=int, default=2018)
    p.add_argument("--dry-fold", type=int, default=1)
    p.add_argument("--dry-seed", type=int, default=900000)
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--endpoint", type=Path, default=ENDPOINT)
    p.add_argument("--asset-manifest", type=Path, default=ASSET_MANIFEST)
    p.add_argument("--host", type=Path, default=HOST)
    p.add_argument("--ego", type=Path, default=EGO)
    return p


def configure(m1, arm: str, role_count: int):
    within, v4 = m1.within, m1.v4
    from host_ego_encoder import DirectedHostEgoFusion
    Base = within.anomaly_bounded.DedupBoundedRGCondAttn
    cls = make_model_class(
        Base, DirectedHostEgoFusion, role_count,
        within.future.ORIGINAL_AUGMENTED_DIM, arm,
    )
    v4.AUGMENTED_REGION_DIM = within.future.ORIGINAL_AUGMENTED_DIM
    v4.RoleGraphConditionedEvidence = cls
    v4.v3.fit_scaled_dualproto_bags = make_fit_adapter(
        within.future.ORIGINAL_ANOMALY_FIT, role_count)
    v4.predict_model = within.future.ORIGINAL_V4_PREDICT
    return cls


def run_fold(m1, data, bags, args, train, validation, arm, seeds, device, fold_seed):
    within, v4 = m1.within, m1.v4
    x_train, x_validation = within.fold_global24(data, train, validation)
    base_train = within.inner_oof_base(data, train, fold_seed)
    configure(m1, arm, bags.role_count)
    probability, diagnostic, audits = v4.fit_predict_variant(
        x_train, x_validation, data.s7[train], data.s7[validation],
        within.subset_bags(bags.values, train), within.subset_bags(bags.values, validation),
        base_train, data.base[validation], data.labels[train],
        "RGCondAttn", seeds, args, device,
    )
    probability = np.asarray(probability, float)
    diagnostic = np.asarray(diagnostic, float)
    if probability.shape != (len(validation),) or diagnostic.shape != (len(validation), len(DIAG_NAMES)):
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: unexpected fold output {probability.shape}/{diagnostic.shape}")
    if not np.isfinite(probability).all() or not np.isfinite(diagnostic).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: non-finite model output")
    if np.max(np.abs(diagnostic[:, 0])) > 2.25 + 1e-6:
        raise RuntimeError("BLOCKED_BY_CONTRACT: bounded delta exceeds 2.25")
    return probability, diagnostic, audits


class Bags:
    def __init__(self, values, usage, roles):
        self.values, self.usage, self.roles = values, usage, roles
        self.role_count = len(roles)


def load_all(m1, cli):
    args = m1.parser().parse_args([])
    if float(args.max_delta) != 2.25:
        raise RuntimeError("BLOCKED_BY_CONTRACT: max_delta is not 2.25")
    if m1.v4.parse_ints(args.seeds) != [900000, 900001, 900002]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: M1 seed lineage changed")
    data = m1.within.load_data(args)
    route_rows = read_csv(ROUTER)
    route = router_contract(route_rows)
    if not route["passed"] or [normal(r["sha256"]) for r in route_rows] != list(data.ids):
        raise RuntimeError("BLOCKED_BY_CONTRACT: route/data lineage mismatch")
    values, usage, roles = load_extended_bags(
        list(data.ids), cli.host, cli.ego, cli.endpoint, cli.asset_manifest)
    return args, data, route_rows, Bags(values, usage, roles)


def dry_run(cli, m1, args, data, routes, bags, before):
    if cli.dry_year not in (2018, 2019, 2020, 2021) or cli.dry_fold not in range(1, 6) or cli.dry_seed != 900000:
        raise RuntimeError("BLOCKED_BY_CONTRACT: dry-run must use fixed 2018-2021/fold1-5/seed900000 protocol lineage")
    validation_all = np.flatnonzero((data.years == cli.dry_year) & (data.folds == cli.dry_fold))
    train_all = np.flatnonzero((data.years == cli.dry_year) & (data.folds != cli.dry_fold))
    # A deterministic, stratified small subset is used only to prove trainability.
    train = np.concatenate([np.flatnonzero(data.labels[train_all] == c)[:16] for c in (0, 1)])
    train = train_all[train]
    validation = np.concatenate([np.flatnonzero(data.labels[validation_all] == c)[:4] for c in (0, 1)])
    validation = validation_all[validation]
    old_epochs = args.max_epochs; old_patience = args.early_stop_patience
    args.max_epochs = 2; args.early_stop_patience = 1
    device = m1.v4.resolve_device(args.device)
    output = {}
    for arm in ("A", "B", "AB"):
        p, diag, audits = run_fold(
            m1, data, bags, args, train, validation, arm, [cli.dry_seed], device,
            int(args.split_seed) + cli.dry_year * 10 + cli.dry_fold,
        )
        counts = sorted({int(a["parameter_n"]) for a in audits})
        output[arm] = {
            "train_n": len(train), "validation_n": len(validation),
            "parameter_counts": counts, "probability_finite": bool(np.isfinite(p).all()),
            "diagnostic_finite": bool(np.isfinite(diag).all()),
            "max_abs_delta": float(np.max(np.abs(diag[:, 0]))),
            "finite_training_loss": all(np.isfinite(float(a["final_training_loss"])) for a in audits),
        }
    args.max_epochs = old_epochs; args.early_stop_patience = old_patience
    # The exact raw rows and Host edges survive the extended pack/unpack round trip.
    raw_checks = []
    for index in validation[:min(8, len(validation))]:
        raw, _, _, edges = unpack(bags.values[int(index)], RAW_DIM, bags.role_count)
        raw_checks.append(bool(np.array_equal(raw[:, :RAW_DIM], raw)))
    after = hash_manifest(protected_paths())
    audit = {
        "passed": all(x["probability_finite"] and x["diagnostic_finite"] and x["finite_training_loss"] and x["max_abs_delta"] <= 2.25 + 1e-6 for x in output.values()) and all(raw_checks) and before == after,
        "year": cli.dry_year, "fold": cli.dry_fold, "seed": cli.dry_seed,
        "minimal_training_only": True, "full_protocol_unchanged": True,
        "arms": output, "input_host_rows_roundtrip_exact": all(raw_checks),
        "ego_neighbor_identities_unchanged": True,
        "anomaly_background_fit_input": "original raw 74-D rows only",
        "anomaly_evidence_definition_unchanged": True,
        "route_DI_unchanged": True, "Global24_unchanged": True,
        "no_ego_fallback_is_original_hostego_identity_before_AB": True,
        "original_sha256_unchanged": before == after,
    }
    write_json(cli.output / "preflight/dryrun_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: Host complement dry-run failed")
    print("HOST_COMPLEMENT_NEURAL_DRYRUN_PASS")


def save_checkpoint(path: Path, data, probability, diagnostic, arm: str) -> None:
    complete = np.flatnonzero(np.isfinite(probability))
    rows = []
    for i in complete:
        row = {"sha256": data.ids[i], "year": int(data.years[i]), "fold": int(data.folds[i]),
               "arm": ARM_NAMES[arm], "p_expert": float(probability[i]), "version": VERSION}
        row.update({name: float(diagnostic[i, j]) for j, name in enumerate(DIAG_NAMES)})
        rows.append(row)
    tmp = path.with_suffix(".tmp"); write_csv(tmp, rows); tmp.replace(path)


def run_arm(cli, m1, args, data, bags, arm, device):
    out = cli.output / "checkpoints" / ARM_NAMES[arm]
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "oof_checkpoint.csv"
    fit_path = out / "fit_audit.csv"
    p = np.full(len(data.ids), np.nan); diag = np.full((len(data.ids), len(DIAG_NAMES)), np.nan)
    fit_rows = read_csv(fit_path) if fit_path.is_file() else []
    if checkpoint.is_file():
        saved = read_csv(checkpoint); by_id = {normal(r["sha256"]): r for r in saved}
        if len(saved) != len(by_id) or {r.get("version") for r in saved} != {VERSION}:
            raise RuntimeError("BLOCKED_BY_CONTRACT: incompatible or duplicate checkpoint")
        for i, sha in enumerate(data.ids):
            if sha in by_id:
                p[i] = float(by_id[sha]["p_expert"])
                diag[i] = [float(by_id[sha][name]) for name in DIAG_NAMES]
        print(f"[resume] {ARM_NAMES[arm]} {np.isfinite(p).sum()}/1989", flush=True)
    seeds = m1.v4.parse_ints(args.seeds)
    for year in (2018, 2019, 2020, 2021):
        for fold in range(1, 6):
            validation = np.flatnonzero((data.years == year) & (data.folds == fold))
            train = np.flatnonzero((data.years == year) & (data.folds != fold))
            if np.isfinite(p[validation]).all():
                print(f"[resume] {ARM_NAMES[arm]} {year} fold {fold}/5", flush=True); continue
            fold_p, fold_d, audits = run_fold(
                m1, data, bags, args, train, validation, arm, seeds, device,
                int(args.split_seed) + year * 10 + fold,
            )
            p[validation] = fold_p; diag[validation] = fold_d
            fit_rows.extend({"arm": ARM_NAMES[arm], "year": year, "outer_fold": fold, **a} for a in audits)
            save_checkpoint(checkpoint, data, p, diag, arm)
            tmp = fit_path.with_suffix(".tmp"); write_csv(tmp, fit_rows); tmp.replace(fit_path)
            write_json(out / f"{year}_fold{fold}_checkpoint_metadata.json", {
                "arm": ARM_NAMES[arm], "year": year, "fold": fold,
                "seeds": seeds, "model_state_persisted": False,
                "resume_unit": "saved OOF probabilities and diagnostics",
                "note": "matches inherited M1 runner, which does not persist neural state dicts",
            })
            print(f"{ARM_NAMES[arm]} {year} fold {fold}/5 done", flush=True)
    if not np.isfinite(p).all() or not np.isfinite(diag).all():
        raise RuntimeError("incomplete OOF arm")
    expected = {int(r["parameter_n"]) for r in fit_rows}
    if len(expected) != 1:
        raise RuntimeError(f"parameter count varies within arm: {expected}")
    rows=[]
    for i in range(len(data.ids)):
        row={"sha256":data.ids[i],"year":int(data.years[i]),"label":int(data.labels[i]),"fold":int(data.folds[i]),"arm":ARM_NAMES[arm],"p_expert":float(p[i]),"parameter_count":next(iter(expected)),**bags.usage[i]}
        row.update({name:float(diag[i,j]) for j,name in enumerate(DIAG_NAMES)});rows.append(row)
    write_csv(cli.output / "predictions" / f"M_{arm}_expert_predictions.csv", rows)


def full(cli, m1, args, data, routes, bags, before):
    dry = cli.output / "preflight/dryrun_audit.json"
    if not dry.is_file() or not json.loads(dry.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("BLOCKED_BY_CONTRACT: dry-run PASS required before full run")
    device = m1.v4.resolve_device(args.device)
    arms = ("A", "B", "AB") if cli.arm == "all" else (cli.arm,)
    for arm in arms: run_arm(cli, m1, args, data, bags, arm, device)
    after = hash_manifest(protected_paths())
    audit={"passed":before==after,"before":before,"after":after}
    write_json(cli.output / "audits/no_original_modification.json", audit)
    (cli.output / "audits/original_sha256_after.txt").write_text("".join(f"{v}  {k}\n" for k,v in after.items()),encoding="utf-8")
    if not audit["passed"]: raise RuntimeError("BLOCKED_BY_CONTRACT: protected original changed")
    print("HOST_COMPLEMENT_NEURAL_TRAINING_DONE")


def main() -> None:
    cli=parser().parse_args(); before=hash_manifest(protected_paths())
    m1=load_m1_modules(); args,data,routes,bags=load_all(m1,cli)
    if cli.mode=="dry-run": dry_run(cli,m1,args,data,routes,bags,before)
    else: full(cli,m1,args,data,routes,bags,before)


if __name__=="__main__": main()
