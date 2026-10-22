#!/usr/bin/env python3
"""Hard, read-only preflight for Host Complement Neural v1."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from audit_common import (
    ASSET_MANIFEST, EGO, ENDPOINT, EXPECTED_M0_SHA256, HOST, LOCKED_M0,
    MATRIX_ROOT, M1_ROOT, ROOT, ROUTER, hash_manifest, load_m1_modules,
    protected_paths, read_csv, router_contract, sha256_file, write_json,
)
from host_complement_model import make_model_class


def architecture(m1, args) -> tuple[dict, dict]:
    within, v4 = m1.within, m1.v4
    Base = within.anomaly_bounded.DedupBoundedRGCondAttn
    _, M1 = m1.anomaly_classes(within.anomaly_bounded, v4)
    from host_ego_encoder import DirectedHostEgoFusion
    role_count = 31
    models = {"M0_LOCKED": M1(24, "RGCondAttn", args)}
    for arm, name in (("A", "M_A_COMPOSITION"), ("B", "M_B_POSITION"), ("AB", "M_AB_COMPOSITION_POSITION")):
        cls = make_model_class(Base, DirectedHostEgoFusion, role_count, within.future.ORIGINAL_AUGMENTED_DIM, arm)
        models[name] = cls(24, "RGCondAttn", args)
    counts = {name: int(sum(p.numel() for p in model.parameters())) for name, model in models.items()}
    if counts["M0_LOCKED"] != 34613:
        raise RuntimeError(f"BLOCKED_BY_ARCHITECTURE: actual M1 parameters={counts['M0_LOCKED']}, expected 34613")
    result = {
        "grounded_from": [str(M1_ROOT / "host_ego_models.py"), str(MATRIX_ROOT / "run_care_rgcondattn_dedup_bounded_source_v1.py")],
        "host_raw_dim": 74,
        "anomaly_augmented_host_dim": int(within.future.ORIGINAL_AUGMENTED_DIM),
        "host_representation_dim": int(args.embedding_dim),
        "ego_representation_dim": int(args.embedding_dim),
        "graph_token_dim": int(args.graph_dim),
        "host_to_ego_to_local": "h_host=region_encoder(original anomaly-augmented Host); h_ego=projection([incoming_mean,outgoing_mean]); h_local=h_host+gamma*h_ego only when a real Host neighbor exists",
        "gamma_gate_position": "after original Host encoder and before local-risk/Critical/Attention",
        "anomaly_evidence_position": "the original fit_scaled_dualproto_bags augments raw 74-D Host rows before neural encoding",
        "anomaly_background_input": "raw 74-D Host representation; A/B fields are stripped before the unchanged original fitter",
        "critical_host_position": "argmax of graph-conditioned local score after Host/Ego and A/B residual fusion",
        "attended_host_position": "softmax-weighted sum after Host/Ego and A/B residual fusion",
        "correction_head_input": "Global24(24)+base_logit(1)+graph_token(32)+critical(24)+attended(24)+summary(8)=113",
        "bounded_formula": "final_logit=stage1_logit+2.25*tanh(raw_delta); probability=sigmoid(final_logit)",
        "threshold": 0.5,
        "parameter_counts": counts,
        "actual_base_class": Base.__name__,
        "actual_m1_class": M1.__name__,
    }
    return result, counts


def protocol(args) -> dict:
    optimizer_source = (MATRIX_ROOT / "run_care_rolegraph_conditioned_dualstream_v4.py")
    if not optimizer_source.is_file(): optimizer_source = MATRIX_ROOT / "vendor/run_care_rolegraph_conditioned_dualstream_v4.py"
    if not optimizer_source.is_file(): optimizer_source = M1_ROOT.parent / "care-rolegraph-conditioned-v4-script/run_care_rolegraph_conditioned_dualstream_v4.py"
    source = optimizer_source.read_text(encoding="utf-8")
    optimizer = "AdamW" if "optim.AdamW" in source else ("Adam" if "optim.Adam" in source else "unknown")
    return {
        "protocol": "independent per-year strict five-fold OOF development/diagnostic",
        "years": [2018, 2019, 2020, 2021], "folds": 5,
        "seeds": [int(v) for v in str(args.seeds).split(",")],
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "max_epochs": args.max_epochs,
        "early_stop_fraction": args.early_stop_fraction,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "inner_folds": args.inner_folds, "split_seed": args.split_seed,
        "hidden_dim": args.hidden_dim, "embedding_dim": args.embedding_dim,
        "graph_dim": args.graph_dim, "graph_heads": args.graph_heads,
        "graph_layers": args.graph_layers, "dropout": args.dropout,
        "grad_clip": args.grad_clip, "optimizer": optimizer,
        "keep_weight": args.keep_weight, "fix_weight": args.fix_weight,
        "keep_consistency": args.keep_consistency,
        "max_delta": args.max_delta,
        "anomaly_fitting": "unchanged ORIGINAL_ANOMALY_FIT, outer-train only",
        "outer_validation_use": "inference only; labels excluded from training and model selection except inherited early-stop split inside outer-train",
        "hyperparameter_search": False, "threshold_search": False,
    }


def overlap_audit() -> dict:
    source = Path("scripts/care-source-local-assets-v2/extract_source_local_assets_v2.py")
    if not source.is_file(): source = Path("care-source-local-assets-v2/extract_source_local_assets_v2.py")
    text = source.read_text(encoding="utf-8")
    checks = {
        "role_count_31_present": "ROLE_COUNT" in text and "31" in text,
        "normalized_role_counts_present": "counts / max(host_n, 1)" in text,
        "log_role_counts_present": "np.log1p(counts)" in text,
        "topology_features_present": "topology" in text,
    }
    if not all(checks.values()):
        raise RuntimeError("BLOCKED_BY_ARCHITECTURE: could not ground Host role overlap in extractor")
    return {
        "source": str(source), "source_sha256": sha256_file(source), "checks": checks,
        "host_raw_dim": 74,
        "role_identity_already_explicit": True,
        "existing_role_features": "31 normalized role counts + 31 log1p role counts",
        "prohibited_A": "role mean embedding or another role identity vector",
        "implemented_A": "unique-role unordered pair interactions only; single-role Host maps exactly to zero",
        "diagnostic_summary_features_used_as_model_input": False,
    }


def main() -> None:
    required = protected_paths()
    hashes = hash_manifest(required)
    if sha256_file(LOCKED_M0) != EXPECTED_M0_SHA256:
        raise RuntimeError("BLOCKED_BY_CONTRACT: frozen M0 SHA256 mismatch")
    route = router_contract(read_csv(ROUTER))
    if not route["passed"]:
        raise RuntimeError("BLOCKED_BY_CONTRACT: DI25 route contract failed: " + str(route))
    m1 = load_m1_modules()
    args = m1.parser().parse_args([])
    if float(args.max_delta) != 2.25 or str(args.seeds) != "900000,900001,900002":
        raise RuntimeError("BLOCKED_BY_CONTRACT: inherited M1 protocol changed")
    arch, counts = architecture(m1, args)
    training = protocol(args)
    overlap = overlap_audit()
    pre = ROOT / "preflight"; pre.mkdir(parents=True, exist_ok=True)
    write_json(pre / "current_m1_architecture.json", arch)
    write_json(pre / "host_role_overlap_audit.json", overlap)
    write_json(pre / "training_protocol.json", training)
    write_json(pre / "contract.json", {
        "passed": True, "evaluation_name": "same-year OOF development/diagnostic",
        "locked_m0_sha256": EXPECTED_M0_SHA256, "router_contract": route,
        "no_apk_reanalysis": True, "allowed_arms": ["M0_LOCKED", "M_A_COMPOSITION", "M_B_POSITION", "M_AB_COMPOSITION_POSITION"],
        "forbidden": ["C", "C2", "DMD", "KEEP", "new correction head", "threshold search", "hyperparameter search"],
        "parameter_counts_pretraining": counts,
    })
    (pre / "original_sha256_before.txt").write_text("".join(f"{v}  {k}\n" for k,v in hashes.items()), encoding="utf-8")
    write_json(pre / "original_sha256_before.json", hashes)
    print("HOST_COMPLEMENT_NEURAL_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
