#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import ROOT, YEARS, norm, dump
from a0_adapter import install_fold_safe_compact76, prepare_imports
from run_frozen_future import parser


def main():
    target = ROOT / "cache/future_s7.npz"
    if target.is_file():
        print("[resume] shared future S7 cache exists")
        return
    args = parser().parse_args()
    resolved = json.loads((ROOT / "resolved_inputs.json").read_text())
    if args.features is None:
        args.features = Path(resolved["sealed_features"])
    args.cache = Path(resolved["endpoint_cache"])
    args.future_inter_cache = Path(resolved["future_inter_cache"])
    prepare_imports()
    import run_care_rolegraph_conditioned_dualstream_v4 as v4
    install_fold_safe_compact76(v4.audit)
    _, _, clean_rows, _, _, _ = v4.audit.load_inputs(args)
    rows = [r for r in clean_rows if str(r.get("metadata_year", "")).strip() in {str(y) for y in YEARS}]
    if len(rows) != 1989:
        raise RuntimeError(f"BLOCKED_BY_CONTRACT: future rows={len(rows)}")
    endpoint_cache = v4.audit.exp.read_cache(args.cache)
    print("Preparing shared label-free future RoleGraph/S7 cache...", flush=True)
    s3, _ = v4.audit.future.reconstruct_structure(rows, endpoint_cache, int(v4.audit.exp.ROLE_VOCABULARY_SIZE))
    inter, _, _ = v4.v3.future_s7.load_or_build_future_inter(rows, s3, endpoint_cache, args)
    s7 = np.concatenate([s3, inter], axis=1)
    if s7.shape != (1989, 1546) or not np.isfinite(s7).all():
        raise RuntimeError("BLOCKED_BY_CONTRACT: future S7 shape/nonfinite " + repr(s7.shape))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, sha256=np.asarray([norm(r["sha256"]) for r in rows]), s7=s7.astype(np.float32))
    dump(ROOT / "cache/future_s7_audit.json", {"rows": 1989, "width": 1546, "future_labels_loaded": False})
    print("FORMAL_FUTURE_S7_CACHE_PASS")
    print("rows=1989 width=1546 future_labels_loaded=false")


if __name__ == "__main__":
    main()
