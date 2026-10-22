#!/usr/bin/env python3
from __future__ import annotations

import json
import numpy as np

from common import DIRECT_METHOD, YEARS, discover, norm, read_csv, top_k


def main():
    assets = discover(); source = read_csv(assets["source"]); future = read_csv(assets["future"])
    if len(source) != 799 or len({norm(row["sha256"]) for row in source}) != 799:
        raise RuntimeError("BLOCKED_BY_CONTRACT: source cohort is not 799 unique APKs")
    if len(future) != 1989 or len({norm(row["sha256"]) for row in future}) != 1989:
        raise RuntimeError("BLOCKED_BY_CONTRACT: future cohort is not 1989 unique APKs")
    source_required = {"sha256", "label", "fold", "Global24__prob", "A25__score", "A25__routed", DIRECT_METHOD + "__expert_prob"}
    future_required = {"sha256", "year", "stage1_prob", "router_score", "routed", "direct_c43_prob"}
    if source_required - set(source[0]): raise RuntimeError("BLOCKED_BY_CONTRACT: source columns missing " + repr(source_required - set(source[0])))
    if future_required - set(future[0]): raise RuntimeError("BLOCKED_BY_CONTRACT: future columns missing " + repr(future_required - set(future[0])))
    fold = np.asarray([int(float(row["fold"])) for row in source]); score = np.asarray([float(row["A25__score"]) for row in source])
    sha = np.asarray([norm(row["sha256"]) for row in source]); reconstructed = top_k(score, sha, fold, .25)
    stored = np.asarray([int(float(row["A25__routed"])) for row in source], bool)
    if not np.array_equal(reconstructed, stored): raise RuntimeError("BLOCKED_BY_CONTRACT: source Top25 mismatch")
    year = np.asarray([int(row["year"]) for row in future]); future_score = np.asarray([float(row["router_score"]) for row in future])
    future_sha = np.asarray([norm(row["sha256"]) for row in future]); future_route = top_k(future_score, future_sha, year, .25)
    stored_future = np.asarray([int(row["routed"]) for row in future], bool)
    if not np.array_equal(future_route, stored_future): raise RuntimeError("BLOCKED_BY_CONTRACT: future Top25 mismatch")
    if set(year.tolist()) != set(YEARS): raise RuntimeError("BLOCKED_BY_CONTRACT: future years changed")
    architecture = json.loads(assets["architecture"].read_text(encoding="utf-8"))
    if "direct" not in json.dumps(architecture).casefold(): raise RuntimeError("BLOCKED_BY_CONTRACT: not Direct-C43")
    print("ELASTIC_MEANOPPOSITION_PREFLIGHT_PASS")
    print({"source": len(source), "future_label_free": len(future), "top25_source": int(stored.sum()),
           "top25_future": int(stored_future.sum()), "future_labels_loaded": False})


if __name__ == "__main__": main()
