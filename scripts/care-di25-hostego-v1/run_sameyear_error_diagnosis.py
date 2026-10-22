#!/usr/bin/env python3
"""Read-only diagnosis of frozen CARE-Droid independent same-year OOF results.

This program never imports a training entry point.  It consumes frozen
per-sample predictions and raw, label-independent Host/Ego caches only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from error_diagnosis_core import (
    DISCLAIMER,
    MAX_DELTA,
    YEARS,
    cohort,
    cohens_d,
    iqr,
    logit,
    residual_primary_failure,
    safe_div,
)


EXPECTED_N = 1989
EXPECTED_YEARS = {2018: 793, 2019: 399, 2020: 397, 2021: 400}
EXPECTED_ROUTES = {2018: 198, 2019: 100, 2020: 99, 2021: 100}
EXPECTED_POOLED = {
    "Global24": (0.926131, 49, 98),
    "M0": (0.945217, 42, 67),
    "M1": (0.946728, 39, 67),
}
ROOT = Path("artifacts/care-di25-hostego-v1/error_diagnosis")
ROUTER = Path(
    "artifacts/care-route-stage2-within-year-matrix-v1/shared/within_year_router_inputs.csv"
)
ROUTER_PROTOCOL = Path(
    "artifacts/care-route-stage2-within-year-matrix-v1/shared/router_protocol.json"
)
M0 = Path(
    "artifacts/care-route-stage2-within-year-matrix-v1/arms/anomaly_bounded/within_year_expert_predictions.csv"
)
M0_PROTOCOL = Path(
    "artifacts/care-route-stage2-within-year-matrix-v1/arms/anomaly_bounded/protocol.json"
)
M1 = Path(
    "artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/within_year_m1_predictions.csv"
)
M1_PROTOCOL = Path(
    "artifacts/care-di25-hostego-within-year-v1/m1_hostego_anomaly_bounded/protocol.json"
)
M1_SAMPLES = Path(
    "artifacts/care-di25-hostego-within-year-v1/results/within_year_sample_predictions.csv"
)
ASSET = Path("artifacts/care-future-local-assets-v2")


def normal(value) -> str:
    return str(value or "").strip().lower()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unique(rows: list[dict], name: str, expected: int = EXPECTED_N) -> dict[str, dict]:
    result = {}
    for row in rows:
        sha = normal(row.get("sha256", row.get("sample_id", "")))
        if not sha or sha in result:
            raise RuntimeError(f"{name}: missing or duplicate SHA {sha!r}")
        result[sha] = row
    if len(rows) != expected or len(result) != expected:
        raise RuntimeError(
            f"{name}: rows={len(rows)} unique={len(result)} expected={expected}"
        )
    return result


def finite_float(value, default=float("nan")) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def metric(y, probability) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(probability, float)
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def scope_masks(years: np.ndarray):
    for year in YEARS:
        yield str(year), years == year
    yield "pooled_2018_2021", np.ones(len(years), dtype=bool)


def percentile_by_year(years: np.ndarray, score: np.ndarray) -> np.ndarray:
    result = np.zeros(len(score), dtype=float)
    for year in YEARS:
        idx = np.flatnonzero(years == year)
        order = np.argsort(score[idx], kind="mergesort")
        ranks = np.empty(len(idx), dtype=float)
        ranks[order] = np.arange(1, len(idx) + 1)
        result[idx] = ranks / len(idx)
    return result


def load_core():
    required = [
        ROUTER,
        ROUTER_PROTOCOL,
        M0,
        M0_PROTOCOL,
        M1,
        M1_PROTOCOL,
        ASSET / "future_asset_gate.txt",
        ASSET / "local_asset_manifest.json",
        ASSET / "host_regions_2018_2021.npz",
        ASSET / "ego_1hop_regions_2018_2021.npz",
        ASSET / "global_role_graph_2018_2021.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "BLOCKED_BY_CONTRACT: missing frozen inputs:\n- " + "\n- ".join(missing)
        )
    if (
        "passed: true"
        not in (ASSET / "future_asset_gate.txt").read_text(encoding="utf-8").lower()
    ):
        raise SystemExit("BLOCKED_BY_CONTRACT: future local-asset gate is not PASS")
    router_rows = read_csv(ROUTER)
    unique(router_rows, "router")
    m0_index = unique(read_csv(M0), "M0")
    m1_index = unique(read_csv(M1), "M1")
    ids = [normal(row["sha256"]) for row in router_rows]
    if set(ids) != set(m0_index) or set(ids) != set(m1_index):
        raise RuntimeError("router/M0/M1 SHA sets differ")
    years = np.asarray([int(row["year"]) for row in router_rows], int)
    if dict(Counter(years.tolist())) != EXPECTED_YEARS:
        raise RuntimeError("same-year cohort counts changed")
    labels = np.asarray([int(float(row["label"])) for row in router_rows], int)
    folds = np.asarray([int(row["fold"]) for row in router_rows], int)
    p0 = np.asarray([float(row["p_stage1"]) for row in router_rows], float)
    di = np.asarray([float(row["U_DI"]) for row in router_rows], float)
    route = np.asarray([int(float(row["route_DI"])) == 1 for row in router_rows], bool)
    route_counts = {year: int(np.sum(route[years == year])) for year in YEARS}
    if route_counts != EXPECTED_ROUTES:
        raise RuntimeError(
            f"frozen same-year DI25 route counts changed: {route_counts}"
        )
    if any(set(folds[years == year].tolist()) != {1, 2, 3, 4, 5} for year in YEARS):
        raise RuntimeError("same-year outer-fold coverage is incomplete")
    router_protocol = json.loads(ROUTER_PROTOCOL.read_text(encoding="utf-8"))
    if (
        router_protocol.get("protocol")
        != "strict independent within-year five-fold OOF"
    ):
        raise RuntimeError(
            "router protocol is not strict independent within-year five-fold OOF"
        )
    m0_protocol = json.loads(M0_PROTOCOL.read_text(encoding="utf-8"))
    if (
        m0_protocol.get("representation") != "anomaly"
        or m0_protocol.get("head") != "bounded"
    ):
        raise RuntimeError("M0 protocol is not frozen Host/Anomaly/Bounded")
    m1_protocol = json.loads(M1_PROTOCOL.read_text(encoding="utf-8"))
    if (
        m1_protocol.get("protocol") != "independent per-year five-fold OOF"
        or int(m1_protocol.get("parameter_count", -1)) != 34613
    ):
        raise RuntimeError(
            "M1 protocol is not the frozen HostEgo/Anomaly/Bounded same-year run"
        )
    pm0 = np.asarray([float(m0_index[sha]["p_expert"]) for sha in ids], float)
    pm1expert = np.asarray(
        [float(m1_index[sha]["p_expert_bounded"]) for sha in ids], float
    )
    pm1 = np.where(route, pm1expert, p0)
    for name, values in (("p_stage1", p0), ("U_DI", di), ("M0", pm0), ("M1", pm1)):
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite frozen field: {name}")
    actual = {
        "Global24": metric(labels, p0),
        "M0": metric(labels, np.where(route, pm0, p0)),
        "M1": metric(labels, pm1),
    }
    for name, (ba, fp, fn) in EXPECTED_POOLED.items():
        row = actual[name]
        if (
            not math.isclose(row["BA"], ba, abs_tol=5e-7)
            or row["FP"] != fp
            or row["FN"] != fn
        ):
            raise RuntimeError(f"locked pooled {name} changed: {row}")
    return (
        ids,
        router_rows,
        m0_index,
        m1_index,
        years,
        folds,
        labels,
        p0,
        di,
        route,
        pm0,
        pm1expert,
        pm1,
        actual,
    )


def load_structure(ids: list[str], years: np.ndarray):
    manifest = json.loads(
        (ASSET / "local_asset_manifest.json").read_text(encoding="utf-8")
    )
    role_names = [str(value) for value in manifest["model_role_order"]]
    if len(role_names) != 31:
        raise RuntimeError("31-role contract changed")
    with np.load(ASSET / "host_regions_2018_2021.npz", allow_pickle=False) as h:
        hids = [normal(value) for value in h["sha256"].astype(str)]
        real_host = np.asarray(h["real_host_count"], np.int64)
    with np.load(ASSET / "ego_1hop_regions_2018_2021.npz", allow_pickle=False) as e:
        eids = [normal(value) for value in e["sha256"].astype(str)]
        ego_offsets = np.asarray(e["ego_offsets"], np.int64)
        status = np.asarray(e["ego_status"], np.int64)
        node_n = np.asarray(e["ego_node_count"], np.int64)
        edge_n = np.asarray(e["ego_edge_count"], np.int64)
        primary = np.asarray(e["host_primary_role"], np.int64)
        in_offsets = np.asarray(e["incoming_offsets"], np.int64)
        out_offsets = np.asarray(e["outgoing_offsets"], np.int64)
        in_pos = np.asarray(e["incoming_host_positions"], np.int64)
        out_pos = np.asarray(e["outgoing_host_positions"], np.int64)
    with np.load(ASSET / "global_role_graph_2018_2021.npz", allow_pickle=False) as g:
        gids = [normal(value) for value in g["sha256"].astype(str)]
        role_nodes = np.asarray(g["role_node_counts"], np.int64)
        role_edges = np.asarray(g["role_edge_counts"], np.int64)
    if hids != ids or eids != ids or gids != ids:
        raise RuntimeError(
            "Host/Ego/global-role cache order differs from frozen router order"
        )
    rows, pair_by_apk = [], []
    for i, sha in enumerate(ids):
        hb, he = int(ego_offsets[i]), int(ego_offsets[i + 1])
        nreal = int(real_host[i])
        if he - hb != nreal:
            raise RuntimeError("real-host/Ego offset mismatch: " + sha)
        local_status = status[hb:he]
        neighbors = []
        pairs = Counter()
        ego_roles = set()
        for center in range(hb, he):
            incoming = in_pos[in_offsets[center] : in_offsets[center + 1]].astype(int)
            outgoing = out_pos[out_offsets[center] : out_offsets[center + 1]].astype(
                int
            )
            neighbors.append((len(incoming), len(outgoing)))
            center_role = int(primary[center])
            ego_roles.add(center_role)
            for other in incoming:
                other_role = int(primary[int(other)])
                ego_roles.add(other_role)
                pairs[(center_role, other_role, "incoming")] += 1
            for other in outgoing:
                other_role = int(primary[int(other)])
                ego_roles.add(other_role)
                pairs[(center_role, other_role, "outgoing")] += 1
        degree = np.asarray([left + right for left, right in neighbors], float)
        incoming = np.asarray([left for left, _ in neighbors], float)
        outgoing = np.asarray([right for _, right in neighbors], float)
        nontrivial = int(np.sum(local_status == 1))
        host_roles = set(primary[hb:he].tolist())
        rows.append(
            {
                "sha256": sha,
                "year": int(years[i]),
                "real_host_count": nreal,
                "nontrivial_ego_host_count": nontrivial,
                "ego_host_ratio": safe_div(nontrivial, nreal),
                "mean_neighbor_count": float(np.mean(degree)) if len(degree) else 0.0,
                "max_neighbor_count": float(np.max(degree)) if len(degree) else 0.0,
                "incoming_neighbor_mean": float(np.mean(incoming))
                if len(incoming)
                else 0.0,
                "outgoing_neighbor_mean": float(np.mean(outgoing))
                if len(outgoing)
                else 0.0,
                "mean_ego_node_count": float(np.mean(node_n[hb:he])) if nreal else 0.0,
                "mean_ego_edge_count": float(np.mean(edge_n[hb:he])) if nreal else 0.0,
                "host_role_diversity_count": len(host_roles),
                "host_role_diversity_ratio": safe_div(len(host_roles), 31),
                "ego_role_diversity_count": len(ego_roles),
                "ego_role_diversity_ratio": safe_div(len(ego_roles), 31),
                "missing_host_indicator": int(nreal == 0),
                "global_role_presence_count": int(np.sum(role_nodes[i] > 0)),
                "global_role_node_total": int(np.sum(role_nodes[i])),
                "global_role_edge_total": int(np.sum(role_edges[i])),
            }
        )
        pair_by_apk.append(pairs)
    return role_names, role_nodes, role_edges, rows, pair_by_apk


def load_optional_compact(ids: list[str]) -> tuple[dict[str, dict], str]:
    candidates = []
    resolved = Path(
        "artifacts/care-route-stage2-within-year-matrix-v1/preflight/resolved_inputs.json"
    )
    if resolved.is_file():
        try:
            path = Path(
                json.loads(resolved.read_text(encoding="utf-8")).get("features", "")
            )
            if path.is_file():
                candidates.append(path)
        except Exception:
            pass
    candidates.extend(
        [Path("artifacts/p1-features/features.csv"), ASSET / "features.csv"]
    )
    idset = set(ids)
    for path in candidates:
        if not path.is_file():
            continue
        rows = read_csv(path)
        by = {}
        for row in rows:
            sha = normal(row.get("sha256", row.get("sample_id", "")))
            if sha in idset and sha not in by:
                by[sha] = row
        if len(by) == len(ids):
            return by, str(path)
    return {}, "unavailable"


def core_rows(
    ids,
    router_rows,
    years,
    folds,
    labels,
    p0,
    di,
    route,
    pm0,
    pm1expert,
    pm1,
    structures,
    compact,
):
    p0pred = p0 >= 0.5
    finalpred = pm1 >= 0.5
    p0correct = p0pred == labels
    finalcorrect = finalpred == labels
    percentile = percentile_by_year(years, di)
    result = []
    for i, sha in enumerate(ids):
        c = cohort(bool(p0correct[i]), bool(route[i]), bool(finalcorrect[i]))
        row = {
            "year": int(years[i]),
            "fold": int(folds[i]),
            "sha256": sha,
            "label": int(labels[i]),
            "Global24_prob": float(p0[i]),
            "Global24_pred": int(p0pred[i]),
            "Global24_correct": int(p0correct[i]),
            "DI_score_if_available": float(di[i]),
            "DI_rank_or_percentile": float(percentile[i]),
            "routed": int(route[i]),
            "M0_expert_prob": float(pm0[i]),
            "M1_expert_prob": float(pm1expert[i]),
            "M1_prob": float(pm1[i]),
            "M1_pred": int(finalpred[i]),
            "M1_correct": int(finalcorrect[i]),
            "cohort": c,
            "Global24_margin": float(abs(p0[i] - 0.5)),
        }
        row.update(
            {k: v for k, v in structures[i].items() if k not in {"sha256", "year"}}
        )
        if compact:
            source = compact[sha]
            for key, value in source.items():
                low = key.lower()
                if any(
                    token in low
                    for token in (
                        "manifest",
                        "permission",
                        "component",
                        "code",
                        "method",
                        "class",
                        "call",
                        "graph",
                        "sensitive",
                        "endpoint",
                        "coverage",
                        "parse",
                        "node",
                        "edge",
                        "role",
                    )
                ):
                    numeric = finite_float(value)
                    if math.isfinite(numeric):
                        row["compact__" + key] = numeric
        result.append(row)
    return result


def error_capture(rows: list[dict]) -> list[dict]:
    output = []
    for scope, selected in [
        (str(y), [r for r in rows if r["year"] == y]) for y in YEARS
    ] + [("pooled_2018_2021", rows)]:
        y = np.asarray([r["label"] for r in selected], int)
        s1 = np.asarray([r["Global24_pred"] for r in selected], int)
        route = np.asarray([r["routed"] for r in selected], bool)
        wrong = s1 != y
        fp = (y == 0) & (s1 == 1)
        fn = (y == 1) & (s1 == 0)
        unrouted = ~route
        row = {
            "scope": scope,
            "N": len(y),
            "positive_n": int(np.sum(y == 1)),
            "negative_n": int(np.sum(y == 0)),
            "Global24_error_n": int(wrong.sum()),
            "Global24_FP": int(fp.sum()),
            "Global24_FN": int(fn.sum()),
            "routed_n": int(route.sum()),
            "unrouted_n": int(unrouted.sum()),
            "routed_Global24_error_n": int(np.sum(route & wrong)),
            "routed_Global24_correct_n": int(np.sum(route & ~wrong)),
            "routed_FP": int(np.sum(route & fp)),
            "routed_FN": int(np.sum(route & fn)),
            "unrouted_Global24_error_n": int(np.sum(unrouted & wrong)),
            "unrouted_FP": int(np.sum(unrouted & fp)),
            "unrouted_FN": int(np.sum(unrouted & fn)),
        }
        row.update(
            {
                "error_recall": safe_div(
                    row["routed_Global24_error_n"], row["Global24_error_n"]
                ),
                "FP_error_recall": safe_div(row["routed_FP"], row["Global24_FP"]),
                "FN_error_recall": safe_div(row["routed_FN"], row["Global24_FN"]),
                "route_precision": safe_div(
                    row["routed_Global24_error_n"], row["routed_n"]
                ),
                "routed_error_rate": safe_div(
                    row["routed_Global24_error_n"], row["routed_n"]
                ),
                "accepted_risk": safe_div(
                    row["unrouted_Global24_error_n"], row["unrouted_n"]
                ),
                "unrouted_FP_rate": safe_div(row["unrouted_FP"], row["unrouted_n"]),
                "unrouted_FN_rate": safe_div(row["unrouted_FN"], row["unrouted_n"]),
            }
        )
        output.append(row)
    return output


def stage2_efficiency(rows: list[dict]) -> list[dict]:
    output = []
    for scope, selected in [
        (str(y), [r for r in rows if r["year"] == y]) for y in YEARS
    ] + [("pooled_2018_2021", rows)]:
        by = Counter(r["cohort"] for r in selected)
        routed_n = sum(r["routed"] for r in selected)
        fp_fixed = sum(r["cohort"] == "C" and r["label"] == 0 for r in selected)
        fn_fixed = sum(r["cohort"] == "C" and r["label"] == 1 for r in selected)
        fp_broken = sum(r["cohort"] == "F" and r["label"] == 0 for r in selected)
        fn_broken = sum(r["cohort"] == "F" and r["label"] == 1 for r in selected)
        output.append(
            {
                "scope": scope,
                "routed_n": routed_n,
                "routed_stage1_errors": by["C"] + by["D"],
                "fixed": by["C"],
                "unfixed_routed_errors": by["D"],
                "broken": by["F"],
                "preserved": by["E"],
                "correction_recall": safe_div(by["C"], by["C"] + by["D"]),
                "correction_precision": safe_div(by["C"], by["C"] + by["F"]),
                "break_rate": safe_div(by["F"], by["E"] + by["F"]),
                "net": by["C"] - by["F"],
                "net_per_routed": safe_div(by["C"] - by["F"], routed_n),
                "FP_fixed": fp_fixed,
                "FN_fixed": fn_fixed,
                "FP_broken": fp_broken,
                "FN_broken": fn_broken,
            }
        )
    return output


def oracle_tables(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ceiling = []
    systems = []
    for scope, selected in [
        (str(y), [r for r in rows if r["year"] == y]) for y in YEARS
    ] + [("pooled_2018_2021", rows)]:
        y = np.asarray([r["label"] for r in selected], int)
        base = np.asarray([r["Global24_prob"] for r in selected], float)
        final = np.asarray([r["M1_prob"] for r in selected], float)
        route = np.asarray([r["routed"] for r in selected], bool)
        s1 = base >= 0.5
        wrong = s1 != y
        finalpred = final >= 0.5
        perfect = np.array(base, copy=True)
        perfect[route] = np.where(y[route] == 1, 1 - 1e-9, 1e-9)
        o2 = np.array(final, copy=True)
        broken = route & ~wrong & (finalpred != y)
        o2[broken] = base[broken]
        o3 = np.array(final, copy=True)
        unfixed = route & wrong & (finalpred != y)
        o3[unfixed] = np.where(y[unfixed] == 1, 1 - 1e-9, 1e-9)
        variants = {
            "O0_Global24": base,
            "O1_PerfectStageII_current_DI25": perfect,
            "O2_CurrentM1_no_broken": o2,
            "O3_CurrentM1_plus_perfect_routed_unfixed": o3,
            "O4_Perfect_classifier_within_DI25": perfect,
        }
        for name, p in variants.items():
            systems.append(
                {
                    "scope": scope,
                    "oracle": name,
                    **metric(y, p),
                    "AUC_interpretation": "label-informed descriptive oracle"
                    if name != "O0_Global24"
                    else "ordinary",
                }
            )
        mb = metric(y, base)
        mm = metric(y, final)
        mo = metric(y, perfect)
        opp = mo["BA"] - mb["BA"]
        realized = mm["BA"] - mb["BA"]
        ceiling.append(
            {
                "scope": scope,
                "Global24_BA": mb["BA"],
                "M1_BA": mm["BA"],
                "DI25_oracle_BA": mo["BA"],
                "oracle_TN": mo["TN"],
                "oracle_FP": mo["FP"],
                "oracle_FN": mo["FN"],
                "oracle_TP": mo["TP"],
                "router_available_gain": opp,
                "M1_realized_gain": realized,
                "realized_fraction_of_router_opportunity": safe_div(realized, opp),
            }
        )
    return ceiling, systems


def confidence_table(rows):
    bins = [
        (0, 0.05, "[0,0.05)"),
        (0.05, 0.10, "[0.05,0.10)"),
        (0.10, 0.20, "[0.10,0.20)"),
        (0.20, 0.30, "[0.20,0.30)"),
        (0.30, 0.40, "[0.30,0.40)"),
        (0.40, 0.5000001, "[0.40,0.50]"),
    ]
    out = []
    scopes = [(str(y), [r for r in rows if r["year"] == y]) for y in YEARS] + [
        ("pooled_2018_2021", rows)
    ]
    for scope, data in scopes:
        for left, right, name in bins:
            chosen = [r for r in data if left <= r["Global24_margin"] < right]
            row = {
                "scope": scope,
                "margin_bucket": name,
                "confidence_band": "low"
                if right <= 0.10
                else ("medium" if right <= 0.30 else "high"),
                "n": len(chosen),
            }
            for key, pred in {
                "correct": lambda r: r["Global24_correct"] == 1,
                "FP": lambda r: r["Global24_pred"] == 1 and r["label"] == 0,
                "FN": lambda r: r["Global24_pred"] == 0 and r["label"] == 1,
            }.items():
                subset = [r for r in chosen if pred(r)]
                row[key] = len(subset)
                row[key + "_routed"] = sum(r["routed"] for r in subset)
                row[key + "_unrouted"] = sum(not r["routed"] for r in subset)
            row["highconf_unrouted_error"] = sum(
                row["confidence_band"] == "high" and r["cohort"] == "B" for r in chosen
            )
            row["highconf_routed_unfixed"] = sum(
                row["confidence_band"] == "high" and r["cohort"] == "D" for r in chosen
            )
            out.append(row)
    return out


def residual_and_bounded(rows):
    residual = []
    bounded = []
    broken = []
    for r in rows:
        base_logit = float(logit(r["Global24_prob"]))
        effective = (
            float(logit(r["M1_expert_prob"]) - base_logit) if r["routed"] else 0.0
        )
        required_abs = abs(base_logit)
        aligned = effective if r["label"] == 1 else -effective
        direction = int(aligned > 0)
        almost = int(abs(effective) < 0.05)
        exceeds = int(required_abs > MAX_DELTA)
        if r["routed"] and not r["Global24_correct"]:
            bounded.append(
                {
                    "year": r["year"],
                    "sha256": r["sha256"],
                    "label": r["label"],
                    "error_type": "FN" if r["label"] == 1 else "FP",
                    "required_delta_to_flip_abs": required_abs,
                    "required_delta_exceeds_2p25": exceeds,
                    "effective_bounded_delta": effective,
                    "correction_direction_correct": direction,
                    "M1_correct": r["M1_correct"],
                }
            )
        if not r["M1_correct"]:
            kind = "FN" if r["label"] == 1 else "FP"
            if r["cohort"] == "B":
                primary = "router_miss"
            elif r["cohort"] == "D":
                primary = residual_primary_failure(r["label"], effective, required_abs)
            elif r["cohort"] == "F":
                primary = "broken_by_stage2"
            else:
                primary = "other"
            residual.append(
                {
                    **r,
                    "final_error_type": kind,
                    "effective_bounded_delta": effective,
                    "raw_delta": "unavailable_not_persisted",
                    "correction_direction_correct": direction,
                    "correction_magnitude": abs(effective),
                    "distance_to_flip": required_abs,
                    "available_bound": MAX_DELTA,
                    "required_delta_exceeds_bound": exceeds,
                    "almost_no_correction_lt_0p05": almost,
                    "primary_failure_type": primary,
                }
            )
        if r["cohort"] == "F":
            near = int(r["Global24_margin"] < 0.10)
            explanation = (
                "F_structure_coverage_insufficient"
                if r["missing_host_indicator"]
                else (
                    "A_stage1_near_boundary"
                    if near
                    else (
                        "B_correction_direction_wrong"
                        if not direction
                        else "UNRESOLVED_model_internal_diagnostics_unavailable"
                    )
                )
            )
            broken.append(
                {
                    **r,
                    "effective_bounded_delta": effective,
                    "raw_delta": "unavailable_not_persisted",
                    "correction_direction_correct": direction,
                    "critical_host": "unavailable_not_persisted",
                    "critical_role": "unavailable_not_persisted",
                    "critical_host_anomaly": "unavailable_not_persisted",
                    "mean_gamma": "unavailable_not_persisted",
                    "max_gamma": "unavailable_not_persisted",
                    "attention_entropy": "unavailable_not_persisted",
                    "max_attention": "unavailable_not_persisted",
                    "broken_explanation": explanation,
                }
            )
    summary = []
    for scope, data in [
        (str(y), [r for r in bounded if r["year"] == y]) for y in YEARS
    ] + [("pooled_2018_2021", bounded)]:
        for error_type in ("ALL", "FP", "FN"):
            chosen = (
                data
                if error_type == "ALL"
                else [r for r in data if r["error_type"] == error_type]
            )
            summary.append(
                {
                    "scope": scope,
                    "error_type": error_type,
                    "routed_stage1_error_n": len(chosen),
                    "required_delta_exceeds_2p25_n": sum(
                        r["required_delta_exceeds_2p25"] for r in chosen
                    ),
                    "required_delta_exceeds_2p25_rate": safe_div(
                        sum(r["required_delta_exceeds_2p25"] for r in chosen),
                        len(chosen),
                    ),
                    "direction_correct_n": sum(
                        r["correction_direction_correct"] for r in chosen
                    ),
                }
            )
    return residual, summary, broken


def hostego_table(rows):
    wanted = [
        "year",
        "fold",
        "sha256",
        "label",
        "cohort",
        "routed",
        "Global24_prob",
        "DI_score_if_available",
        "M1_prob",
        "real_host_count",
        "nontrivial_ego_host_count",
        "ego_host_ratio",
        "mean_neighbor_count",
        "max_neighbor_count",
        "incoming_neighbor_mean",
        "outgoing_neighbor_mean",
        "host_role_diversity_count",
        "host_role_diversity_ratio",
        "ego_role_diversity_count",
        "ego_role_diversity_ratio",
        "missing_host_indicator",
        "global_role_presence_count",
        "global_role_node_total",
        "global_role_edge_total",
    ]
    output = []
    for r in rows:
        row = {k: r.get(k, "") for k in wanted}
        row.update(
            {
                "mean_gamma": "unavailable_not_persisted",
                "max_gamma": "unavailable_not_persisted",
                "median_gamma": "unavailable_not_persisted",
                "attention_entropy": "unavailable_not_persisted",
                "max_attention": "unavailable_not_persisted",
                "top1_attention": "unavailable_not_persisted",
                "top3_attention_sum": "unavailable_not_persisted",
                "mean_anomaly": "unavailable_not_persisted",
                "max_anomaly": "unavailable_not_persisted",
                "critical_host_anomaly": "unavailable_not_persisted",
                "external_distance": "unavailable_not_persisted",
                "internal_distance": "unavailable_not_persisted",
                "shared_commonness": "unavailable_not_persisted",
            }
        )
        for key, value in r.items():
            if key.startswith("compact__"):
                row[key] = value
        output.append(row)
    return output


def cohort_comparison(rows):
    groups = {c: [r for r in rows if r["cohort"] == c] for c in "BCDEF"}
    variables = [
        "Global24_prob",
        "Global24_margin",
        "DI_score_if_available",
        "real_host_count",
        "nontrivial_ego_host_count",
        "ego_host_ratio",
        "mean_neighbor_count",
        "max_neighbor_count",
        "incoming_neighbor_mean",
        "outgoing_neighbor_mean",
        "host_role_diversity_count",
        "ego_role_diversity_count",
        "global_role_node_total",
        "global_role_edge_total",
    ]
    out = []
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        mannwhitneyu = None
    for variable in variables:
        for c in "BCDEF":
            v = np.asarray([finite_float(r.get(variable)) for r in groups[c]], float)
            v = v[np.isfinite(v)]
            out.append(
                {
                    "row_type": "descriptive",
                    "variable": variable,
                    "cohort": c,
                    "comparison": "",
                    "n": len(v),
                    "mean": float(np.mean(v)) if len(v) else float("nan"),
                    "median": float(np.median(v)) if len(v) else float("nan"),
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                    "IQR": iqr(v),
                    "cohens_d": "",
                    "mannwhitney_U": "",
                    "mannwhitney_p": "",
                }
            )
        for other in "DFBE":
            a = np.asarray([finite_float(r.get(variable)) for r in groups["C"]], float)
            b = np.asarray(
                [finite_float(r.get(variable)) for r in groups[other]], float
            )
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            u = p = float("nan")
            if mannwhitneyu is not None and len(a) and len(b):
                test = mannwhitneyu(a, b, alternative="two-sided")
                u = float(test.statistic)
                p = float(test.pvalue)
            out.append(
                {
                    "row_type": "effect",
                    "variable": variable,
                    "cohort": "",
                    "comparison": f"C_vs_{other}",
                    "n": "",
                    "mean": "",
                    "median": "",
                    "std": "",
                    "IQR": "",
                    "cohens_d": cohens_d(a, b),
                    "mannwhitney_U": u,
                    "mannwhitney_p": p,
                }
            )
    return out


def role_enrichment(rows, role_names, role_nodes):
    definitions = {
        "fixed": {"C"},
        "unfixed": {"D"},
        "broken": {"F"},
        "unrouted_error": {"B"},
        "correct_control": {"A", "E"},
    }
    out = []
    for j, name in enumerate(role_names):
        row = {"role_index": j, "role": name}
        rates = {}
        for label, cohorts in definitions.items():
            idx = [i for i, r in enumerate(rows) if r["cohort"] in cohorts]
            values = role_nodes[idx, j] if idx else np.asarray([], int)
            rates[label] = float(np.mean(values > 0)) if len(values) else float("nan")
            row[label + "_n"] = len(values)
            row[label + "_presence_rate"] = rates[label]
            row[label + "_mean_count"] = (
                float(np.mean(values)) if len(values) else float("nan")
            )
            total = role_nodes[idx].sum(axis=1) if idx else np.asarray([], int)
            row[label + "_mean_proportion"] = (
                float(
                    np.mean(
                        np.divide(
                            values,
                            total,
                            out=np.zeros_like(values, dtype=float),
                            where=total > 0,
                        )
                    )
                )
                if len(values)
                else float("nan")
            )
        control = rates["correct_control"]
        for label in ("unfixed", "broken", "unrouted_error"):
            row[label + "_vs_correct_ratio"] = safe_div(rates[label], control)
        out.append(row)
    return out


def pair_enrichment(rows, role_names, pair_by_apk):
    definitions = {
        "fixed": {"C"},
        "unfixed": {"D"},
        "broken": {"F"},
        "correct": {"A", "E"},
    }
    all_pairs = sorted({pair for counter in pair_by_apk for pair in counter})
    out = []
    for pair in all_pairs:
        row = {
            "center_role_index": pair[0],
            "center_role": role_names[pair[0]],
            "neighbor_role_index": pair[1],
            "neighbor_role": role_names[pair[1]],
            "direction": pair[2],
        }
        for name, cohorts in definitions.items():
            idx = [i for i, r in enumerate(rows) if r["cohort"] in cohorts]
            counts = np.asarray([pair_by_apk[i][pair] for i in idx], int)
            row[name + "_apk_n"] = len(idx)
            row[name + "_edge_frequency"] = int(counts.sum())
            row[name + "_presence_rate"] = (
                float(np.mean(counts > 0)) if len(counts) else float("nan")
            )
        out.append(row)
    return out


def residual_clusters(residual):
    features = [
        "Global24_prob",
        "DI_score_if_available",
        "routed",
        "Global24_margin",
        "real_host_count",
        "ego_host_ratio",
        "mean_neighbor_count",
        "max_neighbor_count",
        "incoming_neighbor_mean",
        "outgoing_neighbor_mean",
        "host_role_diversity_count",
        "ego_role_diversity_count",
        "correction_magnitude",
        "effective_bounded_delta",
    ]
    x = np.asarray(
        [[finite_float(r.get(k), 0.0) for k in features] for r in residual], float
    )
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    z = StandardScaler().fit_transform(x)
    assignments = []
    summaries = []
    for k in (2, 3, 4, 5):
        if len(residual) <= k:
            continue
        labels = KMeans(n_clusters=k, random_state=20260920, n_init=20).fit_predict(z)
        sil = (
            float(silhouette_score(z, labels)) if len(set(labels)) > 1 else float("nan")
        )
        for i, r in enumerate(residual):
            assignments.append(
                {
                    "k": k,
                    "silhouette_score": sil,
                    "cluster": int(labels[i]),
                    "sha256": r["sha256"],
                    "year": r["year"],
                    "error_type": r["final_error_type"],
                    "cohort": r["cohort"],
                    **{f: r.get(f, "") for f in features},
                }
            )
        for c in range(k):
            idx = np.flatnonzero(labels == c)
            subset = [residual[i] for i in idx]
            row = {
                "k": k,
                "silhouette_score": sil,
                "cluster": c,
                "n": len(subset),
                "years_present": len(set(r["year"] for r in subset)),
                "year_distribution": ";".join(
                    f"{y}:{sum(r['year']==y for r in subset)}" for y in YEARS
                ),
                "FP_n": sum(r["final_error_type"] == "FP" for r in subset),
                "FN_n": sum(r["final_error_type"] == "FN" for r in subset),
                "routed_n": sum(r["routed"] for r in subset),
            }
            for f in features:
                row["mean_" + f] = (
                    float(np.mean([finite_float(r.get(f), 0.0) for r in subset]))
                    if subset
                    else float("nan")
                )
            summaries.append(row)
    return assignments, summaries


def recurring_patterns(residual, cluster_summary):
    out = []
    patterns = defaultdict(list)
    for r in residual:
        patterns[
            (r["final_error_type"], r["cohort"], r["primary_failure_type"])
        ].append(r)
    for key, subset in patterns.items():
        years = sorted(set(r["year"] for r in subset))
        out.append(
            {
                "pattern_source": "rule_based",
                "pattern": "|".join(key),
                "n": len(subset),
                "years_present_n": len(years),
                "years": ",".join(map(str, years)),
                "recurrence": "cross-year recurring"
                if len(years) >= 3
                else "year-specific",
            }
        )
    for row in cluster_summary:
        out.append(
            {
                "pattern_source": f"KMeans_k{row['k']}",
                "pattern": f"cluster_{row['cluster']}",
                "n": row["n"],
                "years_present_n": row["years_present"],
                "years": row["year_distribution"],
                "recurrence": "cross-year recurring"
                if row["years_present"] >= 3
                else "year-specific",
            }
        )
    return out


def ba_gap(rows):
    out = []
    for scope, data in [
        (str(y), [r for r in rows if r["year"] == y]) for y in YEARS
    ] + [("pooled_2018_2021", rows)]:
        y = np.asarray([r["label"] for r in data], int)
        p = np.asarray([r["M1_prob"] for r in data], float)
        base = np.asarray([r["Global24_prob"] for r in data], float)
        current = metric(y, p)
        variants = {}
        for label, cohorts in {
            "fix_only_unrouted_stage1_errors": {"B"},
            "fix_only_routed_unfixed": {"D"},
            "eliminate_broken": {"F"},
        }.items():
            q = np.array(p, copy=True)
            mask = np.asarray([r["cohort"] in cohorts for r in data], bool)
            if label == "eliminate_broken":
                q[mask] = base[mask]
            else:
                q[mask] = np.where(y[mask] == 1, 1 - 1e-9, 1e-9)
            variants[label] = metric(y, q)["BA"] - current["BA"]
        out.append(
            {
                "scope": scope,
                "target_BA": 0.95,
                "M1_BA": current["BA"],
                "gap_to_95": 0.95 - current["BA"],
                "router_miss_contribution_max_BA_gain": variants[
                    "fix_only_unrouted_stage1_errors"
                ],
                "stage2_unfixed_contribution_max_BA_gain": variants[
                    "fix_only_routed_unfixed"
                ],
                "broken_contribution_max_BA_gain": variants["eliminate_broken"],
            }
        )
    return out


def write_budgets(rows):
    for scope, data in [
        (str(y), [r for r in rows if r["year"] == y]) for y in YEARS
    ] + [("POOLED", rows)]:
        c = Counter(r["cohort"] for r in data)
        errors = c["B"] + c["C"] + c["D"]
        lines = [
            f"CARE-Droid error budget: {scope}",
            DISCLAIMER,
            "",
            f"Global24 total errors = {errors}",
            f"├── routed errors = {c['C']+c['D']}",
            f"│   ├── fixed by M1 = {c['C']}",
            f"│   └── still wrong = {c['D']}",
            f"└── unrouted errors = {c['B']}",
            "",
            f"routed originally-correct = {c['E']+c['F']}",
            f"└── broken by M1 = {c['F']}",
            "",
            f"M1 remaining errors = {c['D']+c['B']+c['F']}",
            f"M1 remaining FP = {sum(not r['M1_correct'] and r['label']==0 for r in data)}",
            f"M1 remaining FN = {sum(not r['M1_correct'] and r['label']==1 for r in data)}",
        ]
        (ROOT / f"ERROR_BUDGET_{scope}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def read_maybe_gzip_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def malscan_audit(ids, rows):
    candidates = []
    for path in Path("artifacts").rglob("*"):
        low = str(path).lower()
        if (
            path.is_file()
            and "malscan" in low
            and "prediction" in path.name.lower()
            and path.suffix.lower() in {".csv", ".gz"}
        ):
            candidates.append(path)
    lines = [
        "MalScan per-sample prediction search",
        f"candidate_files: {len(candidates)}",
    ]
    for p in candidates[:100]:
        lines.append(str(p))
    care = {r["sha256"]: r for r in rows}
    exact = []
    id_keys = ("sha256", "sample_id", "apk_sha256")
    label_keys = ("label", "y_true", "target")
    prob_keys = ("probability", "prob", "p_malware", "malware_probability", "score")
    pred_keys = ("prediction", "pred", "y_pred")
    for path in candidates:
        try:
            candidate = read_maybe_gzip_csv(path)
        except Exception as exc:
            lines.append(f"UNREADABLE {path}: {type(exc).__name__}")
            continue
        if not candidate:
            continue
        names = set(candidate[0])
        id_key = next((k for k in id_keys if k in names), None)
        label_key = next((k for k in label_keys if k in names), None)
        prob_key = next((k for k in prob_keys if k in names), None)
        pred_key = next((k for k in pred_keys if k in names), None)
        if not id_key or not label_key or (not prob_key and not pred_key):
            continue
        by = {normal(r[id_key]): r for r in candidate}
        if len(candidate) != EXPECTED_N or len(by) != EXPECTED_N or set(by) != set(ids):
            continue
        if any(
            int(float(by[sha][label_key])) != int(care[sha]["label"]) for sha in ids
        ):
            lines.append(f"LABEL_MISMATCH {path}")
            continue
        exact.append((path, by, prob_key, pred_key))
    if exact:
        path, by, prob_key, pred_key = exact[0]
        context = ""
        for sibling in list(path.parent.glob("*protocol*.json")) + list(
            path.parent.glob("*REPORT*.txt")
        ):
            try:
                context += (
                    "\n" + sibling.read_text(encoding="utf-8", errors="ignore").lower()
                )
            except Exception:
                pass
        directly_comparable = ("same-year" in context or "within-year" in context) and (
            "five-fold" in context or "5-fold" in context
        )
        status = (
            "DIRECTLY_COMPARABLE"
            if directly_comparable
            else "PROTOCOL_NOT_DIRECTLY_COMPARABLE"
        )
        disagreement = []
        for sha in ids:
            c = care[sha]
            m = by[sha]
            mp = (
                finite_float(m[prob_key])
                if prob_key
                else float(int(float(m[pred_key])))
            )
            mcorrect = int((mp >= 0.5) == int(c["label"]))
            ccorrect = int(c["M1_correct"])
            category = (
                "Both_correct"
                if mcorrect and ccorrect
                else (
                    "CARE_only_correct"
                    if ccorrect
                    else ("MalScan_only_correct" if mcorrect else "Both_wrong")
                )
            )
            disagreement.append(
                {
                    "protocol_status": status,
                    "sha256": sha,
                    "year": c["year"],
                    "label": c["label"],
                    "category": category,
                    "CARE_M1_prob": c["M1_prob"],
                    "MalScan_prob_or_prediction": mp,
                    "Global24_prob": c["Global24_prob"],
                    "DI_score": c["DI_score_if_available"],
                    "routed": c["routed"],
                    "error_type": ""
                    if ccorrect
                    else ("FN" if c["label"] == 1 else "FP"),
                    "real_host_count": c["real_host_count"],
                    "ego_host_ratio": c["ego_host_ratio"],
                    "mean_neighbor_count": c["mean_neighbor_count"],
                    "anomaly": "unavailable_not_persisted",
                    "attention": "unavailable_not_persisted",
                    "effective_bounded_delta": float(
                        logit(c["M1_expert_prob"]) - logit(c["Global24_prob"])
                    )
                    if c["routed"]
                    else 0.0,
                }
            )
        write_csv(ROOT / "M1_vs_MalScan_disagreement.csv", disagreement)
        lines.extend(
            ["", f"exact_SHA_label_candidate: {path}", f"protocol_status: {status}"]
        )
        if not directly_comparable:
            lines.append(
                "No comparative performance conclusion is permitted for this candidate."
            )
    else:
        lines.extend(
            ["", "No exact 1989-SHA, label-matched per-sample MalScan file was found."]
        )
    lines.extend(
        [
            "",
            "No comparative performance conclusion is emitted unless an explicit independent same-year five-fold OOF protocol contract is present.",
            "PROTOCOL_NOT_DIRECTLY_COMPARABLE applies to candidates lacking that complete contract.",
        ]
    )
    (ROOT / "malscan_search_audit.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def bottleneck_labels(ceiling, efficiency):
    annual = []
    for y in YEARS:
        c = next(r for r in ceiling if r["scope"] == str(y))
        e = next(r for r in efficiency if r["scope"] == str(y))
        router = "YES" if c["DI25_oracle_BA"] < 0.95 else "NO"
        stage2 = (
            "YES"
            if c["M1_BA"] < 0.95 <= c["DI25_oracle_BA"]
            else ("MIXED" if e["unfixed_routed_errors"] or e["broken"] else "NO")
        )
        annual.append((y, router, stage2))
    router_values = {v[1] for v in annual}
    stage_values = {v[2] for v in annual}
    return (
        next(iter(router_values)) if len(router_values) == 1 else "MIXED",
        next(iter(stage_values)) if len(stage_values) == 1 else "MIXED",
        annual,
    )


def improvement_candidates(rows, residual, ceiling, efficiency, recurring):
    b = [r for r in rows if r["cohort"] == "B"]
    d = [r for r in rows if r["cohort"] == "D"]
    f = [r for r in rows if r["cohort"] == "F"]
    d_bound = [
        r for r in residual if r["cohort"] == "D" and r["required_delta_exceeds_bound"]
    ]
    d_wrong = [
        r
        for r in residual
        if r["cohort"] == "D" and not r["correction_direction_correct"]
    ]

    def years(data):
        return len(set(r["year"] for r in data))

    candidates = []
    if b:
        candidates.append(
            (
                "Router miss audit with source-only class-conditional reliability calibration",
                b,
                years(b),
                "Router",
                "Add a source-only, nested-OOF class-conditional reliability calibration to the existing DI score while keeping the 25% budget fixed.",
                "The current route never exposes these Stage-I errors to local evidence; high-confidence misses are especially invisible to a pure margin signal.",
                "unrouted FP/FN",
                "May overfit 2017 error composition and alter which correct samples consume the fixed budget.",
                "NO",
            )
        )
    if d_wrong:
        candidates.append(
            (
                "Correction-direction audit and sign-aware auxiliary supervision",
                d_wrong,
                years(d_wrong),
                "Correction",
                "Add a source-only auxiliary sign objective to the existing correction head without changing its input, capacity, or bound.",
                "These routed errors receive a correction in the wrong label-aligned direction.",
                "routed-unfixed errors",
                "Auxiliary supervision can reduce calibration quality or trade FN against FP.",
                "NO",
            )
        )
    if d_bound:
        candidates.append(
            (
                "Bound-limited cases: improve evidence efficiency before reconsidering protection",
                d_bound,
                years(d_bound),
                "Correction",
                "Keep max_delta=2.25 and first test whether the existing head can concentrate correction near the allowed boundary for source-identified bound-limited cases.",
                "Their Stage-I logit distance exceeds the maximum permitted correction; simply increasing the bound would also increase broken risk.",
                "high-confidence routed-unfixed errors",
                "May still be unable to flip mathematically unreachable cases; changing the bound later would require a separate safety study.",
                "NO",
            )
        )
    if f:
        candidates.append(
            (
                "Evidence-consistency safety gate for harmful corrections",
                f,
                years(f),
                "Attention / Correction",
                "Add a source-only abstention gate that retains Global24 when local evidence does not consistently support the proposed correction.",
                "These APKs were originally correct and become wrong only after Stage II.",
                "broken",
                "A conservative gate may also suppress useful corrections and must be nested-OOF evaluated.",
                "NO",
            )
        )
    low_struct = [
        r
        for r in d + b
        if r["real_host_count"] == 0 or r["nontrivial_ego_host_count"] == 0
    ]
    if low_struct:
        candidates.append(
            (
                "Explicit low-coverage fallback",
                low_struct,
                years(low_struct),
                "Static-analysis coverage / Ego",
                "Use an explicit coverage indicator to fall back to the frozen Stage-I decision when Host/Ego evidence is absent, and diagnose router handling separately.",
                "These failures have no Host or no nontrivial Ego support, so the local branch cannot supply the intended structural evidence.",
                "broken or unrepairable routed errors",
                "Could discard useful Host-only evidence and does not address well-covered residual errors.",
                "NO",
            )
        )
    candidates = sorted(candidates, key=lambda x: (-len(x[1]), -x[2]))[:5]
    lines = [
        "# Targeted improvement candidates",
        "",
        DISCLAIMER,
        "",
        "No change is implemented by this diagnostic package. Candidates are hypotheses for a later, separately approved source-only study.",
        "",
    ]
    for i, (
        title,
        data,
        yn,
        bottleneck,
        change,
        why,
        benefit,
        risk,
        future,
    ) in enumerate(candidates, 1):
        priority = (
            "HIGH"
            if yn >= 3 or len(data) >= max(20, int(0.15 * max(len(residual), 1)))
            else ("MEDIUM" if len(data) >= 5 else "LOW")
        )
        lines.extend(
            [
                f"## {i}. {title}",
                "",
                f"Observed failure: {why}",
                f"Affected samples: {len(data)} APKs across {yn}/4 years.",
                f"Bottleneck: {bottleneck}",
                f"Proposed minimal change: {change}",
                f"Why: {why}",
                f"Expected benefit: reduce {benefit}.",
                f"Risk: {risk}",
                f"Requires future label tuning: {future}",
                f"Recommended priority: {priority}",
                "",
            ]
        )
    (ROOT / "TARGETED_IMPROVEMENT_CANDIDATES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def final_report(
    rows,
    capture,
    ceiling,
    efficiency,
    bounded,
    recurring,
    gap,
    availability,
    compact_source,
):
    router_b, stage_b, annual_b = bottleneck_labels(ceiling, efficiency)
    cap = {r["scope"]: r for r in capture}
    eff = {r["scope"]: r for r in efficiency}
    cei = {r["scope"]: r for r in ceiling}
    gaps = {r["scope"]: r for r in gap}
    recurring_rules = [
        r
        for r in recurring
        if r["pattern_source"] == "rule_based"
        and r["recurrence"] == "cross-year recurring"
    ]
    lines = [
        "CARE-Droid FINAL ERROR DIAGNOSIS REPORT",
        "",
        DISCLAIMER,
        "",
        "Protocol: independent same-year five-fold OOF for each of 2018, 2019, 2020, and 2021.",
        "Frozen method: CARE-Droid-DI25-HostEgo-Anomaly-Bounded.",
        "DI25 budget=25%; max_delta_logit=2.25. No model was trained or modified by this program.",
        "",
        f"ROUTER_BOTTLENECK = {router_b}",
        f"STAGE2_BOTTLENECK = {stage_b}",
        "",
        "Annual bottleneck-to-95 diagnostic: "
        + "; ".join(f"{y}: router={rb}, stage2={sb}" for y, rb, sb in annual_b),
        "",
        "Q1-Q4. Global24 errors and DI25 capture:",
    ]
    for y in YEARS:
        r = cap[str(y)]
        lines.append(
            f"{y}: errors={r['Global24_error_n']} (FP={r['Global24_FP']}, FN={r['Global24_FN']}), captured={r['routed_Global24_error_n']}, error_recall={r['error_recall']:.4f}, FP_recall={r['FP_error_recall']:.4f}, FN_recall={r['FN_error_recall']:.4f}."
        )
    p = cap["pooled_2018_2021"]
    e = eff["pooled_2018_2021"]
    c = cei["pooled_2018_2021"]
    g = gaps["pooled_2018_2021"]
    final_fn = [r for r in rows if not r["M1_correct"] and r["label"] == 1]
    final_fp = [r for r in rows if not r["M1_correct"] and r["label"] == 0]
    lines.extend(
        [
            "",
            f"Q5. Errors never entering Stage II: {p['unrouted_Global24_error_n']} (FP={p['unrouted_FP']}, FN={p['unrouted_FN']}).",
            f"Q6-Q8. Routed Stage-I errors fixed={e['fixed']}; unfixed={e['unfixed_routed_errors']}; correction_recall={e['correction_recall']:.4f}; broken={e['broken']}; break_rate={e['break_rate']:.4f}.",
            f"Q9. Final FN={len(final_fn)}: unrouted={sum(r['cohort']=='B' for r in final_fn)}, routed-unfixed={sum(r['cohort']=='D' for r in final_fn)}, broken={sum(r['cohort']=='F' for r in final_fn)}.",
            f"Q10. Final FP={len(final_fp)}: unrouted={sum(r['cohort']=='B' for r in final_fp)}, routed-unfixed={sum(r['cohort']=='D' for r in final_fp)}, broken={sum(r['cohort']=='F' for r in final_fp)}.",
            f"Q11. Pooled DI25 perfect-Stage-II oracle BA={c['DI25_oracle_BA']:.6f}; current M1 BA={c['M1_BA']:.6f}; realized_fraction={c['realized_fraction_of_router_opportunity']:.4f}.",
            "Q12-Q13. The target-specific YES/NO/MIXED labels above use the exact oracle: oracle<0.95 means current routing alone cannot reach 0.95; M1<0.95<=oracle means Stage II leaves sufficient routed opportunity unrealized.",
            "Q14. The three prototype-background values and per-APK anomaly/attention tensors were not persisted by the completed same-year run; no claim about their error pattern is made.",
            "Q15. Host/Ego structural differences between fixed and unfixed cohorts are reported in cohort_feature_comparison.csv; model-internal gamma/attention differences are unavailable.",
            f"Q16. Cross-year recurring rule-based failure patterns={len(recurring_rules)}; see cross_year_failure_patterns.csv.",
            "Q17. MalScan comparison is produced only if an exact 1989-SHA, label-matched, independently same-year five-fold OOF prediction contract is found; see malscan_search_audit.txt.",
            f"Q18. Maximum isolated BA gains from current M1: router misses={g['router_miss_contribution_max_BA_gain']:.6f}, routed-unfixed={g['stage2_unfixed_contribution_max_BA_gain']:.6f}, removing broken={g['broken_contribution_max_BA_gain']:.6f}.",
            "Q19. Evidence-linked minimal candidates are in TARGETED_IMPROVEMENT_CANDIDATES.md.",
            "Q20. No modification or new CARE-Droid variant was implemented; awaiting user confirmation.",
            "",
            "Artifact availability:",
            f"Compact feature source: {compact_source}",
        ]
    )
    lines.extend(f"- {k}: {v}" for k, v in availability.items())
    (ROOT / "FINAL_ERROR_DIAGNOSIS_REPORT.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return router_b, stage_b


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "PROTOCOL_BOUNDARY.txt").write_text(DISCLAIMER + "\n", encoding="utf-8")
    core = load_core()
    (
        ids,
        router_rows,
        m0_index,
        m1_index,
        years,
        folds,
        labels,
        p0,
        di,
        route,
        pm0,
        pm1expert,
        pm1,
        actual,
    ) = core
    role_names, role_nodes, role_edges, structures, pairs = load_structure(ids, years)
    compact, compact_source = load_optional_compact(ids)
    rows = core_rows(
        ids,
        router_rows,
        years,
        folds,
        labels,
        p0,
        di,
        route,
        pm0,
        pm1expert,
        pm1,
        structures,
        compact,
    )
    capture = error_capture(rows)
    efficiency = stage2_efficiency(rows)
    ceiling, oracles = oracle_tables(rows)
    residual, bounded, broken = residual_and_bounded(rows)
    confidence = confidence_table(rows)
    hostego = hostego_table(rows)
    cohort_stats = cohort_comparison(rows)
    roles = role_enrichment(rows, role_names, role_nodes)
    pair_rows = pair_enrichment(rows, role_names, pairs)
    clusters, cluster_summary = residual_clusters(residual)
    recurring = recurring_patterns(residual, cluster_summary)
    gap = ba_gap(rows)
    write_csv(ROOT / "DI25_error_capture_by_year.csv", capture)
    write_csv(ROOT / "DI25_oracle_ceiling.csv", ceiling)
    write_csv(ROOT / "error_cohort_per_apk.csv", rows)
    write_csv(ROOT / "stage2_correction_efficiency.csv", efficiency)
    write_csv(ROOT / "confidence_error_analysis.csv", confidence)
    write_csv(ROOT / "residual_error_analysis.csv", residual)
    write_csv(ROOT / "bounded_constraint_analysis.csv", bounded)
    write_csv(ROOT / "hostego_failure_analysis.csv", hostego)
    write_csv(ROOT / "cohort_feature_comparison.csv", cohort_stats)
    write_csv(ROOT / "role_error_enrichment.csv", roles)
    write_csv(ROOT / "ego_role_pair_enrichment.csv", pair_rows)
    write_csv(ROOT / "broken_case_analysis.csv", broken)
    write_csv(ROOT / "residual_error_clusters.csv", clusters)
    write_csv(ROOT / "residual_error_cluster_summary.csv", cluster_summary)
    write_csv(ROOT / "cross_year_failure_patterns.csv", recurring)
    write_csv(ROOT / "BA_gap_decomposition.csv", gap)
    write_csv(ROOT / "source_oracle_analysis.csv", oracles)
    write_budgets(rows)
    malscan_audit(ids, rows)
    availability = {
        "frozen_predictions": "available_exact",
        "DI_score_rank": "available_exact",
        "Host_Ego_raw_structure": "available_exact",
        "31_role_graph": "available_exact",
        "compact_static_features": "available" if compact else "unavailable",
        "effective_bounded_delta": "recoverable_exact_from_saved_probabilities",
        "raw_delta": "unavailable_not_persisted",
        "gamma": "unavailable_not_persisted",
        "attention": "unavailable_not_persisted",
        "anomaly_prototype_background_values": "unavailable_not_persisted",
        "critical_host": "unavailable_not_persisted",
    }
    write_csv(
        ROOT / "diagnostic_field_availability.csv",
        [{"field": k, "status": v} for k, v in availability.items()],
    )
    improvement_candidates(rows, residual, ceiling, efficiency, recurring)
    router_b, stage_b = final_report(
        rows,
        capture,
        ceiling,
        efficiency,
        bounded,
        recurring,
        gap,
        availability,
        compact_source,
    )
    protocol = {
        "passed": True,
        "diagnostic_only": True,
        "protocol": "independent same-year five-fold OOF",
        "years": list(YEARS),
        "n": EXPECTED_N,
        "frozen_method": "CARE-Droid-DI25-HostEgo-Anomaly-Bounded",
        "DI25_budget": 0.25,
        "max_delta_logit": MAX_DELTA,
        "router_bottleneck": router_b,
        "stage2_bottleneck": stage_b,
        "disclaimer": DISCLAIMER,
        "existing_predictions_modified": False,
        "new_model_trained": False,
        "availability": availability,
    }
    (ROOT / "diagnosis_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    required = [
        "DI25_error_capture_by_year.csv",
        "DI25_oracle_ceiling.csv",
        "error_cohort_per_apk.csv",
        "stage2_correction_efficiency.csv",
        "confidence_error_analysis.csv",
        "residual_error_analysis.csv",
        "bounded_constraint_analysis.csv",
        "hostego_failure_analysis.csv",
        "role_error_enrichment.csv",
        "ego_role_pair_enrichment.csv",
        "broken_case_analysis.csv",
        "residual_error_clusters.csv",
        "cross_year_failure_patterns.csv",
        "BA_gap_decomposition.csv",
        "source_oracle_analysis.csv",
        "TARGETED_IMPROVEMENT_CANDIDATES.md",
        "FINAL_ERROR_DIAGNOSIS_REPORT.txt",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError("diagnosis output incomplete: " + repr(missing))
    print("CARE_DI25_HOSTEGO_ERROR_DIAGNOSIS_PASS")
    print(DISCLAIMER)
    print(f"ROUTER_BOTTLENECK={router_b} STAGE2_BOTTLENECK={stage_b}")
    print(f"Output: {ROOT}")


if __name__ == "__main__":
    main()
