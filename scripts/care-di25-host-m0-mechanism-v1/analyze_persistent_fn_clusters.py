#!/usr/bin/env python3

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


ROOT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis"
)

PAIR = (
    ROOT
    / "analysis/direct_vs_bounded"
    / "paired_overlap.csv"
)

DIAG = (
    ROOT
    / "malben_signed_capture"
    / "malben_diagnostics.csv"
)

REGIONS = Path(
    "artifacts/care-future-local-assets-v2/"
    "regions_2018_2021.npz"
)

OUT = (
    ROOT
    / "analysis/persistent_fn_cluster_audit"
)


def norm(x):
    return str(x).strip().lower()


def hash_array(a):
    a = np.ascontiguousarray(a)

    h = hashlib.sha256()
    h.update(
        str(a.shape).encode("ascii")
    )
    h.update(
        str(a.dtype).encode("ascii")
    )
    h.update(a.tobytes())

    return h.hexdigest()


def sort_rows(a):
    """
    Order-independent representation of one Host bag.
    """
    a = np.asarray(a)

    if len(a) <= 1:
        return np.ascontiguousarray(a)

    # Primary key = column 0, then 1, ...
    order = np.lexsort(
        a.T[::-1]
    )

    return np.ascontiguousarray(
        a[order]
    )


def bag_summary(a):
    """
    Simple diagnostic-only summary.
    Not used by CARE-Droid.
    """
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    if len(a) == 0:
        return np.zeros(
            74 * 4,
            dtype=np.float64,
        )

    return np.concatenate([
        np.mean(a, axis=0),
        np.std(a, axis=0),
        np.min(a, axis=0),
        np.max(a, axis=0),
    ])


def main():

    p = pd.read_csv(PAIR)

    p["sha256"] = (
        p["sha256"]
        .astype(str)
        .map(norm)
    )

    d = pd.read_csv(DIAG)

    d["sha256"] = (
        d["sha256"]
        .astype(str)
        .map(norm)
    )

    # ---------------------------------------------
    # All 2020 routed FN
    # ---------------------------------------------

    fn = p[
        (p["year"] == 2020)
        &
        (p["error_type"] == "FN")
    ].copy()

    if len(fn) != 14:
        raise RuntimeError(
            f"expected 14 routed 2020 FN, got {len(fn)}"
        )

    fn["persistent"] = (
        (
            fn["bounded_state"]
            == "WRONG_DIRECTION"
        )
        &
        (
            fn["direct_state"]
            == "WRONG_DIRECTION"
        )
    )

    persistent = fn[
        fn["persistent"]
    ].copy()

    if len(persistent) != 8:
        raise RuntimeError(
            f"expected 8 persistent FN, got {len(persistent)}"
        )

    # ---------------------------------------------
    # Load raw Host bags
    # ---------------------------------------------

    with np.load(
        REGIONS,
        allow_pickle=False,
    ) as z:

        print("REGION KEYS:")
        print(z.files)
        print()

        ids = [
            norm(x)
            for x in z["sha256"]
            .astype(str)
        ]

        values = np.asarray(
            z["host_values"]
        )

        offsets = np.asarray(
            z["host_offsets"],
            dtype=np.int64,
        )

        counts = np.asarray(
            z["real_host_count"],
            dtype=np.int64,
        )

    if len(ids) != 1989:
        raise RuntimeError(
            f"region cohort changed: {len(ids)}"
        )

    index = {
        sha: i
        for i, sha in enumerate(ids)
    }

    if len(index) != 1989:
        raise RuntimeError(
            "duplicate region SHA"
        )

    # ---------------------------------------------
    # Extract all 14 FN bags
    # ---------------------------------------------

    bags = {}
    rows = []
    summaries = []

    for _, r in fn.iterrows():

        sha = r["sha256"]

        if sha not in index:
            raise RuntimeError(
                "missing SHA in region cache: "
                + sha
            )

        i = index[sha]

        left = int(offsets[i])
        right = int(offsets[i + 1])

        bag = np.asarray(
            values[left:right]
        )

        bags[sha] = bag

        ordered = sort_rows(bag)

        # Exact feature-set identity.
        exact_hash = hash_array(
            ordered
        )

        # Also check equality after tiny numerical
        # rounding, to catch effectively identical bags.
        rounded = np.round(
            ordered.astype(np.float64),
            6,
        )

        rounded_hash = hash_array(
            rounded
        )

        summary = bag_summary(
            bag
        )

        summaries.append(summary)

        rows.append({
            "sha256": sha,
            "persistent":
                bool(r["persistent"]),

            "bounded_state":
                r["bounded_state"],

            "direct_state":
                r["direct_state"],

            "host_rows":
                len(bag),

            "real_host_count":
                int(counts[i]),

            "exact_host_hash":
                exact_hash,

            "rounded6_host_hash":
                rounded_hash,
        })

    audit = pd.DataFrame(rows)

    # ---------------------------------------------
    # Add MalBen diagnostic values
    # ---------------------------------------------

    diagnostic_columns = [
        "sha256",
        "e_mal",
        "e_ben",
        "evidence_difference",
        "mean_evidence_strength",
        "max_abs_margin",
    ]

    audit = audit.merge(
        d[diagnostic_columns],
        on="sha256",
        how="left",
        validate="one_to_one",
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        OUT / "2020_routed_fn_cluster_audit.csv",
        index=False,
    )

    # ---------------------------------------------
    # Exact Host-bag clusters
    # ---------------------------------------------

    print()
    print("=" * 120)
    print("PERSISTENT 8: BASIC SIGNATURES")
    print("=" * 120)

    q = audit[
        audit["persistent"]
    ].copy()

    print(
        q[
            [
                "sha256",
                "host_rows",
                "real_host_count",
                "e_mal",
                "e_ben",
                "evidence_difference",
                "max_abs_margin",
                "exact_host_hash",
            ]
        ]
        .sort_values(
            [
                "exact_host_hash",
                "sha256",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("PERSISTENT 8: UNIQUE SIGNATURE COUNTS")
    print("=" * 120)

    print(
        "persistent APKs             =",
        len(q)
    )

    print(
        "unique exact Host bags     =",
        q["exact_host_hash"].nunique()
    )

    print(
        "unique rounded-6 Host bags =",
        q["rounded6_host_hash"].nunique()
    )

    print()
    print("=" * 120)
    print("EXACT HOST-BAG CLUSTERS")
    print("=" * 120)

    cluster_rows = []

    for h, g in q.groupby(
        "exact_host_hash"
    ):

        members = sorted(
            g["sha256"].tolist()
        )

        cluster_rows.append({
            "host_hash": h,
            "n": len(g),
            "members":
                " | ".join(members),
        })

        if len(g) > 1:

            print()
            print(
                "cluster size =",
                len(g)
            )

            for sha in members:
                print(" ", sha)

    cluster_df = pd.DataFrame(
        cluster_rows
    )

    cluster_df.to_csv(
        OUT / "persistent_exact_host_clusters.csv",
        index=False,
    )

    # ---------------------------------------------
    # Diagnostic signature clusters
    # ---------------------------------------------

    diag_sig = q[
        [
            "e_mal",
            "e_ben",
            "evidence_difference",
            "mean_evidence_strength",
            "max_abs_margin",
        ]
    ].round(6)

    q = q.copy()

    q["malben_diag_signature"] = [
        "|".join(
            f"{float(v):.6f}"
            for v in row
        )
        for row in diag_sig.to_numpy()
    ]

    print()
    print("=" * 120)
    print("MALBEN DIAGNOSTIC SIGNATURE CLUSTERS")
    print("=" * 120)

    for sig, g in q.groupby(
        "malben_diag_signature"
    ):

        if len(g) > 1:

            print()
            print(
                "cluster size =",
                len(g)
            )

            print("signature =", sig)

            for sha in sorted(
                g["sha256"].tolist()
            ):
                print(" ", sha)

    # ---------------------------------------------
    # Near-similarity of raw Host-bag summaries.
    #
    # Standardizer fitted only to the 14 diagnostic
    # FN samples. This is descriptive, not model input.
    # ---------------------------------------------

    X = np.asarray(
        summaries,
        dtype=np.float64,
    )

    Xs = StandardScaler().fit_transform(
        X
    )

    dist = pairwise_distances(
        Xs,
        metric="euclidean",
    )

    fn_ids = audit[
        "sha256"
    ].tolist()

    persistent_ids = set(
        q["sha256"].tolist()
    )

    nearest_rows = []

    for i, sha in enumerate(fn_ids):

        if sha not in persistent_ids:
            continue

        candidate = []

        for j, other in enumerate(fn_ids):

            if i == j:
                continue

            candidate.append(
                (
                    float(dist[i, j]),
                    other,
                    other
                    in persistent_ids,
                )
            )

        candidate.sort()

        for rank, item in enumerate(
            candidate[:5],
            start=1,
        ):
            dd, other, other_persistent = item

            nearest_rows.append({
                "sha256": sha,
                "rank": rank,
                "neighbor_sha256":
                    other,
                "distance":
                    dd,
                "neighbor_persistent":
                    other_persistent,
            })

    nearest = pd.DataFrame(
        nearest_rows
    )

    nearest.to_csv(
        OUT / "persistent_nearest_neighbors.csv",
        index=False,
    )

    print()
    print("=" * 120)
    print("PERSISTENT HOST-SUMMARY NEAREST NEIGHBORS")
    print("=" * 120)

    print(
        nearest.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    print()
    print("saved:", OUT)


if __name__ == "__main__":
    main()
