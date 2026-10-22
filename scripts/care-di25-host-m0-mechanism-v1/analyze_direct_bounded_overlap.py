#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


P = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis/analysis/direct_vs_bounded/"
    "routed_transitions.csv"
)

OUT = Path(
    "artifacts/care-di25-host-m0-future-replay-v1/"
    "mechanism_diagnosis/analysis/direct_vs_bounded/"
    "paired_overlap.csv"
)


d = pd.read_csv(P)

key = ["sha256", "year", "label", "error_type"]

b = (
    d[d["arm"] == "Bounded"]
    [key + ["failure_type", "p_expert", "effective_delta"]]
    .rename(columns={
        "failure_type": "bounded_state",
        "p_expert": "bounded_prob",
        "effective_delta": "bounded_delta",
    })
)

q = (
    d[d["arm"] == "Direct"]
    [key + ["failure_type", "p_expert", "effective_delta"]]
    .rename(columns={
        "failure_type": "direct_state",
        "p_expert": "direct_prob",
        "effective_delta": "direct_delta",
    })
)

x = b.merge(
    q,
    on=key,
    how="inner",
    validate="one_to_one",
)

x["transition"] = (
    x["bounded_state"]
    + " -> "
    + x["direct_state"]
)

x.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 100)
print("ALL ROUTED STAGE-I ERRORS: BOUNDED -> DIRECT")
print("=" * 100)

print(
    pd.crosstab(
        x["bounded_state"],
        x["direct_state"],
        margins=True,
    ).to_string()
)

for year in [2018, 2019, 2020, 2021]:

    print()
    print("=" * 100)
    print(f"{year} FN: BOUNDED -> DIRECT")
    print("=" * 100)

    z = x[
        (x["year"] == year)
        &
        (x["error_type"] == "FN")
    ]

    print(
        pd.crosstab(
            z["bounded_state"],
            z["direct_state"],
            margins=True,
        ).to_string()
    )

    if year == 2020:

        print()
        print("2020 FN SAMPLE-LEVEL TRANSITIONS")
        print()

        print(
            z[
                [
                    "sha256",
                    "bounded_state",
                    "direct_state",
                    "bounded_delta",
                    "direct_delta",
                ]
            ]
            .sort_values(
                ["bounded_state", "direct_state"]
            )
            .to_string(
                index=False,
                float_format=lambda v: f"{v:.4f}",
            )
        )

print()
print("=" * 100)
print("2020 FN WRONG-DIRECTION OVERLAP")
print("=" * 100)

z = x[
    (x["year"] == 2020)
    &
    (x["error_type"] == "FN")
]

bw = set(
    z.loc[
        z["bounded_state"] == "WRONG_DIRECTION",
        "sha256",
    ]
)

dw = set(
    z.loc[
        z["direct_state"] == "WRONG_DIRECTION",
        "sha256",
    ]
)

inter = bw & dw
union = bw | dw

print("Bounded wrong =", len(bw))
print("Direct wrong  =", len(dw))
print("Both wrong    =", len(inter))
print("Bounded-only  =", len(bw - dw))
print("Direct-only   =", len(dw - bw))
print(
    "Jaccard       =",
    len(inter) / len(union) if union else 0.0,
)

print()
print("BOTH-WRONG SHA:")
for sha in sorted(inter):
    print(sha)

print()
print("saved:", OUT)
