#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import test_minimal_negative_crossing as base


INPUT = base.INPUT

OUT = Path(
    "artifacts/"
    "minimal-crossing-conflict-veto"
)

G_NEG = 16.25
EPS_Z = 1e-6
EPS_R = 1e-12

# 极端负残差边界
DELTA_CUTS = np.round(
    np.arange(
        -0.95,
        -0.399,
        0.025,
    ),
    3,
)

# Compact恶意置信度边界
BASE_P_CUTS = np.round(
    np.arange(
        0.65,
        0.951,
        0.025,
    ),
    3,
)

YEARS = (
    2018,
    2019,
    2020,
    2021,
)


def build_common(df):
    p_base = (
        df["base_probability"]
        .to_numpy(dtype=float)
    )

    z_base = base.logit(
        p_base
    )

    semantic_delta = (
        df["semantic_delta"]
        .to_numpy(dtype=float)
    )

    r = (
        df["gate"]
        .to_numpy(dtype=float)
        *
        semantic_delta
    )

    z_current = (
        z_base + r
    )

    required_gain = np.full(
        len(df),
        np.inf,
        dtype=float,
    )

    valid = (
        (z_base > 0)
        &
        (r < -EPS_R)
    )

    required_gain[
        valid
    ] = (
        z_base[valid]
        /
        (-r[valid])
    )

    eligible = (
        valid
        &
        (z_current >= 0)
        &
        (required_gain <= G_NEG)
    )

    return {
        "p_base":
            p_base,

        "z_base":
            z_base,

        "semantic_delta":
            semantic_delta,

        "r":
            r,

        "z_current":
            z_current,

        "required_gain":
            required_gain,

        "eligible":
            eligible,
    }


def build_probability(
    common,
    delta_cut=None,
    base_p_cut=None,
):
    z = (
        common[
            "z_current"
        ].copy()
    )

    eligible = common[
        "eligible"
    ]

    # 默认执行最小跨界。
    z[
        eligible
    ] = -EPS_Z

    veto = np.zeros(
        len(z),
        dtype=bool,
    )

    # ========================================================
    # 方案A：
    # 仅因为semantic_delta过于极端而否决。
    # ========================================================

    if (
        delta_cut is not None
        and base_p_cut is None
    ):
        veto = (
            eligible
            &
            (
                common[
                    "semantic_delta"
                ]
                <= float(delta_cut)
            )
        )

    # ========================================================
    # 方案B：
    # Compact高置信恶意
    # AND
    # 局部残差极端指向良性
    #
    # 同时成立才否决。
    # ========================================================

    elif (
        delta_cut is not None
        and base_p_cut is not None
    ):
        veto = (
            eligible
            &
            (
                common[
                    "p_base"
                ]
                >= float(base_p_cut)
            )
            &
            (
                common[
                    "semantic_delta"
                ]
                <= float(delta_cut)
            )
        )

    # 被否决的样本恢复当前残差模型输出，
    # 不再执行额外最小跨界。
    z[
        veto
    ] = common[
        "z_current"
    ][
        veto
    ]

    return (
        base.sigmoid(z),
        veto,
    )


def evaluate_variant(
    df,
    probability,
):
    (
        per_seed,
        per_year,
    ) = base.evaluate(
        df,
        probability,
    )

    row = {
        "mean_BA":
            float(
                per_year[
                    "BA"
                ].mean()
            ),

        "min_BA":
            float(
                per_year[
                    "BA"
                ].min()
            ),

        "mean_AUC":
            float(
                per_year[
                    "AUC"
                ].mean()
            ),

        "mean_FP":
            float(
                per_year[
                    "FP"
                ].mean()
            ),

        "mean_FN":
            float(
                per_year[
                    "FN"
                ].mean()
            ),
    }

    for year in YEARS:
        r = per_year[
            per_year[
                "year"
            ] == year
        ].iloc[0]

        row[
            f"BA_{year}"
        ] = float(
            r["BA"]
        )

        row[
            f"AUC_{year}"
        ] = float(
            r["AUC"]
        )

        row[
            f"FP_{year}"
        ] = float(
            r["FP"]
        )

        row[
            f"FN_{year}"
        ] = float(
            r["FN"]
        )

    return (
        row,
        per_seed,
        per_year,
    )


def averaged_probability_diagnostic(
    df,
    candidate_probability,
    baseline_probability,
):
    work = df[
        [
            "year",
            "sha256",
            "label",
        ]
    ].copy()

    work[
        "candidate_probability"
    ] = candidate_probability

    work[
        "baseline_probability"
    ] = baseline_probability

    avg = (
        work.groupby(
            [
                "year",
                "sha256",
                "label",
            ],
            as_index=False,
        )
        .agg(
            candidate_probability=(
                "candidate_probability",
                "mean",
            ),

            baseline_probability=(
                "baseline_probability",
                "mean",
            ),
        )
    )

    cp = (
        avg[
            "candidate_probability"
        ] >= 0.5
    ).astype(int)

    bp = (
        avg[
            "baseline_probability"
        ] >= 0.5
    ).astype(int)

    y = avg[
        "label"
    ].astype(int)

    avg[
        "candidate_correct"
    ] = (
        cp == y
    )

    avg[
        "baseline_correct"
    ] = (
        bp == y
    )

    avg[
        "recovered"
    ] = (
        (~avg[
            "baseline_correct"
        ])
        &
        avg[
            "candidate_correct"
        ]
    )

    avg[
        "lost"
    ] = (
        avg[
            "baseline_correct"
        ]
        &
        (~avg[
            "candidate_correct"
        ])
    )

    rows = []

    for year_value in (
        2018,
        2019,
        2020,
        2021,
        "ALL",
    ):
        if year_value == "ALL":
            g = avg
        else:
            g = avg[
                avg[
                    "year"
                ] == year_value
            ]

        rows.append({
            "year":
                year_value,

            "recovered":
                int(
                    g[
                        "recovered"
                    ].sum()
                ),

            "lost":
                int(
                    g[
                        "lost"
                    ].sum()
                ),

            "net":
                int(
                    g[
                        "recovered"
                    ].sum()
                    -
                    g[
                        "lost"
                    ].sum()
                ),

            "recovered_benign":
                int(
                    (
                        g[
                            "recovered"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 0
                        )
                    ).sum()
                ),

            "recovered_malware":
                int(
                    (
                        g[
                            "recovered"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 1
                        )
                    ).sum()
                ),

            "lost_benign":
                int(
                    (
                        g[
                            "lost"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 0
                        )
                    ).sum()
                ),

            "lost_malware":
                int(
                    (
                        g[
                            "lost"
                        ]
                        &
                        (
                            g[
                                "label"
                            ] == 1
                        )
                    ).sum()
                ),
        })

    return (
        pd.DataFrame(rows),
        avg,
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT
    )

    common = build_common(
        df
    )

    # ========================================================
    # 当前单向最小跨界基线
    # ========================================================

    (
        baseline_probability,
        _,
    ) = build_probability(
        common,
    )

    (
        baseline,
        _,
        _,
    ) = evaluate_variant(
        df,
        baseline_probability,
    )

    print(
        "rows =",
        len(df),
        "APK =",
        df[
            "sha256"
        ].nunique(),
    )

    print(
        "eligible records =",
        int(
            common[
                "eligible"
            ].sum()
        ),
    )

    print()
    print(
        "当前最小跨界:"
    )

    print(
        f"meanBA={baseline['mean_BA']:.6f} "
        f"minBA={baseline['min_BA']:.6f} "
        f"AUC={baseline['mean_AUC']:.6f}"
    )

    # ========================================================
    # A. 只根据极端负delta否决
    # ========================================================

    delta_rows = []

    for delta_cut in DELTA_CUTS:

        (
            probability,
            veto,
        ) = build_probability(
            common,
            delta_cut=float(
                delta_cut
            ),
            base_p_cut=None,
        )

        (
            row,
            _,
            _,
        ) = evaluate_variant(
            df,
            probability,
        )

        row[
            "delta_cut"
        ] = float(
            delta_cut
        )

        row[
            "veto_records"
        ] = int(
            veto.sum()
        )

        delta_rows.append(
            row
        )

    delta_df = pd.DataFrame(
        delta_rows
    )

    delta_ranked = (
        delta_df
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # B. 高置信Compact + 极端负delta 同时满足才否决
    # ========================================================

    joint_rows = []

    for p_cut in BASE_P_CUTS:

        for delta_cut in DELTA_CUTS:

            (
                probability,
                veto,
            ) = build_probability(
                common,
                delta_cut=float(
                    delta_cut
                ),
                base_p_cut=float(
                    p_cut
                ),
            )

            (
                row,
                _,
                _,
            ) = evaluate_variant(
                df,
                probability,
            )

            row[
                "base_p_cut"
            ] = float(
                p_cut
            )

            row[
                "delta_cut"
            ] = float(
                delta_cut
            )

            row[
                "veto_records"
            ] = int(
                veto.sum()
            )

            joint_rows.append(
                row
            )

    joint_df = pd.DataFrame(
        joint_rows
    )

    joint_ranked = (
        joint_df
        .sort_values(
            [
                "mean_BA",
                "min_BA",
                "mean_AUC",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    delta_df.to_csv(
        OUT
        / "delta_only_all.csv",
        index=False,
    )

    delta_ranked.to_csv(
        OUT
        / "delta_only_ranked.csv",
        index=False,
    )

    joint_df.to_csv(
        OUT
        / "joint_all.csv",
        index=False,
    )

    joint_ranked.to_csv(
        OUT
        / "joint_ranked.csv",
        index=False,
    )

    # ========================================================
    # 最佳方案样本诊断
    # ========================================================

    best_delta = (
        delta_ranked.iloc[0]
    )

    (
        best_delta_probability,
        best_delta_veto,
    ) = build_probability(
        common,
        delta_cut=float(
            best_delta[
                "delta_cut"
            ]
        ),
        base_p_cut=None,
    )

    (
        delta_diag,
        _,
    ) = averaged_probability_diagnostic(
        df,
        best_delta_probability,
        baseline_probability,
    )

    delta_diag.to_csv(
        OUT
        / "best_delta_only_sample_change.csv",
        index=False,
    )

    best_joint = (
        joint_ranked.iloc[0]
    )

    (
        best_joint_probability,
        best_joint_veto,
    ) = build_probability(
        common,
        delta_cut=float(
            best_joint[
                "delta_cut"
            ]
        ),
        base_p_cut=float(
            best_joint[
                "base_p_cut"
            ]
        ),
    )

    (
        joint_diag,
        _,
    ) = averaged_probability_diagnostic(
        df,
        best_joint_probability,
        baseline_probability,
    )

    joint_diag.to_csv(
        OUT
        / "best_joint_sample_change.csv",
        index=False,
    )

    # ========================================================
    # 输出
    # ========================================================

    print()
    print(
        "=" * 82
    )
    print(
        "只否决极端负残差：前10"
    )
    print(
        "=" * 82
    )

    for _, r in (
        delta_ranked.head(
            10
        ).iterrows()
    ):
        print(
            f"D={r['delta_cut']:.3f} "
            f"vetoN={int(r['veto_records']):3d} "
            f"meanBA={r['mean_BA']:.6f} "
            f"minBA={r['min_BA']:.6f} "
            f"AUC={r['mean_AUC']:.6f} "
            f"BA18={r['BA_2018']:.6f} "
            f"BA19={r['BA_2019']:.6f} "
            f"BA20={r['BA_2020']:.6f} "
            f"BA21={r['BA_2021']:.6f}"
        )

    print()
    print(
        "=" * 82
    )
    print(
        "高置信Compact + 极端负残差联合否决：前15"
    )
    print(
        "=" * 82
    )

    for _, r in (
        joint_ranked.head(
            15
        ).iterrows()
    ):
        print(
            f"P={r['base_p_cut']:.3f} "
            f"D={r['delta_cut']:.3f} "
            f"vetoN={int(r['veto_records']):3d} "
            f"meanBA={r['mean_BA']:.6f} "
            f"minBA={r['min_BA']:.6f} "
            f"AUC={r['mean_AUC']:.6f} "
            f"BA18={r['BA_2018']:.6f} "
            f"BA19={r['BA_2019']:.6f} "
            f"BA20={r['BA_2020']:.6f} "
            f"BA21={r['BA_2021']:.6f}"
        )

    print()
    print(
        "=" * 82
    )
    print(
        "最佳单变量否决"
    )
    print(
        "=" * 82
    )

    print(
        f"D={best_delta['delta_cut']:.3f} "
        f"vetoN={int(best_delta['veto_records'])}"
    )

    print(
        f"meanBA "
        f"{baseline['mean_BA']:.6f}"
        f" -> "
        f"{best_delta['mean_BA']:.6f} "
        f"Δ="
        f"{best_delta['mean_BA']-baseline['mean_BA']:+.6f}"
    )

    print(
        f"minBA "
        f"{baseline['min_BA']:.6f}"
        f" -> "
        f"{best_delta['min_BA']:.6f}"
    )

    print(
        f"AUC "
        f"{baseline['mean_AUC']:.6f}"
        f" -> "
        f"{best_delta['mean_AUC']:.6f}"
    )

    print()
    print(
        "=" * 82
    )
    print(
        "最佳联合否决"
    )
    print(
        "=" * 82
    )

    print(
        f"P={best_joint['base_p_cut']:.3f} "
        f"D={best_joint['delta_cut']:.3f} "
        f"vetoN={int(best_joint['veto_records'])}"
    )

    print(
        f"meanBA "
        f"{baseline['mean_BA']:.6f}"
        f" -> "
        f"{best_joint['mean_BA']:.6f} "
        f"Δ="
        f"{best_joint['mean_BA']-baseline['mean_BA']:+.6f}"
    )

    print(
        f"minBA "
        f"{baseline['min_BA']:.6f}"
        f" -> "
        f"{best_joint['min_BA']:.6f}"
    )

    print(
        f"AUC "
        f"{baseline['mean_AUC']:.6f}"
        f" -> "
        f"{best_joint['mean_AUC']:.6f}"
    )

    print()
    print(
        "=" * 82
    )
    print(
        "最佳联合否决相对当前最小跨界的样本变化"
    )
    print(
        "=" * 82
    )

    for _, r in (
        joint_diag.iterrows()
    ):
        print(
            f"{str(r['year']):>4s} "
            f"恢复正确={int(r['recovered']):3d} "
            f"损失正确={int(r['lost']):3d} "
            f"net={int(r['net']):+3d} "
            f"恢复良性={int(r['recovered_benign']):3d} "
            f"恢复恶意={int(r['recovered_malware']):3d} "
            f"损失良性={int(r['lost_benign']):3d} "
            f"损失恶意={int(r['lost_malware']):3d}"
        )


if __name__ == "__main__":
    main()
