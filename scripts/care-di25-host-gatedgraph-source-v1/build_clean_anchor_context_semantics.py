#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_endpoint_centered_tensor_cache_ag import role_tiers


ANCHOR_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)


def read_rows(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def load_roles(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    tiers = role_tiers(data)

    return tuple(sorted(tiers))


def build_anchor_method_sets(
    structure_rows,
):
    """
    收缩后的结构：

        方法 -> 方法
        方法 -> 五类敏感角色

    对每一个五类敏感角色反向遍历，
    得到所有能够组织该敏感行为的方法。
    """

    predecessors = defaultdict(set)
    direct_role_callers = defaultdict(set)
    all_methods = set()

    for row in structure_rows:
        source = str(
            row.get(
                "source_method_id",
                "",
            )
        ).strip()

        if not source:
            continue

        all_methods.add(source)

        target_type = str(
            row.get(
                "target_type",
                "",
            )
        ).strip()

        if target_type == "method":
            target = str(
                row.get(
                    "target_method_id",
                    "",
                )
            ).strip()

            if target:
                all_methods.add(target)
                predecessors[target].add(source)

        elif target_type == "role":
            role = str(
                row.get(
                    "target_value",
                    "",
                )
            ).strip()

            if role:
                direct_role_callers[
                    role
                ].add(source)

    anchor_methods = {}

    for role in ANCHOR_ROLES:
        seen = set(
            direct_role_callers.get(
                role,
                (),
            )
        )

        queue = deque(seen)

        while queue:
            current = queue.popleft()

            for caller in predecessors.get(
                current,
                (),
            ):
                if caller in seen:
                    continue

                seen.add(caller)
                queue.append(caller)

        anchor_methods[role] = seen

    return (
        anchor_methods,
        all_methods,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--structure",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--context",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--priority",
        type=Path,
        default=Path(
            "configs/protocol/"
            "sensitive_endpoint_priority_v1.yaml"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--audit",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    all_roles = load_roles(
        args.priority
    )

    if len(all_roles) != 31:
        raise RuntimeError(
            "预期31类语义角色，实际="
            + str(len(all_roles))
        )

    missing_anchor = (
        set(ANCHOR_ROLES)
        - set(all_roles)
    )

    if missing_anchor:
        raise RuntimeError(
            "五类敏感角色缺失："
            + str(
                sorted(missing_anchor)
            )
        )

    summary_rows = read_rows(
        args.summary
    )

    structure_rows = read_rows(
        args.structure
    )

    context_rows = read_rows(
        args.context
    )

    structure_by_sha = defaultdict(list)

    for row in structure_rows:
        sha = str(
            row.get(
                "sha256",
                "",
            )
        ).strip().casefold()

        structure_by_sha[sha].append(
            row
        )

    # sha -> method -> role -> 最近距离
    context_by_sha = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in context_rows:
        sha = str(
            row.get(
                "sha256",
                "",
            )
        ).strip().casefold()

        method = str(
            row.get(
                "method_id",
                "",
            )
        ).strip()

        role = str(
            row.get(
                "context_role",
                "",
            )
        ).strip()

        if (
            not sha
            or not method
            or role not in all_roles
        ):
            continue

        distance = int(
            float(
                row.get(
                    "down_semantic_distance",
                    0,
                )
            )
        )

        old = (
            context_by_sha[
                sha
            ][method].get(role)
        )

        if (
            old is None
            or distance < old
        ):
            context_by_sha[
                sha
            ][method][role] = (
                distance
            )

    feature_rows = []
    audit_rows = []

    apk_pair_counts = []
    apk_non_top5_counts = []

    distance_counter = Counter()
    pair_apk_coverage = Counter()

    print(
        "开始构造五类敏感行为"
        "与31类上下文语义关系"
    )

    print(
        "敏感角色 =",
        len(ANCHOR_ROLES),
    )

    print(
        "上下文语义角色 =",
        len(all_roles),
    )

    print(
        "语义关系槽位 =",
        len(ANCHOR_ROLES)
        * len(all_roles),
    )

    print()

    for index, summary in enumerate(
        summary_rows,
        start=1,
    ):
        sha = str(
            summary.get(
                "sha256",
                "",
            )
        ).strip().casefold()

        (
            anchor_method_sets,
            all_methods,
        ) = build_anchor_method_sets(
            structure_by_sha.get(
                sha,
                (),
            )
        )

        features = {
            "sha256": sha,
            "metadata_year": summary.get(
                "year",
                "",
            ),
            "metadata_label": summary.get(
                "label",
                "",
            ),
        }

        nonzero_pairs = 0
        non_top5_pairs = 0

        for anchor in ANCHOR_ROLES:
            anchor_methods = (
                anchor_method_sets[
                    anchor
                ]
            )

            denominator = len(
                anchor_methods
            )

            for context_role in all_roles:
                support_count = 0
                weighted_sum = 0.0
                local_distances = []

                for method in anchor_methods:
                    distance = (
                        context_by_sha[
                            sha
                        ][method].get(
                            context_role
                        )
                    )

                    if distance is None:
                        continue

                    support_count += 1

                    weighted_sum += (
                        1.0
                        / (
                            1.0
                            + float(distance)
                        )
                    )

                    local_distances.append(
                        distance
                    )

                support_ratio = (
                    support_count
                    / denominator
                    if denominator
                    else 0.0
                )

                weighted_support = (
                    weighted_sum
                    / denominator
                    if denominator
                    else 0.0
                )

                support_name = (
                    "semrel_support__"
                    + anchor
                    + "__"
                    + context_role
                )

                weighted_name = (
                    "semrel_weighted__"
                    + anchor
                    + "__"
                    + context_role
                )

                features[
                    support_name
                ] = support_ratio

                features[
                    weighted_name
                ] = weighted_support

                if support_count > 0:
                    nonzero_pairs += 1

                    pair_apk_coverage[
                        (
                            anchor,
                            context_role,
                        )
                    ] += 1

                    if (
                        context_role
                        not in ANCHOR_ROLES
                    ):
                        non_top5_pairs += 1

                    for distance in (
                        local_distances
                    ):
                        distance_counter[
                            distance
                        ] += 1

        apk_pair_counts.append(
            nonzero_pairs
        )

        apk_non_top5_counts.append(
            non_top5_pairs
        )

        audit_rows.append({
            "sha256": sha,
            "kept_method_count":
                len(all_methods),
            "nonzero_semantic_pair_count":
                nonzero_pairs,
            "non_top5_semantic_pair_count":
                non_top5_pairs,
            "pair_density":
                (
                    nonzero_pairs
                    / (
                        len(ANCHOR_ROLES)
                        * len(all_roles)
                    )
                ),
        })

        feature_rows.append(
            features
        )

        print(
            f"{index:2d}/"
            f"{len(summary_rows)} "
            f"方法={len(all_methods):3d} "
            f"非零语义关系="
            f"{nonzero_pairs:3d}/"
            f"{len(ANCHOR_ROLES) * len(all_roles)} "
            f"其中五类外="
            f"{non_top5_pairs:3d}",
            flush=True,
        )

    # --------------------------------------------------
    # 输出正式特征。
    # --------------------------------------------------
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_fields = list(
        feature_rows[0]
    )

    with gzip.open(
        args.output,
        "wt",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=feature_fields,
        )
        writer.writeheader()
        writer.writerows(
            feature_rows
        )

    # --------------------------------------------------
    # 输出逐 APK 检查。
    # --------------------------------------------------
    args.audit.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.audit.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                audit_rows[0]
            ),
        )
        writer.writeheader()
        writer.writerows(
            audit_rows
        )

    total_slots = (
        len(ANCHOR_ROLES)
        * len(all_roles)
    )

    mean_pairs = (
        sum(apk_pair_counts)
        / len(apk_pair_counts)
        if apk_pair_counts
        else 0.0
    )

    mean_non_top5 = (
        sum(apk_non_top5_counts)
        / len(apk_non_top5_counts)
        if apk_non_top5_counts
        else 0.0
    )

    print()
    print(
        "======================================"
    )
    print(
        "语义关系表示检查结果"
    )
    print(
        "======================================"
    )

    print(
        "APK =",
        len(feature_rows),
    )

    print(
        "基础语义关系槽位 =",
        total_slots,
    )

    print(
        "支持比例特征 =",
        total_slots,
    )

    print(
        "距离加权特征 =",
        total_slots,
    )

    print(
        "总语义特征维数 =",
        total_slots * 2,
    )

    print()
    print(
        "平均非零语义关系 =",
        f"{mean_pairs:.2f}",
        "/",
        total_slots,
    )

    print(
        "平均关系密度 =",
        f"{mean_pairs / total_slots:.2%}",
    )

    print(
        "平均五类外语义关系 =",
        f"{mean_non_top5:.2f}",
    )

    print()
    print(
        "按向下语义距离统计："
    )

    for distance in range(5):
        print(
            "  距离",
            distance,
            "=",
            distance_counter[
                distance
            ],
        )

    print()
    print(
        "出现 APK 最多的语义关系："
    )

    for (
        anchor,
        context_role,
    ), count in (
        pair_apk_coverage.most_common(
            20
        )
    ):
        print(
            " ",
            anchor,
            "->",
            context_role,
            ":",
            count,
            "/",
            len(feature_rows),
        )

    print()
    print(
        "保存：",
        args.output,
    )

    print(
        "保存：",
        args.audit,
    )


if __name__ == "__main__":
    main()
