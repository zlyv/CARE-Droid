#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict, deque
from pathlib import Path


TOP5_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

SUFFIXES = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)


def read_csv(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(
            path,
            "rt",
            encoding="utf-8",
            newline="",
        ) as f:
            return list(csv.DictReader(f))

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_gz_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise SystemExit(
            "没有可输出的数据"
        )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def shortest_role_distances(
    method_edges: set[tuple[str, str]],
    role_edges: set[tuple[str, str]],
    role: str,
    max_hops: int,
) -> dict[str, int]:
    """
    收缩后的结构方向：

        调用方法 -> 被调用方法
        调用方法 -> 敏感语义角色

    从某个敏感角色反向找调用方法。

    直接连接角色的方法：
        距离 = 1

    每再经过一个保留内部方法：
        距离 +1

    由于纯转发方法已经被收缩，
    所以这里得到的就是新的“有意义距离”。
    """

    reverse_methods = defaultdict(set)
    direct_callers = defaultdict(set)

    for source, target in method_edges:
        reverse_methods[target].add(
            source
        )

    for source, target_role in role_edges:
        direct_callers[
            target_role
        ].add(source)

    distance = {}
    queue = deque()

    for method in direct_callers.get(
        role,
        (),
    ):
        distance[method] = 1
        queue.append(method)

    while queue:
        current = queue.popleft()

        current_distance = (
            distance[current]
        )

        if current_distance >= max_hops:
            continue

        for caller in reverse_methods.get(
            current,
            (),
        ):
            candidate = (
                current_distance + 1
            )

            previous = distance.get(
                caller
            )

            if (
                previous is None
                or candidate < previous
            ):
                distance[
                    caller
                ] = candidate

                queue.append(
                    caller
                )

    return distance


def feature_names() -> list[str]:
    return [
        f"up_role__{role}__{suffix}"
        for role in TOP5_ROLES
        for suffix in SUFFIXES
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--apk-summary",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-test20/"
            "apk_summary.csv"
        ),
    )

    parser.add_argument(
        "--structure",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-test20/"
            "contracted_structure_edges.csv.gz"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/"
            "clean-local-context-test20/"
            "normalized_top5_20.csv.gz"
        ),
    )

    parser.add_argument(
        "--max-hops",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    summary_rows = read_csv(
        args.apk_summary
    )

    edge_rows = read_csv(
        args.structure
    )

    by_sha_edges = defaultdict(list)

    for row in edge_rows:
        sha = str(
            row.get(
                "sha256",
                "",
            )
        ).strip().casefold()

        if sha:
            by_sha_edges[
                sha
            ].append(row)

    output_rows = []

    role_caller_totals = defaultdict(int)
    role_apk_coverage = defaultdict(int)

    print(
        "开始从收缩结构生成新的五类20维表示"
    )

    print(
        "APK =",
        len(summary_rows),
    )

    print(
        "最大有意义距离 =",
        args.max_hops,
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

        method_edges = set()
        role_edges = set()

        for edge in by_sha_edges.get(
            sha,
            (),
        ):
            source = str(
                edge.get(
                    "source_method_id",
                    "",
                )
            ).strip()

            if not source:
                continue

            target_type = str(
                edge.get(
                    "target_type",
                    "",
                )
            ).strip()

            if target_type == "method":
                target = str(
                    edge.get(
                        "target_method_id",
                        "",
                    )
                ).strip()

                if target:
                    method_edges.add(
                        (
                            source,
                            target,
                        )
                    )

            elif target_type == "role":
                role = str(
                    edge.get(
                        "target_value",
                        "",
                    )
                ).strip()

                if role:
                    role_edges.add(
                        (
                            source,
                            role,
                        )
                    )

        # ----------------------------------------
        # 每一个敏感角色：
        # 方法 -> 该角色的最短有意义距离
        # ----------------------------------------
        role_distances = {
            role:
                shortest_role_distances(
                    method_edges,
                    role_edges,
                    role,
                    args.max_hops,
                )

            for role in TOP5_ROLES
        }

        # ----------------------------------------
        # 五类范围内所有不同调用方法。
        #
        # 新表示中的 caller_ratio 分母。
        # ----------------------------------------
        all_callers = set()

        for distances in (
            role_distances.values()
        ):
            all_callers.update(
                distances
            )

        total_callers = len(
            all_callers
        )

        out = {
            "sha256":
                sha,

            "metadata_year":
                summary.get(
                    "year",
                    summary.get(
                        "metadata_year",
                        "",
                    ),
                ),

            "metadata_label":
                summary.get(
                    "label",
                    summary.get(
                        "metadata_label",
                        "",
                    ),
                ),

            "normalized_top5_total_callers":
                total_callers,
        }

        for role in TOP5_ROLES:
            distances_by_method = (
                role_distances[
                    role
                ]
            )

            distances = list(
                distances_by_method.values()
            )

            caller_count = len(
                distances
            )

            # 诊断列，不进入正式20维。
            out[
                f"normalized_audit__"
                f"{role}__caller_count"
            ] = caller_count

            out[
                f"normalized_audit__"
                f"{role}__max_depth"
            ] = (
                max(distances)
                if distances
                else 0
            )

            prefix = (
                f"up_role__"
                f"{role}__"
            )

            # ====================================
            # 与旧20维相同的四种统计形式。
            # ====================================

            # 1. 调用方法数量取对数
            out[
                prefix + "caller_log"
            ] = math.log1p(
                caller_count
            )

            # 2. 该角色调用方法占
            #    五类所有调用方法并集的比例
            out[
                prefix + "caller_ratio"
            ] = (
                caller_count
                / max(
                    1,
                    total_callers,
                )
            )

            # 3. 平均有意义距离 / 4
            out[
                prefix + "mean_depth_norm"
            ] = (
                sum(distances)
                / len(distances)
                / max(
                    1,
                    args.max_hops,
                )

                if distances
                else 0.0
            )

            # 4. 距离 >= 3 的方法比例
            out[
                prefix + "deep_caller_ratio"
            ] = (
                sum(
                    distance >= 3
                    for distance
                    in distances
                )
                / len(distances)

                if distances
                else 0.0
            )

            role_caller_totals[
                role
            ] += caller_count

            if caller_count:
                role_apk_coverage[
                    role
                ] += 1

        output_rows.append(
            out
        )

        counts = [
            len(
                role_distances[
                    role
                ]
            )
            for role in TOP5_ROLES
        ]

        print(
            f"{index:2d}/"
            f"{len(summary_rows)} "
            f"方法并集={total_callers:3d} | "
            f"麦克风={counts[0]:3d} "
            f"相机={counts[1]:3d} "
            f"通话={counts[2]:3d} "
            f"定位={counts[3]:3d} "
            f"后台={counts[4]:3d}",
            flush=True,
        )

    expected = feature_names()

    if len(expected) != 20:
        raise RuntimeError(
            "20维列数异常："
            f"{len(expected)}"
        )

    for row in output_rows:
        missing = [
            name
            for name in expected
            if name not in row
        ]

        if missing:
            raise RuntimeError(
                "缺失正式特征列："
                + str(missing)
            )

    write_gz_csv(
        args.output,
        output_rows,
    )

    print()
    print(
        "======================================"
    )
    print(
        "新的五类20维表示检查"
    )
    print(
        "======================================"
    )

    print(
        "输出 APK =",
        len(output_rows),
    )

    print(
        "正式特征维数 =",
        len(expected),
    )

    print()
    print("各敏感角色覆盖：")

    for role in TOP5_ROLES:
        print(
            f"  {role:32s} "
            f"APK="
            f"{role_apk_coverage[role]:2d}/"
            f"{len(output_rows)} "
            f"调用方法总数="
            f"{role_caller_totals[role]}"
        )

    union_values = [
        int(
            row[
                "normalized_top5_total_callers"
            ]
        )
        for row in output_rows
    ]

    mean_union = (
        sum(union_values)
        / len(union_values)

        if union_values
        else 0.0
    )

    print()
    print(
        "平均五类调用方法并集 =",
        f"{mean_union:.2f}",
    )

    print(
        "最大五类调用方法并集 =",
        (
            max(union_values)
            if union_values
            else 0
        ),
    )

    print()
    print("正式20维列：")

    for name in expected:
        print(
            " ",
            name,
        )

    print()
    print(
        "保存：",
        args.output,
    )


if __name__ == "__main__":
    main()
