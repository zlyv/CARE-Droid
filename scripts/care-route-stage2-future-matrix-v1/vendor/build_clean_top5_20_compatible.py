#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict, deque
from pathlib import Path


FIVE_ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)

FOUR_STATS = (
    "caller_log",
    "caller_ratio",
    "mean_depth_norm",
    "deep_caller_ratio",
)


def read_csv(path: Path):
    opener = (
        gzip.open
        if str(path).endswith(".gz")
        else open
    )

    with opener(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_gz(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(rows[0])

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


def role_distances(
    method_edges,
    role_edges,
    role,
    max_hops=4,
):
    """
    收缩后的图：

        内部方法 -> 内部方法
        内部方法 -> 五类敏感角色

    从敏感角色反向找调用方法。

    直接调用敏感角色：
        距离 = 1

    每经过一个保留内部方法：
        距离 +1

    纯转发方法已经从图中消失，
    所以不会消耗这里的距离。
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

    distances = {}
    queue = deque()

    for method in direct_callers.get(
        role,
        (),
    ):
        distances[method] = 1
        queue.append(method)

    while queue:
        current = queue.popleft()

        current_distance = (
            distances[current]
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

            old = distances.get(
                caller
            )

            if (
                old is None
                or candidate < old
            ):
                distances[caller] = (
                    candidate
                )
                queue.append(caller)

    return distances


def official_names():
    return [
        f"up_role__{role}__{stat}"
        for role in FIVE_ROLES
        for stat in FOUR_STATS
    ]


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
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--max-hops",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    summary_rows = read_csv(
        args.summary
    )

    structure_rows = read_csv(
        args.structure
    )

    edges_by_sha = defaultdict(list)

    for row in structure_rows:
        sha = str(
            row["sha256"]
        ).strip().casefold()

        edges_by_sha[sha].append(
            row
        )

    outputs = []

    mismatch_count = 0

    print(
        "开始生成严格兼容旧定义的五类20维"
    )
    print(
        "APK =",
        len(summary_rows),
    )
    print()

    for index, summary in enumerate(
        summary_rows,
        start=1,
    ):
        sha = str(
            summary["sha256"]
        ).strip().casefold()

        total_callers = int(
            float(
                summary.get(
                    "kept_method_count",
                    0,
                )
                or 0
            )
        )

        method_edges = set()
        role_edges = set()

        method_ids = set()

        for row in edges_by_sha.get(
            sha,
            (),
        ):
            source = str(
                row.get(
                    "source_method_id",
                    "",
                )
            ).strip()

            if not source:
                continue

            method_ids.add(source)

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
                    method_ids.add(target)

                    method_edges.add(
                        (
                            source,
                            target,
                        )
                    )

            elif target_type == "role":
                role = str(
                    row.get(
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

        role_to_distances = {
            role:
                role_distances(
                    method_edges,
                    role_edges,
                    role,
                    max_hops=
                        args.max_hops,
                )
            for role in FIVE_ROLES
        }

        # -------------------------------------
        # 正确性检查：
        #
        # 五类角色能够覆盖到的方法并集
        # 不应该超过全部保留内部方法。
        # -------------------------------------
        reached_union = set()

        for distances in (
            role_to_distances.values()
        ):
            reached_union.update(
                distances
            )

        if len(reached_union) > total_callers:
            raise RuntimeError(
                f"{sha}: "
                "五类可达方法数超过"
                "全部保留方法数 "
                f"{len(reached_union)} > "
                f"{total_callers}"
            )

        if len(reached_union) != total_callers:
            mismatch_count += 1

        out = {
            "sha256":
                sha,

            "metadata_year":
                summary.get(
                    "year",
                    "",
                ),

            "metadata_label":
                summary.get(
                    "label",
                    "",
                ),

            # 以下是诊断列，不属于正式20维。
            "clean_total_callers":
                total_callers,

            "clean_five_role_reached_union":
                len(reached_union),

            "clean_unassigned_callers":
                (
                    total_callers
                    - len(reached_union)
                ),
        }

        counts = []

        for role in FIVE_ROLES:
            distances_map = (
                role_to_distances[
                    role
                ]
            )

            distances = list(
                distances_map.values()
            )

            caller_count = len(
                distances
            )

            counts.append(
                caller_count
            )

            prefix = (
                f"up_role__"
                f"{role}__"
            )

            # ------------------------------
            # 1. 调用方法数量取对数
            # 与旧版完全同形式。
            # ------------------------------
            out[
                prefix + "caller_log"
            ] = math.log1p(
                caller_count
            )

            # ------------------------------
            # 2. 调用方法比例
            #
            # 旧版：
            # len(callers)
            # /
            # len(internal_callers)
            #
            # 新版严格对应：
            # 该角色保留方法数
            # /
            # 全部保留内部方法数
            # ------------------------------
            out[
                prefix + "caller_ratio"
            ] = (
                caller_count
                / max(
                    1,
                    total_callers,
                )
            )

            # ------------------------------
            # 3. 平均距离 / 4
            # ------------------------------
            out[
                prefix + "mean_depth_norm"
            ] = (
                (
                    sum(distances)
                    / len(distances)
                )
                / max(
                    1,
                    args.max_hops,
                )
                if distances
                else 0.0
            )

            # ------------------------------
            # 4. 距离 >= 3 的比例
            # ------------------------------
            out[
                prefix + "deep_caller_ratio"
            ] = (
                sum(
                    d >= 3
                    for d in distances
                )
                / len(distances)
                if distances
                else 0.0
            )

            # 诊断。
            out[
                f"clean_audit__"
                f"{role}__caller_count"
            ] = caller_count

            out[
                f"clean_audit__"
                f"{role}__max_distance"
            ] = (
                max(distances)
                if distances
                else 0
            )

        outputs.append(out)

        print(
            f"{index:2d}/"
            f"{len(summary_rows)} "
            f"全部保留={total_callers:3d} "
            f"五类可达并集="
            f"{len(reached_union):3d} | "
            f"麦={counts[0]:3d} "
            f"相={counts[1]:3d} "
            f"通={counts[2]:3d} "
            f"定={counts[3]:3d} "
            f"后台={counts[4]:3d}",
            flush=True,
        )

    official = official_names()

    if len(official) != 20:
        raise RuntimeError(
            f"正式列不是20维："
            f"{len(official)}"
        )

    for row in outputs:
        for name in official:
            if name not in row:
                raise RuntimeError(
                    "缺少正式列："
                    + name
                )

    write_gz(
        args.output,
        outputs,
    )

    print()
    print(
        "======================================"
    )
    print(
        "严格兼容20维生成结果"
    )
    print(
        "======================================"
    )

    print(
        "APK =",
        len(outputs),
    )

    print(
        "正式维数 =",
        len(official),
    )

    print(
        "五类并集 != 全部保留方法 的 APK =",
        mismatch_count,
        "/",
        len(outputs),
    )

    print()
    print("正式20列：")

    for name in official:
        print(" ", name)

    print()
    print(
        "保存：",
        args.output,
    )


if __name__ == "__main__":
    main()
