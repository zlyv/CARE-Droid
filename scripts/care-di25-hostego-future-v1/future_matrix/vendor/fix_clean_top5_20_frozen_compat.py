#!/usr/bin/env python3

import csv
import gzip
import math
from pathlib import Path


OLD_PATH = Path(
    "artifacts/sensitive-upstream-stream-full/"
    "sensitive_upstream_features.csv.gz"
)

NEW_PATH = Path(
    "artifacts/clean-local-context-test20/"
    "clean_top5_20_compatible.csv.gz"
)

OUT_PATH = Path(
    "artifacts/clean-local-context-test20/"
    "clean_top5_20_frozen_compat.csv.gz"
)

ROLES = (
    "microphone_access",
    "camera_access",
    "call_control_or_recording",
    "location_access",
    "background_service_control",
)


def read_gz(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


old_rows = read_gz(OLD_PATH)
new_rows = read_gz(NEW_PATH)

old_by_sha = {
    str(row["sha256"]).strip().casefold(): row
    for row in old_rows
}

outputs = []

print(
    "开始生成冻结旧模型兼容版20维"
)
print()

for index, row in enumerate(
    new_rows,
    start=1,
):
    sha = str(
        row["sha256"]
    ).strip().casefold()

    if sha not in old_by_sha:
        raise SystemExit(
            f"旧缓存缺少 APK: {sha}"
        )

    old = old_by_sha[sha]

    # ---------------------------------------
    # 旧模型真正认识的 total_callers。
    #
    # 旧特征：
    # up_global_internal_caller_log
    # = log1p(total_callers)
    # ---------------------------------------
    old_global_log = float(
        old[
            "up_global_internal_caller_log"
        ]
    )

    recovered = math.expm1(
        old_global_log
    )

    old_total_callers = int(
        round(recovered)
    )

    # 数值恢复检查。
    reconstructed_log = math.log1p(
        old_total_callers
    )

    if abs(
        reconstructed_log
        - old_global_log
    ) > 1e-6:
        raise RuntimeError(
            f"{sha}: 无法稳定恢复旧 total_callers "
            f"log={old_global_log} "
            f"count={old_total_callers}"
        )

    out = dict(row)

    out[
        "frozen_compat_old_total_callers"
    ] = old_total_callers

    for role in ROLES:
        caller_count = int(
            float(
                row[
                    f"clean_audit__"
                    f"{role}__caller_count"
                ]
            )
        )

        prefix = (
            f"up_role__{role}__"
        )

        # ===================================
        # 关键修正：
        #
        # 分子 = 新的净化方法数
        # 分母 = 旧模型训练时使用的旧 total_callers
        #
        # 因此唯一变化来自净化本身，
        # 不再来自分母空间变化。
        # ===================================
        out[
            prefix + "caller_ratio"
        ] = (
            caller_count
            / max(
                1,
                old_total_callers,
            )
        )

    outputs.append(out)

    print(
        f"{index:2d}/"
        f"{len(new_rows)} "
        f"旧总调用方法="
        f"{old_total_callers:4d} "
        f"净化五类方法="
        f"{row['clean_total_callers']}",
        flush=True,
    )


fields = []

for row in outputs:
    for key in row:
        if key not in fields:
            fields.append(key)


OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with gzip.open(
    OUT_PATH,
    "wt",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )
    writer.writeheader()
    writer.writerows(outputs)


print()
print(
    "保存：",
    OUT_PATH,
)
