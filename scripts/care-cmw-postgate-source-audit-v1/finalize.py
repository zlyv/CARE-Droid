#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import OUTPUT, dump_json, file_record, read_json


def main() -> None:
    before = read_json(OUTPUT / "input_asset_manifest_before.json")["assets"]
    after = [file_record(Path(row["absolute_path"]), row["purpose"]) for row in before]
    changed = [row["absolute_path"] for row, new in zip(before, after) if row != new]
    dump_json(OUTPUT / "input_asset_manifest_after.json", {"assets": after})
    dump_json(OUTPUT / "input_integrity_check.json", {"status": "PASS" if not changed else "FAIL",
              "changed": changed, "asset_count": len(after)})
    print("CMW_POSTGATE_INPUT_INTEGRITY_" + ("PASS" if not changed else "FAIL"), flush=True)
    if changed:
        raise RuntimeError("frozen input assets changed")


if __name__ == "__main__":
    main()
