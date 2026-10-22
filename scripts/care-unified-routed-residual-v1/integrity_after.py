#!/usr/bin/env python3
from common import FORMAL, MEMBER, OUTPUT, asset_record, dump_json, read_json


def main():
    before = read_json(OUTPUT / "baseline_asset_manifest_before.json")
    after = [asset_record(MEMBER, "strict source member OOF probabilities"),
             asset_record(FORMAL, "formal RF-DI25 A0-C43 OOF predictions")]
    changed = [a["path"] for a, b in zip(after, before) if a["sha256"] != b["sha256"] or a["size"] != b["size"]]
    dump_json(OUTPUT / "baseline_asset_manifest_after.json", after)
    dump_json(OUTPUT / "baseline_integrity_check.json", {"status": "PASS" if not changed else "FAIL", "changed": changed})
    if changed:
        raise RuntimeError("BLOCKED_BY_CONTRACT: formal input changed " + repr(changed))
    print("UNIFIED_ROUTED_RESIDUAL_INTEGRITY_PASS")


if __name__ == "__main__":
    main()
