#!/usr/bin/env python3
from __future__ import annotations

from guard_core import classify_method, match_role, parse_method_reference


def main() -> None:
    reference = parse_method_reference("v0, Landroid/content/Context;->checkSelfPermission(Ljava/lang/String;)I")
    assert reference == "landroid/content/context;->checkselfpermission(ljava/lang/string;)i"
    mapping = {"rules": [{"rule_kind": "api", "class_prefixes": ["Landroid/telephony/"],
                           "method_names": ["getDeviceId"], "rule_id": "r", "role": "device_id", "confidence": 1.0}]}
    assert match_role("Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;", mapping)["role"] == "device_id"
    instructions = [{"name": "if-eqz", "output": "v0, +4"}, {"name": "invoke-virtual", "output": "v1, X"}]
    user = classify_method("onClick", instructions, 1, [])
    assert user["user_callback"] and user["branch_local_12"] and user["condition_bearing"]
    background = classify_method("onReceive", [{"name": "invoke-virtual", "output": ""}], 0, [])
    assert background["background_callback"] and not background["user_callback"]
    neutral = classify_method("a", [{"name": "invoke-virtual", "output": ""}], 0, [])
    assert neutral["context"] == "NO_RELIABLE_CONDITION"
    print("SENSITIVE_GUARD_CONTEXT_SELFTEST_PASS")


if __name__ == "__main__":
    main()
