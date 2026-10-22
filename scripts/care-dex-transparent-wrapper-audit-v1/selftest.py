#!/usr/bin/env python3
from __future__ import annotations

from wrapper_core import analyze_forwarder, match_role, register_numbers


def method(instructions, descriptor="(II)V", registers=2, inputs=2, name="forward"):
    return {
        "name": name, "access": "public static", "descriptor": descriptor,
        "registers_size": registers, "ins_size": inputs,
        "instructions": [{"name": op, "output": output} for op, output in instructions],
    }


def main() -> None:
    positive = method([
        ("invoke-static", "v0, v1, Landroid/telephony/SmsManager;->send(II)V"),
        ("return-void", ""),
    ])
    result = analyze_forwarder(positive)
    assert result["candidate"] and result["tier"] == "VERIFIED_STRICT"

    moved = method([
        ("move", "v0, v1"),
        ("invoke-static", "v0, Landroid/provider/Settings;->read(I)I"),
        ("move-result", "v0"),
        ("return", "v0"),
    ], descriptor="(I)I", registers=2, inputs=1)
    result = analyze_forwarder(moved)
    assert result["candidate"] and result["return_type"] == "i"

    for unsafe in [
        method([("if-eqz", "v0, +4"), ("invoke-static", "v0, v1, Lx;->f(II)V"), ("return-void", "")]),
        method([("sput", "v0, Lx;->field I"), ("invoke-static", "v0, v1, Lx;->f(II)V"), ("return-void", "")]),
        method([("const/4", "v0, 1"), ("invoke-static", "v0, v1, Lx;->f(II)V"), ("return-void", "")]),
        method([("invoke-static", "v0, Lx;->a(I)V"), ("invoke-static", "v1, Lx;->b(I)V"), ("return-void", "")]),
    ]:
        assert not analyze_forwarder(unsafe)["candidate"]

    dynamic = method([("invoke-virtual", "v0, v1, Lx;->f(I)V"), ("return-void", "")])
    result = analyze_forwarder(dynamic)
    assert result["candidate"] and result["tier"] == "DECLARED_TARGET_ONLY"
    assert register_numbers("v3 ... v6, Lx;->f(IIII)V") == [3, 4, 5, 6]

    mapping = {"rules": [{
        "rule_id": "R", "role": "sms", "confidence": 1.0,
        "class_prefixes": ["Landroid/telephony/SmsManager;"], "method_names": ["send"],
    }]}
    assert match_role("Landroid/telephony/SmsManager;->send(II)V", mapping)["role"] == "sms"
    assert match_role("Landroid/telephony/SmsManager;->other(II)V", mapping) is None
    print("DEX_TRANSPARENT_WRAPPER_SELFTEST_PASS positive,negative,dispatch,range,role", flush=True)


if __name__ == "__main__":
    main()
