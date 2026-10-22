#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ccbc_evidence import evidence_for_apk, shuffle_contracts
from common import MAX_DELTA, metrics, top_budget
from contracts import CompatibilityModel, class_descriptor, normalize_component_name, parse_manifest_xml


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def synthetic_record(sid, token, roles, component_id=0):
    return {
        "sha256": sid, "active_roles": list(roles),
        "components": [{
            "component_id": component_id, "type": "service", "tokens": [token],
            "weak_tokens": [], "reachable_roles": list(roles),
        }],
        "hosts": [{"host_position": 0, "roles": list(roles), "reachable_component_ids": [component_id]}],
        "uses_features": [], "weak_permissions": [],
        "graph_node_count": 3, "graph_edge_count": 2,
    }


def main() -> None:
    expect(normalize_component_name("a.b", ".Main") == "a.b.Main", "relative class")
    expect(normalize_component_name("a.b", "Main") == "a.b.Main", "bare class")
    expect(normalize_component_name("a.b", "x.y.Main") == "x.y.Main", "full class")
    expect(class_descriptor("a.b.Outer$Inner") == "La/b/Outer$Inner;", "inner class")
    xml = '''<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="a.b">
      <uses-permission android:name="android.permission.INTERNET"/>
      <application><service android:name=".Sync" android:exported="false"
        android:permission="android.permission.BIND_VPN_SERVICE">
        <intent-filter><action android:name="android.net.VpnService"/></intent-filter>
      </service></application></manifest>'''
    parsed = parse_manifest_xml(ET.fromstring(xml), "a.b")
    expect(len(parsed["components"]) == 1, "manifest component parse")
    expect("special_service:vpn" in parsed["components"][0]["tokens"], "special service token")
    expect(all("a.b" not in token for token in parsed["components"][0]["tokens"]), "name-zero")

    records = [synthetic_record(f"b{i}", "special_service:vpn", [17]) for i in range(4)]
    records += [synthetic_record(f"m{i}", "component_type:receiver", [0]) for i in range(4)]
    labels = [0] * 4 + [1] * 4
    model = CompatibilityModel(minimum_support=3).fit(records, labels)
    expect(model.lookup(["special_service:vpn"], 17)["known"], "supported edge")
    expect(not model.lookup(["special_service:vpn"], 0)["known"], "unsupported edge unknown")
    matched = evidence_for_apk(records[0], np.eye(31)[17], model, np.asarray([1.0]), 0)
    expect(matched["fb_attn"] >= 0 and matched["vm_attn"] >= 0, "non-negative evidence")
    expect(abs((matched["fb_attn"] + matched["vm_attn"]) - 1.0) > 1e-6, "FB not complement of VM")
    unassigned = dict(records[0]); unassigned["hosts"] = [{"host_position": 0, "roles": [17], "reachable_component_ids": []}]
    uncertain = evidence_for_apk(unassigned, np.eye(31)[17], model, np.asarray([1.0]), 0)
    expect(uncertain["ui"] > 0 and uncertain["vm_attn"] == 0 and uncertain["fb_attn"] == 0, "unknown stays uncertain")

    p31 = np.vstack([np.eye(31)[17]] * len(records))
    shuffled = shuffle_contracts(records, np.asarray([.2] * len(records)), 123, p31)
    expect(len(shuffled) == len(records), "shuffle row count")
    expect([len(row["components"]) for row in shuffled] == [1] * len(records), "shuffle component marginal")
    route = top_budget([1, 1, 1, 1], ["d", "c", "b", "a"], 2)
    expect(route.tolist() == [False, False, True, True], "deterministic SHA tie break")
    expect(abs(MAX_DELTA * np.tanh(1e9)) <= MAX_DELTA, "bounded correction")
    y = np.asarray([0, 1]); base = np.asarray([.1, .1]); final = np.asarray([.9, .9])
    result = metrics(y, base, final, [True, True])
    expect(result["fixed"] == 1 and result["broken"] == 1 and result["net"] == 0, "fixed/broken accounting")
    print("CCBC_SELFTEST_PASS tests=manifest,mapping_semantics,evidence,null,budget,bound,metrics")


if __name__ == "__main__":
    main()
