#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


ANDROID_NS = "{http://schemas.android.com/apk/res/android}"
FRAMEWORK_SCHEMES = {
    "http", "https", "tel", "sms", "smsto", "geo", "content", "file",
    "package", "market", "mailto", "sip", "voicemail",
}
SPECIAL_PERMISSIONS = {
    "android.permission.BIND_ACCESSIBILITY_SERVICE": "accessibility",
    "android.permission.BIND_VPN_SERVICE": "vpn",
    "android.permission.BIND_NOTIFICATION_LISTENER_SERVICE": "notification_listener",
    "android.permission.BIND_INPUT_METHOD": "input_method",
    "android.permission.BIND_DEVICE_ADMIN": "device_admin",
    "android.permission.BIND_CALL_SCREENING_SERVICE": "call_screening",
    "android.permission.BIND_TELECOM_CONNECTION_SERVICE": "telecom_connection",
    "android.permission.BIND_INCALL_SERVICE": "incall",
    "android.permission.BIND_CHOOSER_TARGET_SERVICE": "chooser_target",
    "android.permission.BIND_AUTOFILL_SERVICE": "autofill",
    "android.permission.BIND_DREAM_SERVICE": "dream",
    "android.permission.BIND_JOB_SERVICE": "job",
}
FRAMEWORK_META = {
    "android.accessibilityservice", "android.app.device_admin",
    "android.service.notification.default_filter_types",
    "android.service.notification.disabled_filter_types",
    "android.view.im", "android.net.vpn", "android.telecom",
}
WEAK_PERMISSION_PREFIX = "weak_permission:"


def local_name(tag: str) -> str:
    return str(tag).split("}")[-1]


def attr(element: Any, name: str, default: str = "") -> str:
    return str(element.get(ANDROID_NS + name, element.get(name, default)) or default).strip()


def normalize_component_name(package: str, name: str) -> str:
    package = str(package or "").strip()
    name = str(name or "").strip()
    if not name:
        return ""
    if name.startswith("."):
        return package + name
    if "." not in name and package:
        return package + "." + name
    return name


def class_descriptor(name: str) -> str:
    return "L" + str(name).replace(".", "/") + ";"


def bool_token(value: str, default: str) -> str:
    text = str(value or "").strip().casefold()
    if text in {"true", "1"}:
        return "true"
    if text in {"false", "0"}:
        return "false"
    return default


def framework_name(value: str) -> str | None:
    value = str(value or "").strip()
    lower = value.casefold()
    if lower.startswith("android.") or lower.startswith("com.android."):
        return lower
    return None


def permission_bucket(value: str) -> str:
    value = str(value or "").strip()
    if value.startswith("android.permission."):
        return WEAK_PERMISSION_PREFIX + value.split(".")[-1].casefold()
    return ""


def intent_tokens(intent_filter: Any) -> list[str]:
    tokens: set[str] = set()
    for child in list(intent_filter):
        kind = local_name(child.tag)
        if kind == "action":
            value = framework_name(attr(child, "name"))
            if value:
                tokens.add("action:" + value)
        elif kind == "category":
            value = framework_name(attr(child, "name"))
            if value:
                tokens.add("category:" + value)
        elif kind == "data":
            scheme = attr(child, "scheme").casefold()
            if scheme:
                tokens.add("scheme:" + (scheme if scheme in FRAMEWORK_SCHEMES else "custom"))
            mime = attr(child, "mimeType").casefold()
            if mime:
                tokens.add("mime_top:" + mime.split("/", 1)[0])
    return sorted(tokens)


def _component_tokens(component: Any, component_type: str) -> tuple[list[str], list[str]]:
    tokens = {"component_type:" + component_type}
    weak = set()
    exported = bool_token(attr(component, "exported"), "implicit")
    enabled = bool_token(attr(component, "enabled"), "true")
    tokens.add("exported:" + exported)
    tokens.add("enabled:" + enabled)
    permission = attr(component, "permission")
    if permission in SPECIAL_PERMISSIONS:
        tokens.add("special_service:" + SPECIAL_PERMISSIONS[permission])
    bucket = permission_bucket(permission)
    if bucket:
        weak.add(bucket)
    foreground = attr(component, "foregroundServiceType")
    for value in re.split(r"[|,\s]+", foreground):
        value = value.strip().casefold()
        if value:
            tokens.add("foreground_type:" + value)
    for child in list(component):
        kind = local_name(child.tag)
        if kind == "intent-filter":
            tokens.update(intent_tokens(child))
        elif kind == "meta-data":
            name = attr(child, "name").casefold()
            if name in FRAMEWORK_META or name.startswith("android."):
                tokens.add("framework_meta:" + name)
    return sorted(tokens), sorted(weak)


def parse_manifest_xml(root: Any, package: str) -> dict[str, Any]:
    components = []
    weak_permissions = set()
    uses_features = set()
    for element in root.iter():
        kind = local_name(element.tag)
        if kind in {"uses-permission", "uses-permission-sdk-23"}:
            bucket = permission_bucket(attr(element, "name"))
            if bucket:
                weak_permissions.add(bucket)
        elif kind == "uses-feature":
            value = framework_name(attr(element, "name"))
            if value:
                uses_features.add("uses_feature:" + value)
        elif kind in {"activity", "activity-alias", "service", "receiver", "provider"}:
            name = normalize_component_name(package, attr(element, "name"))
            if not name:
                continue
            target = normalize_component_name(package, attr(element, "targetActivity"))
            component_type = "activity" if kind == "activity-alias" else kind
            tokens, weak = _component_tokens(element, component_type)
            components.append({
                "type": component_type,
                "manifest_kind": kind,
                "class_name": name,
                "target_class_name": target,
                "class_descriptor": class_descriptor(target or name),
                "tokens": tokens,
                "weak_tokens": weak,
                "exported": next((x.split(":", 1)[1] for x in tokens if x.startswith("exported:")), "implicit"),
                "enabled": next((x.split(":", 1)[1] for x in tokens if x.startswith("enabled:")), "true"),
            })
    return {
        "package_present_for_mapping_only": bool(package),
        "components": components,
        "weak_permissions": sorted(weak_permissions),
        "uses_features": sorted(uses_features),
    }


LIFECYCLE_METHODS = {
    "activity": {
        "onCreate", "onStart", "onResume", "onPause", "onStop", "onDestroy",
        "onNewIntent", "onActivityResult", "onRequestPermissionsResult",
    },
    "service": {
        "onCreate", "onStart", "onStartCommand", "onBind", "onRebind", "onUnbind",
        "onTaskRemoved", "onHandleIntent", "onDestroy", "onAccessibilityEvent",
        "onNotificationPosted", "onNotificationRemoved", "onCallScreening",
    },
    "receiver": {"onReceive"},
    "provider": {"onCreate", "query", "insert", "update", "delete", "openFile", "call", "bulkInsert"},
}


def lifecycle_names(component_type: str) -> set[str]:
    return set(LIFECYCLE_METHODS.get(component_type, set()))


def contract_schema() -> dict[str, Any]:
    return {
        "version": "CCBC_CONTRACT_SCHEMA_V1",
        "component_types": ["activity", "service", "receiver", "provider"],
        "token_families": {
            "component_type": {"source": "AndroidManifest component tag", "coarse": True, "fine": True},
            "exported": {"source": "android:exported", "coarse": True, "fine": True},
            "enabled": {"source": "android:enabled", "coarse": True, "fine": True},
            "action": {"source": "framework intent-filter action only", "coarse": True, "fine": True},
            "category": {"source": "framework intent-filter category only", "coarse": True, "fine": True},
            "scheme": {"source": "allowlisted scheme or custom bucket", "coarse": True, "fine": True},
            "mime_top": {"source": "top-level MIME bucket", "coarse": True, "fine": True},
            "foreground_type": {"source": "android:foregroundServiceType", "coarse": True, "fine": True},
            "special_service": {"source": "framework BIND_* permission", "coarse": True, "fine": True},
            "framework_meta": {"source": "framework-defined meta-data name", "coarse": True, "fine": True},
            "uses_feature": {"source": "framework uses-feature", "coarse": True, "fine": False},
            "weak_permission": {"source": "framework uses-permission", "coarse": False, "fine": False, "null_only": True},
        },
        "forbidden_model_fields": [
            "raw package name", "raw component class name", "developer", "certificate",
            "market", "file hash", "SDK/vendor name", "free text", "APK/DEX size",
            "file/method/API count",
        ],
        "class_name_use": "mapping only; never emitted as model feature",
        "missing_policy": "unknown; never treated as mismatch",
        "permissions_policy": "weak auxiliary and permission-only null; never standalone benign support",
    }


@dataclass
class CompatibilityEntry:
    support_count: int
    contract_apk_count: int
    posterior: float
    background: float
    specificity: float
    valid: bool


class CompatibilityModel:
    """Fold-local benign normative component-contract/Role table."""

    def __init__(self, minimum_support: int = 3, prior_strength: float = 5.0):
        self.minimum_support = int(minimum_support)
        self.prior_strength = float(prior_strength)
        self.entries: dict[tuple[str, int], CompatibilityEntry] = {}
        self.background = [0.0] * 31

    def fit(self, records: list[dict[str, Any]], labels: list[int]) -> "CompatibilityModel":
        benign = [record for record, label in zip(records, labels) if int(label) == 0]
        if not benign:
            raise RuntimeError("No benign outer-train APK for compatibility fitting")
        role_apks = [0] * 31
        token_apks: dict[str, int] = defaultdict(int)
        pair_apks: dict[tuple[str, int], int] = defaultdict(int)
        for record in benign:
            active_roles = set(int(value) for value in record.get("active_roles", []))
            for role in active_roles:
                role_apks[role] += 1
            observed_tokens = set()
            observed_pairs = set()
            for component in record.get("components", []):
                tokens = set(component.get("tokens", []))
                reachable_roles = set(int(value) for value in component.get("reachable_roles", []))
                observed_tokens.update(tokens)
                observed_pairs.update((token, role) for token in tokens for role in reachable_roles)
            for token in observed_tokens:
                token_apks[token] += 1
            for pair in observed_pairs:
                pair_apks[pair] += 1
        n = len(benign)
        self.background = [(value + 1.0) / (n + 2.0) for value in role_apks]
        for token, token_count in token_apks.items():
            for role in range(31):
                support = int(pair_apks.get((token, role), 0))
                background = float(self.background[role])
                posterior = (support + self.prior_strength * background) / (token_count + self.prior_strength)
                specificity = max(0.0, posterior - background) / max(1.0 - background, 1e-6)
                self.entries[(token, role)] = CompatibilityEntry(
                    support_count=support,
                    contract_apk_count=int(token_count),
                    posterior=float(posterior),
                    background=background,
                    specificity=float(specificity),
                    valid=bool(support >= self.minimum_support),
                )
        return self

    def lookup(self, tokens: list[str], role: int) -> dict[str, Any]:
        candidates = [self.entries[(token, int(role))] for token in tokens if (token, int(role)) in self.entries and self.entries[(token, int(role))].valid]
        if not candidates:
            return {"known": False, "compatibility": 0.0, "specificity": 0.0, "support": 0}
        best = max(candidates, key=lambda entry: (entry.posterior, entry.specificity, entry.support_count))
        return {
            "known": True,
            "compatibility": float(best.posterior),
            "specificity": float(best.specificity),
            "support": int(best.support_count),
            "background": float(best.background),
        }

    def to_rows(self, fold: int) -> list[dict[str, Any]]:
        rows = []
        for (token, role), entry in sorted(self.entries.items()):
            rows.append({
                "fold": int(fold), "contract_token": token, "role": int(role),
                "role_name": ROLE_NAME_SAFE(role), "benign_support_count": entry.support_count,
                "contract_benign_apk_count": entry.contract_apk_count,
                "smoothed_role_probability": entry.posterior,
                "background_role_probability": entry.background,
                "specificity": entry.specificity,
                "prototype_valid": int(entry.valid),
            })
        return rows


def ROLE_NAME_SAFE(role: int) -> str:
    from common import ROLE_NAMES
    return ROLE_NAMES[int(role)] if 0 <= int(role) < len(ROLE_NAMES) else "unknown"

