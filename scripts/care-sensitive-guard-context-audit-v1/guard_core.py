#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any

from common import canonical

METHOD_REFERENCE = re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+", re.I)

USER_CALLBACKS = {
    "onclick", "onlongclick", "onitemclick", "onitemlongclick", "onchildclick",
    "onpreferenceclick", "ontouch", "onkey", "oneditoraction", "oncheckedchanged",
    "onoptionsitemselected", "oncontextitemselected", "ondialogclick", "onactivityresult",
}
BACKGROUND_CALLBACKS = {
    "onreceive", "onstartcommand", "onhandleintent", "doinbackground", "dowork",
    "onstartjob", "onperformSync", "onlocationchanged", "onmessageReceived",
    "onbind", "ontransact", "onalarm", "ontaskremoved",
}

PERMISSION_MARKERS = (
    "->checkselfpermission(", "->checkcallingpermission(", "->checkcallingorselfpermission(",
    "->enforcepermission(", "->enforcecallingpermission(", "->shouldshowrequestpermissionrationale(",
)
CONSENT_MARKERS = (
    "->requestpermissions(", "landroid/app/alertdialog", "landroidx/appcompat/app/alertdialog",
    "->setpositivebutton(", "->setnegativebutton(", "->show(",
)
BACKGROUND_SCHEDULING_MARKERS = (
    "landroid/app/alarmmanager;->set", "landroidx/work/", "lcom/google/firebase/jobdispatcher/",
    "landroid/app/job/jobscheduler;->schedule", "landroid/os/handler;->post",
    "ljava/lang/thread;->start", "ljava/util/timer;->schedule",
)


def parse_method_reference(output: str) -> str | None:
    matches = METHOD_REFERENCE.findall(canonical(output))
    return canonical(matches[-1]) if matches else None


def split_reference(reference: str) -> tuple[str, str, str] | None:
    match = re.fullmatch(r"(L[^;]+;)->([^\s(]+)(\([^)]*\).+)", canonical(reference), re.I)
    return match.groups() if match else None


def api_rule_matches(rule: dict[str, Any], class_name: str, method_name: str) -> bool:
    if rule.get("rule_kind", "api") != "api":
        return False
    class_prefixes = [canonical(x) for x in rule.get("class_prefixes", [])]
    class_contains = [canonical(x) for x in rule.get("class_contains", [])]
    method_names = [canonical(x) for x in rule.get("method_names", [])]
    method_prefixes = [canonical(x) for x in rule.get("method_prefixes", [])]
    class_ok = any(class_name.startswith(x) for x in class_prefixes) or any(x in class_name for x in class_contains)
    method_ok = method_name in method_names or any(method_name.startswith(x) for x in method_prefixes)
    return bool(class_ok and method_ok and (class_prefixes or class_contains))


def match_role(reference: str, mapping: dict[str, Any]) -> dict[str, Any] | None:
    parts = split_reference(reference)
    if not parts:
        return None
    class_name, method_name, _ = parts
    for rule in mapping.get("rules", []):
        if api_rule_matches(rule, class_name, method_name):
            return {"rule_id": str(rule.get("rule_id", "")), "role": str(rule.get("role", "other")),
                    "confidence": float(rule.get("confidence", 0.0))}
    return None


def contains_marker(references: list[str], markers: tuple[str, ...]) -> bool:
    return any(marker in reference for reference in references for marker in markers)


def classify_method(method_name: str, instructions: list[dict[str, Any]], sensitive_index: int,
                    references: list[str]) -> dict[str, Any]:
    name = canonical(method_name)
    preceding = instructions[max(0, sensitive_index - 12):sensitive_index]
    preceding_names = [canonical(row.get("name")) for row in preceding]
    all_names = [canonical(row.get("name")) for row in instructions]
    branch_local = any(op.startswith("if-") or op in {"packed-switch", "sparse-switch"} for op in preceding_names)
    branch_any = any(op.startswith("if-") or op in {"packed-switch", "sparse-switch"} for op in all_names)
    user_callback = name in {canonical(x) for x in USER_CALLBACKS}
    background_callback = name in {canonical(x) for x in BACKGROUND_CALLBACKS}
    permission_check = contains_marker(references, PERMISSION_MARKERS)
    consent_ui = contains_marker(references, CONSENT_MARKERS)
    background_scheduling = contains_marker(references, BACKGROUND_SCHEDULING_MARKERS)
    directional = user_callback or background_callback
    condition_bearing = directional or branch_local or permission_check or consent_ui or background_scheduling
    if user_callback and not background_callback:
        context = "USER_CALLBACK"
    elif background_callback and not user_callback:
        context = "BACKGROUND_CALLBACK"
    elif user_callback and background_callback:
        context = "CONFLICTING_CALLBACK"
    elif condition_bearing:
        context = "CONDITION_PRESENT_UNATTRIBUTED"
    else:
        context = "NO_RELIABLE_CONDITION"
    return {
        "context": context,
        "user_callback": user_callback,
        "background_callback": background_callback,
        "branch_local_12": branch_local,
        "branch_any_method": branch_any,
        "permission_check_colocated": permission_check,
        "consent_ui_colocated": consent_ui,
        "background_scheduling_colocated": background_scheduling,
        "condition_bearing": condition_bearing,
        "directional_callback": directional,
    }
