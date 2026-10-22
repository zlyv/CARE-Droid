#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from common import canonical

# This is intentionally identical to the frozen TRCBE source-v1 taxonomy.
RULES = [
    {"family":"VIEW_CLICK","required_type":"Landroid/view/View$OnClickListener;","name":"onClick","descriptor":"(Landroid/view/View;)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"DIALOG_CLICK","required_type":"Landroid/content/DialogInterface$OnClickListener;","name":"onClick","descriptor":"(Landroid/content/DialogInterface;I)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"MENU_CLICK","required_type":"Landroid/view/MenuItem$OnMenuItemClickListener;","name":"onMenuItemClick","descriptor":"(Landroid/view/MenuItem;)Z","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"ITEM_CLICK","required_type":"Landroid/widget/AdapterView$OnItemClickListener;","name":"onItemClick","descriptor":"(Landroid/widget/AdapterView;Landroid/view/View;IJ)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"ITEM_SELECTED","required_type":"Landroid/widget/AdapterView$OnItemSelectedListener;","name":"onItemSelected","descriptor":"(Landroid/widget/AdapterView;Landroid/view/View;IJ)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"LONG_CLICK","required_type":"Landroid/view/View$OnLongClickListener;","name":"onLongClick","descriptor":"(Landroid/view/View;)Z","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"CHECKED_CHANGE","required_type":"Landroid/widget/CompoundButton$OnCheckedChangeListener;","name":"onCheckedChanged","descriptor":"(Landroid/widget/CompoundButton;Z)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"ACTIVITY_RESULT","required_type":"Landroid/app/Activity;","name":"onActivityResult","descriptor":"(IILandroid/content/Intent;)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"ANDROIDX_ACTIVITY_RESULT","required_type":"Landroidx/activity/result/ActivityResultCallback;","name":"onActivityResult","descriptor":"(Ljava/lang/Object;)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"BROADCAST_RECEIVER","required_type":"Landroid/content/BroadcastReceiver;","name":"onReceive","descriptor":"(Landroid/content/Context;Landroid/content/Intent;)V","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"SERVICE_START","required_type":"Landroid/app/Service;","name":"onStartCommand","descriptor":"(Landroid/content/Intent;II)I","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"JOB_SERVICE","required_type":"Landroid/app/job/JobService;","name":"onStartJob","descriptor":"(Landroid/app/job/JobParameters;)Z","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"WORK_MANAGER","required_type":"Landroidx/work/Worker;","name":"doWork","descriptor":"()Landroidx/work/ListenableWorker$Result;","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"FIREBASE_MESSAGE","required_type":"Lcom/google/firebase/messaging/FirebaseMessagingService;","name":"onMessageReceived","descriptor":"(Lcom/google/firebase/messaging/RemoteMessage;)V","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"CONTENT_PROVIDER_CREATE","required_type":"Landroid/content/ContentProvider;","name":"onCreate","descriptor":"()Z","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"INTENT_SERVICE","required_type":"Landroid/app/IntentService;","name":"onHandleIntent","descriptor":"(Landroid/content/Intent;)V","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH"},
    {"family":"APPLICATION_CREATE","required_type":"Landroid/app/Application;","name":"onCreate","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH"},
    {"family":"ACTIVITY_CREATE","required_type":"Landroid/app/Activity;","name":"onCreate","descriptor":"(Landroid/os/Bundle;)V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH"},
    {"family":"ACTIVITY_START","required_type":"Landroid/app/Activity;","name":"onStart","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH"},
    {"family":"ACTIVITY_RESUME","required_type":"Landroid/app/Activity;","name":"onResume","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH"},
    {"family":"SERVICE_CREATE","required_type":"Landroid/app/Service;","name":"onCreate","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH"},
]
for rule in RULES:
    rule["required_type"] = canonical(rule["required_type"])
    rule["name"] = canonical(rule["name"])
    rule["descriptor"] = canonical(rule["descriptor"])


def classify(type_closure: set[str], name: str, descriptor: str) -> dict[str, Any] | None:
    name, descriptor = canonical(name), canonical(descriptor)
    for rule in RULES:
        if rule["required_type"] in type_closure and name == rule["name"] and descriptor == rule["descriptor"]:
            return rule
    return None


def schema() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "source": "byte-for-byte semantic copy of TRCBE source-v1 frozen rules",
        "label_free": True,
        "exact_type_and_signature_required": True,
        "method_name_only_enters_direction": False,
        "rules": RULES,
    }
