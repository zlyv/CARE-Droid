#!/usr/bin/env python3
from __future__ import annotations

from typing import Any
from common import canonical

# Frozen copy of the source-only TRCBE root taxonomy. Do not tune on labels/future.
RULES = [
    ("VIEW_CLICK", "Landroid/view/View$OnClickListener;", "onClick", "(Landroid/view/View;)V", "USER"),
    ("DIALOG_CLICK", "Landroid/content/DialogInterface$OnClickListener;", "onClick", "(Landroid/content/DialogInterface;I)V", "USER"),
    ("MENU_CLICK", "Landroid/view/MenuItem$OnMenuItemClickListener;", "onMenuItemClick", "(Landroid/view/MenuItem;)Z", "USER"),
    ("ITEM_CLICK", "Landroid/widget/AdapterView$OnItemClickListener;", "onItemClick", "(Landroid/widget/AdapterView;Landroid/view/View;IJ)V", "USER"),
    ("ITEM_SELECTED", "Landroid/widget/AdapterView$OnItemSelectedListener;", "onItemSelected", "(Landroid/widget/AdapterView;Landroid/view/View;IJ)V", "USER"),
    ("LONG_CLICK", "Landroid/view/View$OnLongClickListener;", "onLongClick", "(Landroid/view/View;)Z", "USER"),
    ("CHECKED_CHANGE", "Landroid/widget/CompoundButton$OnCheckedChangeListener;", "onCheckedChanged", "(Landroid/widget/CompoundButton;Z)V", "USER"),
    ("ACTIVITY_RESULT", "Landroid/app/Activity;", "onActivityResult", "(IILandroid/content/Intent;)V", "USER"),
    ("ANDROIDX_ACTIVITY_RESULT", "Landroidx/activity/result/ActivityResultCallback;", "onActivityResult", "(Ljava/lang/Object;)V", "USER"),
    ("BROADCAST_RECEIVER", "Landroid/content/BroadcastReceiver;", "onReceive", "(Landroid/content/Context;Landroid/content/Intent;)V", "AUTO"),
    ("SERVICE_START", "Landroid/app/Service;", "onStartCommand", "(Landroid/content/Intent;II)I", "AUTO"),
    ("JOB_SERVICE", "Landroid/app/job/JobService;", "onStartJob", "(Landroid/app/job/JobParameters;)Z", "AUTO"),
    ("WORK_MANAGER", "Landroidx/work/Worker;", "doWork", "()Landroidx/work/ListenableWorker$Result;", "AUTO"),
    ("FIREBASE_MESSAGE", "Lcom/google/firebase/messaging/FirebaseMessagingService;", "onMessageReceived", "(Lcom/google/firebase/messaging/RemoteMessage;)V", "AUTO"),
    ("CONTENT_PROVIDER_CREATE", "Landroid/content/ContentProvider;", "onCreate", "()Z", "AUTO"),
    ("INTENT_SERVICE", "Landroid/app/IntentService;", "onHandleIntent", "(Landroid/content/Intent;)V", "AUTO"),
    ("APPLICATION_CREATE", "Landroid/app/Application;", "onCreate", "()V", "NEUTRAL"),
    ("ACTIVITY_CREATE", "Landroid/app/Activity;", "onCreate", "(Landroid/os/Bundle;)V", "NEUTRAL"),
    ("ACTIVITY_START", "Landroid/app/Activity;", "onStart", "()V", "NEUTRAL"),
    ("ACTIVITY_RESUME", "Landroid/app/Activity;", "onResume", "()V", "NEUTRAL"),
    ("SERVICE_CREATE", "Landroid/app/Service;", "onCreate", "()V", "NEUTRAL"),
]
RULES = [{"family": f, "required_type": canonical(t), "name": canonical(n),
          "descriptor": canonical(d), "category": c} for f, t, n, d, c in RULES]


def classify(type_closure: set[str], name: str, descriptor: str) -> dict[str, Any] | None:
    name, descriptor = canonical(name), canonical(descriptor)
    for rule in RULES:
        if rule["required_type"] in type_closure and name == rule["name"] and descriptor == rule["descriptor"]:
            return rule
    return None
