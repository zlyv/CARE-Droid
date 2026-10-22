#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from common import canonical

RULES=[
 {"family":"VIEW_CLICK","required_type":"Landroid/view/View$OnClickListener;","name":"onClick","descriptor":"(Landroid/view/View;)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"framework listener type and exact callback signature","direction":True,"risk":"programmatic performClick can invoke it"},
 {"family":"DIALOG_CLICK","required_type":"Landroid/content/DialogInterface$OnClickListener;","name":"onClick","descriptor":"(Landroid/content/DialogInterface;I)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"framework dialog listener and signature","direction":True,"risk":"dialogs may be programmatically driven"},
 {"family":"MENU_CLICK","required_type":"Landroid/view/MenuItem$OnMenuItemClickListener;","name":"onMenuItemClick","descriptor":"(Landroid/view/MenuItem;)Z","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"menu listener and signature","direction":True,"risk":"programmatic menu dispatch"},
 {"family":"ITEM_CLICK","required_type":"Landroid/widget/AdapterView$OnItemClickListener;","name":"onItemClick","descriptor":"(Landroid/widget/AdapterView;Landroid/view/View;IJ)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"list item listener and signature","direction":True,"risk":"programmatic selection"},
 {"family":"ITEM_SELECTED","required_type":"Landroid/widget/AdapterView$OnItemSelectedListener;","name":"onItemSelected","descriptor":"(Landroid/widget/AdapterView;Landroid/view/View;IJ)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"selection listener and signature","direction":True,"risk":"selection may be restored automatically"},
 {"family":"LONG_CLICK","required_type":"Landroid/view/View$OnLongClickListener;","name":"onLongClick","descriptor":"(Landroid/view/View;)Z","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"framework long-click listener and signature","direction":True,"risk":"programmatic dispatch"},
 {"family":"CHECKED_CHANGE","required_type":"Landroid/widget/CompoundButton$OnCheckedChangeListener;","name":"onCheckedChanged","descriptor":"(Landroid/widget/CompoundButton;Z)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"widget change listener and signature","direction":True,"risk":"state can be changed programmatically"},
 {"family":"ACTIVITY_RESULT","required_type":"Landroid/app/Activity;","name":"onActivityResult","descriptor":"(IILandroid/content/Intent;)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"activity result callback with structural subtype evidence","direction":True,"risk":"result may originate from a non-user flow"},
 {"family":"ANDROIDX_ACTIVITY_RESULT","required_type":"Landroidx/activity/result/ActivityResultCallback;","name":"onActivityResult","descriptor":"(Ljava/lang/Object;)V","category":"USER_MEDIATED_HIGH_CONFIDENCE","confidence":"HIGH","reason":"AndroidX result callback interface and exact signature","direction":True,"risk":"generic object erases result kind"},

 {"family":"BROADCAST_RECEIVER","required_type":"Landroid/content/BroadcastReceiver;","name":"onReceive","descriptor":"(Landroid/content/Context;Landroid/content/Intent;)V","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"framework receiver subtype and signature","direction":True,"risk":"broadcast may follow a user action"},
 {"family":"SERVICE_START","required_type":"Landroid/app/Service;","name":"onStartCommand","descriptor":"(Landroid/content/Intent;II)I","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"service lifecycle callback","direction":True,"risk":"service may be explicitly started by foreground UI"},
 {"family":"JOB_SERVICE","required_type":"Landroid/app/job/JobService;","name":"onStartJob","descriptor":"(Landroid/app/job/JobParameters;)Z","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"scheduled job callback","direction":True,"risk":"job scheduling may follow user action"},
 {"family":"WORK_MANAGER","required_type":"Landroidx/work/Worker;","name":"doWork","descriptor":"()Landroidx/work/ListenableWorker$Result;","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"WorkManager worker callback","direction":True,"risk":"work may have been enqueued by UI"},
 {"family":"FIREBASE_MESSAGE","required_type":"Lcom/google/firebase/messaging/FirebaseMessagingService;","name":"onMessageReceived","descriptor":"(Lcom/google/firebase/messaging/RemoteMessage;)V","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"push message callback","direction":True,"risk":"push may reflect a requested service"},
 {"family":"CONTENT_PROVIDER_CREATE","required_type":"Landroid/content/ContentProvider;","name":"onCreate","descriptor":"()Z","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"provider initialization callback","direction":True,"risk":"provider access may be user-mediated indirectly"},
 {"family":"INTENT_SERVICE","required_type":"Landroid/app/IntentService;","name":"onHandleIntent","descriptor":"(Landroid/content/Intent;)V","category":"AUTONOMOUS_HIGH_CONFIDENCE","confidence":"HIGH","reason":"background intent service callback","direction":True,"risk":"intent may originate from foreground UI"},

 {"family":"APPLICATION_CREATE","required_type":"Landroid/app/Application;","name":"onCreate","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH","reason":"process lifecycle is not direction-specific","direction":False,"risk":"none"},
 {"family":"ACTIVITY_CREATE","required_type":"Landroid/app/Activity;","name":"onCreate","descriptor":"(Landroid/os/Bundle;)V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH","reason":"activity creation can be user launch or system restoration","direction":False,"risk":"none"},
 {"family":"ACTIVITY_START","required_type":"Landroid/app/Activity;","name":"onStart","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH","reason":"ambiguous lifecycle","direction":False,"risk":"none"},
 {"family":"ACTIVITY_RESUME","required_type":"Landroid/app/Activity;","name":"onResume","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH","reason":"ambiguous lifecycle","direction":False,"risk":"none"},
 {"family":"SERVICE_CREATE","required_type":"Landroid/app/Service;","name":"onCreate","descriptor":"()V","category":"NEUTRAL_OR_AMBIGUOUS","confidence":"HIGH","reason":"service construction does not reveal trigger origin","direction":False,"risk":"none"},
]

for rule in RULES:
    rule["required_type"]=canonical(rule["required_type"]);rule["name"]=canonical(rule["name"]);rule["descriptor"]=canonical(rule["descriptor"])

def schema()->dict[str,Any]:
    return {"schema_version":"1.0.0","label_free":True,"future_used":False,"exact_type_and_signature_required":True,"method_name_only_enters_direction":False,"categories":["USER_MEDIATED_HIGH_CONFIDENCE","AUTONOMOUS_HIGH_CONFIDENCE","NEUTRAL_OR_AMBIGUOUS","UNKNOWN"],"rules":RULES,"known_gaps":["XML android:onClick not structurally recoverable without Manifest","reflection-registered listeners","some dynamic receivers","Kotlin/Compose lambdas without framework interface ancestry","callbacks farther than the frozen endpoint graph upstream bound"]}

def classify(type_closure:set[str],name:str,descriptor:str)->dict[str,Any]|None:
    name=canonical(name);descriptor=canonical(descriptor)
    for rule in RULES:
        if rule["required_type"] in type_closure and name==rule["name"] and descriptor==rule["descriptor"]:return rule
    return None

CALLBACK_NAMES={r["name"] for r in RULES}

