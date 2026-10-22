#!/usr/bin/env python3
from __future__ import annotations
import re

FAMILIES={
"UI_INTERACTION":("landroid/view/","landroid/widget/","landroid/app/alertdialog","landroidx/appcompat/"),
"WEB_CONTENT":("landroid/webkit/",),
"MEDIA_PRESENTATION":("landroid/media/mediaplayer","landroid/media/audiotrack","landroid/graphics/","landroid/opengl/"),
"CAMERA_MEDIA":("landroid/hardware/camera","landroid/media/mediarecorder","landroid/media/audiorecord"),
"LOCATION_GEO":("landroid/location/","landroid/net/uri;->parse(ljava/lang/string;)"),
"TELEPHONY_COMMUNICATION":("landroid/telephony/","landroid/telecom/"),
"CONTACTS_CALENDAR":("landroid/provider/contactscontract","landroid/provider/calendarcontract"),
"BLUETOOTH_NFC_USB":("landroid/bluetooth/","landroid/nfc/","landroid/hardware/usb/"),
"SENSOR_MOTION":("landroid/hardware/sensormanager","landroid/hardware/sensor"),
"NOTIFICATION":("landroid/app/notification","landroidx/core/app/notification"),
"ACCOUNT_AUTH":("landroid/accounts/","landroid/security/keystore/","ljava/security/"),
"DOCUMENT_WORKFLOW":("landroid/provider/documentscontract","landroid/app/downloadmanager","landroid/os/storage/"),
"BACKGROUND_SCHEDULING":("landroid/app/job/","landroid/app/alarmmanager","landroidx/work/"),
"ACCESSIBILITY_INPUT":("landroid/accessibilityservice/","landroid/view/accessibility/","landroid/view/inputmethod/"),
"VPN_CONNECTIVITY":("landroid/net/vpnservice","landroid/net/connectivitymanager","landroid/net/wifi/"),
"PACKAGE_MANAGEMENT":("landroid/content/pm/","landroid/app/admin/"),
"DATABASE_CONTENT":("landroid/database/","landroid/content/contentresolver"),
"IPC_COMPONENT":("landroid/content/context;->start","landroid/content/context;->bind","landroid/content/context;->sendbroadcast","landroid/os/binder"),
}
METHOD_REF=re.compile(r"L[^;]+;->[^\s(]+\([^)]*\)[^\s,}\]]+",re.I)
def classify(ref):
    x=str(ref).replace(" ","").casefold();return [k for k,v in FAMILIES.items() if any(t in x for t in v)]
def schema():return {"version":"FUNCTION_ROLE_EXPLANATION_V1","families":{k:list(v) for k,v in FAMILIES.items()},"scope":"coarse framework functional families; raw API identities are discarded","future_rows_loaded":0}

