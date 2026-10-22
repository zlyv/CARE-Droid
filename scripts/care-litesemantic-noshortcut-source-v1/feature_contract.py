#!/usr/bin/env python3
"""Label-free, identity-free LiteSemantic feature contract."""
from __future__ import annotations

PERMISSION_GROUPS={
 "sms":("READ_SMS","RECEIVE_SMS","SEND_SMS","RECEIVE_MMS","RECEIVE_WAP_PUSH"),
 "phone":("READ_PHONE_STATE","READ_PHONE_NUMBERS","CALL_PHONE","ANSWER_PHONE_CALLS","READ_CALL_LOG","WRITE_CALL_LOG"),
 "contacts_calendar":("READ_CONTACTS","WRITE_CONTACTS","GET_ACCOUNTS","READ_CALENDAR","WRITE_CALENDAR"),
 "location":("ACCESS_FINE_LOCATION","ACCESS_COARSE_LOCATION","ACCESS_BACKGROUND_LOCATION"),
 "microphone":("RECORD_AUDIO",),"camera":("CAMERA",),
 "storage":("READ_EXTERNAL_STORAGE","WRITE_EXTERNAL_STORAGE","MANAGE_EXTERNAL_STORAGE"),
 "admin_overlay":("SYSTEM_ALERT_WINDOW","REQUEST_INSTALL_PACKAGES","BIND_DEVICE_ADMIN","BIND_ACCESSIBILITY_SERVICE"),
}

MANIFEST_NAMES=(
 *("perm_presence__"+k for k in PERMISSION_GROUPS),
 "has_activity","has_service","has_receiver","has_provider",
 "component_share_activity","component_share_service","component_share_receiver","component_share_provider",
 "has_boot_context","has_accessibility_context","has_device_admin_context","has_overlay_context",
)


def arm_indices(feature_names):
    names=list(feature_names)
    manifest=[i for i,n in enumerate(names) if n.startswith("manifest__")]
    presence=[i for i,n in enumerate(names) if n.startswith("role_presence__")]
    proportion=[i for i,n in enumerate(names) if n.startswith("role_proportion__")]
    if len(manifest)!=len(MANIFEST_NAMES) or len(presence)!=31 or len(proportion)!=31:
        raise RuntimeError("LiteSemantic feature contract mismatch")
    return {
      "L0_MANIFEST":manifest,
      "L1_ROLE_PRESENCE":presence,
      "L2_ROLE_PROPORTION":proportion,
      "L3_FUSED":manifest+presence+proportion,
    }
