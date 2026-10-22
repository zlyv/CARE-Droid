# CARE-Droid high-confidence observability audit v1

This is a read-only matched case-control audit over 2017-2021 Direct-C43 high-confidence errors. It inspects static observability gaps that were not represented reliably by P31/C43: native/JNI, reflection, dynamic loading, embedded payloads, known packer traces, namespace-observable SDK dominance, Manifest-component code coverage, multidex, and identifier obfuscation proxies.

Labels, including future labels, are used only to select diagnostic error/control pairs. The audit never trains a model, changes a threshold, changes routing, or authorizes a CARE-Droid feature. Any descriptive mechanism must be confirmed separately using source-only development or untouched external evidence.

Revision r2 reuses the validated APK-lineage/index discovery surface from the completed CCBC future diagnostics. It accepts CSV, JSON, JSONL and JSONL.GZ indices, verifies every selected SHA/path pair against an existing APK, and optionally honors `HCOA_SOURCE_APK_INDEX`, `HCOA_FUTURE_APK_INDEX`, `HCOA_SOURCE_APK_ROOTS` and `HCOA_FUTURE_APK_ROOTS`. This revision does not change pair selection or the audit protocol.
