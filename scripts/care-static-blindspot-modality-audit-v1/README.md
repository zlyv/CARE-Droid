# CARE-Droid static blind-spot modality audit v1

This read-only audit asks whether the 93 LR/RF/XGB/SVM/Direct-C43 consensus
errors are enriched for static modalities that the frozen system may not fully
observe: native code, multidex, embedded code payloads, dynamic loading,
reflection, native loading, indirect invokes, encrypted/high-entropy payloads,
and Manifest-to-DEX component gaps.

The source decision uses only 2017.  Each source consensus error is compared
with same-label, same-fold controls matched on all five model scores plus APK,
method and invoke scale.  2018--2021 are diagnostics only.  No classifier,
route, threshold, label or formal asset is modified.

Exact API references are used transiently to map aggregate observability
families and are not exported as classification features.  Cached per-APK
records make interrupted runs resumable.  GPU is not used.
