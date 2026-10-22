# CARE-Droid P31 lineage and Host-direction annotatability audit v1

This package deliberately does not train a new classifier or correction head.

It answers two prerequisite questions:

1. Are P31-equivalent cross-fold samples also related at the APK code/Manifest/certificate level, or only equivalent after P31 compression?
2. Do the existing routed full graphs, endpoint roles and frozen C43 attention expose enough Critical-Host context for a blinded human direction audit?

The package is source-2017-only and does not load future rows. It reuses the completed strict group-validation output and the completed routed full-graph cache. It writes only to `outputs/care-p31-lineage-host-audit-v1`.

Main outputs:

- `lineage_summary.json`
- `lineage_pair_audit.csv`
- `p31_exposure_performance.csv`
- `host_annotation_feasibility.json`
- `host_review_items_blinded.csv`
- `host_review_form_blinded.csv`
- `host_review_key_sealed.csv`
- `annotation_protocol.md` (also shipped with the package)
- `final_decision.json`
- `report.md`

Do not open the sealed key before annotations are frozen.
