# CARE-Droid BFA-V2

Read-only reuse of BFA-V1 models and frozen 2017–2021 features. Audits B1/B3 APK-level overlap, then retrains only B13 with the same class-blind training-window External/Internal anchors and recomputed Residual. No APK files are read or executed. All new files remain under `outputs/care_background_complementarity_v2/`.

The first three B0/B1/B3 feature configurations must replay BFA-V1 exactly before training. The pipeline is resumable but never overwrites an existing checkpoint.
