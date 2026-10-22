# CARE-Droid SCBE V1

Independent, checkpoint-resumable experiment on the historical 2017–2021 cache.
No APK is parsed or executed. Formal scripts, historical checkpoints, routing
outputs and Prototype Banks are read-only. All new files are confined to
`outputs/care_scbe_v1/` and this new script directory.

The method uses the formal 107-dimensional standardized Host space and the
original 32/24 centers. It estimates a 24-dimensional External similarity from
32-dimensional Shared similarity plus Role31, then uses an APK-cross-fitted
residual and APK-level, smoothed category-support weights. It changes only the
existing External scalar and its deterministic Residual scalar in the 142-wide
Host tensor. Internal and Commonness remain bitwise unchanged.

This is an experiment, not a replacement for the formal model. The old
Prototype Bank includes each training APK's Hosts, so cross-fitting the
conditional predictor and support labels does **not** remove every source-bank
self-influence. This limitation is audited, never described as leak-free.
