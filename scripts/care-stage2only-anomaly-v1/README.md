# Independent Stage-II-only Anomaly control

This strict 2017 outer-fold OOF control invokes Stage II for all 799 APKs and
contains no Global24, Stage-I probability, router, bounded correction, FixKeep,
or MalBen role/class prototypes. Its 81-D representation is
`[g_i, z_anomaly, h_critical, e_anomaly]`, where `z_anomaly` is the
attention-weighted local embedding and `e_anomaly` is the attention-weighted
established anomaly residual.

The folds, three seeds, graph/local widths, early-stopping protocol, final
feature width, StandardScaler, and L2 Logistic Regression are identical to the
completed StageIIOnly-MalBen experiment. No APK re-extraction is performed.

`preflight_stage2only_anomaly.py` independently validates the passed 799-APK
asset gate, source lineage, region cache, endpoint cache, and completed paired
MalBen OOF file. It does not depend on an optional aggregate gate shell script.

The archive vendors the exact Stage-II-only training core and source-label
compatibility helper used by this control. The vendored core imports the
CARE role-graph v4 module and its complete historical Python dependency bundle
from the same project snapshot. It has no import-time dependency on scripts
already being present elsewhere on the server.
