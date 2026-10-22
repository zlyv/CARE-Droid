# CARE-Droid Global24 utility and cost source audit v1

Strict 2017 five-fold OOF audit. It never reads 2018--2021 and never trains
Stage II.

Stage-I utility:

- frozen official Global24 L2 control;
- replayed FULL24;
- leave-one-group-out for Manifest, Code/Call, Sensitive, and Graph;
- group-only sufficiency arms.

Router utility:

- fixed official Stage-I decisions;
- calibrated LR/RF/XGBoost/SVM committees using FULL24 or one group removed;
- exact foldwise 25% routing budget;
- error recall, precision, high-confidence error recall, and exact paired counts.

Cost boundary:

- measures cached Global24 matrix/model wall time;
- records artifact sizes and feature acquisition provenance;
- does not rerun APK parsing/callgraph extraction merely to manufacture timing;
- raw APK-to-Global24 phase timing is explicitly reported unavailable.

The launcher uses an existing environment containing NumPy, scikit-learn,
PyTorch, and XGBoost. It installs nothing.
