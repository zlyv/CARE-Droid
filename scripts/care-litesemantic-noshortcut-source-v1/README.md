# CARE-Droid LiteSemantic-NoShortcut source v1

Strict 2017-only feasibility experiment for a genuinely low-cost first layer.
It parses the manifest and DEX reference tables but never constructs an
Androguard `Analysis`, call graph, RoleGraph, Host, or Ego.

Model inputs contain only semantic binary/ratio features. Raw APK bytes, DEX,
class, method-reference, string, permission, component, and mapped-API counts
are audit-only. Package names, certificates, timestamps, SDK versions, years,
and dataset-source identities are excluded.

Arms: Manifest20, RolePresence31, RoleProportion31, and fused82. Each receives
strict outer-fold OOF LR/RF/XGB/SVM probabilities with train-only inner-OOF
Platt calibration. DI-style mean opposition uses exact foldwise 25% budgets.

The shortcut gate reports feature-size correlations and OOF prediction of
extreme APK/method/class-size quartiles. A source result cannot authorize future
evaluation or replacement of frozen CARE-Droid.

Androguard/Loguru parser DEBUG output is disabled before any APK or DEX parser
is imported. The console retains only bounded progress messages and failures.
