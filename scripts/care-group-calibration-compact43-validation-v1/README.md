# CARE group-safe calibration and Compact43 strict validation v1

Source-only confirmatory diagnostic. It does not modify the frozen CARE-Droid method.

The package:

- reproduces the prior P31 similarity groups;
- audits connected-component chaining and group purity;
- compares raw, group-OOF Platt-calibrated, and inverse-group-size weighted RF;
- creates strict inner-OOF Stage-I contexts for every outer training fold;
- validates the predeclared Compact43 mean+std+max representation;
- compares mean/std/max components, count-only, count-residualized Compact43, and five matched shuffles;
- repeats the direction probe under original fixed folds and group-disjoint stress folds;
- does not run bounded correction or future diagnostics.

