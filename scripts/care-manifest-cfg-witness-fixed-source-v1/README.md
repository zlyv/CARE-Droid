# Manifest Stage-I + CFG Witness fixed-source evaluation

This audit combines existing frozen predictions without retraining:

- Manifest135-only LR/RF/XGB/SVM Stage-I models trained on 2017;
- Manifest-probability-only disagreement and uncertainty routes frozen from 2017;
- Direct-C43 W0 and fixed-capacity W1/W2/W3 experts trained on 2017;
- 2017 OOF plus fixed-source evaluation on 2018--2021.

No APK is read or executed. No future label changes training, routing, thresholds, or models.
