# CARE-Droid P31-LR C43-RQ Provenance6 source v1

Strict 2017 source-only, five-fold OOF representation experiment.

- Stage-I classifier: calibrated P31 Logistic Regression only.
- Router: four-member LR/RF/XGB/SVM committee is used only to compute LR-anchored MeanOpposition; exact foldwise top 25% (200/799).
- Stage-II: C43 Relation-Query, Argmax Critical Host, unchanged scalar bounded correction (`max_delta=2.25`), trained on all outer-training APKs.
- P0: six provenance fields fixed to zero.
- P1: six real Host/endpoint provenance fields.
- P2: P1 provenance shuffled within APK and existing primary role, without labels.
- The same Compact43+Prov6 representation is used for validation Hosts and all three fold-safe backgrounds.
- Relation31 remains attention-only. No Ego tensor, future row, Action3 head, threshold search, or route change.

Run on the server from the project root:

```bash
unzip -o ../care-litelocal-p31lr-c43-prov6-source-v1.zip -d scripts/
bash scripts/care-litelocal-p31lr-c43-prov6-source-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31lr-c43-prov6-source-v1.console.log
```

P0 and P1 run concurrently on GPU0/GPU1. P2 starts on GPU0 after both finish.
