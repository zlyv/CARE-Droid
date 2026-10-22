# CARE-Droid P31-DI25 + C43-RQ complete diagnostics v1

This package fills the missing end-to-end 2018-2021 evaluation. The previously
reported P31 diagnostics contained Stage-I and routing only. This package adds
the source-selected `C43_RELATION_QUERY` bounded correction.

It runs two separate protocols:

1. frozen 2017 training followed by label-free model prediction on 2018-2021;
2. independent same-year five-fold OOF, where every Stage-I, Router input,
   background, graph scaler, and Stage-II model is fit inside that year.

The same-year protocol is not temporal generalization. Future labels are joined
to frozen saved predictions only by `summarize.py`. The failed TopK and aligned
localization candidates are not used. No Ego tensor is loaded.

Run from the repository root (the zip is uploaded to `/home/boyuanjie/projects`):

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-litelocal-p31-c43-complete-diagnostics-v1.zip -d scripts/
bash scripts/care-litelocal-p31-c43-complete-diagnostics-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-p31-c43-complete-diagnostics-v1.console.log
```

Per-job logs:

```bash
tail -F artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/logs/frozen.gpu0.log
tail -F artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/logs/within_2018_2019.gpu1.log
tail -F artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/logs/within_2020_2021.gpu0.log
```

Final report:

`artifacts/care-litelocal-p31-c43-complete-diagnostics-v1/results/REPORT.txt`
