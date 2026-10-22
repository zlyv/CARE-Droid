# CARE-Droid GraphGlobal path 2x2 source ablation v1

This package tests one narrowly defined mechanism: whether Stage-II
`GraphGlobal` is useful for selecting Host evidence, for directly controlling
the correction direction/magnitude, for both, or for neither.

| Arm | GraphGlobal -> attention | GraphGlobal -> delta |
|---|---:|---:|
| `N0_NO_GRAPHGLOBAL` | no | no |
| `D_DELTA_ONLY` | no | yes |
| `Q_QUERY_ONLY` | yes | no |
| `QD_QUERY_AND_DELTA` | yes | yes |

All arms use the same Node/Edge-only role encoder, separate GraphGlobal
encoder, Host-only anomaly representation, 113-D correction head, seeds,
strict 2017 five-fold OOF protocol, frozen DI25 route (200/799), and bounded
logit correction `2.25*tanh(raw_delta)`. No Ego or 2018--2021 row is read.

`Q_QUERY_ONLY` does **not** mean that every global signal is absent from the
correction head: Global24 and the Stage-I logit remain fixed inputs. It means
only that the Stage-II GraphGlobal block has no direct post-selection delta
path. Its influence may propagate through the Host evidence it selects.

The self-test contains two hard mechanism checks:

1. changing GraphGlobal cannot change the Node/Edge role token or encoded Host
   vectors;
2. after selected Host evidence is frozen, changing GraphGlobal cannot change
   Q-only delta, while it must change D-enabled delta.

Historical M0 is not one of the four factorial arms because it injects
GraphGlobal inside the role encoder. Keep it as an external reference.

## Upload and run

Upload the ZIP to `/home/boyuanjie/projects/`, then:

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-graphglobal-path-source-v1.zip -d scripts/

# quick fail-closed checks
/home/boyuanjie/envs/tif/bin/python scripts/care-di25-graphglobal-path-source-v1/selftest.py
/home/boyuanjie/envs/tif/bin/python scripts/care-di25-graphglobal-path-source-v1/preflight.py

# long run
bash scripts/care-di25-graphglobal-path-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-graphglobal-path-source-v1.console.log
```

Do not launch a future diagnostic until the source-side decision rule is
frozen. The four arms answer the path question; they do not by themselves
establish temporal generalization.

