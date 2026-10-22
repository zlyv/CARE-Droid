# CARE-Droid HostRel74 semantic-background source v1

Strict 2017 source-only capacity-matched test of fixed-width relational Host
features. It does not load any future row or Ego tensor and does not instantiate
an Ego encoder.

- B0: locked original H0 control; not retrained.
- B1: locked original Host74 plus hard role-conditioned backgrounds; not retrained.
- R1: `SelfRole31 + DirectedNeighborRole31 + frozen tail12` with the same hard
  role-conditioned three backgrounds rebuilt in the same HostRel74 space.

The relation block is
`tanh(log1p(outgoing normalized-neighbor role mass) - log1p(incoming normalized-neighbor role mass))`.
It uses only raw retained Host-to-Host edges in the existing endpoint cache; no
path contraction or APK reanalysis is performed. Model width remains 74 and
the neural parameter count remains 33,036.

```bash
cd /home/boyuanjie/projects/tasa-droid
unzip -o ../care-di25-hostrel-semantic-background-source-v1.zip -d scripts/
bash scripts/care-di25-hostrel-semantic-background-source-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostrel-semantic-background-source-v1.console.log
```

Set `CARE_GPU=1` before launching to use GPU 1 instead of the default GPU 0.
