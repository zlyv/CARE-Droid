# CARE-Droid CCBC source v1

Independent, source-only implementation of **Component-Contract Bidirectional Counterevidence**.
It never modifies the frozen P31-RF / RF-DI25 / A0 C43-RQ code or its artifacts. The launcher
selects a new `outputs/care-ccbc-source-vN` directory whenever an earlier run exists.

## Run

```bash
cd /home/boyuanjie/projects/tasa-droid
bash scripts/care-ccbc-source-v1/launch_tmux.sh
```

The runner automatically prefers `/home/boyuanjie/envs/tif/bin/python`. For another environment:

```bash
export CCBC_PYTHON=/absolute/path/to/python
```

If APK discovery needs help, set either an index or roots before launching:

```bash
export CCBC_APK_INDEX=/absolute/path/to/sha256_to_apk.csv
# or
export CCBC_APK_ROOTS=/root/one:/root/two
```

Optional explicit cache paths are documented by the `CCBC_*` discovery variables in `common.py`.

## Monitor and collect

Use the exact paths printed by `launch_tmux.sh`. When complete:

```bash
bash scripts/care-ccbc-source-v1/collect_results.sh outputs/care-ccbc-source-v1
```

Send the complete output plus, if failed, the last 160 lines of the console log. Do not rerun with
changed thresholds. A failure caused by schema/path mismatch should be repaired without changing the
research definition and recorded as the same version; a feature-definition change requires v2.

## Evidence boundary

- Selection is strictly 2017, five-fold OOF; outer-train Stage-I state is inner OOF.
- Future rows loaded during selection: zero.
- Same-call-graph reachability is an execution-scope approximation, not data flow or runtime proof.
- 2018–2021 are not launched by this source pipeline. They may be run only after the source protocol
  and protocol hash are frozen; they are historical diagnostics, not untouched tests.
- The final source conclusion is automatically restricted to `SUPPORTED`, `PARTIALLY_SUPPORTED`,
  `NOT_SUPPORTED`, or `BLOCKED`.
