# LCC-FA v2: exceptional CFG and cyclic control observability

Independent, label-free audit of the same 200 source-2017 P31-SVM MeanOpposition
Top25% APKs as v1. No model training, conditional probe or APK execution.

## Engineering changes

- Decode DEX try/handler tables: PC code units become bytes; handler_off stays bytes.
- Add every declared handler for a protected potentially throwing instruction.
  Keep exceptional EXIT even for catch-all; exception types are not resolved.
- Cyclic CFGs use strong postdominance via a least fixed point over all maximal
  paths. Infinite paths avoiding a call invalidate that postdominance claim.
- Erase permission symbols on exceptional edges. Cyclic permission def-use
  remains unsupported/PARTIAL. Unknown is never NO_GUARD or a class label.
- Export matched event transitions and permission/unknown/partial diagnostics.

The Host set, Role mapping, event identity, strict control definition, permission
contract, route, folds and EVERY v1 coverage threshold remain unchanged. No
entry recovery, callbacks, full taint, new condition taxonomy, classifier, gate,
future data or correction rule is added. No APK is rewritten or re-signed.

## Prerequisites and protection

Requires completed v1 output and its scan caches for label-free comparison.
Preflight verifies identical selection and thresholds. Writes are restricted to
`outputs/care-local-control-constraint-feasibility-v2`. Original APKs, formal
inputs and v1 reports/caches are hashed before and after. v1 caches are never
reused as v2 results; v2 resume requires matching engine and protocol hashes.

Upload `care-local-control-constraint-feasibility-v2-package-r1.zip` to
`/home/boyuanjie/projects/`, then run:

```bash
cd /home/boyuanjie/projects/tasa-droid
if [ -e scripts/care-local-control-constraint-feasibility-v2 ] ||
   [ -e outputs/care-local-control-constraint-feasibility-v2 ]; then
  echo "v2 already exists: do not overwrite; inspect it or resume its launcher"
else
  unzip -q ../care-local-control-constraint-feasibility-v2-package-r1.zip -d scripts/ &&
  chmod +x scripts/care-local-control-constraint-feasibility-v2/*.sh &&
  LCC_WORKERS=4 bash scripts/care-local-control-constraint-feasibility-v2/launch_tmux.sh
fi
tail -F outputs/care-local-control-constraint-feasibility-v2.console.log
```

The existence guard does not exit the interactive shell. Do not kill unrelated
tmux sessions or remove historical outputs. Optional `LCC_V1_OUTPUT` identifies
an existing v1 directory; the launcher forwards it to workers. No GPU, Git,
downloads or installation of new dependencies are required.

After completion:

```bash
bash scripts/care-local-control-constraint-feasibility-v2/collect_results.sh
```

`CARE_LCC_FEASIBILITY_DONE` means the audit completed, including negative gate
results. `CARE_LCC_FEASIBILITY_FAILED` means a technical failure; inspect the
explicit reason before resuming. The package never starts supervised training.

## Timing and limits

v1 measured about 23 minutes of summed per-APK scan time, NOT parallel elapsed
time. v2 adds fixed-point work and v1-cache hashes; actual wall time must be
measured again. Start with four CPU workers. Per-APK timeout: 1200 seconds;
method cap: 4096 instructions; fixed-point cap: 2,500,000 updates per analysis.
Failures/timeouts/caps remain in all denominators.

## New outputs

- `v1_v2_comparison.json`: changes in coverage, inventory and verified facts.
- `v1_v2_event_transitions.csv`: every old/new event, without labels/predictions.
- `method_cfg_audit.csv`: exception regions, cycles, work counts, def-use reasons.
- Existing coverage/fold/event/proof tables, final decision, report and manifests.

Passing authorizes only specification of a separate incremental-information
experiment. Failing freezes this version. More parsed CFGs alone do not prove
class direction, improved detection, consent, runtime feasibility or whole-
program semantic equivalence. This package does not establish method novelty.

## References

- https://source.android.com/docs/core/runtime/dex-format
- https://source.android.com/docs/core/runtime/dalvik-bytecode
- https://www.sable.mcgill.ca/soot/doc/soot/toolkits/graph/ExceptionalUnitGraph.html
