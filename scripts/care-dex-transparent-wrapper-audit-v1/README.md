# CARE-Droid DEX transparent-wrapper audit v1

This package performs a 30-APK, source-2017-only, label-free feasibility audit of conservative DEX transparent-wrapper normalization.

It does not train a classifier, rewrite an APK, modify the formal CARE-Droid pipeline, read future cohorts, or claim whole-program semantic equivalence. The only possible positive decision is `GO_FULL_200_AUDIT`, which authorizes coverage expansion under the same frozen rules—not model integration.

## Strict contract

`VERIFIED_STRICT` requires one exact-dispatch invoke (`static`, `direct`, or `super`), straight-line bytecode, exact input-register forwarding, direct return propagation, and no operation other than register moves, move-result, nop and return. Virtual/interface calls are reported separately as `DECLARED_TARGET_ONLY` and never count toward the strict gate.

## Server commands

```bash
cd /home/boyuanjie/projects/tasa-droid

test ! -e scripts/care-dex-transparent-wrapper-audit-v1 || {
  echo "脚本目录已存在，停止以避免覆盖"
  exit 1
}

test ! -e outputs/care-dex-transparent-wrapper-audit-v1 || {
  echo "输出目录已存在，停止以避免覆盖"
  exit 1
}

unzip -q ../care-dex-transparent-wrapper-audit-v1-package-r2.zip -d scripts/
chmod +x scripts/care-dex-transparent-wrapper-audit-v1/*.sh
bash scripts/care-dex-transparent-wrapper-audit-v1/launch_tmux.sh
tail -F outputs/care-dex-transparent-wrapper-audit-v1.console.log
```

After completion:

```bash
bash scripts/care-dex-transparent-wrapper-audit-v1/collect_results.sh
```

Expected runtime is primarily APK parsing and hashing. On the server that previously built 200 full DEX graphs in minutes, the 30-APK pilot should normally finish in roughly 5–20 minutes; large or malformed APKs may extend this. No GPU is used.

## Optional explicit paths

Only use these if automatic discovery fails:

```bash
TW_APK_INDEX=/absolute/path/apk_paths.json \
TW_MAPPING=/absolute/path/semantic_mapping_v1.yaml \
bash scripts/care-dex-transparent-wrapper-audit-v1/launch_tmux.sh
```

