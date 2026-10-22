# CARE-Droid LiteLocalSemantic source v1

Strict 2017-only diagnostic of method-local semantic relations without constructing an inter-method call graph.

The extractor scans DEX bytecode instructions and maps `invoke-*` targets through the frozen 31-role semantic mapping. It records role prevalence across sensitive methods, unordered role co-occurrence inside one method, and directed adjacent role transitions in bytecode order. Consecutive duplicate invoke-role events are collapsed before transitions are counted; multiple roles assigned to one invoke are treated as an unordered event rather than given an invented order.

Model arms:

- `P31`: role prevalence among sensitive methods.
- `COOC16`: 465 co-occurrence values compressed by fold-safe TruncatedSVD.
- `TRANS16`: 961 directed transitions compressed by fold-safe TruncatedSVD.
- `FUSED63`: P31 + COOC16 + TRANS16.

No APK size, method/class count, package name, certificate, SDK, year, Host, RoleGraph, Ego, future row, or Stage-II input enters a classifier. Size variables are retained only for shortcut audits.

Run from `/home/boyuanjie/projects/tasa-droid` after extracting this package into `scripts/`.

```bash
bash scripts/care-litelocal-semantic-source-v1/launch_tmux.sh
tail -F artifacts/care-litelocal-semantic-source-v1.console.log
```

For inner-OOF calibration, every SVD is fitted on the corresponding inner-fit subset; the final outer-validation representation is fitted only on outer-train. The first extraction is resumable through per-APK NPZ shards. No future run is launched automatically.
