# RCDB-V1

2017-only, fixed-fivefold Direct-C43 experiment. The 2018–2021 cohorts are
read only after the full-2017 models have been frozen. This code never opens
or executes APKs; it uses the previously audited feature cache.

`M2_RCDB` and `M3_BALANCED_PROTOTYPE` use the same 12+12 Role quotas,
candidate KMeans, nonnegative one-slot response adapter, Direct-C43 network,
and optimizer. They differ only in candidate selection: M2 uses APK-level
class contrast and Shared nonredundancy; M3 chooses highest own-class coverage.

The adapter maps a signed distance contrast to a nonnegative scalar using
train-only calibration. Thus the derived Residual remains algebraically valid,
but it is **not** the original external-distance residual and must not be
interpreted as a standalone benign certificate.

No historical model, cache, bank, APK, or script is modified.
