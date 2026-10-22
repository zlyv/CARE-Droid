# CARE-Droid frozen M1 Stage-II mechanism diagnosis

This package reruns the already frozen same-year M1 method only to persist
forward-pass internals.  It must not be used to select or alter the method.

Run on the server with:

```bash
bash scripts/care-di25-hostego-stage2-mechanism-v1/launch_tmux.sh
tail -F artifacts/care-di25-hostego-stage2-mechanism-v1.console.log
```

The hard reproduction gate compares the newly produced probabilities with the
locked same-year M1 predictions at tolerance `1e-8`.  Analysis is not run if
that gate or the internal delta identities fail.

The reported `raw_delta` and `bounded_delta` are the ensemble-effective values:

`bounded_delta = logit(mean_seed_probability) - logit(p_stage1)`

`raw_delta = atanh(bounded_delta / 2.25)`

The actual seed-level outputs are also retained (`seed_*` columns).  Every
seed-level pair is separately checked against `bounded = 2.25*tanh(raw)`.
