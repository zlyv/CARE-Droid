# CARE-Droid final instrumentation replay

This is the last approved numerically-equivalent replay of the frozen
`CARE-Droid-DI25-HostEgo-Anomaly-Bounded` same-year OOF experiment.  It does
not define or train a new method.  It persists all three seed outputs, routed
Host internals, and reloadable inference checkpoints.

The launcher first runs the fixed `2018/fold 1` dry run.  Full replay starts
only after three independent seed outputs and checkpoint reload pass.
