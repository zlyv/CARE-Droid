# CARE-Droid ICDE Stage-A audit v1

This package performs the read-only infrastructure feasibility audit only.

- It does not install, launch, stop, clear, or uninstall APKs.
- It never selects a physical/unknown adb device.
- It does not start an emulator, change firewall rules, install SDKs, or create a network route.
- It hashes and statically validates source APKs and formal lineage.
- It distinguishes tool discovery from verified record/replay, isolation, attribution, and Role mapping capabilities.
- It never runs Git and writes only under a new `outputs/care-icde-source-v*` directory.

Stage B is prohibited unless all 14 Stage-A hard requirements pass. A later pilot package must be tailored to the audited sandbox rather than guessing an execution backend.

