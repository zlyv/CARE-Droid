# Implementation verification, 2026-10-14

11 local tests passed with `D:/anaconda/envs/maskdroid/python.exe` (Python 3.10). All 8 Python files parsed successfully. Tests verify fixed source/future/within seeds, unchanged 4:1 weighting and class/prior permutation controls, per-year/pooled/macro tables, all 80 within-year arm/fold rows, rejecting altered deployment routes or missing arms, and zero paired intervals for identical controls.

A synthetic four-arm callback checks disjoint inner fit/hold IDs, training/test SHA separation, nested calibrated routing, identical B0 epoch counts and optimizer exposure for the three controls, and that only training labels enter these fits. A replay check rejects both numerical drift and a threshold flip within numerical tolerance. The within-year adapter test executes its installer and parser wrappers to ensure a Direct head with keep/fix 1/1 and consistency 0. The live server callback also checks the Direct head's declared correction semantics before fitting.

These tests use synthetic data and a fake neural backend. No actual future-year performance has been computed locally. Local PyTorch cannot load its DLL, and the local Bash runtime has a signal-pipe permission issue. The server pipeline performs shell syntax checks, selftests, asset checks and a true historical frozen B0 neural replay before weighted future fitting. No claim of successful GPU or runtime dependency verification is made by this local artifact.

Original source-v2 files and formal CARE-Droid files were read-only. The uploaded diagnostic depends on those server files and checks source-v2 implementation hashes against the completed source manifest. No APK extraction or execution is part of this package.
