# CARE-Droid Signed Host MIL v1

Strict 2017 source-only comparison of the frozen SVM/C43 line with:

- M1: ordinary attention-based MIL;
- M2: signed malware/benign evidence MIL;
- M3: M2 plus CLAM-style weak instance constraints;
- N1: count-only;
- N2: outer-train label shuffle;
- N3: fold/Stage-I-side/Host-count-matched bag shuffle.

All models receive only the first 43 columns of the frozen A0 Host cache. They
train on all outer-training APK bags and are evaluated OOF. Routed-only training
is prohibited. Future rows are never loaded.

The source direction gate runs before bounded correction. If it fails, the
pipeline stops scientifically: it still writes the complete negative probe but
does not manufacture an end-to-end improvement. Checkpoints make the 20 matched
null repetitions resumable.
