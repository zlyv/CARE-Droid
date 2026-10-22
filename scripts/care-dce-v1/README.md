# DCE-V1

Frozen Direct-C43 113-D pre-head tensor, 2017-only fixed fivefold protocol.
Stage-I training directions are generated inside each outer training fold
with four-way cross-fitting and nested three-way SVM calibration. The final
full-2017 head uses the historical fivefold source OOF Stage-I probabilities.

D0 is the frozen Direct-C43 checkpoint. D1 uses two always-active linear
adapters (228 trainable parameters); D2/D3 use two direction-selected linear
adapters (228); D4 uses one 114-to-2 linear adapter summed to a scalar (230).
The pretrained C43 encoder **and its original shared classifier** are frozen.
All new arms predict a direct malware logit, not a bounded Stage-I correction.

No APK is opened, installed, rewritten, or executed. All writes are restricted
to `DCE_OUTPUT`. Future labels are masked until final-2017 freezing succeeds.
