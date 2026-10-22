# CFG Witness fixed-source future evaluation v2

Corrected temporal protocol:

- train Stage-I, router committee, Direct-C43 and CFG-Witness arms using 2017 only;
- keep the 0.5 decision threshold and Top25 MeanOpposition route frozen;
- evaluate the same frozen models independently on 2018, 2019, 2020 and 2021;
- do not perform rolling training or use future labels for preprocessing/routing;
- reuse existing T1 checkpoints and require exact 2018 replay before later-year inference.

This package reads cached features only. It does not read, install, execute or rewrite APKs.
