import numpy as np
from pathlib import Path

path = Path("/home/boyuanjie/projects/tasa-droid/outputs/care_crossyear_layerwise_diagnosis_v2/embeddings/T3_C1/validation.npz")
with np.load(path, allow_pickle=False) as data:
    for key in data.files:
        print(key, data[key].shape, data[key].dtype)
