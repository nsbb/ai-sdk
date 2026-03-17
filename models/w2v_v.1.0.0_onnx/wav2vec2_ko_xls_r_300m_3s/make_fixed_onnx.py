#!/usr/bin/env python3
"""Convert dynamic-shape Korean Wav2Vec2 XLS-R-300M ONNX to fixed 3s shape."""
import onnx
from onnx import TensorProto, helper
import numpy as np
import sys

SRC = "/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/model.onnx"
DST = "wav2vec2_ko_3s.onnx"
INPUT_LENGTH = 48000  # 3 seconds @ 16kHz

print(f"Loading {SRC} ...")
model = onnx.load(SRC)

# Fix input shape
for inp in model.graph.input:
    if inp.name == "input":
        dims = inp.type.tensor_type.shape.dim
        dims[0].dim_value = 1
        dims[1].dim_value = INPUT_LENGTH
        print(f"Input fixed: {inp.name} -> [1, {INPUT_LENGTH}]")

# Fix output shape - output seq_len = 48000 / 320 = 150
for out in model.graph.output:
    if out.name == "output":
        dims = out.type.tensor_type.shape.dim
        dims[0].dim_value = 1
        dims[1].dim_value = 149
        dims[2].dim_value = 2617
        print(f"Output fixed: {out.name} -> [1, 150, 2617]")

print(f"Saving to {DST} ...")
onnx.save(model, DST)
import os
print(f"Done! Size: {os.path.getsize(DST) / 1e9:.2f} GB")

# Verify with onnxruntime
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(DST)
    dummy = np.zeros((1, INPUT_LENGTH), dtype=np.float32)
    out = sess.run(None, {"input": dummy})[0]
    print(f"Verification: input [1, {INPUT_LENGTH}] -> output {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
except Exception as e:
    print(f"Verification failed: {e}")
