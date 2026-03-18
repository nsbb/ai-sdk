#!/usr/bin/env python3
"""Export English wav2vec2-base-960h as 3s ONNX (opset 12, eager attention)."""
import torch
import numpy as np
from transformers import Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-base-960h"
OUTPUT = "wav2vec2_en_3s.onnx"
SEQ_LEN = 48000  # 3 seconds @ 16kHz

print(f"Loading {MODEL_ID} with eager attention...")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, attn_implementation="eager")
model.eval()

dummy = torch.randn(1, SEQ_LEN)
print(f"Input shape: {dummy.shape}")

with torch.no_grad():
    out = model(dummy)
    print(f"Output shape: {out.logits.shape}")  # [1, 149, 32]

print(f"Exporting to {OUTPUT} (opset 12)...")
torch.onnx.export(
    model,
    dummy,
    OUTPUT,
    opset_version=12,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes=None,
    do_constant_folding=True,
)
print(f"Done: {OUTPUT}")

# Verify
import onnxruntime as ort
sess = ort.InferenceSession(OUTPUT)
inp = np.random.randn(1, SEQ_LEN).astype(np.float32)
out = sess.run(None, {"input_values": inp})
print(f"ONNX verify: input {inp.shape} -> output {out[0].shape}, dtype={out[0].dtype}")
