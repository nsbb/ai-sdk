#!/usr/bin/env python3
"""Download wav2vec2-base-korean and export to fixed-shape ONNX."""
import torch
import numpy as np
import json
import os

MODEL_ID = "."  # Load from local directory (was "Kkonjeong/wav2vec2-base-korean")
INPUT_LENGTH = 48000  # 3 seconds @ 16kHz
OUTPUT_NAME = "wav2vec2_ko_base_3s.onnx"

print(f"Downloading {MODEL_ID}...")
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

model.eval()

# Get vocab info
vocab = processor.tokenizer.get_vocab()
print(f"Vocab size: {len(vocab)}")
print(f"Sample vocab: {list(vocab.items())[:20]}")

# Save vocab
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
print(f"Saved vocab.json ({len(vocab)} entries)")

# Save processor config
processor.save_pretrained("processor_config")
print("Saved processor config")

# Test inference
dummy_input = torch.randn(1, INPUT_LENGTH)
with torch.no_grad():
    output = model(dummy_input)
    logits = output.logits
    print(f"Input: [1, {INPUT_LENGTH}] -> Output: {logits.shape}")

# Export to ONNX
print(f"\nExporting to ONNX ({OUTPUT_NAME})...")
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_NAME,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {1: "sequence_length"},
        "logits": {1: "output_length"}
    },
    opset_version=14,
    do_constant_folding=True,
)
print(f"ONNX exported: {os.path.getsize(OUTPUT_NAME) / 1e6:.1f} MB")

# Verify ONNX
import onnxruntime as ort
sess = ort.InferenceSession(OUTPUT_NAME)
onnx_out = sess.run(None, {"input_values": dummy_input.numpy()})[0]
max_diff = np.abs(logits.numpy() - onnx_out).max()
print(f"PyTorch vs ONNX max diff: {max_diff:.6f}")

# Check output shape
print(f"\nONNX I/O:")
for i in sess.get_inputs():
    print(f"  Input: {i.name} {i.shape} {i.type}")
for o in sess.get_outputs():
    print(f"  Output: {o.name} {o.shape} {o.type}")

# Fix shape for Acuity
import onnx
onnx_model = onnx.load(OUTPUT_NAME)
for inp in onnx_model.graph.input:
    if inp.name == "input_values":
        dims = inp.type.tensor_type.shape.dim
        dims[0].dim_value = 1
        dims[1].dim_value = INPUT_LENGTH

output_seq_len = logits.shape[1]
for out in onnx_model.graph.output:
    if out.name == "logits":
        dims = out.type.tensor_type.shape.dim
        dims[0].dim_value = 1
        dims[1].dim_value = output_seq_len
        dims[2].dim_value = len(vocab)

onnx.save(onnx_model, OUTPUT_NAME)
print(f"\nFixed ONNX shape: input [1, {INPUT_LENGTH}] -> output [1, {output_seq_len}, {len(vocab)}]")

# Verify fixed shape
sess2 = ort.InferenceSession(OUTPUT_NAME)
test_input = np.zeros((1, INPUT_LENGTH), dtype=np.float32)
test_out = sess2.run(None, {"input_values": test_input})[0]
print(f"Verification: {test_out.shape}")
