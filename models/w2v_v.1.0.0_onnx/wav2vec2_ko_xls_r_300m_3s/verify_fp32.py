#!/usr/bin/env python3
"""Verify Acuity FP32 simulation vs ONNX FP32."""
import numpy as np
import json
import onnxruntime as ort

with open("/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/vocab.json") as f:
    vocab_raw = json.load(f)
vocab = {v: k for k, v in vocab_raw.items()}
pad_id = vocab_raw.get("<pad>", 2616)

def ctc_decode(logits_2d):
    tokens = np.argmax(logits_2d, axis=1)
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    text = ''.join(vocab.get(int(t), '') for t in deduped if int(t) not in {0, pad_id})
    return text.replace('|', ' '), tokens

# Acuity FP32
acuity_data = np.fromfile(
    "inf_float/iter_0_attach_Add__lm_head_Add_out0_0_out0_1_149_2617.tensor",
    dtype=np.float32
).reshape(149, 2617)
acuity_text, acuity_tokens = ctc_decode(acuity_data)
print(f'Acuity FP32: "{acuity_text}"')
print(f'  Non-pad: {sum(1 for t in acuity_tokens if t != pad_id)}/{len(acuity_tokens)}')

# ONNX FP32
audio = np.load("test_audio.npy")
sess = ort.InferenceSession("wav2vec2_ko_3s.onnx")
onnx_logits = sess.run(None, {"input": audio})[0][0]
onnx_text, onnx_tokens = ctc_decode(onnx_logits)
print(f'ONNX FP32:   "{onnx_text}"')
print(f'  Non-pad: {sum(1 for t in onnx_tokens if t != pad_id)}/{len(onnx_tokens)}')

# Comparison
max_diff = np.abs(acuity_data - onnx_logits).max()
mean_diff = np.abs(acuity_data - onnx_logits).mean()
agree = np.sum(acuity_tokens == onnx_tokens)
print(f'\nAcuity vs ONNX:')
print(f'  Max diff: {max_diff:.6f}')
print(f'  Mean diff: {mean_diff:.6f}')
print(f'  Argmax agreement: {agree}/{len(acuity_tokens)}')

# Now run uint8 simulation
try:
    acuity_uint8 = np.fromfile(
        "inf_uint8/iter_0_attach_Add__lm_head_Add_out0_0_out0_1_149_2617.tensor",
        dtype=np.float32
    ).reshape(149, 2617)
    uint8_text, uint8_tokens = ctc_decode(acuity_uint8)
    print(f'\nAcuity uint8: "{uint8_text}"')
    print(f'  Non-pad: {sum(1 for t in uint8_tokens if t != pad_id)}/{len(uint8_tokens)}')
except FileNotFoundError:
    print("\nAcuity uint8 simulation not run yet")
