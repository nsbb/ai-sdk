#!/usr/bin/env python3
"""Verify Acuity FP32 simulation vs ONNX FP32."""
import numpy as np
import json
import onnxruntime as ort

with open("/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/vocab.json") as f:
    vocab_raw = json.load(f)
vocab = {v: k for k, v in vocab_raw.items()}
pad_id = vocab_raw.get("<pad>", 2616)

SEQ_LEN = 149
VOCAB_SIZE = 2617

def ctc_decode(logits_2d):
    tokens = np.argmax(logits_2d, axis=1)
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    text = ''.join(vocab.get(int(t), '') for t in deduped if int(t) not in {0, pad_id})
    return text.replace('|', ' '), tokens

# Acuity FP32 (text format, one float per line)
import sys
tensor_dir = sys.argv[1] if len(sys.argv) > 1 else "inf_float"
acuity_vals = np.loadtxt(f"{tensor_dir}/iter_0_attach_Add__lm_head_Add_out0_0_out0_1_149_2617.tensor")
print(f"Acuity tensor: {len(acuity_vals)} values ({len(acuity_vals) / VOCAB_SIZE:.1f} x {VOCAB_SIZE})")

# Could be either [149, 2617] or [2617, 149] flattened
# 389932 lines but we need 389933 (149*2617). Off by one — check last line
if len(acuity_vals) == SEQ_LEN * VOCAB_SIZE - 1:
    acuity_vals = np.append(acuity_vals, 0.0)  # Pad last value

acuity_logits = acuity_vals[:SEQ_LEN * VOCAB_SIZE].reshape(SEQ_LEN, VOCAB_SIZE)
acuity_text, acuity_tokens = ctc_decode(acuity_logits)
print(f'Acuity FP32 [seq,vocab]: "{acuity_text}"')
non_pad = sum(1 for t in acuity_tokens if t != pad_id)
print(f'  Non-pad tokens: {non_pad}/{len(acuity_tokens)}')

# Try transposed
acuity_logits_t = acuity_vals[:SEQ_LEN * VOCAB_SIZE].reshape(VOCAB_SIZE, SEQ_LEN).T
acuity_text_t, acuity_tokens_t = ctc_decode(acuity_logits_t)
print(f'Acuity FP32 [vocab,seq]: "{acuity_text_t}"')
non_pad_t = sum(1 for t in acuity_tokens_t if t != pad_id)
print(f'  Non-pad tokens: {non_pad_t}/{len(acuity_tokens_t)}')

# ONNX FP32
audio = np.load("test_audio.npy")
sess = ort.InferenceSession("wav2vec2_ko_3s.onnx")
onnx_logits = sess.run(None, {"input": audio})[0][0]
onnx_text, onnx_tokens = ctc_decode(onnx_logits)
print(f'\nONNX FP32:   "{onnx_text}"')

# Compare whichever layout matches better
for name, logits in [("[seq,vocab]", acuity_logits), ("[vocab,seq]", acuity_logits_t)]:
    max_diff = np.abs(logits - onnx_logits).max()
    mean_diff = np.abs(logits - onnx_logits).mean()
    agree = np.sum(np.argmax(logits, axis=1) == onnx_tokens)
    print(f'\n{name} vs ONNX: max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}, agree={agree}/{SEQ_LEN}')
