#!/usr/bin/env python3
"""Compare ONNX FP32 vs NPU uint8 output for Korean Wav2Vec2."""
import numpy as np
import json
import onnxruntime as ort

# Load vocab
with open("/home/nsbb/travail/wav2vec/w2v_v.1.0.0_onnx/vocab.json") as f:
    vocab_raw = json.load(f)
vocab = {v: k for k, v in vocab_raw.items()}
pad_id = vocab_raw.get("<pad>", 2616)

SEQ_LEN = 149
VOCAB_SIZE = 2617
NPU_SCALE = 0.219490
NPU_ZP = 122

# ONNX inference
audio = np.load("test_audio.npy")
sess = ort.InferenceSession("wav2vec2_ko_3s.onnx")
onnx_logits = sess.run(None, {"input": audio})[0][0]  # [149, 2617]

# NPU output
npu_data = np.fromfile("output_0.dat", dtype=np.uint8).reshape(SEQ_LEN, VOCAB_SIZE)
npu_logits = (npu_data.astype(np.float32) - NPU_ZP) * NPU_SCALE

print("=== ONNX FP32 vs NPU uint8 comparison ===\n")

# Decode both
def ctc_decode(logits_2d):
    tokens = np.argmax(logits_2d, axis=1)
    deduped = [tokens[0]]
    for t in tokens[1:]:
        if t != deduped[-1]:
            deduped.append(t)
    text = ''.join(vocab.get(int(t), '') for t in deduped if int(t) not in {0, pad_id})
    return text.replace('|', ' '), tokens

onnx_text, onnx_tokens = ctc_decode(onnx_logits)
npu_text, npu_tokens = ctc_decode(npu_logits)

print(f"ONNX FP32: '{onnx_text}'")
print(f"NPU uint8: '{npu_text}'")

# Compare specific positions
print(f"\n=== Position-by-position comparison ===")
onnx_argmax = np.argmax(onnx_logits, axis=1)
npu_argmax = np.argmax(npu_logits, axis=1)

agree = np.sum(onnx_argmax == npu_argmax)
print(f"Argmax agreement: {agree}/{SEQ_LEN} ({100*agree/SEQ_LEN:.1f}%)")

# Find positions where ONNX has non-pad tokens
non_pad_positions = np.where(onnx_argmax != pad_id)[0]
print(f"\nONNX non-pad positions ({len(non_pad_positions)}):")
for pos in non_pad_positions:
    onnx_top = np.argsort(-onnx_logits[pos])[:5]
    npu_top = np.argsort(-npu_logits[pos])[:5]
    onnx_char = vocab.get(int(onnx_argmax[pos]), '?')
    npu_char = vocab.get(int(npu_argmax[pos]), '?')

    # Gap between 1st and 2nd in ONNX
    onnx_gap = onnx_logits[pos, onnx_top[0]] - onnx_logits[pos, onnx_top[1]]
    npu_gap = npu_logits[pos, npu_top[0]] - npu_logits[pos, npu_top[1]]

    print(f"  Pos {pos:3d}: ONNX='{onnx_char}'({onnx_argmax[pos]}) gap={onnx_gap:.2f} | "
          f"NPU='{npu_char}'({npu_argmax[pos]}) gap={npu_gap:.2f}")

    # Show pad logit comparison
    onnx_pad_val = onnx_logits[pos, pad_id]
    npu_pad_val = npu_logits[pos, pad_id]
    onnx_best_val = onnx_logits[pos, onnx_argmax[pos]]
    npu_pad_minus_best = npu_pad_val - npu_logits[pos, npu_top[1] if npu_top[0] == pad_id else npu_top[0]]
    print(f"         ONNX: best={onnx_best_val:.2f}, pad={onnx_pad_val:.2f} | "
          f"NPU: pad={npu_pad_val:.2f}, 2nd={npu_logits[pos, npu_top[1]]:.2f}")

# Overall logit range comparison
print(f"\n=== Logit range comparison ===")
print(f"ONNX: [{onnx_logits.min():.2f}, {onnx_logits.max():.2f}]")
print(f"NPU:  [{npu_logits.min():.2f}, {npu_logits.max():.2f}]")
print(f"ONNX pad logit mean: {onnx_logits[:, pad_id].mean():.2f}")
print(f"NPU pad logit mean:  {npu_logits[:, pad_id].mean():.2f}")
print(f"ONNX non-pad logit mean: {np.delete(onnx_logits, pad_id, axis=1).mean():.4f}")
print(f"NPU non-pad logit mean:  {np.delete(npu_logits, pad_id, axis=1).mean():.4f}")
